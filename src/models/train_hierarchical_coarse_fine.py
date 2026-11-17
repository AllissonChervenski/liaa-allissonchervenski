"""Train, Calibrate and Evaluate Hierarchical Coarse-to-Fine Architecture (M11).

Implements the two-stage Coarse-to-Fine pipeline with Confidence Gating (tau = 0.80)
and Fallback Option A (dispatch to Fine Specialist CNN-1D on uncertainty):
  - Stage 1: Macro-LDA (4 classes on NinaPro DB2, 3 classes on Ozdemir)
  - Stage 2:
      * Fast Path: Sub-LDA (E1 17 classes / E3 9 classes) or Rest (0)
      * Deep Path: FineSpecialistSemgCnn1D (E2 23 classes / Ozdemir 3 classes)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_quantization_and_channels import compute_evaluation_metrics  # noqa: E402
from semg_baseline.lda import CanonicalLdaTrainer, PriorType  # noqa: E402
from semg_baseline.scaling import FeatureScaler  # noqa: E402
from src.models.hierarchical_taxonomy import (  # noqa: E402
    NinaProDB2Taxonomy,
    OzdemirTaxonomy,
)
from src.models.train_cnn1d_quantized import (  # noqa: E402
    QuantizedCnn1DArtifacts,
    quantize_windows_for_c99,
    set_deterministic_seed,
    simulate_c99_int8_cnn1d,
    train_and_quantize_dataset,
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@dataclass
class HierarchicalModelArtifacts:
    """Container for all calibrated components of M11."""

    dataset_name: str
    num_channels: int
    confidence_threshold: float
    macro_lda_trainer: CanonicalLdaTrainer
    sub_ldas: dict[int, CanonicalLdaTrainer]
    fine_cnn_artifacts: QuantizedCnn1DArtifacts
    taxonomy_name: str
    scaler: FeatureScaler

    # Performance summary
    top1_accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    fast_path_ratio: float = 0.0
    deep_path_ratio: float = 0.0
    fine_accuracy: float = 0.0
    macro_accuracy: float = 0.0


class HierarchicalCoarseFineClassifier:
    """Inference engine for M11 Hierarchical Coarse-to-Fine Pipeline."""

    def __init__(
        self,
        artifacts: HierarchicalModelArtifacts,
        use_quantized_sim: bool = True,
    ) -> None:
        self.artifacts = artifacts
        self.use_quantized_sim = use_quantized_sim
        self.tau = artifacts.confidence_threshold
        self.taxonomy = (
            NinaProDB2Taxonomy if artifacts.taxonomy_name == "ninapro" else OzdemirTaxonomy
        )
        self.fine_macro_id = 2  # Macro 2 is Fine Grasps in both taxonomies

        # Extract Macro-LDA linear weights and intercepts
        macro_art = artifacts.macro_lda_trainer.get_model_artifacts()
        self.macro_w = macro_art.coef_.astype(np.float32)  # [n_macro, n_features]
        self.macro_b = macro_art.intercept_.astype(np.float32)  # [n_macro]

        # Extract Sub-LDAs
        self.sub_ldas: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for m_id, sub_trainer in artifacts.sub_ldas.items():
            sub_art = sub_trainer.get_model_artifacts()
            self.sub_ldas[m_id] = (
                sub_art.coef_.astype(np.float32),
                sub_art.intercept_.astype(np.float32),
            )

    def predict_batch(
        self,
        raw_windows: np.ndarray,
        aux_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels for a batch of windows.

        Args:
            raw_windows: Array of shape (N, 400, C)
            aux_features: Array of shape (N, 5 * C)

        Returns:
            preds: Array of shape (N,) with global predicted labels
            p_max: Array of shape (N,) with macro confidence
            macro_preds: Array of shape (N,) with predicted macro group
            fast_path_mask: Array of shape (N,) boolean, True if resolved by LDA/Rest
        """
        n_samples = len(raw_windows)
        aux_scaled = self.artifacts.scaler.transform(aux_features).astype(np.float32)

        # 1. Macro-LDA discriminant scores and probabilities
        macro_scores = aux_scaled @ self.macro_w.T + self.macro_b[None, :]  # [N, n_macro]
        macro_probs = softmax(macro_scores, axis=1)  # [N, n_macro]
        p_max = np.max(macro_probs, axis=1)  # [N]
        macro_preds = np.argmax(macro_probs, axis=1)  # [N]

        preds = np.zeros(n_samples, dtype=np.int64)
        fast_path_mask = np.zeros(n_samples, dtype=bool)

        # 2. Decision Logic with Option A (Fallback to Fine Specialist on low confidence)
        # Sort macro probabilities descending
        sorted_macro = np.argsort(-macro_probs, axis=1)

        # 2A. High Confidence (P_max >= tau)
        for i in range(n_samples):
            top1_m = sorted_macro[i, 0]
            top2_m = sorted_macro[i, 1]
            p1 = macro_probs[i, top1_m]

            if p1 >= self.tau:
                if top1_m == 0:
                    preds[i] = 0
                    fast_path_mask[i] = True
                elif top1_m == self.fine_macro_id:
                    # Deep Path: Fine Specialist
                    preds[i] = -1  # Placeholder to batch-evaluate
                    fast_path_mask[i] = False
                else:
                    # Sub-LDA
                    sub_w, sub_b = self.sub_ldas[top1_m]
                    sub_score = aux_scaled[i : i + 1] @ sub_w.T + sub_b[None, :]
                    local_p = int(np.argmax(sub_score, axis=1)[0])
                    preds[i] = self.taxonomy.sub_lda_local_to_global(local_p, top1_m)
                    fast_path_mask[i] = True
            else:
                # Ambiguous window (P_max < tau) -> Fallback Option A
                # If Fine Specialist (Macro 2) is a candidate (top1 or top2):
                if top1_m == self.fine_macro_id or top2_m == self.fine_macro_id:
                    preds[i] = -1  # Placeholder to batch-evaluate on Fine CNN
                    fast_path_mask[i] = False
                else:
                    # Ambiguity is strictly among non-fine groups
                    if top1_m == 0:
                        preds[i] = 0
                    else:
                        sub_w, sub_b = self.sub_ldas[top1_m]
                        sub_score = aux_scaled[i : i + 1] @ sub_w.T + sub_b[None, :]
                        local_p = int(np.argmax(sub_score, axis=1)[0])
                        preds[i] = self.taxonomy.sub_lda_local_to_global(local_p, top1_m)
                    fast_path_mask[i] = True

        # Batch-evaluate deep path windows on Fine Specialist CNN
        deep_indices = np.where(~fast_path_mask)[0]
        if len(deep_indices) > 0:
            raw_deep = raw_windows[deep_indices]
            aux_deep = aux_features[deep_indices]
            cnn_art = self.artifacts.fine_cnn_artifacts
            x_q, aux_q = quantize_windows_for_c99(
                raw_windows=raw_deep,
                aux_features=aux_deep,
                in_mean=cnn_art.in_mean,
                in_inv_std=cnn_art.in_inv_std,
                in_quant_scale=cnn_art.in_quant_scale,
                aux_mean=cnn_art.aux_mean,
                aux_inv_std=cnn_art.aux_inv_std,
                aux_quant_scale=cnn_art.aux_quant_scale,
            )
            local_preds, _ = simulate_c99_int8_cnn1d(cnn_art, x_q, aux_q)
            global_fine_preds = self.taxonomy.fine_local_to_global(local_preds)
            preds[deep_indices] = global_fine_preds

        return preds, p_max, macro_preds, fast_path_mask


def train_hierarchical_model(
    dataset_name: str,
    subject_id: str,
    train_win: np.ndarray,
    train_lbl: np.ndarray,
    test_win: np.ndarray,
    test_lbl: np.ndarray,
    X_tr_aux: np.ndarray,
    X_te_aux: np.ndarray,
    num_channels: int,
    confidence_threshold: float = 0.80,
    cnn_epochs: int = 35,
    seed: int = 42,
) -> tuple[HierarchicalModelArtifacts, np.ndarray]:
    """Train all components of M11 and evaluate on test pool."""
    set_deterministic_seed(seed)
    taxonomy_name = "ninapro" if "nina" in dataset_name.lower() else "ozdemir"
    taxonomy = NinaProDB2Taxonomy if taxonomy_name == "ninapro" else OzdemirTaxonomy

    # 1. Feature Scaler
    scaler = FeatureScaler()
    X_tr_aux_s = scaler.fit_transform(X_tr_aux)
    X_te_aux_s = scaler.transform(X_te_aux)

    # 2. Map labels to Macro
    y_tr_macro = taxonomy.to_macro_labels(train_lbl)
    y_te_macro = taxonomy.to_macro_labels(test_lbl)

    num_macro = (
        NinaProDB2Taxonomy.NUM_MACRO_CLASSES
        if taxonomy_name == "ninapro"
        else OzdemirTaxonomy.NUM_MACRO_CLASSES
    )

    # 3. Train Macro-LDA
    macro_lda = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_macro)
    macro_lda.fit(X_tr_aux_s, y_tr_macro)
    macro_preds_te = macro_lda.predict(X_te_aux_s)
    macro_acc = float(np.mean(macro_preds_te == y_te_macro))

    # 4. Train Sub-LDAs for non-fine macro groups
    sub_ldas: dict[int, CanonicalLdaTrainer] = {}
    fine_macro_id = 2

    for m_id, group in taxonomy.GROUPS.items():
        if m_id == 0 or m_id == fine_macro_id:
            continue
        mask_tr, y_tr_local = taxonomy.to_sub_lda_labels(train_lbl, macro_id=m_id)
        if len(y_tr_local) > 0 and len(np.unique(y_tr_local)) > 1:
            sub_trainer = CanonicalLdaTrainer(
                prior_type=PriorType.UNIFORM, num_classes=group.num_classes
            )
            sub_trainer.fit(X_tr_aux_s[mask_tr], y_tr_local)
            sub_ldas[m_id] = sub_trainer

    # 5. Train FineSpecialistSemgCnn1D
    mask_fine_tr, y_tr_fine = taxonomy.to_fine_specialist_labels(train_lbl)
    mask_fine_te, y_te_fine = taxonomy.to_fine_specialist_labels(test_lbl)

    fine_group = taxonomy.GROUPS[fine_macro_id]
    num_fine_classes = fine_group.num_classes

    fine_cnn_art, _, _, _, _ = train_and_quantize_dataset(
        dataset_name=f"{dataset_name}_fine",
        subject_id=subject_id,
        train_win=train_win[mask_fine_tr],
        train_lbl=y_tr_fine,
        test_win=test_win[mask_fine_te],
        test_lbl=y_te_fine,
        X_tr_aux=X_tr_aux[mask_fine_tr],
        X_te_aux=X_te_aux[mask_fine_te],
        num_classes=num_fine_classes,
        epochs=cnn_epochs,
        seed=seed,
    )

    # Evaluate Fine Specialist standalone accuracy
    x_q_fine, aux_q_fine = quantize_windows_for_c99(
        raw_windows=test_win[mask_fine_te],
        aux_features=X_te_aux[mask_fine_te],
        in_mean=fine_cnn_art.in_mean,
        in_inv_std=fine_cnn_art.in_inv_std,
        in_quant_scale=fine_cnn_art.in_quant_scale,
        aux_mean=fine_cnn_art.aux_mean,
        aux_inv_std=fine_cnn_art.aux_inv_std,
        aux_quant_scale=fine_cnn_art.aux_quant_scale,
    )
    fine_preds_local, _ = simulate_c99_int8_cnn1d(fine_cnn_art, x_q_fine, aux_q_fine)
    fine_acc = float(np.mean(fine_preds_local == y_te_fine))

    # 6. Build Artifacts
    artifacts = HierarchicalModelArtifacts(
        dataset_name=dataset_name,
        num_channels=num_channels,
        confidence_threshold=confidence_threshold,
        macro_lda_trainer=macro_lda,
        sub_ldas=sub_ldas,
        fine_cnn_artifacts=fine_cnn_art,
        taxonomy_name=taxonomy_name,
        scaler=scaler,
        macro_accuracy=macro_acc,
        fine_accuracy=fine_acc,
    )

    # 7. Evaluate Integrated M11 Pipeline
    classifier = HierarchicalCoarseFineClassifier(artifacts, use_quantized_sim=True)
    y_pred, p_max, macro_preds, fast_path_mask = classifier.predict_batch(test_win, X_te_aux)

    acc, bal_acc, macro_f1 = compute_evaluation_metrics(test_lbl, y_pred)
    artifacts.top1_accuracy = acc
    artifacts.balanced_accuracy = bal_acc
    artifacts.macro_f1 = macro_f1
    artifacts.fast_path_ratio = float(np.mean(fast_path_mask))
    artifacts.deep_path_ratio = 1.0 - artifacts.fast_path_ratio

    return artifacts, y_pred
