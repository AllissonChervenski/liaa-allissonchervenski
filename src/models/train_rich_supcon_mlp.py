#!/usr/bin/env python3
"""Train and INT8-Quantize M14: Feature-Rich SupCon Residual MLP (636 Features).

Combines the 636-dimensional multi-domain feature representation
(`EnhancedSemgFeatureExtractor`: Nonlinear TD, Hjorth, HOS, AR-4, 5-Band PSD,
EMGCC Cepstrum, 4-Level Haar DWT, 4x Sub-window Envelope, and Cross-Channel
Spatial Synergy) with Two-Stage Class-Balanced SupCon + Prototype Initialization
and BatchNorm-folded Symmetric INT8 Q24 simulation.

Designed for ESP32-S3:
  - Peak activation SRAM arena < 5 KB (4x–6x smaller than 2D/1D CNNs)
  - Classifier MACs ~160k (~0.12 ms classifier + ~0.27 ms DSP = ~0.39 ms total)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semg_baseline.lda import CanonicalLdaTrainer, PriorType  # noqa: E402
from semg_baseline.scaling import FeatureScaler  # noqa: E402
from src.models.train_cnn1d_quantized import (  # noqa: E402
    c_round_half_away,
    requantize_q24_numpy,
)
from src.models.train_cnn2d_circular import (  # noqa: E402
    fold_conv_bn_weights,
    set_deterministic_seed,
)
from src.models.train_supcon_cnn2d import SupConLoss, SupConProjectionHead  # noqa: E402


class RichFeatureResNetMlp(nn.Module):
    """Two-stage BatchNorm Residual MLP with Canonical LDA Skip Projection."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden1: int = 192,
        hidden2: int = 128,
        dropout1: float = 0.25,
        dropout2: float = 0.20,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)

        self.fc1 = nn.Linear(in_features, hidden1, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.drop1 = nn.Dropout(p=dropout1)

        self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.drop2 = nn.Dropout(p=dropout2)

        self.head = nn.Linear(hidden2, num_classes, bias=True)
        self.skip = nn.Linear(in_features, num_classes, bias=False)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hidden representations (h1, h2) for SupCon and PTQ calibration."""
        h1 = self.drop1(F.relu(self.bn1(self.fc1(x))))
        h2 = self.drop2(F.relu(self.bn2(self.fc2(h1))))
        return h1, h2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h2 = self.forward_features(x)
        return self.head(h2) + self.skip(x)


@dataclass(frozen=True)
class QuantizedRichMlpArtifacts:
    """Folded symmetric INT8 parameters and calibration scales for M14 Rich MLP."""

    dataset_name: str
    subject_id: str
    num_channels: int
    in_features: int
    hidden1: int
    hidden2: int
    num_classes: int
    total_params: int
    feat_mean: np.ndarray
    feat_inv_std: np.ndarray
    in_quant_scale: float

    fc1_w_q: np.ndarray
    fc1_b_q: np.ndarray
    fc1_eff_scale: float

    fc2_w_q: np.ndarray
    fc2_b_q: np.ndarray
    fc2_eff_scale: float

    head_w_q: np.ndarray
    head_b_q: np.ndarray
    head_eff_scale: float

    skip_w_q: np.ndarray
    skip_eff_scale: float

    fp32_top1_acc: float
    fp32_bal_acc: float
    fp32_macro_f1: float
    int8_top1_acc: float
    int8_bal_acc: float
    int8_macro_f1: float


def quantize_features_rich_mlp(
    raw_features: np.ndarray,
    art: QuantizedRichMlpArtifacts,
) -> np.ndarray:
    """Standardize and symmetrically quantize feature matrix (N, F) -> int8 (N, F)."""
    z = (raw_features.astype(np.float32) - art.feat_mean[None, :]) * art.feat_inv_std[None, :]
    inv_s = np.float32(1.0 / art.in_quant_scale)
    return np.clip(c_round_half_away(z * inv_s), -127, 127).astype(np.int8)


def simulate_c99_int8_rich_mlp(
    art: QuantizedRichMlpArtifacts,
    x_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact INT8/Q24 simulation of RichFeatureResNetMlp matching C99 target."""
    m1_q24 = int(c_round_half_away(art.fc1_eff_scale * float(1 << 24)).item())
    m2_q24 = int(c_round_half_away(art.fc2_eff_scale * float(1 << 24)).item())
    m_skip_q16 = int(c_round_half_away(art.skip_eff_scale * 65536.0).item())

    x_i32 = x_q.astype(np.int32)

    # Layer 1: Linear + ReLU + Q24 requantization
    acc1 = x_i32 @ art.fc1_w_q.astype(np.int32).T + art.fc1_b_q.astype(np.int32)[None, :]
    h1_q = np.clip(requantize_q24_numpy(acc1, m1_q24), 0, 127).astype(np.int8)

    # Layer 2: Linear + ReLU + Q24 requantization
    acc2 = h1_q.astype(np.int32) @ art.fc2_w_q.astype(np.int32).T + art.fc2_b_q.astype(np.int32)[None, :]
    h2_q = np.clip(requantize_q24_numpy(acc2, m2_q24), 0, 127).astype(np.int8)

    # Head + Skip residual branch
    head_acc = h2_q.astype(np.int32) @ art.head_w_q.astype(np.int32).T + art.head_b_q.astype(np.int32)[None, :]
    skip_acc = x_i32 @ art.skip_w_q.astype(np.int32).T
    skip_scaled = ((skip_acc.astype(np.int64) * m_skip_q16 + 32768) >> 16).astype(np.int32)

    logits = head_acc + skip_scaled
    preds = np.argmax(logits, axis=1).astype(np.int32)
    return preds, logits


def train_and_quantize_rich_supcon_mlp(
    dataset_name: str,
    subject_id: str,
    num_channels: int,
    X_tr: np.ndarray,
    tr_lbl: np.ndarray,
    X_te: np.ndarray,
    te_lbl: np.ndarray,
    num_classes: int,
    hidden1: int = 192,
    hidden2: int = 128,
    supcon_epochs: int = 20,
    finetune_epochs: int = 30,
    batch_size: int = 512,
    temperature: float = 0.10,
    seed: int = 42,
) -> tuple[QuantizedRichMlpArtifacts, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train M14 Feature-Rich SupCon MLP, fold BatchNorm, quantize to INT8, and evaluate."""
    set_deterministic_seed(seed)
    torch.set_num_threads(4)

    in_features = int(X_tr.shape[1])

    scaler = FeatureScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    sdict = scaler.to_dict()
    feat_mean = np.asarray(sdict["mean"], dtype=np.float32)
    feat_scale = np.asarray(sdict["scale"], dtype=np.float32)
    feat_inv_std = np.where(feat_scale != 0.0, 1.0 / feat_scale, 0.0).astype(np.float32)
    for z_idx in sdict["zero_variance_indices"]:
        if 0 <= z_idx < in_features:
            feat_inv_std[z_idx] = 0.0

    # Canonical LDA warm-start for skip branch
    lda = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_classes)
    lda.fit(X_tr_s, tr_lbl)
    lda_art = lda.get_model_artifacts()

    t_x_tr = torch.from_numpy(X_tr_s)
    t_y_tr = torch.from_numpy(tr_lbl.astype(np.int64))
    t_x_te = torch.from_numpy(X_te_s)

    model = RichFeatureResNetMlp(
        in_features=in_features,
        num_classes=num_classes,
        hidden1=hidden1,
        hidden2=hidden2,
    )
    proj = SupConProjectionHead(in_dim=hidden2, hidden_dim=128, out_dim=64)
    gen = torch.Generator().manual_seed(seed)
    n_tr = t_x_tr.shape[0]

    # =========================================================================
    # STAGE 1: Class-Balanced SupCon over Augmented Multi-Domain Feature Views
    # =========================================================================
    if supcon_epochs > 0:
        backbone_params = (
            list(model.fc1.parameters()) + list(model.bn1.parameters()) +
            list(model.fc2.parameters()) + list(model.bn2.parameters())
        )
        opt_sc = torch.optim.AdamW(backbone_params + list(proj.parameters()), lr=3.0e-3, weight_decay=2.0e-3)
        sched_sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sc, T_max=supcon_epochs)
        crit_sc = SupConLoss(temperature=temperature, balance_classes=True)

        model.train()
        proj.train()
        for ep in range(1, supcon_epochs + 1):
            perm = torch.randperm(n_tr, generator=gen)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n_tr, batch_size):
                b_idx = perm[start : start + batch_size]
                if len(b_idx) < 4:
                    continue
                bx = t_x_tr[b_idx]
                by = t_y_tr[b_idx]

                # Two stochastic feature-dropout + Gaussian perturbation views
                mask1 = (torch.rand(bx.shape, generator=gen) > 0.10).float()
                mask2 = (torch.rand(bx.shape, generator=gen) > 0.10).float()
                v1 = bx * mask1 + 0.08 * torch.randn(bx.shape, generator=gen)
                v2 = bx * mask2 + 0.08 * torch.randn(bx.shape, generator=gen)

                _, h2_1 = model.forward_features(v1)
                _, h2_2 = model.forward_features(v2)
                z1 = proj(h2_1)
                z2 = proj(h2_2)
                loss = crit_sc(torch.stack([z1, z2], dim=1), by)

                opt_sc.zero_grad()
                loss.backward()
                opt_sc.step()
                ep_loss += float(loss.item())
                n_batches += 1
            sched_sc.step()
            if ep % 5 == 0 or ep == supcon_epochs:
                print(
                    f"    [{dataset_name}][Stage1-SupCon] Epoch {ep:02d}/{supcon_epochs} | loss={ep_loss / max(1, n_batches):.4f}",
                    flush=True,
                )

    # =========================================================================
    # STAGE 2: Prototype Initialization + End-to-End Fine-Tuning
    # =========================================================================
    model.eval()
    with torch.no_grad():
        _, all_h2 = model.forward_features(t_x_tr)
        protos = torch.zeros((num_classes, hidden2), dtype=torch.float32)
        for cls_id in range(num_classes):
            c_mask = t_y_tr == cls_id
            if torch.any(c_mask):
                protos[cls_id] = F.normalize(all_h2[c_mask].mean(dim=0), dim=0) * 0.25
        model.head.weight.copy_(protos)
        model.head.bias.copy_(torch.from_numpy(lda_art.intercept_.astype(np.float32)))
        model.skip.weight.copy_(torch.from_numpy(lda_art.coef_.astype(np.float32)))

    opt_ft = torch.optim.AdamW(
        [
            {"params": list(model.fc1.parameters()) + list(model.bn1.parameters()) + list(model.fc2.parameters()) + list(model.bn2.parameters()), "lr": 1.5e-3, "weight_decay": 2.0e-3},
            {"params": model.head.parameters(), "lr": 3.0e-3, "weight_decay": 1.0e-2},
            {"params": model.skip.parameters(), "lr": 2.0e-4, "weight_decay": 1.0e-2},
        ]
    )
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=max(1, finetune_epochs))
    crit_ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.train()
    for ep in range(1, finetune_epochs + 1):
        perm = torch.randperm(n_tr, generator=gen)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, batch_size):
            b_idx = perm[start : start + batch_size]
            bx = t_x_tr[b_idx]
            by = t_y_tr[b_idx]
            bx_aug = bx + 0.06 * torch.randn(bx.shape, generator=gen)

            opt_ft.zero_grad()
            logits = model(bx_aug)
            loss = crit_ce(logits, by)
            loss.backward()
            opt_ft.step()
            ep_loss += float(loss.item())
            n_batches += 1
        sched_ft.step()
        if ep % 10 == 0 or ep == finetune_epochs:
            print(
                f"    [{dataset_name}][Stage2-FineTune] Epoch {ep:02d}/{finetune_epochs} | loss={ep_loss / max(1, n_batches):.4f}",
                flush=True,
            )

    # FP32 Evaluation
    model.eval()
    with torch.no_grad():
        fp32_logits = model(t_x_te).numpy()
    fp32_preds = np.argmax(fp32_logits, axis=1)

    fp32_acc = float(accuracy_score(te_lbl, fp32_preds))
    fp32_bal = float(balanced_accuracy_score(te_lbl, fp32_preds))
    fp32_f1 = float(f1_score(te_lbl, fp32_preds, average="macro", zero_division=0))

    # Fold BatchNorm layers & Calibrate Symmetric INT8 Scales
    with torch.no_grad():
        calib_idx = np.linspace(0, n_tr - 1, min(1500, n_tr), dtype=int)
        h1_cal, h2_cal = model.forward_features(t_x_tr[calib_idx])

        s_in = max(float(np.percentile(np.abs(X_tr_s[calib_idx]), 99.95)) / 127.0, 1e-6)
        s_h1 = max(float(np.percentile(h1_cal.numpy(), 99.95)) / 127.0, 1e-6)
        s_h2 = max(float(np.percentile(h2_cal.numpy(), 99.95)) / 127.0, 1e-6)

        w1_f, b1_f = fold_conv_bn_weights(
            model.fc1.weight.numpy(),
            model.fc1.bias.numpy(),
            model.bn1.weight.numpy(),
            model.bn1.bias.numpy(),
            model.bn1.running_mean.numpy(),
            model.bn1.running_var.numpy(),
        )
        s_w1 = max(float(np.max(np.abs(w1_f))) / 127.0, 1e-8)
        w1_q = np.clip(c_round_half_away(w1_f / s_w1), -127, 127).astype(np.int8)
        b1_q = c_round_half_away(b1_f / (s_in * s_w1)).astype(np.int32)
        fc1_eff = (s_in * s_w1) / s_h1

        w2_f, b2_f = fold_conv_bn_weights(
            model.fc2.weight.numpy(),
            model.fc2.bias.numpy(),
            model.bn2.weight.numpy(),
            model.bn2.bias.numpy(),
            model.bn2.running_mean.numpy(),
            model.bn2.running_var.numpy(),
        )
        s_w2 = max(float(np.max(np.abs(w2_f))) / 127.0, 1e-8)
        w2_q = np.clip(c_round_half_away(w2_f / s_w2), -127, 127).astype(np.int8)
        b2_q = c_round_half_away(b2_f / (s_h1 * s_w2)).astype(np.int32)
        fc2_eff = (s_h1 * s_w2) / s_h2

        wh_fp, bh_fp = model.head.weight.numpy(), model.head.bias.numpy()
        s_wh = max(float(np.max(np.abs(wh_fp))) / 127.0, 1e-8)
        wh_q = np.clip(c_round_half_away(wh_fp / s_wh), -127, 127).astype(np.int8)
        head_eff = s_h2 * s_wh
        bh_q = c_round_half_away(bh_fp / head_eff).astype(np.int32)

        wsk_fp = model.skip.weight.numpy()
        s_wsk = max(float(np.max(np.abs(wsk_fp))) / 127.0, 1e-8)
        wsk_q = np.clip(c_round_half_away(wsk_fp / s_wsk), -127, 127).astype(np.int8)
        skip_eff = (s_in * s_wsk) / head_eff

    folded_params = w1_q.size + b1_q.size + w2_q.size + b2_q.size + wh_q.size + bh_q.size + wsk_q.size

    art = QuantizedRichMlpArtifacts(
        dataset_name=dataset_name,
        subject_id=subject_id,
        num_channels=num_channels,
        in_features=in_features,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=num_classes,
        total_params=folded_params,
        feat_mean=feat_mean,
        feat_inv_std=feat_inv_std,
        in_quant_scale=float(s_in),
        fc1_w_q=w1_q,
        fc1_b_q=b1_q,
        fc1_eff_scale=float(fc1_eff),
        fc2_w_q=w2_q,
        fc2_b_q=b2_q,
        fc2_eff_scale=float(fc2_eff),
        head_w_q=wh_q,
        head_b_q=bh_q,
        head_eff_scale=float(head_eff),
        skip_w_q=wsk_q,
        skip_eff_scale=float(skip_eff),
        fp32_top1_acc=fp32_acc,
        fp32_bal_acc=fp32_bal,
        fp32_macro_f1=fp32_f1,
        int8_top1_acc=0.0,
        int8_bal_acc=0.0,
        int8_macro_f1=0.0,
    )

    x_q_te = quantize_features_rich_mlp(X_te, art)
    int8_preds, int8_logits = simulate_c99_int8_rich_mlp(art, x_q_te)

    int8_acc = float(accuracy_score(te_lbl, int8_preds))
    int8_bal = float(balanced_accuracy_score(te_lbl, int8_preds))
    int8_f1 = float(f1_score(te_lbl, int8_preds, average="macro", zero_division=0))

    art_final = replace(
        art,
        int8_top1_acc=int8_acc,
        int8_bal_acc=int8_bal,
        int8_macro_f1=int8_f1,
    )
    return art_final, fp32_preds, fp32_logits, int8_preds, int8_logits
