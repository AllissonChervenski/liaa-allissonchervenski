"""Automated Test Suite for Hierarchical Coarse-to-Fine Pipeline (M11)."""

from __future__ import annotations

import numpy as np

from src.models.hierarchical_taxonomy import (
    NinaProDB2Taxonomy,
    OzdemirTaxonomy,
)
from src.models.train_hierarchical_coarse_fine import (
    HierarchicalCoarseFineClassifier,
    HierarchicalModelArtifacts,
    softmax,
)
from semg_baseline.lda import CanonicalLdaTrainer, PriorType
from semg_baseline.scaling import FeatureScaler
from src.models.train_cnn1d_quantized import QuantizedCnn1DArtifacts


def test_ninapro_taxonomy_consistency() -> None:
    """Validate 50-class mapping integrity for NinaPro DB2."""
    y = np.arange(50)
    y_macro = NinaProDB2Taxonomy.to_macro_labels(y)

    assert len(y_macro) == 50
    assert y_macro[0] == NinaProDB2Taxonomy.MACRO_REST
    assert np.all(y_macro[1:18] == NinaProDB2Taxonomy.MACRO_BASIC_WRIST)
    assert np.all(y_macro[18:41] == NinaProDB2Taxonomy.MACRO_FINE_GRASPS)
    assert np.all(y_macro[41:50] == NinaProDB2Taxonomy.MACRO_FORCE_PATTERNS)

    # Test fine specialist mapping
    mask_fine, y_fine = NinaProDB2Taxonomy.to_fine_specialist_labels(y)
    assert np.sum(mask_fine) == 23
    assert len(y_fine) == 23
    assert np.all(y_fine == np.arange(23))
    assert np.all(NinaProDB2Taxonomy.fine_local_to_global(y_fine) == np.arange(18, 41))

    # Test Sub-LDA mapping for E1
    mask_e1, y_e1 = NinaProDB2Taxonomy.to_sub_lda_labels(y, macro_id=1)
    assert np.sum(mask_e1) == 17
    assert len(y_e1) == 17
    for local_idx, global_expected in enumerate(range(1, 18)):
        assert NinaProDB2Taxonomy.sub_lda_local_to_global(local_idx, macro_id=1) == global_expected


def test_ozdemir_taxonomy_consistency() -> None:
    """Validate 10-class mapping integrity for Ozdemir et al."""
    y = np.arange(10)
    y_macro = OzdemirTaxonomy.to_macro_labels(y)

    assert len(y_macro) == 10
    assert y_macro[0] == OzdemirTaxonomy.MACRO_REST
    assert np.all(y_macro[[1, 2, 3, 4, 8, 9]] == OzdemirTaxonomy.MACRO_WRIST_POSTURES)
    assert np.all(y_macro[[5, 6, 7]] == OzdemirTaxonomy.MACRO_HAND_GRIP)

    # Test fine specialist mapping
    mask_fine, y_fine = OzdemirTaxonomy.to_fine_specialist_labels(y)
    assert np.sum(mask_fine) == 3
    assert np.all(OzdemirTaxonomy.fine_local_to_global(y_fine) == np.array([5, 6, 7]))


def test_softmax_numerical_stability() -> None:
    """Ensure softmax is normalized and numerically stable on extreme logits."""
    x = np.array([[1000.0, 1000.0], [-1000.0, -1000.0]])
    sm = softmax(x, axis=1)
    assert np.allclose(np.sum(sm, axis=1), 1.0)
    assert np.allclose(sm[0], [0.5, 0.5])
    assert np.allclose(sm[1], [0.5, 0.5])


def test_hierarchical_gating_and_fallback_option_a() -> None:
    """Test that low confidence (< 0.80) triggers Fallback Option A (Fine Specialist CNN)."""
    # Create synthetic test artifacts
    num_channels = 4
    num_aux = 5 * num_channels
    num_macro = 3

    scaler = FeatureScaler()
    dummy_x = np.random.randn(50, num_aux).astype(np.float32)
    scaler.fit(dummy_x)

    macro_lda = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_macro)
    y_macro_dummy = np.random.randint(0, num_macro, size=50)
    macro_lda.fit(scaler.transform(dummy_x), y_macro_dummy)

    # Dummy sub-LDA for wrist
    sub_lda = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=6)
    sub_lda.fit(scaler.transform(dummy_x), np.random.randint(0, 6, size=50))

    # Dummy quantized CNN artifacts
    cnn_art = QuantizedCnn1DArtifacts(
        dataset_name="test_ozdemir",
        subject_id="S1",
        in_channels=num_channels,
        num_classes=3,
        num_aux=num_aux,
        total_params=5000,
        in_mean=np.zeros(num_channels, dtype=np.float32),
        in_inv_std=np.ones(num_channels, dtype=np.float32),
        in_quant_scale=0.01,
        aux_mean=np.zeros(num_aux, dtype=np.float32),
        aux_inv_std=np.ones(num_aux, dtype=np.float32),
        aux_quant_scale=0.01,
        conv1_eff_scale=0.01,
        conv2_eff_scale=0.01,
        fc_eff_scale=0.01,
        aux_fc_eff_scale=0.01,
        conv1_weights_q=np.zeros((16, num_channels, 7), dtype=np.int8),
        conv1_bias_q=np.zeros(16, dtype=np.int32),
        conv2_weights_q=np.zeros((24, 16, 5), dtype=np.int8),
        conv2_bias_q=np.zeros(24, dtype=np.int32),
        fc_weights_q=np.zeros((3, 24), dtype=np.int8),
        fc_bias_q=np.zeros(3, dtype=np.int32),
        aux_fc_weights_q=np.zeros((3, num_aux), dtype=np.int8),
        fp32_top1_acc=0.8,
        fp32_bal_acc=0.8,
        fp32_macro_f1=0.8,
        int8_top1_acc=0.8,
        int8_bal_acc=0.8,
        int8_macro_f1=0.8,
    )

    artifacts = HierarchicalModelArtifacts(
        dataset_name="ozdemir",
        num_channels=num_channels,
        confidence_threshold=0.80,
        macro_lda_trainer=macro_lda,
        sub_ldas={1: sub_lda},
        fine_cnn_artifacts=cnn_art,
        taxonomy_name="ozdemir",
        scaler=scaler,
    )

    classifier = HierarchicalCoarseFineClassifier(artifacts, use_quantized_sim=True)

    # Mock high certainty rest:
    classifier.macro_b = np.array([100.0, -100.0, -100.0], dtype=np.float32)
    classifier.macro_w = np.zeros_like(classifier.macro_w)
    raw_win = np.zeros((10, 400, num_channels), dtype=np.float32)
    aux_feat = np.zeros((10, num_aux), dtype=np.float32)

    preds, p_max, macro_preds, fast_path = classifier.predict_batch(raw_win, aux_feat)
    assert np.all(p_max >= 0.80)
    assert np.all(macro_preds == 0)
    assert np.all(preds == 0)
    assert np.all(fast_path)  # Rest is fast path

    # Mock low certainty with Fine Specialist in contention (P_max ~ 0.50 < 0.80) -> Fallback Option A
    classifier.macro_b = np.array([-10.0, 0.0, 0.0], dtype=np.float32)
    preds, p_max, macro_preds, fast_path = classifier.predict_batch(raw_win, aux_feat)
    assert np.all(p_max < 0.80)
    assert not np.any(fast_path)  # Deep path taken when Fine Specialist is in contention
    assert np.all(np.isin(preds, [5, 6, 7]))  # Dispatched to Fine Specialist classes!
