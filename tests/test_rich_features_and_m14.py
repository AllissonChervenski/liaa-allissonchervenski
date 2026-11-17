"""Unit tests for EnhancedSemgFeatureExtractor and M14 Rich SupCon MLP."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.enhanced_features import EnhancedSemgFeatureExtractor
from src.models.train_rich_supcon_mlp import train_and_quantize_rich_supcon_mlp


def test_enhanced_feature_dimensions() -> None:
    """Validate total feature dimension across 12CH, 4CH, and 2CH topologies."""
    ext12 = EnhancedSemgFeatureExtractor(epsilon=np.ones(12) * 0.01, num_channels=12)
    assert ext12.total_dim == 636  # 12*42 + 66 + 66

    ext4 = EnhancedSemgFeatureExtractor(epsilon=np.ones(4) * 0.01, num_channels=4)
    assert ext4.total_dim == 180  # 4*42 + 6 + 6

    ext2 = EnhancedSemgFeatureExtractor(epsilon=np.ones(2) * 0.01, num_channels=2)
    assert ext2.total_dim == 86  # 2*42 + 1 + 1


def test_enhanced_feature_isolation_and_zero_variance() -> None:
    """Verify strict per-window isolation and finite output on flat/zero windows."""
    rng = np.random.default_rng(123)
    win = rng.standard_normal((4, 400, 12)).astype(np.float32)
    win[1] = 0.0  # Flat zero window

    ext = EnhancedSemgFeatureExtractor(epsilon=np.ones(12) * 0.01, num_channels=12)
    feats_a = ext.extract(win)
    assert feats_a.shape == (4, 636)
    assert np.all(np.isfinite(feats_a))

    # Mutate window 2 & 3 -> window 0 features must remain bit-identical
    win_mod = win.copy()
    win_mod[2:] *= 5.0
    feats_b = ext.extract(win_mod)
    np.testing.assert_array_equal(feats_a[0], feats_b[0])


def test_enhanced_feature_invalid_shapes() -> None:
    """Ensure shape mismatches raise ValueError."""
    with pytest.raises(ValueError, match="Expected epsilon shape"):
        EnhancedSemgFeatureExtractor(epsilon=[0.1, 0.2], num_channels=4)

    ext = EnhancedSemgFeatureExtractor(epsilon=[0.1, 0.1], num_channels=2)
    with pytest.raises(ValueError, match="Expected 3D windows"):
        ext.extract(np.zeros((10, 2)))


def test_m14_rich_supcon_mlp_pipeline_and_int8_parity() -> None:
    """Smoke test M14 SupCon training + BatchNorm folding + INT8 Q24 simulation."""
    rng = np.random.default_rng(99)
    n_tr, n_te = 60, 30
    c, num_cls = 4, 3

    tr_win = rng.standard_normal((n_tr, 400, c)).astype(np.float32)
    te_win = rng.standard_normal((n_te, 400, c)).astype(np.float32)
    tr_lbl = np.tile(np.arange(num_cls), n_tr // num_cls + 1)[:n_tr]
    te_lbl = np.tile(np.arange(num_cls), n_te // num_cls + 1)[:n_te]

    for i in range(n_tr):
        tr_win[i, :, tr_lbl[i]] += 2.0
    for i in range(n_te):
        te_win[i, :, te_lbl[i]] += 2.0

    ext = EnhancedSemgFeatureExtractor(epsilon=np.ones(c) * 0.05, num_channels=c)
    X_tr = ext.extract(tr_win)
    X_te = ext.extract(te_win)

    art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_rich_supcon_mlp(
        dataset_name="synthetic_m14",
        subject_id="S0",
        num_channels=c,
        X_tr=X_tr,
        tr_lbl=tr_lbl,
        X_te=X_te,
        te_lbl=te_lbl,
        num_classes=num_cls,
        hidden1=64,
        hidden2=32,
        supcon_epochs=2,
        finetune_epochs=2,
        batch_size=16,
        seed=42,
    )

    assert art.in_features == 180
    assert fp32_p.shape == (n_te,)
    assert int8_p.shape == (n_te,)
    assert fp32_l.shape == (n_te, num_cls)
    assert int8_l.shape == (n_te, num_cls)
    assert art.fp32_top1_acc >= 0.80
    assert art.int8_top1_acc >= 0.80
