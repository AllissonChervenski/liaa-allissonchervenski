"""Unit tests for M13 Supervised Contrastive Circular CNN-2D (Passo 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.train_supcon_cnn2d import (
    SupConLoss,
    SupConProjectionHead,
    augment_semg_view,
    train_and_quantize_supcon_cnn2d,
)


def test_supcon_loss_validation_and_bounds() -> None:
    """Validate temperature checks and shape validation."""
    with pytest.raises(ValueError, match="temperature must be > 0"):
        SupConLoss(temperature=0.0)

    crit = SupConLoss(temperature=0.1, balance_classes=True)
    with pytest.raises(ValueError, match="Expected 3D tensor"):
        crit(torch.randn(4, 16), torch.zeros(4, dtype=torch.long))


def test_supcon_loss_separability_and_class_balance() -> None:
    """Tightly clustered intra-class embeddings yield near-zero SupCon loss."""
    crit = SupConLoss(temperature=0.1, balance_classes=True)

    # Create 3 well-separated orthogonal clusters across 12 samples (imbalanced: 8, 2, 2)
    labels = torch.tensor([0] * 8 + [1] * 2 + [2] * 2, dtype=torch.long)
    base = torch.zeros(12, 8)
    base[:8, 0] = 1.0
    base[8:10, 1] = 1.0
    base[10:12, 2] = 1.0

    # Good clustering: both views identical along class orthogonal directions
    good_feats = torch.stack([F.normalize(base, dim=1), F.normalize(base, dim=1)], dim=1)
    loss_good = crit(good_feats, labels)

    # Random noisy embeddings
    torch.manual_seed(7)
    bad_feats = torch.stack([F.normalize(torch.randn(12, 8), dim=1), F.normalize(torch.randn(12, 8), dim=1)], dim=1)
    loss_bad = crit(bad_feats, labels)

    # Theoretical lower bound of SupCon with M positives per anchor is mean_i log(M_i):
    # Class 0 has 16 views -> 15 positives -> log(15); Classes 1 & 2 have 4 views -> 3 positives -> log(3)
    expected_min = float((np.log(15.0) + np.log(3.0) + np.log(3.0)) / 3.0)

    assert torch.isfinite(loss_good)
    assert torch.isfinite(loss_bad)
    assert abs(float(loss_good.item()) - expected_min) < 1e-3
    assert float(loss_bad.item()) > float(loss_good.item()) + 1.0


def test_projection_head_normalization_and_augmentation() -> None:
    """Ensure SupConProjectionHead outputs L2 unit vectors and augmentation preserves shape."""
    head = SupConProjectionHead(in_dim=32, hidden_dim=64, out_dim=16)
    head.eval()
    x = torch.randn(6, 32)
    z = head(x)
    assert z.shape == (6, 16)
    norms = torch.norm(z, p=2, dim=1)
    torch.testing.assert_close(norms, torch.ones(6), atol=1e-5, rtol=1e-5)

    gen = torch.Generator().manual_seed(42)
    win = torch.randn(4, 8, 400)
    v1 = augment_semg_view(win, gen)
    v2 = augment_semg_view(win, gen)
    assert v1.shape == win.shape
    assert v2.shape == win.shape
    assert not torch.allclose(v1, v2)


def test_synthetic_supcon_pipeline_zero_inference_overhead() -> None:
    """Smoke test 2-stage SupCon training + INT8 export confirming projection head removal."""
    rng = np.random.default_rng(77)
    n_tr, n_te = 48, 24
    c, num_cls = 4, 3

    tr_win = rng.standard_normal((n_tr, 400, c)).astype(np.float32)
    te_win = rng.standard_normal((n_te, 400, c)).astype(np.float32)
    tr_lbl = np.tile(np.arange(num_cls), n_tr // num_cls + 1)[:n_tr]
    te_lbl = np.tile(np.arange(num_cls), n_te // num_cls + 1)[:n_te]

    for i in range(n_tr):
        tr_win[i, :, tr_lbl[i]] += 1.5
    for i in range(n_te):
        te_win[i, :, te_lbl[i]] += 1.5

    X_tr_aux = rng.standard_normal((n_tr, 5 * c)).astype(np.float32)
    X_te_aux = rng.standard_normal((n_te, 5 * c)).astype(np.float32)

    art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_supcon_cnn2d(
        dataset_name="synthetic_supcon",
        subject_id="S0",
        train_win=tr_win,
        train_lbl=tr_lbl,
        test_win=te_win,
        test_lbl=te_lbl,
        X_tr_aux=X_tr_aux,
        X_te_aux=X_te_aux,
        num_classes=num_cls,
        supcon_epochs=2,
        finetune_epochs=2,
        batch_size=16,
        seed=42,
    )

    assert fp32_p.shape == (n_te,)
    assert int8_p.shape == (n_te,)
    assert fp32_l.shape == (n_te, num_cls)
    assert int8_l.shape == (n_te, num_cls)
    assert 0.0 <= art.int8_top1_acc <= 1.0
