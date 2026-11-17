"""Unit tests for Spatial-Temporal Circular CNN-2D (M12)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.train_cnn2d_circular import (
    CompactCircularSemgCnn2D,
    circular_pad_height,
    circular_pad_height_numpy,
    fold_conv_bn_weights,
    infer_ring_topology,
    train_and_quantize_cnn2d_circular,
)


def test_ring_topology_inference() -> None:
    """Validate anatomical channel partitioning across configurations."""
    assert infer_ring_topology(12) == (8, 4)
    assert infer_ring_topology(8) == (8, 0)
    assert infer_ring_topology(4) == (4, 0)
    assert infer_ring_topology(2) == (2, 0)


def test_circular_padding_equivalence() -> None:
    """Ensure PyTorch and NumPy circular padding wrap electrodes identically."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((2, 3, 8, 10)).astype(np.float32)

    np_padded = circular_pad_height_numpy(arr, pad=1)
    th_padded = circular_pad_height(torch.from_numpy(arr), pad=1).numpy()

    assert np_padded.shape == (2, 3, 10, 10)
    np.testing.assert_allclose(np_padded, th_padded)

    # Check wrap boundary conditions: row 0 == original row 7; row 9 == original row 0
    np.testing.assert_allclose(np_padded[:, :, 0, :], arr[:, :, 7, :])
    np.testing.assert_allclose(np_padded[:, :, 9, :], arr[:, :, 0, :])

    # For < 3 height, padding should be a no-op
    arr_2ch = rng.standard_normal((2, 3, 2, 10)).astype(np.float32)
    assert circular_pad_height_numpy(arr_2ch, pad=1).shape == (2, 3, 2, 10)


def test_batchnorm_folding_equivalence() -> None:
    """Verify Conv + BatchNorm mathematical folding is exact in eval mode."""
    torch.manual_seed(0)
    conv = nn.Conv2d(2, 4, kernel_size=(3, 5), bias=True)
    bn = nn.BatchNorm2d(4)
    # Run dummy train step to populate running stats
    x_dummy = torch.randn(8, 2, 6, 20)
    _ = bn(conv(x_dummy))

    conv.eval()
    bn.eval()

    x_test = torch.randn(4, 2, 6, 20)
    with torch.no_grad():
        y_ref = bn(conv(x_test)).numpy()

        w_fold, b_fold = fold_conv_bn_weights(
            conv.weight.numpy(),
            conv.bias.numpy(),
            bn.weight.numpy(),
            bn.bias.numpy(),
            bn.running_mean.numpy(),
            bn.running_var.numpy(),
            eps=bn.eps,
        )
        conv_folded = nn.Conv2d(2, 4, kernel_size=(3, 5), bias=True)
        conv_folded.weight.copy_(torch.from_numpy(w_fold))
        conv_folded.bias.copy_(torch.from_numpy(b_fold))
        y_fold = conv_folded(x_test).numpy()

    np.testing.assert_allclose(y_ref, y_fold, atol=1e-5, rtol=1e-5)


def test_model_forward_shapes_all_channels() -> None:
    """Verify forward pass dimensions for 12CH, 4CH, and 2CH topologies."""
    for c, num_classes in [(12, 50), (4, 10), (2, 10)]:
        num_aux = 5 * c
        model = CompactCircularSemgCnn2D(in_channels=c, num_classes=num_classes, num_aux=num_aux)
        x = torch.randn(4, c, 400)
        aux = torch.randn(4, num_aux)
        logits = model(x, aux)
        assert logits.shape == (4, num_classes)


def test_synthetic_train_and_int8_quantization() -> None:
    """Smoke test full training + INT8 folding/simulation pipeline."""
    rng = np.random.default_rng(123)
    n_tr, n_te = 40, 20
    c, num_cls = 4, 3

    tr_win = rng.standard_normal((n_tr, 400, c)).astype(np.float32)
    te_win = rng.standard_normal((n_te, 400, c)).astype(np.float32)
    tr_lbl = np.tile(np.arange(num_cls), n_tr // num_cls + 1)[:n_tr]
    te_lbl = np.tile(np.arange(num_cls), n_te // num_cls + 1)[:n_te]

    # Make classes mildly separable
    for i in range(n_tr):
        tr_win[i, :, tr_lbl[i]] += 1.5
    for i in range(n_te):
        te_win[i, :, te_lbl[i]] += 1.5

    X_tr_aux = rng.standard_normal((n_tr, 5 * c)).astype(np.float32)
    X_te_aux = rng.standard_normal((n_te, 5 * c)).astype(np.float32)

    art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_cnn2d_circular(
        dataset_name="synthetic",
        subject_id="S0",
        train_win=tr_win,
        train_lbl=tr_lbl,
        test_win=te_win,
        test_lbl=te_lbl,
        X_tr_aux=X_tr_aux,
        X_te_aux=X_te_aux,
        num_classes=num_cls,
        epochs=3,
        batch_size=16,
        seed=42,
    )

    assert art.total_params > 0
    assert fp32_p.shape == (n_te,)
    assert int8_p.shape == (n_te,)
    assert fp32_l.shape == (n_te, num_cls)
    assert int8_l.shape == (n_te, num_cls)
    assert 0.0 <= art.fp32_top1_acc <= 1.0
    assert 0.0 <= art.int8_top1_acc <= 1.0
