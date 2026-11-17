#!/usr/bin/env python3
"""Train and INT8-Quantize Spatial-Temporal Circular CNN-2D (M12) for Forearm sEMG.

Exploits the anatomical circumferential electrode ring geometry of forearm sEMG:
  - NinaPro DB2 (12 channels): 8 circumferential ring electrodes (Ch 0..7) wrapped
    circularly around the forearm (Ch 7 <-> Ch 0) + 4 anatomically targeted
    electrodes (Ch 8..11: FDS, EDC, Biceps, Triceps).
  - Ozdemir et al. (4 channels): 4 forearm electrodes wrapped circularly.
  - Reduced 2-channel fallback: 2-channel spatial strip without wrap.

Architecture (`CompactCircularSemgCnn2D`):
  - Ring Branch: Circular spatial padding along electrode axis + Conv2D(1->16, k=(3,9), s=(1,2))
    + BN + ReLU + MaxPool2D((1,4)) + CircularPad + Conv2D(16->24, k=(3,5), s=(1,2)) + BN + ReLU.
    Retains 4 azimuthal forearm quadrants after temporal GAP (`24 * 4 = 96` features)
    to preserve flexor/extensor spatial localization.
  - Anatomical Branch (when C > ring_channels): Conv1D(C_anat->12, k=9, s=2) + BN + ReLU
    + MaxPool1D(4) + Conv1D(12->16, k=5, s=2) + BN + ReLU + GAP (`16` features).
  - Auxiliary Residual Branch: Linear projection over 5*C standardized TD features
    warm-started from Canonical LDA weights.

Includes BatchNorm folding and exact symmetric INT8 quantization + Q24 simulator
ready for ESP32-S3 deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import sys
from pathlib import Path
import numpy as np
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


def set_deterministic_seed(seed: int = 42) -> None:
    """Fix all random seeds for strict scientific reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def infer_ring_topology(in_channels: int) -> tuple[int, int]:
    """Determine (ring_channels, anat_channels) from total channel count."""
    if in_channels >= 8:
        return 8, in_channels - 8
    return in_channels, 0


def circular_pad_height(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """Apply deterministic circular wrap padding along height (electrode ring axis).

    Args:
        x: Tensor of shape (N, F, H, W).
        pad: Number of rows to wrap on top and bottom.
    """
    if pad <= 0 or x.shape[2] < 3:
        return x
    top = x[:, :, -pad:, :]
    bottom = x[:, :, :pad, :]
    return torch.cat([top, x, bottom], dim=2)


def circular_pad_height_numpy(x: np.ndarray, pad: int = 1) -> np.ndarray:
    """NumPy mirror of deterministic circular padding along height (axis 2)."""
    if pad <= 0 or x.shape[2] < 3:
        return x
    return np.concatenate([x[:, :, -pad:, :], x, x[:, :, :pad, :]], axis=2)


def fold_conv_bn_weights(
    conv_w: np.ndarray,
    conv_b: np.ndarray | None,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mathematically fold BatchNorm parameters into Conv weights and biases."""
    scale = bn_weight / np.sqrt(bn_var + eps)
    w_shape = [-1] + [1] * (conv_w.ndim - 1)
    w_fold = conv_w * scale.reshape(w_shape)
    b_raw = conv_b if conv_b is not None else np.zeros_like(bn_mean)
    b_fold = (b_raw - bn_mean) * scale + bn_bias
    return w_fold.astype(np.float32), b_fold.astype(np.float32)


class CompactCircularSemgCnn2D(nn.Module):
    """Spatial-Temporal Circular CNN-2D for Forearm sEMG Decoding on ESP32-S3."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_aux: int,
        dropout_p: float = 0.15,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_aux = num_aux
        self.ring_channels, self.anat_channels = infer_ring_topology(in_channels)

        # Determine spatial kernel height for ring branch
        self.use_circular = self.ring_channels >= 3
        k_h1 = 3 if self.use_circular else min(2, self.ring_channels)
        k_h2 = 3 if self.use_circular else 1

        # 1. Circumferential Ring 2D-CNN Branch
        self.ring_conv1 = nn.Conv2d(1, 12, kernel_size=(k_h1, 9), stride=(1, 3), padding=(0, 4), bias=True)
        self.ring_bn1 = nn.BatchNorm2d(12)
        self.ring_pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.ring_conv2 = nn.Conv2d(12, 24, kernel_size=(k_h2, 5), stride=(1, 2), padding=(0, 2), bias=True)
        self.ring_bn2 = nn.BatchNorm2d(24)

        # Azimuthal quadrant retention
        if self.ring_channels == 8:
            self.spatial_quad_pool = nn.AvgPool1d(kernel_size=2, stride=2)
            self.ring_feat_dim = 24 * 4  # 96
        elif self.use_circular:
            self.spatial_quad_pool = None
            self.ring_feat_dim = 24 * self.ring_channels
        else:
            self.spatial_quad_pool = None
            self.ring_feat_dim = 24

        # 2. Optional Anatomical Electrode 1D-CNN Branch (for C > 8, e.g., NinaPro 12CH)
        if self.anat_channels > 0:
            self.anat_conv1 = nn.Conv1d(self.anat_channels, 12, kernel_size=9, stride=3, padding=4, bias=True)
            self.anat_bn1 = nn.BatchNorm1d(12)
            self.anat_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.anat_conv2 = nn.Conv1d(12, 16, kernel_size=5, stride=2, padding=2, bias=True)
            self.anat_bn2 = nn.BatchNorm1d(16)
            self.anat_feat_dim = 16
        else:
            self.anat_feat_dim = 0

        self.total_conv_feat_dim = self.ring_feat_dim + self.anat_feat_dim
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_conv = nn.Linear(self.total_conv_feat_dim, num_classes, bias=True)
        self.fc_aux = nn.Linear(num_aux, num_classes, bias=False)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute spatial-temporal features and intermediate activations for PTQ."""
        # x shape: (N, C, 400)
        x_ring = x[:, : self.ring_channels, :].unsqueeze(1)  # (N, 1, C_ring, 400)
        if self.use_circular:
            x_ring_p = circular_pad_height(x_ring, pad=1)
        else:
            x_ring_p = x_ring

        r_a1 = F.relu(self.ring_bn1(self.ring_conv1(x_ring_p)))
        r_p1 = self.ring_pool1(r_a1)  # (N, 16, H1, 50)

        if self.use_circular:
            r_p1_p = circular_pad_height(r_p1, pad=1)
        else:
            r_p1_p = r_p1
        r_a2 = F.relu(self.ring_bn2(self.ring_conv2(r_p1_p)))  # (N, 24, H2, 25)

        # Global Temporal Average Pooling -> (N, 24, H2)
        r_tgap = r_a2.mean(dim=-1)
        if self.spatial_quad_pool is not None:
            r_spatial = self.spatial_quad_pool(r_tgap)  # (N, 24, 4)
        else:
            r_spatial = r_tgap  # (N, 24, H2)
        r_flat = r_spatial.reshape(x.shape[0], -1)  # (N, ring_feat_dim)

        feats = [r_flat]
        out_dict: dict[str, torch.Tensor] = {
            "r_a1": r_a1,
            "r_a2": r_a2,
            "r_flat": r_flat,
        }

        if self.anat_channels > 0:
            x_anat = x[:, self.ring_channels :, :]  # (N, C_anat, 400)
            a_a1 = F.relu(self.anat_bn1(self.anat_conv1(x_anat)))
            a_p1 = self.anat_pool1(a_a1)
            a_a2 = F.relu(self.anat_bn2(self.anat_conv2(a_p1)))
            a_gap = a_a2.mean(dim=-1)  # (N, 16)
            feats.append(a_gap)
            out_dict["a_a1"] = a_a1
            out_dict["a_a2"] = a_a2
            out_dict["a_gap"] = a_gap

        combined = torch.cat(feats, dim=1)
        out_dict["combined"] = combined
        return out_dict

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        feat_dict = self.forward_features(x)
        comb = self.dropout(feat_dict["combined"])
        return self.fc_conv(comb) + self.fc_aux(aux)


@dataclass(frozen=True)
class QuantizedCnn2DCircularArtifacts:
    """Folded symmetric INT8 parameters and calibration scales for M12 Circular CNN-2D."""

    dataset_name: str
    subject_id: str
    in_channels: int
    ring_channels: int
    anat_channels: int
    num_classes: int
    num_aux: int
    total_params: int
    use_circular: bool
    in_mean: np.ndarray
    in_inv_std: np.ndarray
    in_quant_scale: float
    aux_mean: np.ndarray
    aux_inv_std: np.ndarray
    aux_quant_scale: float

    # Ring branch quantized parameters (with folded BN)
    r_conv1_w_q: np.ndarray
    r_conv1_b_q: np.ndarray
    r_conv1_eff_scale: float
    r_conv2_w_q: np.ndarray
    r_conv2_b_q: np.ndarray
    r_conv2_eff_scale: float
    r_gap_scale: float

    # Anatomical branch quantized parameters (empty if anat_channels == 0)
    a_conv1_w_q: np.ndarray
    a_conv1_b_q: np.ndarray
    a_conv1_eff_scale: float
    a_conv2_w_q: np.ndarray
    a_conv2_b_q: np.ndarray
    a_conv2_eff_scale: float
    a_gap_scale: float

    # Classifier head
    fc_w_q: np.ndarray
    fc_b_q: np.ndarray
    fc_eff_scale: float
    aux_fc_w_q: np.ndarray
    aux_fc_eff_scale: float

    # Validation metrics
    fp32_top1_acc: float
    fp32_bal_acc: float
    fp32_macro_f1: float
    int8_top1_acc: float
    int8_bal_acc: float
    int8_macro_f1: float


def quantize_windows_cnn2d(
    raw_windows: np.ndarray,
    aux_features: np.ndarray,
    art: QuantizedCnn2DCircularArtifacts,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize raw windows (N, 400, C) -> int8 (N, C, 400) and aux -> int8 (N, F)."""
    w_f32 = raw_windows.astype(np.float32)
    mean_f32 = art.in_mean.astype(np.float32)[None, None, :]
    inv_std_f32 = art.in_inv_std.astype(np.float32)[None, None, :]
    inv_s_in = np.float32(1.0 / art.in_quant_scale)

    x_norm = (w_f32 - mean_f32) * inv_std_f32
    x_trans = np.transpose(x_norm, (0, 2, 1))
    x_q = np.clip(c_round_half_away(x_trans * inv_s_in), -127, 127).astype(np.int8)

    aux_f32 = aux_features.astype(np.float32)
    aux_z = (aux_f32 - art.aux_mean.astype(np.float32)[None, :]) * art.aux_inv_std.astype(np.float32)[None, :]
    inv_s_aux = np.float32(1.0 / art.aux_quant_scale)
    aux_q = np.clip(c_round_half_away(aux_z * inv_s_aux), -127, 127).astype(np.int8)
    return x_q, aux_q


def _conv2d_int8_q24(
    x_q: np.ndarray,
    w_q: np.ndarray,
    b_q: np.ndarray,
    mult_q24: int,
    stride_w: int = 3,
    pad_w: int = 4,
    circular_h: bool = True,
) -> np.ndarray:
    """Bit-exact INT8 2D convolution with circular height padding and Q24 requantization.

    Uses chunked exact float64 `F.conv2d` (exact for all integers up to 2^53) for 50x speedup.
    """
    if circular_h and x_q.shape[2] >= 3:
        x_p = circular_pad_height_numpy(x_q, pad=1)
    else:
        x_p = x_q

    if pad_w > 0:
        x_p = np.pad(x_p, ((0, 0), (0, 0), (0, 0), (pad_w, pad_w)), mode="constant", constant_values=0)

    w_t = torch.from_numpy(w_q.astype(np.float64))
    b_t = torch.from_numpy(b_q.astype(np.float64))
    out_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x_p), 2048):
            xb = torch.from_numpy(x_p[start : start + 2048].astype(np.float64))
            acc = F.conv2d(xb, w_t, b_t, stride=(1, stride_w)).numpy().astype(np.int64)
            rq = requantize_q24_numpy(acc, mult_q24)
            out_chunks.append(np.clip(rq, 0, 127).astype(np.int8))

    return np.concatenate(out_chunks, axis=0)


def _conv1d_int8_q24(
    x_q: np.ndarray,
    w_q: np.ndarray,
    b_q: np.ndarray,
    mult_q24: int,
    stride: int = 3,
    pad: int = 4,
) -> np.ndarray:
    """Bit-exact INT8 1D convolution with Q24 requantization."""
    if pad > 0:
        x_p = np.pad(x_q, ((0, 0), (0, 0), (pad, pad)), mode="constant", constant_values=0)
    else:
        x_p = x_q

    w_t = torch.from_numpy(w_q.astype(np.float64))
    b_t = torch.from_numpy(b_q.astype(np.float64))
    out_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x_p), 2048):
            xb = torch.from_numpy(x_p[start : start + 2048].astype(np.float64))
            acc = F.conv1d(xb, w_t, b_t, stride=stride).numpy().astype(np.int64)
            rq = requantize_q24_numpy(acc, mult_q24)
            out_chunks.append(np.clip(rq, 0, 127).astype(np.int8))

    return np.concatenate(out_chunks, axis=0)


def simulate_c99_int8_cnn2d_circular(
    art: QuantizedCnn2DCircularArtifacts,
    x_q: np.ndarray,
    aux_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact INT8/Q24 simulation of CompactCircularSemgCnn2D matching C99 target."""
    n_samples = x_q.shape[0]
    n_cls = art.num_classes
    c_ring = art.ring_channels
    c_anat = art.anat_channels

    # 1. Ring Branch
    x_ring = x_q[:, :c_ring, :][:, None, :, :]  # (N, 1, c_ring, 400)
    r_mult1 = int(c_round_half_away(art.r_conv1_eff_scale * float(1 << 24)).item())
    r_mult2 = int(c_round_half_away(art.r_conv2_eff_scale * float(1 << 24)).item())

    r_out1 = _conv2d_int8_q24(
        x_ring, art.r_conv1_w_q, art.r_conv1_b_q, r_mult1,
        stride_w=3, pad_w=4, circular_h=art.use_circular,
    )  # (N, 12, H1, 134)

    # MaxPool2d(1, 2) along time
    t1_len = (r_out1.shape[3] // 2) * 2
    r_p1 = r_out1[:, :, :, :t1_len].reshape(n_samples, r_out1.shape[1], r_out1.shape[2], t1_len // 2, 2).max(axis=4)

    r_out2 = _conv2d_int8_q24(
        r_p1, art.r_conv2_w_q, art.r_conv2_b_q, r_mult2,
        stride_w=2, pad_w=2, circular_h=art.use_circular,
    )  # (N, 24, H2, 34)

    # Temporal GAP over T2
    t2_len = r_out2.shape[3]
    r_tgap = (np.sum(r_out2.astype(np.int32), axis=3) + (t2_len // 2)) // t2_len  # (N, 24, H2)

    if c_ring == 8:
        # Spatial quadrant pooling: pair adjacent electrodes (0,1), (2,3), (4,5), (6,7)
        r_spatial = (r_tgap[:, :, 0:8:2] + r_tgap[:, :, 1:8:2] + 1) // 2  # (N, 24, 4)
    else:
        r_spatial = r_tgap
    r_flat = np.clip(r_spatial.reshape(n_samples, -1), 0, 127).astype(np.int8)

    feats_list = [r_flat]

    # 2. Anatomical Branch (if c_anat > 0)
    if c_anat > 0:
        x_anat = x_q[:, c_ring:, :]  # (N, c_anat, 400)
        a_mult1 = int(c_round_half_away(art.a_conv1_eff_scale * float(1 << 24)).item())
        a_mult2 = int(c_round_half_away(art.a_conv2_eff_scale * float(1 << 24)).item())

        a_out1 = _conv1d_int8_q24(x_anat, art.a_conv1_w_q, art.a_conv1_b_q, a_mult1, stride=3, pad=4)
        ta1_len = (a_out1.shape[2] // 2) * 2
        a_p1 = a_out1[:, :, :ta1_len].reshape(n_samples, a_out1.shape[1], ta1_len // 2, 2).max(axis=3)

        a_out2 = _conv1d_int8_q24(a_p1, art.a_conv2_w_q, art.a_conv2_b_q, a_mult2, stride=2, pad=2)
        ta2_len = a_out2.shape[2]
        a_gap = np.clip((np.sum(a_out2.astype(np.int32), axis=2) + (ta2_len // 2)) // ta2_len, 0, 127).astype(np.int8)

        feats_list.append(a_gap)

    combined_q = np.concatenate(feats_list, axis=1)  # (N, D)

    # 3. Classifier Head + Auxiliary TD Residual
    w_fc = art.fc_w_q.reshape(n_cls, -1).astype(np.int32)
    b_fc = art.fc_b_q.astype(np.int32)
    w_aux = art.aux_fc_w_q.reshape(n_cls, art.num_aux).astype(np.int32)
    mult_aux_q16 = int(c_round_half_away(art.aux_fc_eff_scale * 65536.0).item())

    fc_acc = combined_q.astype(np.int32) @ w_fc.T + b_fc[None, :]
    aux_acc = aux_q.astype(np.int32) @ w_aux.T
    aux_scaled = ((aux_acc.astype(np.int64) * mult_aux_q16 + 32768) >> 16).astype(np.int32)

    logits = fc_acc + aux_scaled
    preds = np.argmax(logits, axis=1).astype(np.int32)
    return preds, logits


def train_and_quantize_cnn2d_circular(
    dataset_name: str,
    subject_id: str,
    train_win: np.ndarray,
    train_lbl: np.ndarray,
    test_win: np.ndarray,
    test_lbl: np.ndarray,
    X_tr_aux: np.ndarray,
    X_te_aux: np.ndarray,
    num_classes: int,
    epochs: int = 35,
    batch_size: int = 512,
    seed: int = 42,
    augment: bool = True,
) -> tuple[QuantizedCnn2DCircularArtifacts, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train CompactCircularSemgCnn2D, fold BatchNorm, quantize to INT8, and evaluate."""
    set_deterministic_seed(seed)
    torch.set_num_threads(4)

    in_channels = int(train_win.shape[2])
    num_aux = int(X_tr_aux.shape[1])

    # Training-only input standardization
    in_mean = np.mean(train_win, axis=(0, 1), dtype=np.float64).astype(np.float32)
    in_std = np.std(train_win, axis=(0, 1), dtype=np.float64).astype(np.float32)
    in_std = np.where(in_std < 1e-8, 1.0, in_std).astype(np.float32)
    in_inv_std = (1.0 / in_std).astype(np.float32)

    scaler = FeatureScaler()
    X_tr_aux_s = scaler.fit_transform(X_tr_aux).astype(np.float32)
    X_te_aux_s = scaler.transform(X_te_aux).astype(np.float32)
    sdict = scaler.to_dict()
    aux_mean = np.asarray(sdict["mean"], dtype=np.float32)
    aux_scale = np.asarray(sdict["scale"], dtype=np.float32)
    aux_inv_std = np.where(aux_scale != 0.0, 1.0 / aux_scale, 0.0).astype(np.float32)
    for z_idx in sdict["zero_variance_indices"]:
        if 0 <= z_idx < num_aux:
            aux_inv_std[z_idx] = 0.0

    # Canonical LDA warm-start for auxiliary linear branch
    lda_trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_classes)
    lda_trainer.fit(X_tr_aux_s, train_lbl)
    lda_art = lda_trainer.get_model_artifacts()

    X_tr_cnn = np.transpose((train_win - in_mean[None, None, :]) * in_inv_std[None, None, :], (0, 2, 1)).astype(np.float32)
    X_te_cnn = np.transpose((test_win - in_mean[None, None, :]) * in_inv_std[None, None, :], (0, 2, 1)).astype(np.float32)

    t_x_tr = torch.from_numpy(X_tr_cnn)
    t_aux_tr = torch.from_numpy(X_tr_aux_s)
    t_y_tr = torch.from_numpy(train_lbl.astype(np.int64))

    model = CompactCircularSemgCnn2D(in_channels=in_channels, num_classes=num_classes, num_aux=num_aux, dropout_p=0.15)

    with torch.no_grad():
        model.fc_aux.weight.copy_(torch.from_numpy(lda_art.coef_.astype(np.float32)))
        model.fc_conv.bias.copy_(torch.from_numpy(lda_art.intercept_.astype(np.float32)))
        nn.init.normal_(model.fc_conv.weight, mean=0.0, std=0.01)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "fc_" not in n], "lr": 3.0e-3, "weight_decay": 1e-3},
            {"params": model.fc_conv.parameters(), "lr": 1.5e-3, "weight_decay": 1e-2},
            {"params": model.fc_aux.parameters(), "lr": 2.0e-4, "weight_decay": 1e-2},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    gen = torch.Generator().manual_seed(seed)
    n_tr = t_x_tr.shape[0]

    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_tr, generator=gen)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, batch_size):
            b_idx = perm[start : start + batch_size]
            bx = t_x_tr[b_idx]
            baux = t_aux_tr[b_idx]
            by = t_y_tr[b_idx]

            if augment:
                # Mild deterministic amplitude jitter + Gaussian noise on training batch only
                amp_scale = 0.90 + 0.20 * torch.rand((bx.shape[0], bx.shape[1], 1), generator=gen)
                noise = 0.03 * torch.randn(bx.shape, generator=gen)
                bx = bx * amp_scale + noise

            optimizer.zero_grad()
            logits = model(bx, baux)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        if ep % 5 == 0 or ep == epochs:
            print(f"    [{dataset_name}] Epoch {ep:02d}/{epochs} | loss={ep_loss / max(1, n_batches):.4f}", flush=True)

    return evaluate_and_quantize_cnn2d_circular(
        model=model,
        dataset_name=dataset_name,
        subject_id=subject_id,
        X_tr_cnn=X_tr_cnn,
        X_tr_aux_s=X_tr_aux_s,
        test_win=test_win,
        test_lbl=test_lbl,
        X_te_cnn=X_te_cnn,
        X_te_aux=X_te_aux,
        X_te_aux_s=X_te_aux_s,
        in_mean=in_mean,
        in_inv_std=in_inv_std,
        aux_mean=aux_mean,
        aux_inv_std=aux_inv_std,
    )


def evaluate_and_quantize_cnn2d_circular(
    model: CompactCircularSemgCnn2D,
    dataset_name: str,
    subject_id: str,
    X_tr_cnn: np.ndarray,
    X_tr_aux_s: np.ndarray,
    test_win: np.ndarray,
    test_lbl: np.ndarray,
    X_te_cnn: np.ndarray,
    X_te_aux: np.ndarray,
    X_te_aux_s: np.ndarray,
    in_mean: np.ndarray,
    in_inv_std: np.ndarray,
    aux_mean: np.ndarray,
    aux_inv_std: np.ndarray,
) -> tuple[QuantizedCnn2DCircularArtifacts, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate FP32, fold BatchNorm layers, quantize to INT8, and run C99 simulation."""
    from dataclasses import replace
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    in_channels = model.in_channels
    num_classes = model.num_classes
    num_aux = model.num_aux
    n_tr = len(X_tr_cnn)

    t_x_tr = torch.from_numpy(X_tr_cnn)
    t_x_te = torch.from_numpy(X_te_cnn)
    t_aux_te = torch.from_numpy(X_te_aux_s)

    model.eval()
    with torch.no_grad():
        fp32_logits_list = []
        for start in range(0, len(t_x_te), 1024):
            fp32_logits_list.append(model(t_x_te[start : start + 1024], t_aux_te[start : start + 1024]))
        fp32_logits = torch.cat(fp32_logits_list, dim=0).numpy()
    fp32_preds = np.argmax(fp32_logits, axis=1)

    fp32_acc = float(accuracy_score(test_lbl, fp32_preds))
    fp32_bal = float(balanced_accuracy_score(test_lbl, fp32_preds))
    fp32_f1 = float(f1_score(test_lbl, fp32_preds, average="macro", zero_division=0))

    with torch.no_grad():
        calib_idx = np.linspace(0, n_tr - 1, min(1500, n_tr), dtype=int)
        feat_cal = model.forward_features(t_x_tr[calib_idx])

        s_in = max(float(np.percentile(np.abs(X_tr_cnn[calib_idx]), 99.95)) / 127.0, 1e-6)
        s_aux = max(float(np.percentile(np.abs(X_tr_aux_s[calib_idx]), 99.95)) / 127.0, 1e-6)

        s_r_act1 = max(float(np.percentile(feat_cal["r_a1"].numpy(), 99.95)) / 127.0, 1e-6)
        rw1_f, rb1_f = fold_conv_bn_weights(
            model.ring_conv1.weight.numpy(),
            model.ring_conv1.bias.numpy(),
            model.ring_bn1.weight.numpy(),
            model.ring_bn1.bias.numpy(),
            model.ring_bn1.running_mean.numpy(),
            model.ring_bn1.running_var.numpy(),
        )
        s_rw1 = max(float(np.max(np.abs(rw1_f))) / 127.0, 1e-8)
        rw1_q = np.clip(c_round_half_away(rw1_f / s_rw1), -127, 127).astype(np.int8)
        rb1_q = c_round_half_away(rb1_f / (s_in * s_rw1)).astype(np.int32)
        r_conv1_eff = (s_in * s_rw1) / s_r_act1

        s_r_act2 = max(float(np.percentile(feat_cal["r_a2"].numpy(), 99.95)) / 127.0, 1e-6)
        rw2_f, rb2_f = fold_conv_bn_weights(
            model.ring_conv2.weight.numpy(),
            model.ring_conv2.bias.numpy(),
            model.ring_bn2.weight.numpy(),
            model.ring_bn2.bias.numpy(),
            model.ring_bn2.running_mean.numpy(),
            model.ring_bn2.running_var.numpy(),
        )
        s_rw2 = max(float(np.max(np.abs(rw2_f))) / 127.0, 1e-8)
        rw2_q = np.clip(c_round_half_away(rw2_f / s_rw2), -127, 127).astype(np.int8)
        rb2_q = c_round_half_away(rb2_f / (s_r_act1 * s_rw2)).astype(np.int32)
        r_conv2_eff = (s_r_act1 * s_rw2) / s_r_act2

        if model.anat_channels > 0:
            s_a_act1 = max(float(np.percentile(feat_cal["a_a1"].numpy(), 99.95)) / 127.0, 1e-6)
            aw1_f, ab1_f = fold_conv_bn_weights(
                model.anat_conv1.weight.numpy(),
                model.anat_conv1.bias.numpy(),
                model.anat_bn1.weight.numpy(),
                model.anat_bn1.bias.numpy(),
                model.anat_bn1.running_mean.numpy(),
                model.anat_bn1.running_var.numpy(),
            )
            s_aw1 = max(float(np.max(np.abs(aw1_f))) / 127.0, 1e-8)
            aw1_q = np.clip(c_round_half_away(aw1_f / s_aw1), -127, 127).astype(np.int8)
            ab1_q = c_round_half_away(ab1_f / (s_in * s_aw1)).astype(np.int32)
            a_conv1_eff = (s_in * s_aw1) / s_a_act1

            s_a_act2 = max(float(np.percentile(feat_cal["a_a2"].numpy(), 99.95)) / 127.0, 1e-6)
            aw2_f, ab2_f = fold_conv_bn_weights(
                model.anat_conv2.weight.numpy(),
                model.anat_conv2.bias.numpy(),
                model.anat_bn2.weight.numpy(),
                model.anat_bn2.bias.numpy(),
                model.anat_bn2.running_mean.numpy(),
                model.anat_bn2.running_var.numpy(),
            )
            s_aw2 = max(float(np.max(np.abs(aw2_f))) / 127.0, 1e-8)
            aw2_q = np.clip(c_round_half_away(aw2_f / s_aw2), -127, 127).astype(np.int8)
            ab2_q = c_round_half_away(ab2_f / (s_a_act1 * s_aw2)).astype(np.int32)
            a_conv2_eff = (s_a_act1 * s_aw2) / s_a_act2
        else:
            s_a_act2 = s_r_act2
            aw1_q = np.empty(0, dtype=np.int8)
            ab1_q = np.empty(0, dtype=np.int32)
            a_conv1_eff = 1.0
            aw2_q = np.empty(0, dtype=np.int8)
            ab2_q = np.empty(0, dtype=np.int32)
            a_conv2_eff = 1.0

        wfc_fp = model.fc_conv.weight.numpy().copy()
        bfc_fp = model.fc_conv.bias.numpy().copy()
        if model.anat_channels > 0:
            wfc_fp[:, model.ring_feat_dim :] *= (s_a_act2 / s_r_act2)

        s_wfc = max(float(np.max(np.abs(wfc_fp))) / 127.0, 1e-8)
        wfc_q = np.clip(c_round_half_away(wfc_fp / s_wfc), -127, 127).astype(np.int8)
        fc_eff = s_r_act2 * s_wfc
        bfc_q = c_round_half_away(bfc_fp / fc_eff).astype(np.int32)

        waux_fp = model.fc_aux.weight.numpy()
        s_waux = max(float(np.max(np.abs(waux_fp))) / 127.0, 1e-8)
        waux_q = np.clip(c_round_half_away(waux_fp / s_waux), -127, 127).astype(np.int8)
        aux_fc_eff = (s_aux * s_waux) / fc_eff

    folded_params = (
        rw1_q.size + rb1_q.size + rw2_q.size + rb2_q.size +
        aw1_q.size + ab1_q.size + aw2_q.size + ab2_q.size +
        wfc_q.size + bfc_q.size + waux_q.size
    )

    art = QuantizedCnn2DCircularArtifacts(
        dataset_name=dataset_name,
        subject_id=subject_id,
        in_channels=in_channels,
        ring_channels=model.ring_channels,
        anat_channels=model.anat_channels,
        num_classes=num_classes,
        num_aux=num_aux,
        total_params=folded_params,
        use_circular=model.use_circular,
        in_mean=in_mean,
        in_inv_std=in_inv_std,
        in_quant_scale=float(s_in),
        aux_mean=aux_mean,
        aux_inv_std=aux_inv_std,
        aux_quant_scale=float(s_aux),
        r_conv1_w_q=rw1_q,
        r_conv1_b_q=rb1_q,
        r_conv1_eff_scale=float(r_conv1_eff),
        r_conv2_w_q=rw2_q,
        r_conv2_b_q=rb2_q,
        r_conv2_eff_scale=float(r_conv2_eff),
        r_gap_scale=float(s_r_act2),
        a_conv1_w_q=aw1_q,
        a_conv1_b_q=ab1_q,
        a_conv1_eff_scale=float(a_conv1_eff),
        a_conv2_w_q=aw2_q,
        a_conv2_b_q=ab2_q,
        a_conv2_eff_scale=float(a_conv2_eff),
        a_gap_scale=float(s_a_act2),
        fc_w_q=wfc_q,
        fc_b_q=bfc_q,
        fc_eff_scale=float(fc_eff),
        aux_fc_w_q=waux_q,
        aux_fc_eff_scale=float(aux_fc_eff),
        fp32_top1_acc=fp32_acc,
        fp32_bal_acc=fp32_bal,
        fp32_macro_f1=fp32_f1,
        int8_top1_acc=0.0,
        int8_bal_acc=0.0,
        int8_macro_f1=0.0,
    )

    x_q_te, aux_q_te = quantize_windows_cnn2d(test_win, X_te_aux, art)
    int8_preds, int8_logits = simulate_c99_int8_cnn2d_circular(art, x_q_te, aux_q_te)

    int8_acc = float(accuracy_score(test_lbl, int8_preds))
    int8_bal = float(balanced_accuracy_score(test_lbl, int8_preds))
    int8_f1 = float(f1_score(test_lbl, int8_preds, average="macro", zero_division=0))

    art_final = replace(
        art,
        int8_top1_acc=int8_acc,
        int8_bal_acc=int8_bal,
        int8_macro_f1=int8_f1,
    )
    return art_final, fp32_preds, fp32_logits, int8_preds, int8_logits
