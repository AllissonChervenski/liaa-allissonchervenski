#!/usr/bin/env python3
"""Train, Quantize (INT8 Symmetric PTQ/QAT), and Generate Binary Oracles for 1D-CNN.

Trains the compact 1D-CNN architecture on the frozen LOTO splits of:
  1. Primary Dataset: Ozdemir et al. (4 channels, 10 classes, window=400, stride=100)
  2. Secondary Dataset: NinaPro DB2 (12 channels, 50 classes, window=400, stride=100)

Topology (strictly within <= 15,000 params, <= 30 kB Flash, <= 16 kB SRAM):
  - Conv1D Layer 1: in_channels -> 16 filters, kernel_size=7, stride=2, ReLU
  - MaxPool1D: kernel_size=2, stride=2
  - Conv1D Layer 2: 16 -> 24 filters, kernel_size=5, stride=2, ReLU
  - Global Average Pooling (GAP): temporal reduction T=47 -> 1
  - Dense / FC Layer: 24 (+ 5*C pre-integrated TD residual branch) -> num_classes

Performs symmetric INT8 quantization of weights and activations, simulates the
exact integer Q24/INT8 arithmetic in Python to create the deterministic oracle,
and writes:
  - `reports/cnn1d_qemu/cnn1d_model_artifacts.npz` (quantized weights, scales, metrics)
  - `reports/cnn1d_qemu/cnn1d_test_vectors.bin` (50 consecutive test windows per dataset
    with PyTorch / INT8 reference predictions and logits)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import struct
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ozdemir_quantization_and_channels import (  # noqa: E402
    V_FS_MV as OZDEMIR_V_FS_MV,
    compute_evaluation_metrics,
    load_ozdemir_recording,
    quantize_signal as quantize_ozdemir,
    segment_ozdemir_windows,
)
from scripts.benchmark_quantization_and_channels import (  # noqa: E402
    V_FS_VOLTS as NINAPRO_V_FS_VOLTS,
    quantize_signal as quantize_ninapro,
)
from semg_baseline.feature004_multi_exercise import (  # noqa: E402
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.features import TimeDomainFeatureExtractor  # noqa: E402
from semg_baseline.lda import CanonicalLdaTrainer, PriorType  # noqa: E402
from semg_baseline.scaling import FeatureScaler  # noqa: E402
from semg_baseline.thresholds import ThresholdCalibrator  # noqa: E402
from semg_dataset.contract import RecordingData  # noqa: E402
from semg_dataset.loader import NinaProDB2Loader  # noqa: E402


def set_deterministic_seed(seed: int = 42) -> None:
    """Fix all random seeds for strict scientific reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class CompactSemgCnn1D(nn.Module):
    """Compact 1D-CNN for sEMG gesture recognition on ESP32-S3 silicon.

    Combines a two-stage 1D convolutional temporal feature learner with a
    lightweight residual linear projection for rapid convergence on LOTO splits.
    """

    def __init__(self, in_channels: int, num_classes: int, num_aux: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_aux = num_aux

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 24, kernel_size=5, stride=2, padding=0, bias=True)
        self.relu2 = nn.ReLU()
        self.fc_conv = nn.Linear(24, num_classes, bias=True)
        self.fc_aux = nn.Linear(num_aux, num_classes, bias=False)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return intermediate activations (act1, act2, gap) for PTQ scale calibration."""
        a1 = self.relu1(self.conv1(x))
        p1 = self.pool1(a1)
        a2 = self.relu2(self.conv2(p1))
        gap = a2.mean(dim=-1)
        return a1, a2, gap

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        _, _, gap = self.forward_features(x)
        return self.fc_conv(gap) + self.fc_aux(aux)


@dataclass(frozen=True)
class QuantizedCnn1DArtifacts:
    """Exportable symmetric INT8 parameters and calibration scales for C99 engine."""

    dataset_name: str
    subject_id: str
    in_channels: int
    num_classes: int
    num_aux: int
    total_params: int
    in_mean: np.ndarray
    in_inv_std: np.ndarray
    in_quant_scale: float
    aux_mean: np.ndarray
    aux_inv_std: np.ndarray
    aux_quant_scale: float
    conv1_weights_q: np.ndarray
    conv1_bias_q: np.ndarray
    conv1_eff_scale: float
    conv2_weights_q: np.ndarray
    conv2_bias_q: np.ndarray
    conv2_eff_scale: float
    fc_weights_q: np.ndarray
    fc_bias_q: np.ndarray
    fc_eff_scale: float
    aux_fc_weights_q: np.ndarray
    aux_fc_eff_scale: float
    fp32_top1_acc: float
    fp32_bal_acc: float
    fp32_macro_f1: float
    int8_top1_acc: float
    int8_bal_acc: float
    int8_macro_f1: float


def c_round_half_away(val: np.ndarray | float) -> np.ndarray:
    """Match C99 lrintf / round half-way away from zero for deterministic quantization."""
    arr = np.asarray(val, dtype=np.float64)
    return np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5)).astype(np.int64)


def requantize_q24_numpy(acc: np.ndarray, mult_q24: int) -> np.ndarray:
    """Bit-exact NumPy mirror of C99 `requantize_q24(int32_t acc, int32_t mult_q24)`."""
    prod = acc.astype(np.int64) * int(mult_q24)
    pos = (prod + (1 << 23)) >> 24
    neg = -(((-prod) + (1 << 23)) >> 24)
    return np.where(prod >= 0, pos, neg).astype(np.int32)


def simulate_c99_int8_cnn1d(
    art: QuantizedCnn1DArtifacts,
    x_q: np.ndarray,
    aux_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact Python reference oracle matching `embedded/firmware/src/cnn1d_engine.c`.

    Args:
        art: Quantized model parameters.
        x_q: Pre-quantized int8 input windows of shape (N, in_channels, 400).
        aux_q: Pre-quantized int8 auxiliary vectors of shape (N, num_aux).

    Returns:
        preds: Array of shape (N,) in int32.
        logits: Raw integer accumulator logits of shape (N, num_classes) in int32.
    """
    n_samples = x_q.shape[0]
    in_ch = art.in_channels
    n_cls = art.num_classes

    mult1_q24 = int(c_round_half_away(art.conv1_eff_scale * float(1 << 24)).item())
    mult2_q24 = int(c_round_half_away(art.conv2_eff_scale * float(1 << 24)).item())
    mult_aux_q16 = int(c_round_half_away(art.aux_fc_eff_scale * 65536.0).item())

    w1 = art.conv1_weights_q.reshape(16, in_ch, 7).astype(np.int32)
    b1 = art.conv1_bias_q.astype(np.int32)

    w2 = art.conv2_weights_q.reshape(24, 16, 5).astype(np.int32)
    b2 = art.conv2_bias_q.astype(np.int32)

    w_fc = art.fc_weights_q.reshape(n_cls, 24).astype(np.int32)
    b_fc = art.fc_bias_q.astype(np.int32)
    w_aux = art.aux_fc_weights_q.reshape(n_cls, art.num_aux).astype(np.int32)

    # 1. Conv1D (stride=2, kernel=7) -> [N, 16, 197]
    out1 = np.zeros((n_samples, 16, 197), dtype=np.int8)
    x_i32 = x_q.astype(np.int32)
    for t_out in range(197):
        t_in = t_out * 2
        patch = x_i32[:, :, t_in : t_in + 7]  # [N, C, 7]
        acc = np.einsum("nck,fck->nf", patch, w1, optimize=True) + b1[None, :]
        rq = requantize_q24_numpy(acc, mult1_q24)
        out1[:, :, t_out] = np.clip(rq, 0, 127).astype(np.int8)

    # 2. MaxPool1D (2, stride=2) -> [N, 16, 98]
    p1 = np.maximum(out1[:, :, 0:196:2], out1[:, :, 1:196:2])

    # 3. Conv1D Layer 2 (stride=2, kernel=5) -> [N, 24, 47]
    out2 = np.zeros((n_samples, 24, 47), dtype=np.int8)
    p1_i32 = p1.astype(np.int32)
    for t_out in range(47):
        t_in = t_out * 2
        patch = p1_i32[:, :, t_in : t_in + 5]  # [N, 16, 5]
        acc = np.einsum("nck,fck->nf", patch, w2, optimize=True) + b2[None, :]
        rq = requantize_q24_numpy(acc, mult2_q24)
        out2[:, :, t_out] = np.clip(rq, 0, 127).astype(np.int8)

    # 4. Global Average Pooling (GAP across T=47) -> [N, 24]
    sum_gap = np.sum(out2.astype(np.int32), axis=2)
    gap_q = np.clip((sum_gap + 23) // 47, 0, 127).astype(np.int8)

    # 5. Dense Layer + Auxiliary Skip Projection
    fc_acc = gap_q.astype(np.int32) @ w_fc.T + b_fc[None, :]
    aux_acc = aux_q.astype(np.int32) @ w_aux.T
    aux_scaled = ((aux_acc.astype(np.int64) * mult_aux_q16 + 32768) >> 16).astype(np.int32)
    logits = fc_acc + aux_scaled
    preds = np.argmax(logits, axis=1).astype(np.int32)
    return preds, logits


def quantize_windows_for_c99(
    raw_windows: np.ndarray,
    aux_features: np.ndarray,
    in_mean: np.ndarray,
    in_inv_std: np.ndarray,
    in_quant_scale: float,
    aux_mean: np.ndarray,
    aux_inv_std: np.ndarray,
    aux_quant_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize raw windows (N, 400, C) -> int8 (N, C, 400) and aux -> int8 (N, F)."""
    w_f32 = raw_windows.astype(np.float32)
    mean_f32 = in_mean.astype(np.float32)[None, None, :]
    inv_std_f32 = in_inv_std.astype(np.float32)[None, None, :]
    inv_s_in = np.float32(1.0 / in_quant_scale)

    x_norm = (w_f32 - mean_f32) * inv_std_f32
    x_trans = np.transpose(x_norm, (0, 2, 1))  # [N, C, 400]
    x_q = np.clip(c_round_half_away(x_trans * inv_s_in), -127, 127).astype(np.int8)

    aux_f32 = aux_features.astype(np.float32)
    aux_z = (aux_f32 - aux_mean.astype(np.float32)[None, :]) * aux_inv_std.astype(np.float32)[None, :]
    inv_s_aux = np.float32(1.0 / aux_quant_scale)
    aux_q = np.clip(c_round_half_away(aux_z * inv_s_aux), -127, 127).astype(np.int8)
    return x_q, aux_q


def train_and_quantize_dataset(
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
    seed: int = 42,
) -> tuple[QuantizedCnn1DArtifacts, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train CompactSemgCnn1D in PyTorch, apply symmetric INT8 PTQ, and validate oracle."""
    set_deterministic_seed(seed)
    in_channels = int(train_win.shape[2])
    num_aux = int(X_tr_aux.shape[1])

    # Channel-wise normalization statistics strictly from training windows
    in_mean = np.mean(train_win, axis=(0, 1), dtype=np.float64).astype(np.float32)
    in_std = np.std(train_win, axis=(0, 1), dtype=np.float64).astype(np.float32)
    in_std = np.where(in_std < 1e-8, 1.0, in_std).astype(np.float32)
    in_inv_std = (1.0 / in_std).astype(np.float32)

    # Auxiliary TD feature normalization + Canonical LDA warm-start prior
    scaler = FeatureScaler()
    X_tr_aux_s = scaler.fit_transform(X_tr_aux).astype(np.float32)
    X_te_aux_s = scaler.transform(X_te_aux).astype(np.float32)
    scaler_dict = scaler.to_dict()
    aux_mean = np.asarray(scaler_dict["mean"], dtype=np.float32)
    aux_scale = np.asarray(scaler_dict["scale"], dtype=np.float32)
    aux_inv_std = np.where(aux_scale != 0.0, 1.0 / aux_scale, 0.0).astype(np.float32)
    for z_idx in scaler_dict["zero_variance_indices"]:
        if 0 <= z_idx < num_aux:
            aux_inv_std[z_idx] = 0.0

    lda_trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_classes)
    lda_trainer.fit(X_tr_aux_s, train_lbl)
    lda_art = lda_trainer.get_model_artifacts()

    # Prepare normalized tensors [N, C, 400] for PyTorch
    X_tr_cnn = np.transpose((train_win - in_mean[None, None, :]) * in_inv_std[None, None, :], (0, 2, 1)).astype(
        np.float32
    )
    X_te_cnn = np.transpose((test_win - in_mean[None, None, :]) * in_inv_std[None, None, :], (0, 2, 1)).astype(
        np.float32
    )

    model = CompactSemgCnn1D(in_channels=in_channels, num_classes=num_classes, num_aux=num_aux)
    total_params = sum(p.numel() for p in model.parameters())

    # Initialize auxiliary residual projection and bias from regularized LDA
    with torch.no_grad():
        model.fc_aux.weight.copy_(torch.from_numpy(lda_art.coef_.astype(np.float32)))
        model.fc_conv.bias.copy_(torch.from_numpy(lda_art.intercept_.astype(np.float32)))
        nn.init.normal_(model.fc_conv.weight, mean=0.0, std=0.01)

    # Train convolutional layers + fine-tune classifier with cosine schedule
    ds_train = TensorDataset(
        torch.from_numpy(X_tr_cnn),
        torch.from_numpy(X_tr_aux_s),
        torch.from_numpy(train_lbl.astype(np.int64)),
    )
    loader = DataLoader(ds_train, batch_size=128, shuffle=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.conv1.parameters(), "lr": 1.5e-3, "weight_decay": 1e-3},
            {"params": model.conv2.parameters(), "lr": 1.5e-3, "weight_decay": 1e-3},
            {"params": model.fc_conv.parameters(), "lr": 8e-4, "weight_decay": 1e-2},
            {"params": model.fc_aux.parameters(), "lr": 1e-4, "weight_decay": 1e-2},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for bx, baux, by in loader:
            optimizer.zero_grad()
            logits = model(bx, baux)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        te_logits_fp32 = model(torch.from_numpy(X_te_cnn), torch.from_numpy(X_te_aux_s)).numpy()
        fp32_preds = np.argmax(te_logits_fp32, axis=1)
        fp32_acc, fp32_bal, fp32_f1 = compute_evaluation_metrics(test_lbl, fp32_preds)

        # Activation range calibration on training split (99.9th percentile for symmetric INT8)
        calib_idx = np.linspace(0, len(X_tr_cnn) - 1, min(1500, len(X_tr_cnn)), dtype=int)
        cal_x = torch.from_numpy(X_tr_cnn[calib_idx])
        a1_cal, a2_cal, _ = model.forward_features(cal_x)

        max_in = float(np.percentile(np.abs(X_tr_cnn[calib_idx]), 99.95))
        s_in = max(max_in / 127.0, 1e-6)

        max_aux = float(np.percentile(np.abs(X_tr_aux_s[calib_idx]), 99.95))
        s_aux = max(max_aux / 127.0, 1e-6)

        max_a1 = float(np.percentile(a1_cal.numpy(), 99.95))
        s_act1 = max(max_a1 / 127.0, 1e-6)

        max_a2 = float(np.percentile(a2_cal.numpy(), 99.95))
        s_act2 = max(max_a2 / 127.0, 1e-6)

        # Symmetric INT8 quantization of weights and int32 biases
        w1_fp = model.conv1.weight.numpy()
        b1_fp = model.conv1.bias.numpy()
        s_w1 = max(float(np.max(np.abs(w1_fp))) / 127.0, 1e-8)
        w1_q = np.clip(c_round_half_away(w1_fp / s_w1), -127, 127).astype(np.int8)
        b1_q = c_round_half_away(b1_fp / (s_in * s_w1)).astype(np.int32)
        conv1_eff_scale = (s_in * s_w1) / s_act1

        w2_fp = model.conv2.weight.numpy()
        b2_fp = model.conv2.bias.numpy()
        s_w2 = max(float(np.max(np.abs(w2_fp))) / 127.0, 1e-8)
        w2_q = np.clip(c_round_half_away(w2_fp / s_w2), -127, 127).astype(np.int8)
        b2_q = c_round_half_away(b2_fp / (s_act1 * s_w2)).astype(np.int32)
        conv2_eff_scale = (s_act1 * s_w2) / s_act2

        wfc_fp = model.fc_conv.weight.numpy()
        bfc_fp = model.fc_conv.bias.numpy()
        s_wfc = max(float(np.max(np.abs(wfc_fp))) / 127.0, 1e-8)
        wfc_q = np.clip(c_round_half_away(wfc_fp / s_wfc), -127, 127).astype(np.int8)
        fc_eff_scale = s_act2 * s_wfc
        bfc_q = c_round_half_away(bfc_fp / fc_eff_scale).astype(np.int32)

        waux_fp = model.fc_aux.weight.numpy()
        s_waux = max(float(np.max(np.abs(waux_fp))) / 127.0, 1e-8)
        waux_q = np.clip(c_round_half_away(waux_fp / s_waux), -127, 127).astype(np.int8)
        aux_fc_eff_scale = (s_aux * s_waux) / fc_eff_scale

    # Quantize test inputs and evaluate bit-exact INT8 C99 simulation
    x_te_q, aux_te_q = quantize_windows_for_c99(
        raw_windows=test_win,
        aux_features=X_te_aux,
        in_mean=in_mean,
        in_inv_std=in_inv_std,
        in_quant_scale=s_in,
        aux_mean=aux_mean,
        aux_inv_std=aux_inv_std,
        aux_quant_scale=s_aux,
    )

    temp_art = QuantizedCnn1DArtifacts(
        dataset_name=dataset_name,
        subject_id=subject_id,
        in_channels=in_channels,
        num_classes=num_classes,
        num_aux=num_aux,
        total_params=total_params,
        in_mean=in_mean,
        in_inv_std=in_inv_std,
        in_quant_scale=float(s_in),
        aux_mean=aux_mean,
        aux_inv_std=aux_inv_std,
        aux_quant_scale=float(s_aux),
        conv1_weights_q=w1_q.ravel(),
        conv1_bias_q=b1_q.ravel(),
        conv1_eff_scale=float(conv1_eff_scale),
        conv2_weights_q=w2_q.ravel(),
        conv2_bias_q=b2_q.ravel(),
        conv2_eff_scale=float(conv2_eff_scale),
        fc_weights_q=wfc_q.ravel(),
        fc_bias_q=bfc_q.ravel(),
        fc_eff_scale=float(fc_eff_scale),
        aux_fc_weights_q=waux_q.ravel(),
        aux_fc_eff_scale=float(aux_fc_eff_scale),
        fp32_top1_acc=round(fp32_acc, 6),
        fp32_bal_acc=round(fp32_bal, 6),
        fp32_macro_f1=round(fp32_f1, 6),
        int8_top1_acc=0.0,
        int8_bal_acc=0.0,
        int8_macro_f1=0.0,
    )

    int8_preds, int8_logits = simulate_c99_int8_cnn1d(temp_art, x_te_q, aux_te_q)
    int8_acc, int8_bal, int8_f1 = compute_evaluation_metrics(test_lbl, int8_preds)

    final_art = QuantizedCnn1DArtifacts(
        **{
            **asdict(temp_art),
            "int8_top1_acc": round(int8_acc, 6),
            "int8_bal_acc": round(int8_bal, 6),
            "int8_macro_f1": round(int8_f1, 6),
        }
    )
    return final_art, x_te_q, aux_te_q, int8_preds, int8_logits


def write_oracle_test_vectors(
    bin_path: Path,
    dataset_code: int,
    art: QuantizedCnn1DArtifacts,
    x_te_q: np.ndarray,
    aux_te_q: np.ndarray,
    test_lbl: np.ndarray,
    oracle_preds: np.ndarray,
    oracle_logits: np.ndarray,
    num_vectors: int = 50,
) -> None:
    """Write binary test oracle containing `num_vectors` consecutive test windows."""
    n_vec = min(num_vectors, len(test_lbl))
    # Select 50 consecutive windows spanning active gesture transitions
    start_idx = 0
    header = struct.pack(
        "<IIIIII",
        0x434E4E31,  # Magic "CNN1"
        dataset_code,
        art.in_channels,
        art.num_classes,
        art.num_aux,
        n_vec,
    )
    with bin_path.open("wb") as f:
        f.write(header)
        for i in range(start_idx, start_idx + n_vec):
            f.write(struct.pack("<ii", int(test_lbl[i]), int(oracle_preds[i])))
            f.write(np.ascontiguousarray(oracle_logits[i], dtype="<i4").tobytes())
            f.write(np.ascontiguousarray(x_te_q[i], dtype="<i1").tobytes())
            f.write(np.ascontiguousarray(aux_te_q[i], dtype="<i1").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and quantize 1D-CNN in INT8 for ESP32 QEMU.")
    parser.add_argument("--ozdemir-subject", default="S1")
    parser.add_argument("--ninapro-subject", default="S4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "cnn1d_qemu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=====================================================================")
    print(" STEP 1: TRAINING & INT8 SYMMETRIC QUANTIZATION OF 1D-CNN (HOST)")
    print("=====================================================================")

    # 1. Ozdemir 4CH (10 classes)
    raw_oz = load_ozdemir_recording(args.ozdemir_subject, ROOT / "data" / "raw" / "ozdemir", use_filtered=True)
    q_oz = quantize_ozdemir(raw_oz, precision="int16", v_max=OZDEMIR_V_FS_MV)
    win_oz, lbl_oz, reps_oz = segment_ozdemir_windows(q_oz, fs=2000)
    tr_mask_oz = np.isin(reps_oz, [1, 3, 4])
    te_mask_oz = np.isin(reps_oz, [2, 5])
    tr_win_oz, tr_lbl_oz = win_oz[tr_mask_oz], lbl_oz[tr_mask_oz]
    te_win_oz, te_lbl_oz = win_oz[te_mask_oz], lbl_oz[te_mask_oz]

    cal_oz = ThresholdCalibrator(default_r=0.5, num_channels=4).calibrate(
        train_emg=tr_win_oz, train_labels=tr_lbl_oz, rest_label=0, r_multiplier=0.5
    )
    ext_oz = TimeDomainFeatureExtractor(epsilon=cal_oz.epsilon_channels, num_channels=4)
    X_tr_oz = ext_oz.extract(tr_win_oz)
    X_te_oz = ext_oz.extract(te_win_oz)

    oz_art, oz_xq, oz_auxq, oz_preds, oz_logits = train_and_quantize_dataset(
        dataset_name="Ozdemir_4CH",
        subject_id=args.ozdemir_subject,
        train_win=tr_win_oz,
        train_lbl=tr_lbl_oz,
        test_win=te_win_oz,
        test_lbl=te_lbl_oz,
        X_tr_aux=X_tr_oz,
        X_te_aux=X_te_oz,
        num_classes=10,
        epochs=40,
        seed=args.seed,
    )
    print(
        f"[Ozdemir {args.ozdemir_subject} 4CH] Params: {oz_art.total_params:,} | "
        f"PyTorch FP32 Acc: {oz_art.fp32_top1_acc*100:.2f}% | "
        f"Quantized INT8 Acc: {oz_art.int8_top1_acc*100:.2f}% (F1: {oz_art.int8_macro_f1*100:.2f}%)"
    )

    # 2. NinaPro DB2 12CH (50 classes)
    loader_ni = NinaProDB2Loader(raw_dir=ROOT / "data" / "raw" / "ninapro_db2")
    raw_cache_ni: dict[str, RecordingData] = {
        eid: loader_ni.load_recording(args.ninapro_subject, eid) for eid in ("E1", "E2", "E3")
    }

    def quantized_load_ni(sid: str, eid: str) -> RecordingData:
        rec = raw_cache_ni[eid]
        from dataclasses import replace as dc_replace
        return dc_replace(rec, emg=quantize_ninapro(rec.emg, precision="int16", v_max=NINAPRO_V_FS_VOLTS))

    runner_ni = MultiExerciseWithinSubjectRunner()
    subj_ni = MultiExerciseSubject(subject_id=args.ninapro_subject, load_recording=quantized_load_ni)
    tr_win_ni, tr_lbl_ni, _, _ = runner_ni.prepare_repetition_windows(
        subj_ni, repetitions=runner_ni.TRAINING_REPETITIONS
    )
    te_win_ni, te_lbl_ni, _, _ = runner_ni.prepare_repetition_windows(
        subj_ni, repetitions=runner_ni.TEST_REPETITIONS
    )

    cal_ni = ThresholdCalibrator(default_r=0.5, num_channels=12).calibrate(
        train_emg=tr_win_ni, train_labels=tr_lbl_ni, rest_label=0, r_multiplier=0.5
    )
    ext_ni = TimeDomainFeatureExtractor(epsilon=cal_ni.epsilon_channels, num_channels=12)
    X_tr_ni = ext_ni.extract(tr_win_ni)
    X_te_ni = ext_ni.extract(te_win_ni)

    ni_art, ni_xq, ni_auxq, ni_preds, ni_logits = train_and_quantize_dataset(
        dataset_name="NinaPro_DB2_12CH",
        subject_id=args.ninapro_subject,
        train_win=tr_win_ni,
        train_lbl=tr_lbl_ni,
        test_win=te_win_ni,
        test_lbl=te_lbl_ni,
        X_tr_aux=X_tr_ni,
        X_te_aux=X_te_ni,
        num_classes=50,
        epochs=35,
        seed=args.seed,
    )
    print(
        f"[NinaPro {args.ninapro_subject} 12CH] Params: {ni_art.total_params:,} | "
        f"PyTorch FP32 Acc: {ni_art.fp32_top1_acc*100:.2f}% | "
        f"Quantized INT8 Acc: {ni_art.int8_top1_acc*100:.2f}% (F1: {ni_art.int8_macro_f1*100:.2f}%)"
    )

    # Save oracle binary test vectors (50 consecutive windows + full test set)
    oz_vec_50 = args.out_dir / "cnn1d_test_vectors_ozdemir_50.bin"
    oz_vec_full = args.out_dir / "cnn1d_test_vectors_ozdemir_full.bin"
    ni_vec_50 = args.out_dir / "cnn1d_test_vectors_ninapro_50.bin"
    ni_vec_full = args.out_dir / "cnn1d_test_vectors_ninapro_full.bin"

    write_oracle_test_vectors(oz_vec_50, 0, oz_art, oz_xq, oz_auxq, te_lbl_oz, oz_preds, oz_logits, num_vectors=50)
    write_oracle_test_vectors(
        oz_vec_full, 0, oz_art, oz_xq, oz_auxq, te_lbl_oz, oz_preds, oz_logits, num_vectors=len(te_lbl_oz)
    )
    write_oracle_test_vectors(ni_vec_50, 1, ni_art, ni_xq, ni_auxq, te_lbl_ni, ni_preds, ni_logits, num_vectors=50)
    write_oracle_test_vectors(
        ni_vec_full, 1, ni_art, ni_xq, ni_auxq, te_lbl_ni, ni_preds, ni_logits, num_vectors=len(te_lbl_ni)
    )

    # Also copy canonical 50-window test vector file to cnn1d_test_vectors.bin
    (args.out_dir / "cnn1d_test_vectors.bin").write_bytes(oz_vec_50.read_bytes())

    # Save NPZ and JSON summary for export_cnn1d_c_header.py
    npz_path = args.out_dir / "cnn1d_model_artifacts.npz"
    np.savez(
        npz_path,
        oz_in_mean=oz_art.in_mean,
        oz_in_inv_std=oz_art.in_inv_std,
        oz_in_quant_scale=np.array([oz_art.in_quant_scale], dtype=np.float32),
        oz_aux_mean=oz_art.aux_mean,
        oz_aux_inv_std=oz_art.aux_inv_std,
        oz_aux_quant_scale=np.array([oz_art.aux_quant_scale], dtype=np.float32),
        oz_conv1_w=oz_art.conv1_weights_q,
        oz_conv1_b=oz_art.conv1_bias_q,
        oz_conv1_s=np.array([oz_art.conv1_eff_scale], dtype=np.float32),
        oz_conv2_w=oz_art.conv2_weights_q,
        oz_conv2_b=oz_art.conv2_bias_q,
        oz_conv2_s=np.array([oz_art.conv2_eff_scale], dtype=np.float32),
        oz_fc_w=oz_art.fc_weights_q,
        oz_fc_b=oz_art.fc_bias_q,
        oz_fc_s=np.array([oz_art.fc_eff_scale], dtype=np.float32),
        oz_aux_w=oz_art.aux_fc_weights_q,
        oz_aux_s=np.array([oz_art.aux_fc_eff_scale], dtype=np.float32),
        ni_in_mean=ni_art.in_mean,
        ni_in_inv_std=ni_art.in_inv_std,
        ni_in_quant_scale=np.array([ni_art.in_quant_scale], dtype=np.float32),
        ni_aux_mean=ni_art.aux_mean,
        ni_aux_inv_std=ni_art.aux_inv_std,
        ni_aux_quant_scale=np.array([ni_art.aux_quant_scale], dtype=np.float32),
        ni_conv1_w=ni_art.conv1_weights_q,
        ni_conv1_b=ni_art.conv1_bias_q,
        ni_conv1_s=np.array([ni_art.conv1_eff_scale], dtype=np.float32),
        ni_conv2_w=ni_art.conv2_weights_q,
        ni_conv2_b=ni_art.conv2_bias_q,
        ni_conv2_s=np.array([ni_art.conv2_eff_scale], dtype=np.float32),
        ni_fc_w=ni_art.fc_weights_q,
        ni_fc_b=ni_art.fc_bias_q,
        ni_fc_s=np.array([ni_art.fc_eff_scale], dtype=np.float32),
        ni_aux_w=ni_art.aux_fc_weights_q,
        ni_aux_s=np.array([ni_art.aux_fc_eff_scale], dtype=np.float32),
    )

    summary_json: dict[str, Any] = {
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "ozdemir": {
            "subject_id": oz_art.subject_id,
            "in_channels": oz_art.in_channels,
            "num_classes": oz_art.num_classes,
            "total_params": oz_art.total_params,
            "fp32_top1_acc": oz_art.fp32_top1_acc,
            "fp32_bal_acc": oz_art.fp32_bal_acc,
            "fp32_macro_f1": oz_art.fp32_macro_f1,
            "int8_top1_acc": oz_art.int8_top1_acc,
            "int8_bal_acc": oz_art.int8_bal_acc,
            "int8_macro_f1": oz_art.int8_macro_f1,
            "num_test_windows": int(len(te_lbl_oz)),
        },
        "ninapro": {
            "subject_id": ni_art.subject_id,
            "in_channels": ni_art.in_channels,
            "num_classes": ni_art.num_classes,
            "total_params": ni_art.total_params,
            "fp32_top1_acc": ni_art.fp32_top1_acc,
            "fp32_bal_acc": ni_art.fp32_bal_acc,
            "fp32_macro_f1": ni_art.fp32_macro_f1,
            "int8_top1_acc": ni_art.int8_top1_acc,
            "int8_bal_acc": ni_art.int8_bal_acc,
            "int8_macro_f1": ni_art.int8_macro_f1,
            "num_test_windows": int(len(te_lbl_ni)),
        },
    }
    (args.out_dir / "cnn1d_host_training_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )
    print(f"Saved artifacts to {npz_path} and test vectors to {args.out_dir} in {time.time() - t0:.2f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
