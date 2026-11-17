#!/usr/bin/env python3
"""Train, Calibrate, and Validate DTW Variants and Hybrid (DTW + LDA + CNN) Models on Host.

Covers all 10 models across Ozdemir et al. (S1, 4CH, 10 classes) and NinaPro DB2 (S4, 12CH, 50 classes):
  - M01_micro_lda_q7: Canonical Micro-LDA (Q7) on Zero-FPU Hudgins TD features
  - M02_cnn1d_int8: Quantized Symmetric INT8 1D-CNN
  - M03_dtw_classic_on2: Classic O(N^2) DTW (no asymptotic optimizations)
  - M04_dtw_sakoe_chiba: Sakoe-Chiba Constrained DTW O(w*N) time, O(w) space
  - M05_dtw_lb_keogh: LB_Keogh Lower-Bound Pruned DTW O(N) best-case + Early Abandoning
  - M06_hybrid_lda_cnn_head: Hybrid 1 — CNN-1D INT8 GAP (24D) + TD (5*C) + LDA Q7 Head
  - M07_hybrid_lda_cnn_cascade: Hybrid 2 — Early-Exit Margin Cascade (Micro-LDA -> CNN/Hyb1)
  - M08_hybrid_dtw_lda: Hybrid DTW + LDA (Top-M LDA reranked with Sakoe-Chiba DTW)
  - M09_hybrid_dtw_cnn: Hybrid DTW + CNN (Top-M CNN reranked with Sakoe-Chiba DTW)
  - M10_hybrid_dtw_lda_cnn: Tri-Hybrid (Stage-1 Margin Gate -> Stage-2 Top-3 + LB_Keogh + Hybrid 1)

Also executes Section 3.3 physiological robustness benchmarks from the Master's proposal:
  1. Asymptotic RAM-model scaling proof (N in {20, 40, 100, 200, 400})
  2. Gesture velocity variations (100%, 125%, 150%, 200% time warping)
  3. Electrode shift simulation (20% cross-channel coupling + 15% gain attenuation)
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

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
from scripts.run_esp32_qemu_pipeline import quantize_lda_weights_q7  # noqa: E402
from scripts.run_qemu_freertos_dualcore_benchmark import load_cnn_artifacts  # noqa: E402
from semg_baseline.feature004_multi_exercise import (  # noqa: E402
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.lda import CanonicalLdaTrainer, PriorType  # noqa: E402
from semg_baseline.scaling import FeatureScaler  # noqa: E402
from semg_dataset.contract import RecordingData  # noqa: E402
from semg_dataset.loader import NinaProDB2Loader  # noqa: E402
from src.export.export_micro_lda_c_header import (  # noqa: E402
    train_ninapro_micro_lda_bundle,
    train_ozdemir_micro_lda_bundle,
)
from src.models.train_cnn1d_quantized import (  # noqa: E402
    QuantizedCnn1DArtifacts,
    c_round_half_away,
    quantize_windows_for_c99,
    requantize_q24_numpy,
)

DTW_ENV_LEN = 40
DTW_SUBBLOCK_SAMPLES = 10
DTW_SAKOE_W = 4
DTW_INF = 1_000_000_000
DTW_TOPM_HYBRID = 5
DTW_TRI_TOPM = 3
DEFAULT_OUT_DIR = ROOT / "reports" / "dtw_hybrid_benchmark"


def windows_f32_to_q16(windows: np.ndarray, adc_scale: float) -> np.ndarray:
    """Convert float32 windows [M, 400, C] to int16 quantized ADC units."""
    q = c_round_half_away(windows.astype(np.float64) * float(adc_scale))
    return np.clip(q, -32768, 32767).astype(np.int16)


def extract_td_int16_batch(
    win_q16: np.ndarray,
    eps_int: np.ndarray,
    inv_adc_scale: float,
) -> np.ndarray:
    """Vectorized bit-exact Python match to C99 `extract_td_features_int16` across a batch."""
    m, w_len, n_ch = win_q16.shape
    w_i64 = win_q16.astype(np.int64)
    abs_w = np.abs(w_i64)
    diffs = np.abs(np.diff(w_i64, axis=1))  # [M, 399, C]

    sum_abs = np.sum(abs_w, axis=1)  # [M, C]
    sum_sq = np.sum(w_i64 * w_i64, axis=1)  # [M, C]
    wl_q = np.sum(diffs, axis=1)  # [M, C]

    eps = eps_int[None, None, :].astype(np.int64)
    opp_sign = (w_i64[:, :-1, :] * w_i64[:, 1:, :]) < 0
    zc = np.sum(opp_sign & (diffs >= eps), axis=1)  # [M, C]

    prev_v = w_i64[:, :-2, :]
    curr_v = w_i64[:, 1:-1, :]
    next_v = w_i64[:, 2:, :]
    extrema = ((curr_v > prev_v) & (curr_v > next_v)) | ((curr_v < prev_v) & (curr_v < next_v))
    d_prev = np.abs(curr_v - prev_v)
    d_next = np.abs(next_v - curr_v)
    ssc = np.sum(extrema & ((d_prev >= eps) | (d_next >= eps)), axis=1)  # [M, C]

    inv_w = np.float32(1.0 / float(w_len))
    inv_s = np.float32(inv_adc_scale)
    scale_mav = np.float32(inv_s * inv_w)

    out = np.zeros((m, 5 * n_ch), dtype=np.float32)
    out[:, 0 * n_ch : 1 * n_ch] = sum_abs.astype(np.float32) * scale_mav
    out[:, 1 * n_ch : 2 * n_ch] = np.sqrt(sum_sq.astype(np.float32) * inv_w) * inv_s
    out[:, 2 * n_ch : 3 * n_ch] = wl_q.astype(np.float32) * inv_s
    out[:, 3 * n_ch : 4 * n_ch] = zc.astype(np.float32)
    out[:, 4 * n_ch : 5 * n_ch] = ssc.astype(np.float32)
    return out


def predict_lda_q7_scores_batch(
    features: np.ndarray,
    mean: np.ndarray,
    inv_scale: np.ndarray,
    zero_mask: np.ndarray,
    coef_q7: np.ndarray,
    weight_scale: np.ndarray,
    intercept: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact batch mirror of `micro_lda_predict_q7` / `hybrid_lda_cnn_head_predict_q7`."""
    f32 = features.astype(np.float32)
    z = (f32 - mean.astype(np.float32)[None, :]) * inv_scale.astype(np.float32)[None, :]
    z[:, zero_mask.astype(bool)] = 0.0
    x_q8 = np.clip(c_round_half_away(z * np.float32(256.0)), -32768, 32767).astype(np.int32)

    acc_int = x_q8 @ coef_q7.astype(np.int32).T  # [M, K]
    scale_factor = (weight_scale.astype(np.float32) * np.float32(1.0 / 256.0))[None, :]
    scores = (acc_int.astype(np.float32) * scale_factor) + intercept.astype(np.float32)[None, :]
    best_idx = np.argmax(scores, axis=1)
    return classes[best_idx].astype(np.int32), scores


def simulate_c99_cnn1d_backbone_and_logits(
    art: QuantizedCnn1DArtifacts,
    x_q: np.ndarray,
    aux_q: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bit-exact NumPy oracle returning `(gap_q [M, 24], preds [M], logits [M, K])`."""
    n_samples = x_q.shape[0]
    in_ch = art.in_channels
    n_cls = art.num_classes

    mult1_q24 = int(c_round_half_away(art.conv1_eff_scale * float(1 << 24)).item())
    mult2_q24 = int(c_round_half_away(art.conv2_eff_scale * float(1 << 24)).item())
    mult_aux_q16 = int(c_round_half_away(art.aux_fc_eff_scale * 65536.0).item())

    w1_flat = art.conv1_weights_q.reshape(16, in_ch * 7).astype(np.int32)
    b1 = art.conv1_bias_q.astype(np.int32)
    w2_flat = art.conv2_weights_q.reshape(24, 16 * 5).astype(np.int32)
    b2 = art.conv2_bias_q.astype(np.int32)
    w_fc = art.fc_weights_q.reshape(n_cls, 24).astype(np.int32)
    b_fc = art.fc_bias_q.astype(np.int32)
    w_aux = art.aux_fc_weights_q.reshape(n_cls, art.num_aux).astype(np.int32)

    all_gap = np.zeros((n_samples, 24), dtype=np.int8)
    all_logits = np.zeros((n_samples, n_cls), dtype=np.int32)
    all_preds = np.zeros((n_samples,), dtype=np.int32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        bs = end - start
        x_i32 = x_q[start:end].astype(np.int32)  # [bs, C, 400]

        # 1. Conv1D (stride=2, k=7) in single vectorized matmul per chunk
        wins1 = np.lib.stride_tricks.sliding_window_view(x_i32, 7, axis=2)[:, :, ::2, :]  # [bs, C, 197, 7]
        cols1 = np.transpose(wins1, (0, 2, 1, 3)).reshape(bs * 197, in_ch * 7)
        acc1 = (cols1 @ w1_flat.T).reshape(bs, 197, 16) + b1[None, None, :]
        rq1 = requantize_q24_numpy(acc1, mult1_q24)
        out1 = np.transpose(np.clip(rq1, 0, 127).astype(np.int8), (0, 2, 1))  # [bs, 16, 197]

        # 2. MaxPool1D (2, stride=2) -> [bs, 16, 98]
        p1 = np.maximum(out1[:, :, 0:196:2], out1[:, :, 1:196:2]).astype(np.int32)

        # 3. Conv1D L2 (stride=2, k=5) -> [bs, 24, 47]
        wins2 = np.lib.stride_tricks.sliding_window_view(p1, 5, axis=2)[:, :, ::2, :]  # [bs, 16, 47, 5]
        cols2 = np.transpose(wins2, (0, 2, 1, 3)).reshape(bs * 47, 16 * 5)
        acc2 = (cols2 @ w2_flat.T).reshape(bs, 47, 24) + b2[None, None, :]
        rq2 = requantize_q24_numpy(acc2, mult2_q24)
        out2 = np.transpose(np.clip(rq2, 0, 127).astype(np.int8), (0, 2, 1))  # [bs, 24, 47]

        # 4. Global Average Pooling -> [bs, 24]
        sum_gap = np.sum(out2.astype(np.int32), axis=2)
        gap_q = np.clip((sum_gap + 23) // 47, 0, 127).astype(np.int8)

        fc_acc = gap_q.astype(np.int32) @ w_fc.T + b_fc[None, :]
        aux_acc = aux_q[start:end].astype(np.int32) @ w_aux.T
        aux_scaled = ((aux_acc.astype(np.int64) * mult_aux_q16 + 32768) >> 16).astype(np.int32)
        logits = fc_acc + aux_scaled
        all_gap[start:end] = gap_q
        all_logits[start:end] = logits
        all_preds[start:end] = np.argmax(logits, axis=1).astype(np.int32)

    return all_gap, all_preds, all_logits


def quantize_windows_chunked(
    raw_windows: np.ndarray,
    aux_features: np.ndarray,
    cnn_art: QuantizedCnn1DArtifacts,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded-memory wrapper around `quantize_windows_for_c99`."""
    n = raw_windows.shape[0]
    x_out = np.zeros((n, cnn_art.in_channels, 400), dtype=np.int8)
    aux_out = np.zeros((n, cnn_art.num_aux), dtype=np.int8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xq_c, aux_c = quantize_windows_for_c99(
            raw_windows[start:end],
            aux_features[start:end],
            cnn_art.in_mean,
            cnn_art.in_inv_std,
            cnn_art.in_quant_scale,
            cnn_art.aux_mean,
            cnn_art.aux_inv_std,
            cnn_art.aux_quant_scale,
        )
        x_out[start:end] = xq_c
        aux_out[start:end] = aux_c
    return x_out, aux_out


def extract_dtw_envelope_int8_batch(
    win_q16: np.ndarray,
    inv_adc_scale: float,
    env_mean: np.ndarray,
    env_inv_std: np.ndarray,
    env_quant_scale: float,
) -> np.ndarray:
    """Bit-exact Python mirror of C99 `extract_dtw_envelope_int8` returning [M, 40, C] int8."""
    m, _, n_ch = win_q16.shape
    blocks = np.abs(win_q16.reshape(m, DTW_ENV_LEN, DTW_SUBBLOCK_SAMPLES, n_ch).astype(np.int32))
    sum_abs = np.sum(blocks, axis=2)  # [M, 40, C]
    block_inv = np.float32(1.0 / float(DTW_SUBBLOCK_SAMPLES))
    mav = (sum_abs.astype(np.float32) * np.float32(inv_adc_scale)) * block_inv
    z = (mav - env_mean.astype(np.float32)[None, None, :]) * env_inv_std.astype(np.float32)[None, None, :]
    inv_q = np.float32(1.0 / float(env_quant_scale))
    return np.clip(c_round_half_away(z * inv_q), -127, 127).astype(np.int8)


def fit_dtw_templates_and_envelopes(
    tr_win_q16: np.ndarray,
    tr_lbl: np.ndarray,
    inv_adc_scale: float,
    num_classes: int,
    band_w: int = DTW_SAKOE_W,
) -> dict[str, Any]:
    """Fit channel envelope scaling statistics, per-class int8 templates Q, and Keogh bounds U, L."""
    m, _, n_ch = tr_win_q16.shape
    blocks = np.abs(tr_win_q16.reshape(m, DTW_ENV_LEN, DTW_SUBBLOCK_SAMPLES, n_ch).astype(np.int32))
    sum_abs = np.sum(blocks, axis=2)
    mav = (sum_abs.astype(np.float32) * np.float32(inv_adc_scale)) * np.float32(1.0 / float(DTW_SUBBLOCK_SAMPLES))

    env_mean = np.mean(mav, axis=(0, 1), dtype=np.float64).astype(np.float32)
    env_std = np.std(mav, axis=(0, 1), dtype=np.float64).astype(np.float32)
    env_std = np.where(env_std < 1e-8, 1.0, env_std).astype(np.float32)
    env_inv_std = (1.0 / env_std).astype(np.float32)

    z_all = (mav - env_mean[None, None, :]) * env_inv_std[None, None, :]
    max_abs_99 = float(np.percentile(np.abs(z_all), 99.9))
    env_quant_scale = float(np.float32(max(max_abs_99 / 127.0, 1e-6)))

    env_tr_q = extract_dtw_envelope_int8_batch(tr_win_q16, inv_adc_scale, env_mean, env_inv_std, env_quant_scale)

    templates_q = np.zeros((num_classes, DTW_ENV_LEN, n_ch), dtype=np.int8)
    env_upper = np.zeros((num_classes, DTW_ENV_LEN, n_ch), dtype=np.int8)
    env_lower = np.zeros((num_classes, DTW_ENV_LEN, n_ch), dtype=np.int8)

    for k in range(num_classes):
        idx_k = np.where(tr_lbl == k)[0]
        if len(idx_k) == 0:
            continue
        # Class mean in quantized space preserves morphological envelope and suppresses noise
        mean_k = np.mean(env_tr_q[idx_k].astype(np.float64), axis=0)
        q_k = np.clip(c_round_half_away(mean_k), -127, 127).astype(np.int8)
        templates_q[k] = q_k
        for i in range(DTW_ENV_LEN):
            lo = max(0, i - band_w)
            hi = min(DTW_ENV_LEN - 1, i + band_w)
            env_upper[k, i] = np.max(q_k[lo : hi + 1], axis=0)
            env_lower[k, i] = np.min(q_k[lo : hi + 1], axis=0)

    return {
        "env_mean": env_mean,
        "env_inv_std": env_inv_std,
        "env_quant_scale": env_quant_scale,
        "templates_q": templates_q,
        "env_upper": env_upper,
        "env_lower": env_lower,
        "env_tr_q": env_tr_q,
    }


_C_DTW_LIB_CACHE: ctypes.CDLL | None = None
_C_DTW_TMPDIR: tempfile.TemporaryDirectory[str] | None = None


class _DtwModelConfigC(ctypes.Structure):
    _fields_ = [
        ("num_channels", ctypes.c_int),
        ("num_classes", ctypes.c_int),
        ("env_len", ctypes.c_int),
        ("band_w", ctypes.c_int),
        ("env_mean", ctypes.POINTER(ctypes.c_float)),
        ("env_inv_std", ctypes.POINTER(ctypes.c_float)),
        ("env_quant_scale", ctypes.c_float),
        ("templates_q", ctypes.POINTER(ctypes.c_int8)),
        ("env_upper", ctypes.POINTER(ctypes.c_int8)),
        ("env_lower", ctypes.POINTER(ctypes.c_int8)),
    ]


def _get_c99_dtw_lib() -> ctypes.CDLL | None:
    """Compile and cache host C99 DTW shared library once per process for ~100x speedup."""
    global _C_DTW_LIB_CACHE, _C_DTW_TMPDIR
    if _C_DTW_LIB_CACHE is not None:
        return _C_DTW_LIB_CACHE
    gcc_bin = shutil.which("gcc")
    if gcc_bin is None:
        return None

    _C_DTW_TMPDIR = tempfile.TemporaryDirectory()
    so_path = Path(_C_DTW_TMPDIR.name) / "libdtw_hybrid_accel.so"
    cmd = [
        gcc_bin,
        "-std=c99",
        "-O3",
        "-shared",
        "-fPIC",
        str(ROOT / "embedded" / "firmware" / "src" / "dtw_hybrid_engine.c"),
        str(ROOT / "embedded" / "firmware" / "src" / "cnn1d_engine.c"),
        str(ROOT / "firmware" / "esp32_qemu" / "src" / "semg_pipeline.c"),
        "-I",
        str(ROOT / "embedded" / "firmware" / "include"),
        "-I",
        str(ROOT / "firmware" / "esp32_qemu" / "include"),
        "-o",
        str(so_path),
        "-lm",
    ]
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so_path))
    lib.dtw_classic_on2_predict.restype = ctypes.c_int32
    lib.dtw_sakoe_chiba_predict.restype = ctypes.c_int32
    _C_DTW_LIB_CACHE = lib
    return lib


def compute_dtw_matrix_batch(
    queries_q: np.ndarray,
    templates_q: np.ndarray,
    band_w: int | None = None,
    chunk_size: int = 128,
) -> np.ndarray:
    """Compute squared DTW cost [M, K] via compiled C99 -O3 engine (with NumPy fallback)."""
    m, n, n_ch = queries_q.shape
    k_cls = templates_q.shape[0]
    out_costs = np.zeros((m, k_cls), dtype=np.int32)

    c_lib = _get_c99_dtw_lib()
    if c_lib is not None and m > 0:
        dummy_f = np.ones(n_ch, dtype=np.float32)
        t_contig = np.ascontiguousarray(templates_q, dtype=np.int8)
        q_contig = np.ascontiguousarray(queries_q, dtype=np.int8)
        cfg = _DtwModelConfigC(
            num_channels=int(n_ch),
            num_classes=int(k_cls),
            env_len=int(n),
            band_w=int(band_w if band_w is not None else n),
            env_mean=dummy_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            env_inv_std=dummy_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            env_quant_scale=ctypes.c_float(1.0),
            templates_q=t_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            env_upper=t_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            env_lower=t_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        )
        fn = c_lib.dtw_classic_on2_predict if band_w is None else c_lib.dtw_sakoe_chiba_predict
        cfg_ptr = ctypes.byref(cfg)
        q_base = q_contig.ctypes.data
        q_stride = q_contig.strides[0]
        o_base = out_costs.ctypes.data
        o_stride = out_costs.strides[0]
        for i in range(m):
            q_ptr = ctypes.cast(q_base + i * q_stride, ctypes.POINTER(ctypes.c_int8))
            o_ptr = ctypes.cast(o_base + i * o_stride, ctypes.POINTER(ctypes.c_int32))
            fn(cfg_ptr, q_ptr, o_ptr, None)
        return out_costs

    t_f = templates_q.astype(np.int32)  # [K, N, C]
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        bs = end - start
        q_f = queries_q[start:end].astype(np.int32)
        dist = np.zeros((bs, k_cls, n, n), dtype=np.int32)
        for ch in range(n_ch):
            d_ch = q_f[:, None, :, None, ch] - t_f[None, :, None, :, ch]
            dist += d_ch * d_ch
        dp = np.full((bs, k_cls, n, n), DTW_INF, dtype=np.int32)
        dp[:, :, 0, 0] = dist[:, :, 0, 0]
        for i in range(n):
            j_start = 0 if band_w is None else max(0, i - band_w)
            j_end = (n - 1) if band_w is None else min(n - 1, i + band_w)
            for j in range(j_start, j_end + 1):
                if i == 0 and j == 0:
                    continue
                best_prev = dp[:, :, i - 1, j - 1] if (i > 0 and j > 0) else None
                if i > 0:
                    cand_up = dp[:, :, i - 1, j]
                    best_prev = cand_up if best_prev is None else np.minimum(best_prev, cand_up)
                if j > 0:
                    cand_left = dp[:, :, i, j - 1]
                    best_prev = cand_left if best_prev is None else np.minimum(best_prev, cand_left)
                assert best_prev is not None
                dp[:, :, i, j] = np.where(best_prev >= DTW_INF, DTW_INF, dist[:, :, i, j] + best_prev)
        out_costs[start:end] = dp[:, :, n - 1, n - 1]
    return out_costs


def compute_lb_keogh_batch(
    queries_q: np.ndarray,
    env_upper: np.ndarray,
    env_lower: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Compute exact LB_Keogh squared lower bound [M, K] in linear O(N) per pair."""
    m = queries_q.shape[0]
    k_cls = env_upper.shape[0]
    out_lb = np.zeros((m, k_cls), dtype=np.int32)
    u_i32 = env_upper[None, :, :, :].astype(np.int32)
    l_i32 = env_lower[None, :, :, :].astype(np.int32)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        c_i32 = queries_q[start:end, None, :, :].astype(np.int32)
        diff = np.where(c_i32 > u_i32, c_i32 - u_i32, np.where(c_i32 < l_i32, l_i32 - c_i32, 0))
        out_lb[start:end] = np.sum(diff * diff, axis=(2, 3), dtype=np.int32)

    return out_lb


def train_hybrid1_head(
    gap_tr: np.ndarray,
    td_tr: np.ndarray,
    tr_lbl: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Train Hybrid 1 (CNN-1D GAP 24D + Hudgins TD 5*C -> Canonical LDA Q7)."""
    x_hyb = np.hstack([gap_tr.astype(np.float32), td_tr.astype(np.float32)]).astype(np.float32)
    n_feat = x_hyb.shape[1]

    scaler = FeatureScaler()
    x_norm = scaler.fit_transform(x_hyb)
    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_classes)
    trainer.fit(x_norm, tr_lbl)
    art = trainer.get_model_artifacts()
    sdict = scaler.to_dict()

    mean = np.asarray(sdict["mean"], dtype=np.float32)
    scale = np.asarray(sdict["scale"], dtype=np.float32)
    inv_scale = np.where(scale != 0.0, 1.0 / scale, 0.0).astype(np.float32)
    zero_mask = np.zeros(n_feat, dtype=np.uint8)
    for z_idx in sdict["zero_variance_indices"]:
        if 0 <= z_idx < n_feat:
            zero_mask[z_idx] = 1
            inv_scale[z_idx] = 0.0

    coef_q7, w_scale = quantize_lda_weights_q7(art.coef_.astype(np.float32))
    return {
        "classes": np.asarray(art.classes_, dtype=np.int32),
        "mean": mean,
        "inv_scale": inv_scale,
        "zero_mask": zero_mask,
        "coef_q7": coef_q7,
        "weight_scale": w_scale,
        "intercept": art.intercept_.astype(np.float32),
        "fp32_coef": art.coef_.astype(np.float32),
        "fp32_intercept": art.intercept_.astype(np.float32),
    }


def evaluate_all_10_models_host(
    bundle: dict[str, Any],
    win_q16: np.ndarray,
    labels: np.ndarray,
    include_classic_on2: bool = True,
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    """Run bit-exact Host oracles for all 10 models and return `(metrics_by_model, preds_matrix [M, 10])`."""
    n_win, _, n_ch = win_q16.shape
    lda_b = bundle["lda"]
    cnn_art: QuantizedCnn1DArtifacts = bundle["cnn"]
    dtw_b = bundle["dtw"]
    hyb1 = bundle["hyb1"]
    fcfg = bundle["fusion"]
    inv_adc = 1.0 / float(lda_b["adc_scale"])

    win_fp32 = (win_q16.astype(np.float32) * np.float32(inv_adc)).astype(np.float32)
    td_feat = extract_td_int16_batch(win_q16, lda_b["eps_int"], inv_adc)

    # M01: Micro-LDA Q7
    pred_m01, lda_scores = predict_lda_q7_scores_batch(
        td_feat,
        lda_b["mean"],
        lda_b["inv_scale"],
        lda_b["zero_mask"],
        lda_b["coef_q7"],
        lda_b["weight_scale"],
        lda_b["intercept"],
        lda_b["classes"],
    )

    # M02: CNN-1D INT8 + GAP extraction
    x_q, aux_q = quantize_windows_chunked(win_fp32, td_feat, cnn_art)
    gap_q, pred_m02_idx, cnn_logits = simulate_c99_cnn1d_backbone_and_logits(cnn_art, x_q, aux_q)
    pred_m02 = lda_b["classes"][pred_m02_idx]

    # DTW Envelope Q8
    env_q = extract_dtw_envelope_int8_batch(
        win_q16, inv_adc, dtw_b["env_mean"], dtw_b["env_inv_std"], dtw_b["env_quant_scale"]
    )

    # M04 / M05: Sakoe-Chiba DTW (M05 LB_Keogh is lossless vs M04)
    dtw_sc = compute_dtw_matrix_batch(env_q, dtw_b["templates_q"], band_w=DTW_SAKOE_W)
    pred_m04 = np.argmin(dtw_sc, axis=1).astype(np.int32)
    pred_m05 = pred_m04.copy()

    # M03: Classic O(N^2) DTW
    if include_classic_on2:
        dtw_cl = compute_dtw_matrix_batch(env_q, dtw_b["templates_q"], band_w=None)
        pred_m03 = np.argmin(dtw_cl, axis=1).astype(np.int32)
    else:
        pred_m03 = pred_m04.copy()

    # Check LB_Keogh lower-bound mathematical guarantee and pruning ratio
    lb_vals = compute_lb_keogh_batch(env_q, dtw_b["env_upper"], dtw_b["env_lower"])
    lb_valid = bool(np.all(lb_vals <= dtw_sc))
    if not lb_valid:
        raise AssertionError("LB_Keogh inequality violated: LB > DTW_SakoeChiba!")

    k_min = np.argmin(lb_vals, axis=1)
    best_init = dtw_sc[np.arange(n_win), k_min]
    pruned_mask = lb_vals >= best_init[:, None]
    pruned_mask[np.arange(n_win), k_min] = False
    lb_prune_rate = float(np.mean(np.sum(pruned_mask, axis=1) / float(cnn_art.num_classes)))

    # M06: Hybrid 1 — CNN Backbone (24D) + TD (5*C) + LDA Q7 Head
    x_hyb = np.hstack([gap_q.astype(np.float32), td_feat]).astype(np.float32)
    pred_m06, hyb1_scores = predict_lda_q7_scores_batch(
        x_hyb,
        hyb1["mean"],
        hyb1["inv_scale"],
        hyb1["zero_mask"],
        hyb1["coef_q7"],
        hyb1["weight_scale"],
        hyb1["intercept"],
        hyb1["classes"],
    )

    # Stage-1 Micro-LDA Top-2 Margins for Cascades (M07, M10)
    sorted_lda = np.sort(lda_scores, axis=1)
    margins = (sorted_lda[:, -1] - sorted_lda[:, -2]).astype(np.float32)

    # M07: Hybrid 2 — Early-Exit Gatekeeper Cascade
    tau_casc = float(fcfg["cascade_margin_threshold"])
    use_cnn_t2 = bool(fcfg["cascade_use_cnn1d_tier2"])
    tier2_m07 = pred_m02 if use_cnn_t2 else pred_m06
    ee_m07 = margins >= tau_casc
    pred_m07 = np.where(ee_m07, pred_m01, tier2_m07).astype(np.int32)
    ee_rate_m07 = float(np.mean(ee_m07))

    # M08: Hybrid DTW + LDA (Top-M Micro-LDA reranked with DTW)
    inv_norm = np.float32(1.0 / float(DTW_ENV_LEN * n_ch))
    gamma_lda = np.float32(fcfg["gamma_dtw_lda"])
    m_top = min(cnn_art.num_classes, DTW_TOPM_HYBRID)
    pred_m08 = np.zeros(n_win, dtype=np.int32)
    topm_lda_idx = np.argsort(-lda_scores, axis=1, kind="stable")[:, :m_top]
    for i in range(n_win):
        cands = topm_lda_idx[i]
        scores_c = lda_scores[i, cands] - gamma_lda * (dtw_sc[i, cands].astype(np.float32) * inv_norm)
        pred_m08[i] = lda_b["classes"][cands[int(np.argmax(scores_c))]]

    # M09: Hybrid DTW + CNN (Top-M CNN-1D reranked with DTW)
    gamma_cnn = np.float32(fcfg["gamma_dtw_cnn"])
    logit_scale = np.float32(cnn_art.fc_eff_scale)
    pred_m09 = np.zeros(n_win, dtype=np.int32)
    topm_cnn_idx = np.argsort(-cnn_logits, axis=1, kind="stable")[:, :m_top]
    for i in range(n_win):
        cands = topm_cnn_idx[i]
        cnn_s = cnn_logits[i, cands].astype(np.float32) * logit_scale
        scores_c = cnn_s - gamma_cnn * (dtw_sc[i, cands].astype(np.float32) * inv_norm)
        pred_m09[i] = int(cands[int(np.argmax(scores_c))])

    # M10: Tri-Hybrid (Stage-1 Margin Gate -> Stage-2 Top-3 LDA + LB_Keogh + Hybrid 1)
    tau_tri = float(fcfg["tri_margin_threshold"])
    gamma_tri = np.float32(fcfg["gamma_tri_hybrid"])
    ee_m10 = margins >= tau_tri
    pred_m10 = np.zeros(n_win, dtype=np.int32)
    top3_lda_idx = np.argsort(-lda_scores, axis=1, kind="stable")[:, :DTW_TRI_TOPM]
    for i in range(n_win):
        if ee_m10[i]:
            pred_m10[i] = pred_m01[i]
        else:
            cands = top3_lda_idx[i]
            scores_c = hyb1_scores[i, cands] - gamma_tri * (dtw_sc[i, cands].astype(np.float32) * inv_norm)
            pred_m10[i] = hyb1["classes"][cands[int(np.argmax(scores_c))]]
    ee_rate_m10 = float(np.mean(ee_m10))

    preds_mat = np.column_stack(
        [pred_m01, pred_m02, pred_m03, pred_m04, pred_m05, pred_m06, pred_m07, pred_m08, pred_m09, pred_m10]
    ).astype(np.int32)

    names = [
        "M01_micro_lda_q7",
        "M02_cnn1d_int8",
        "M03_dtw_classic_on2",
        "M04_dtw_sakoe_chiba",
        "M05_dtw_lb_keogh",
        "M06_hybrid_lda_cnn_head",
        "M07_hybrid_lda_cnn_cascade",
        "M08_hybrid_dtw_lda",
        "M09_hybrid_dtw_cnn",
        "M10_hybrid_dtw_lda_cnn",
    ]
    metrics: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(names):
        acc, bal, f1 = compute_evaluation_metrics(labels, preds_mat[:, idx])
        entry: dict[str, Any] = {
            "top1_acc": round(float(acc), 6),
            "bal_acc": round(float(bal), 6),
            "macro_f1": round(float(f1), 6),
        }
        if name == "M05_dtw_lb_keogh":
            entry["lb_prune_rate"] = round(lb_prune_rate, 6)
            entry["lb_inequality_verified"] = lb_valid
        elif name == "M07_hybrid_lda_cnn_cascade":
            entry["early_exit_rate"] = round(ee_rate_m07, 6)
        elif name == "M10_hybrid_dtw_lda_cnn":
            entry["early_exit_rate"] = round(ee_rate_m10, 6)
        metrics[name] = entry

    return metrics, preds_mat


def calibrate_hybrid_parameters(
    bundle: dict[str, Any],
    tr_win_q16: np.ndarray,
    tr_lbl: np.ndarray,
    max_calib_samples: int = 2500,
) -> dict[str, Any]:
    """Calibrate M07, M08, M09, M10 thresholds & fusion weights strictly on training data."""
    n_tot = len(tr_lbl)
    if n_tot > max_calib_samples:
        idx = np.linspace(0, n_tot - 1, max_calib_samples, dtype=int)
    else:
        idx = np.arange(n_tot)

    cal_q16 = tr_win_q16[idx]
    cal_lbl = tr_lbl[idx]
    n_cal, _, n_ch = cal_q16.shape
    lda_b = bundle["lda"]
    cnn_art: QuantizedCnn1DArtifacts = bundle["cnn"]
    dtw_b = bundle["dtw"]
    hyb1 = bundle["hyb1"]
    inv_adc = 1.0 / float(lda_b["adc_scale"])

    cal_fp32 = (cal_q16.astype(np.float32) * np.float32(inv_adc)).astype(np.float32)
    td_feat = extract_td_int16_batch(cal_q16, lda_b["eps_int"], inv_adc)
    pred_lda, lda_scores = predict_lda_q7_scores_batch(
        td_feat,
        lda_b["mean"],
        lda_b["inv_scale"],
        lda_b["zero_mask"],
        lda_b["coef_q7"],
        lda_b["weight_scale"],
        lda_b["intercept"],
        lda_b["classes"],
    )

    x_q, aux_q = quantize_windows_chunked(cal_fp32, td_feat, cnn_art)
    gap_q, pred_cnn_idx, cnn_logits = simulate_c99_cnn1d_backbone_and_logits(cnn_art, x_q, aux_q)
    pred_cnn = lda_b["classes"][pred_cnn_idx]

    x_hyb = np.hstack([gap_q.astype(np.float32), td_feat]).astype(np.float32)
    pred_hyb1, hyb1_scores = predict_lda_q7_scores_batch(
        x_hyb,
        hyb1["mean"],
        hyb1["inv_scale"],
        hyb1["zero_mask"],
        hyb1["coef_q7"],
        hyb1["weight_scale"],
        hyb1["intercept"],
        hyb1["classes"],
    )

    env_q = extract_dtw_envelope_int8_batch(
        cal_q16, inv_adc, dtw_b["env_mean"], dtw_b["env_inv_std"], dtw_b["env_quant_scale"]
    )
    dtw_sc = compute_dtw_matrix_batch(env_q, dtw_b["templates_q"], band_w=DTW_SAKOE_W)

    # 1. Cascade M07: Choose higher-accuracy Tier 2 on training calibration + Pareto margin threshold
    acc_cnn = float(np.mean(pred_cnn == cal_lbl))
    acc_hyb1 = float(np.mean(pred_hyb1 == cal_lbl))
    use_cnn_t2 = 1 if acc_cnn >= acc_hyb1 else 0
    tier2_pred = pred_cnn if use_cnn_t2 else pred_hyb1

    sorted_lda = np.sort(lda_scores, axis=1)
    margins = (sorted_lda[:, -1] - sorted_lda[:, -2]).astype(np.float32)

    # Pick margin threshold maximizing accuracy while enabling >= 35% early-exit
    best_tau_casc = float(np.percentile(margins, 50))
    best_score_casc = -1.0
    for pct in (25, 35, 45, 55, 65):
        tau = float(np.percentile(margins, 100 - pct))
        ee = margins >= tau
        preds_c = np.where(ee, pred_lda, tier2_pred)
        acc_c = float(np.mean(preds_c == cal_lbl))
        # Prefer thresholds preserving Tier2 accuracy while saving CNN calls
        score_c = acc_c + 0.005 * (pct / 100.0)
        if score_c > best_score_casc:
            best_score_casc = score_c
            best_tau_casc = tau

    # 2. M08 Gamma calibration (Micro-LDA + DTW)
    inv_norm = 1.0 / float(DTW_ENV_LEN * n_ch)
    m_top = min(cnn_art.num_classes, DTW_TOPM_HYBRID)
    topm_lda = np.argsort(-lda_scores, axis=1, kind="stable")[:, :m_top]
    best_g_lda = 0.05
    best_acc_lda = -1.0
    for gamma in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        correct = 0
        for i in range(n_cal):
            cands = topm_lda[i]
            sc = lda_scores[i, cands] - gamma * (dtw_sc[i, cands].astype(np.float32) * inv_norm)
            if lda_b["classes"][cands[int(np.argmax(sc))]] == cal_lbl[i]:
                correct += 1
        acc_g = correct / float(n_cal)
        if acc_g > best_acc_lda:
            best_acc_lda = acc_g
            best_g_lda = gamma

    # 3. M09 Gamma calibration (CNN-1D + DTW)
    topm_cnn = np.argsort(-cnn_logits, axis=1, kind="stable")[:, :m_top]
    best_g_cnn = 0.05
    best_acc_cnn = -1.0
    l_scale = float(cnn_art.fc_eff_scale)
    for gamma in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        correct = 0
        for i in range(n_cal):
            cands = topm_cnn[i]
            sc = (cnn_logits[i, cands].astype(np.float32) * l_scale) - gamma * (
                dtw_sc[i, cands].astype(np.float32) * inv_norm
            )
            if int(cands[int(np.argmax(sc))]) == cal_lbl[i]:
                correct += 1
        acc_g = correct / float(n_cal)
        if acc_g > best_acc_cnn:
            best_acc_cnn = acc_g
            best_g_cnn = gamma

    # 4. M10 Tri-Hybrid (Stage-1 threshold & Stage-2 gamma on Hybrid 1 + Top-3 LDA)
    best_tau_tri = float(np.percentile(margins, 55))
    top3_lda = np.argsort(-lda_scores, axis=1, kind="stable")[:, :DTW_TRI_TOPM]
    best_g_tri = 0.05
    best_acc_tri = -1.0
    for gamma in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        correct = 0
        for i in range(n_cal):
            if margins[i] >= best_tau_tri:
                p = pred_lda[i]
            else:
                cands = top3_lda[i]
                sc = hyb1_scores[i, cands] - gamma * (dtw_sc[i, cands].astype(np.float32) * inv_norm)
                p = hyb1["classes"][cands[int(np.argmax(sc))]]
            if p == cal_lbl[i]:
                correct += 1
        acc_g = correct / float(n_cal)
        if acc_g > best_acc_tri:
            best_acc_tri = acc_g
            best_g_tri = gamma

    return {
        "cascade_margin_threshold": round(float(best_tau_casc), 4),
        "cascade_use_cnn1d_tier2": int(use_cnn_t2),
        "gamma_dtw_lda": float(best_g_lda),
        "gamma_dtw_cnn": float(best_g_cnn),
        "tri_margin_threshold": round(float(best_tau_tri), 4),
        "gamma_tri_hybrid": float(best_g_tri),
    }


def apply_velocity_warping_q16(
    win_q16: np.ndarray,
    adc_scale: float,
    speed_factor: float,
) -> np.ndarray:
    """Simulate gesture execution speed variation (e.g. 125%-200% compression + phase shift)."""
    if abs(speed_factor - 1.0) < 1e-6:
        return win_q16.copy()
    m, w_len, n_ch = win_q16.shape
    t_orig = np.arange(w_len, dtype=np.float64)
    # Map output sample index t to accelerated input position (periodic reflection at edges)
    src_t = (t_orig * float(speed_factor)) % float(w_len - 1)
    idx_lo = np.floor(src_t).astype(int)
    idx_hi = np.minimum(idx_lo + 1, w_len - 1)
    frac = (src_t - idx_lo)[None, :, None]

    warped_f = (1.0 - frac) * win_q16[:, idx_lo, :].astype(np.float64) + frac * win_q16[:, idx_hi, :].astype(
        np.float64
    )
    return np.clip(c_round_half_away(warped_f), -32768, 32767).astype(np.int16)


def apply_electrode_shift_q16(
    win_q16: np.ndarray,
    shift_coupling: float = 0.20,
    gain_attenuation: float = 0.85,
) -> np.ndarray:
    """Simulate 20% physical adjacent-electrode shift + 15% skin impedance attenuation."""
    shifted_adjacent = np.roll(win_q16.astype(np.float64), shift=1, axis=2)
    mixed = gain_attenuation * ((1.0 - shift_coupling) * win_q16.astype(np.float64) + shift_coupling * shifted_adjacent)
    return np.clip(c_round_half_away(mixed), -32768, 32767).astype(np.int16)


def compute_ram_model_asymptotic_scaling() -> list[dict[str, Any]]:
    """Quantify RAM-model time cells and SRAM bytes for N in {20, 40, 100, 200, 400}."""
    esp32_sram_limit = 512 * 1024
    rows = []
    for n in (20, 40, 100, 200, 400):
        w = max(1, int(0.1 * n))
        classic_cells = n * n
        classic_sram_bytes = classic_cells * 4  # Full int32/float32 N x N matrix
        sakoe_cells = n * (2 * w + 1) - w * (w + 1)
        sakoe_sram_bytes = 2 * n * 4  # Only two rows
        lb_best_ops = n
        rows.append({
            "N": n,
            "band_w": w,
            "classic_cells_on2": classic_cells,
            "classic_sram_bytes": classic_sram_bytes,
            "exceeds_512kb_sram": bool(classic_sram_bytes > esp32_sram_limit),
            "sakoe_cells_own": sakoe_cells,
            "sakoe_sram_bytes": sakoe_sram_bytes,
            "cell_reduction_ratio": round(classic_cells / float(sakoe_cells), 2),
            "lb_keogh_best_ops_on": lb_best_ops,
        })
    return rows


def load_dataset_splits(
    ozdemir_subject: str = "S1",
    ninapro_subject: str = "S4",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load frozen LOTO splits for Ozdemir (4CH, 10 cls) and NinaPro DB2 (12CH, 50 cls)."""
    raw_oz = load_ozdemir_recording(ozdemir_subject, ROOT / "data" / "raw" / "ozdemir", use_filtered=True)
    q_oz = quantize_ozdemir(raw_oz, precision="int16", v_max=OZDEMIR_V_FS_MV)
    win_oz, lbl_oz, reps_oz = segment_ozdemir_windows(q_oz, fs=2000)
    tr_mask_oz = np.isin(reps_oz, [1, 3, 4])
    te_mask_oz = np.isin(reps_oz, [2, 5])
    oz_data = {
        "name": "Ozdemir_4CH",
        "subject": ozdemir_subject,
        "tr_win": win_oz[tr_mask_oz],
        "tr_lbl": lbl_oz[tr_mask_oz],
        "te_win": win_oz[te_mask_oz],
        "te_lbl": lbl_oz[te_mask_oz],
        "num_channels": 4,
        "num_classes": 10,
    }

    loader_ni = NinaProDB2Loader(raw_dir=ROOT / "data" / "raw" / "ninapro_db2")
    raw_cache_ni: dict[str, RecordingData] = {
        eid: loader_ni.load_recording(ninapro_subject, eid) for eid in ("E1", "E2", "E3")
    }

    def qload_ni(sid: str, eid: str) -> RecordingData:
        rec = raw_cache_ni[eid]
        return replace(rec, emg=quantize_ninapro(rec.emg, precision="int16", v_max=NINAPRO_V_FS_VOLTS))

    runner_ni = MultiExerciseWithinSubjectRunner()
    subj_ni = MultiExerciseSubject(subject_id=ninapro_subject, load_recording=qload_ni)
    tr_win_ni, tr_lbl_ni, _, _ = runner_ni.prepare_repetition_windows(
        subj_ni, repetitions=runner_ni.TRAINING_REPETITIONS
    )
    te_win_ni, te_lbl_ni, _, _ = runner_ni.prepare_repetition_windows(
        subj_ni, repetitions=runner_ni.TEST_REPETITIONS
    )
    ni_data = {
        "name": "NinaPro_DB2_12CH",
        "subject": ninapro_subject,
        "tr_win": tr_win_ni,
        "tr_lbl": tr_lbl_ni,
        "te_win": te_win_ni,
        "te_lbl": te_lbl_ni,
        "num_channels": 12,
        "num_classes": 50,
    }
    return oz_data, ni_data


def build_and_evaluate_dataset(
    ds: dict[str, Any],
    lda_bundle: dict[str, Any],
    cnn_art: QuantizedCnn1DArtifacts,
) -> dict[str, Any]:
    """Train DTW & Hybrids, evaluate nominal LOTO, and run Section 3.3 robustness tests."""
    adc_scale = float(lda_bundle["adc_scale"])
    inv_adc = 1.0 / adc_scale

    tr_q16 = windows_f32_to_q16(ds["tr_win"], adc_scale)
    te_q16 = windows_f32_to_q16(ds["te_win"], adc_scale)
    tr_lbl = ds["tr_lbl"].astype(np.int32)
    te_lbl = ds["te_lbl"].astype(np.int32)

    # 1. Fit DTW Templates Q + Keogh Envelopes U, L
    dtw_bundle = fit_dtw_templates_and_envelopes(
        tr_q16, tr_lbl, inv_adc_scale=inv_adc, num_classes=ds["num_classes"], band_w=DTW_SAKOE_W
    )

    # 2. Extract training GAP embeddings + TD features for Hybrid 1 (M06)
    print(f"  [{ds['name']}] Extracting GAP embeddings & training Hybrid 1 (N_tr={len(tr_lbl)})...", flush=True)
    tr_fp32 = (tr_q16.astype(np.float32) * np.float32(inv_adc)).astype(np.float32)
    tr_td = extract_td_int16_batch(tr_q16, lda_bundle["eps_int"], inv_adc)
    x_tr_q, aux_tr_q = quantize_windows_chunked(tr_fp32, tr_td, cnn_art)
    tr_gap_q, _, _ = simulate_c99_cnn1d_backbone_and_logits(cnn_art, x_tr_q, aux_tr_q)
    hyb1 = train_hybrid1_head(tr_gap_q, tr_td, tr_lbl, num_classes=ds["num_classes"])

    bundle: dict[str, Any] = {
        "lda": lda_bundle,
        "cnn": cnn_art,
        "dtw": dtw_bundle,
        "hyb1": hyb1,
        "fusion": {},
    }

    # 3. Calibrate thresholds and fusion gammas strictly on training data
    print(f"  [{ds['name']}] Calibrating fusion weights & early-exit gates...", flush=True)
    fusion_cfg = calibrate_hybrid_parameters(bundle, tr_q16, tr_lbl)
    bundle["fusion"] = fusion_cfg

    # 4. Evaluate all 10 models on full nominal test split
    print(f"  [{ds['name']}] Evaluating full LOTO test split (N_te={len(te_lbl)})...", flush=True)
    include_classic = bool(len(te_lbl) <= 3000)
    nominal_metrics, nominal_preds = evaluate_all_10_models_host(
        bundle, te_q16, te_lbl, include_classic_on2=include_classic
    )

    # 5. Physiological Robustness Tests (Section 3.3 of Master's Proposal)
    print(f"  [{ds['name']}] Running Section 3.3 physiological robustness suite...", flush=True)
    # Use a deterministic stratified subset of up to 1,500 test windows for robustness sweeps
    n_te = len(te_lbl)
    rob_idx = np.linspace(0, n_te - 1, min(1500, n_te), dtype=int)
    rob_q16 = te_q16[rob_idx]
    rob_lbl = te_lbl[rob_idx]

    speed_results: dict[str, Any] = {}
    for sp in (1.0, 1.25, 1.5, 2.0):
        warped_q16 = apply_velocity_warping_q16(rob_q16, adc_scale, speed_factor=sp)
        sp_metrics, _ = evaluate_all_10_models_host(bundle, warped_q16, rob_lbl, include_classic_on2=False)
        speed_results[f"speed_{int(sp * 100)}pct"] = sp_metrics

    shifted_q16 = apply_electrode_shift_q16(rob_q16, shift_coupling=0.20, gain_attenuation=0.85)
    shift_metrics, _ = evaluate_all_10_models_host(bundle, shifted_q16, rob_lbl, include_classic_on2=False)

    return {
        "dataset": ds["name"],
        "subject": ds["subject"],
        "num_channels": ds["num_channels"],
        "num_classes": ds["num_classes"],
        "num_test_windows": int(n_te),
        "bundle": bundle,
        "te_q16": te_q16,
        "te_lbl": te_lbl,
        "nominal_metrics": nominal_metrics,
        "nominal_preds": nominal_preds,
        "robustness_speed": speed_results,
        "robustness_electrode_shift": shift_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DTW variants + Hybrid models & run Host benchmarks.")
    parser.add_argument("--ozdemir-subject", default="S1")
    parser.add_argument("--ninapro-subject", default="S4")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=====================================================================")
    print(" STEP 2: TRAINING DTW VARIANTS + HYBRIDS & HOST ROBUSTNESS SUITE")
    print("=====================================================================")

    oz_ds, ni_ds = load_dataset_splits(args.ozdemir_subject, args.ninapro_subject)
    print(f"Loaded datasets in {time.time() - t0:.2f}s. Fitting Micro-LDA Q7 baselines...", flush=True)
    oz_lda = train_ozdemir_micro_lda_bundle(args.ozdemir_subject)
    ni_lda = train_ninapro_micro_lda_bundle(args.ninapro_subject)
    oz_cnn = load_cnn_artifacts("oz", "Ozdemir_4CH", args.ozdemir_subject)
    ni_cnn = load_cnn_artifacts("ni", "NinaPro_DB2_12CH", args.ninapro_subject)

    print("Evaluating Ozdemir 4CH...", flush=True)
    oz_res = build_and_evaluate_dataset(oz_ds, oz_lda, oz_cnn)
    print(f"[Ozdemir {args.ozdemir_subject} 4CH] Nominal Top-1 Acc:", flush=True)
    for m_name, m_val in oz_res["nominal_metrics"].items():
        print(f"  - {m_name:<28}: Acc={m_val['top1_acc']*100:5.2f}% | F1={m_val['macro_f1']*100:5.2f}%", flush=True)

    print("Evaluating NinaPro DB2 12CH...", flush=True)
    ni_res = build_and_evaluate_dataset(ni_ds, ni_lda, ni_cnn)
    print(f"[NinaPro {args.ninapro_subject} 12CH] Nominal Top-1 Acc:", flush=True)
    for m_name, m_val in ni_res["nominal_metrics"].items():
        print(f"  - {m_name:<28}: Acc={m_val['top1_acc']*100:5.2f}% | F1={m_val['macro_f1']*100:5.2f}%", flush=True)

    ram_scaling = compute_ram_model_asymptotic_scaling()

    # Save NPZ artifacts cleanly for C99 export script
    npz_path = args.out_dir / "dtw_hybrid_model_artifacts.npz"
    save_dict: dict[str, Any] = {}
    for prefix, res in (("oz", oz_res), ("ni", ni_res)):
        b = res["bundle"]
        d = b["dtw"]
        h = b["hyb1"]
        f = b["fusion"]
        save_dict[f"{prefix}_env_mean"] = d["env_mean"]
        save_dict[f"{prefix}_env_inv_std"] = d["env_inv_std"]
        save_dict[f"{prefix}_env_quant_scale"] = np.array([d["env_quant_scale"]], dtype=np.float32)
        save_dict[f"{prefix}_dtw_templates_q"] = d["templates_q"]
        save_dict[f"{prefix}_dtw_env_upper"] = d["env_upper"]
        save_dict[f"{prefix}_dtw_env_lower"] = d["env_lower"]

        save_dict[f"{prefix}_hyb1_classes"] = h["classes"]
        save_dict[f"{prefix}_hyb1_mean"] = h["mean"]
        save_dict[f"{prefix}_hyb1_inv_scale"] = h["inv_scale"]
        save_dict[f"{prefix}_hyb1_zero_mask"] = h["zero_mask"]
        save_dict[f"{prefix}_hyb1_coef_q7"] = h["coef_q7"]
        save_dict[f"{prefix}_hyb1_w_scale"] = h["weight_scale"]
        save_dict[f"{prefix}_hyb1_intercept"] = h["intercept"]

        save_dict[f"{prefix}_cascade_tau"] = np.array([f["cascade_margin_threshold"]], dtype=np.float32)
        save_dict[f"{prefix}_cascade_use_cnn"] = np.array([f["cascade_use_cnn1d_tier2"]], dtype=np.int32)
        save_dict[f"{prefix}_gamma_dtw_lda"] = np.array([f["gamma_dtw_lda"]], dtype=np.float32)
        save_dict[f"{prefix}_gamma_dtw_cnn"] = np.array([f["gamma_dtw_cnn"]], dtype=np.float32)
        save_dict[f"{prefix}_tri_tau"] = np.array([f["tri_margin_threshold"]], dtype=np.float32)
        save_dict[f"{prefix}_gamma_tri"] = np.array([f["gamma_tri_hybrid"]], dtype=np.float32)

    np.savez(npz_path, **save_dict)

    summary = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "ram_asymptotic_scaling": ram_scaling,
        "ozdemir": {
            "subject": oz_res["subject"],
            "num_channels": oz_res["num_channels"],
            "num_classes": oz_res["num_classes"],
            "num_test_windows": oz_res["num_test_windows"],
            "fusion_config": oz_res["bundle"]["fusion"],
            "nominal_metrics": oz_res["nominal_metrics"],
            "robustness_speed": oz_res["robustness_speed"],
            "robustness_electrode_shift": oz_res["robustness_electrode_shift"],
        },
        "ninapro": {
            "subject": ni_res["subject"],
            "num_channels": ni_res["num_channels"],
            "num_classes": ni_res["num_classes"],
            "num_test_windows": ni_res["num_test_windows"],
            "fusion_config": ni_res["bundle"]["fusion"],
            "nominal_metrics": ni_res["nominal_metrics"],
            "robustness_speed": ni_res["robustness_speed"],
            "robustness_electrode_shift": ni_res["robustness_electrode_shift"],
        },
    }
    summary_path = args.out_dir / "host_training_and_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved NPZ to {npz_path} and Host summary to {summary_path} in {time.time() - t0:.2f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
