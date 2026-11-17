#!/usr/bin/env python3
"""Multi-Channel & Multi-Precision Benchmark for all 10 sEMG Models (DTW, LDA, CNN & Hybrids).

Evaluates all 10 models across the complete channel-reduction and quantization grid:
  - Ozdemir et al. (S1, 10 classes): Channels C in {4, 3, 2, 1} x Precisions {float32, float16, int32, int16, int8}
  - NinaPro DB2 (S4, 50 classes): Channels C in {12, 10, 8, 6, 4, 2, 1} x Precisions {float32, float16, int32, int16, int8}

Total: 55 scenarios x 10 models = 550 full LOTO host evaluations + cycle-exact QEMU (`esp32s3` @ 240 MHz)
profiling across all 11 `(dataset, C)` configurations for Latency, SRAM, Flash, Energy, IRC, and IEE.

Outputs:
  - `reports/dtw_hybrid_benchmark/channel_and_quantization_benchmark_results.json`
  - `reports/dtw_hybrid_benchmark/channel_and_quantization_master_report.md`
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ozdemir_quantization_and_channels import (  # noqa: E402
    CHANNEL_OPTIONS as OZ_CHANNELS,
    PRECISION_OPTIONS,
    V_FS_MV as OZ_V_FS_MV,
    get_channel_feature_indices,
    load_ozdemir_recording,
    quantize_signal as quantize_ozdemir,
    segment_ozdemir_windows,
)
from scripts.benchmark_quantization_and_channels import (  # noqa: E402
    CHANNEL_OPTIONS as NI_CHANNELS,
    V_FS_VOLTS as NI_V_FS_VOLTS,
    quantize_signal as quantize_ninapro,
)
from scripts.run_dtw_and_hybrids_benchmark import (  # noqa: E402
    build_toolchain_env,
    model_footprint_kb,
)
from scripts.run_esp32_qemu_pipeline import quantize_lda_weights_q7  # noqa: E402
from scripts.run_qemu_cnn1d_benchmark import ESP32S3_ACTIVE_POWER_MW, compute_irc_iee  # noqa: E402
from scripts.run_qemu_freertos_dualcore_benchmark import load_cnn_artifacts  # noqa: E402
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
from src.models.train_cnn1d_quantized import (  # noqa: E402
    CompactSemgCnn1D,
    QuantizedCnn1DArtifacts,
    c_round_half_away,
    set_deterministic_seed,
)
from src.models.train_dtw_and_hybrids import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DTW_ENV_LEN,
    DTW_SAKOE_W,
    DTW_SUBBLOCK_SAMPLES,
    DTW_TOPM_HYBRID,
    DTW_TRI_TOPM,
    compute_dtw_matrix_batch,
    compute_evaluation_metrics,
    compute_lb_keogh_batch,
    predict_lda_q7_scores_batch,
    quantize_windows_chunked,
    simulate_c99_cnn1d_backbone_and_logits,
    train_hybrid1_head,
)

MODEL_NAMES = [
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


def train_fast_compact_cnn1d(
    dataset_name: str,
    subject_id: str,
    train_win: np.ndarray,
    train_lbl: np.ndarray,
    X_tr_aux: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int = 512,
    seed: int = 42,
) -> QuantizedCnn1DArtifacts:
    """Train & INT8-quantize CompactSemgCnn1D for reduced channel count C cleanly and fast."""
    set_deterministic_seed(seed)
    torch.set_num_threads(4)
    in_channels = int(train_win.shape[2])
    num_aux = int(X_tr_aux.shape[1])

    in_mean = np.mean(train_win, axis=(0, 1), dtype=np.float64).astype(np.float32)
    in_std = np.std(train_win, axis=(0, 1), dtype=np.float64).astype(np.float32)
    in_std = np.where(in_std < 1e-8, 1.0, in_std).astype(np.float32)
    in_inv_std = (1.0 / in_std).astype(np.float32)

    scaler = FeatureScaler()
    X_tr_aux_s = scaler.fit_transform(X_tr_aux).astype(np.float32)
    sdict = scaler.to_dict()
    aux_mean = np.asarray(sdict["mean"], dtype=np.float32)
    aux_scale = np.asarray(sdict["scale"], dtype=np.float32)
    aux_inv_std = np.where(aux_scale != 0.0, 1.0 / aux_scale, 0.0).astype(np.float32)
    for z_idx in sdict["zero_variance_indices"]:
        if 0 <= z_idx < num_aux:
            aux_inv_std[z_idx] = 0.0

    lda_trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_classes)
    lda_trainer.fit(X_tr_aux_s, train_lbl)
    lda_art = lda_trainer.get_model_artifacts()

    X_tr_cnn = np.transpose((train_win - in_mean[None, None, :]) * in_inv_std[None, None, :], (0, 2, 1)).astype(
        np.float32
    )
    t_x = torch.from_numpy(X_tr_cnn)
    t_aux = torch.from_numpy(X_tr_aux_s)
    t_y = torch.from_numpy(train_lbl.astype(np.int64))

    model = CompactSemgCnn1D(in_channels=in_channels, num_classes=num_classes, num_aux=num_aux)
    total_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        model.fc_aux.weight.copy_(torch.from_numpy(lda_art.coef_.astype(np.float32)))
        model.fc_conv.bias.copy_(torch.from_numpy(lda_art.intercept_.astype(np.float32)))
        nn.init.normal_(model.fc_conv.weight, mean=0.0, std=0.01)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.conv1.parameters(), "lr": 2.5e-3, "weight_decay": 1e-3},
            {"params": model.conv2.parameters(), "lr": 2.5e-3, "weight_decay": 1e-3},
            {"params": model.fc_conv.parameters(), "lr": 1.2e-3, "weight_decay": 1e-2},
            {"params": model.fc_aux.parameters(), "lr": 2.0e-4, "weight_decay": 1e-2},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    n_tr = t_x.shape[0]

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_tr, generator=gen)
        for start in range(0, n_tr, batch_size):
            b_idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(t_x[b_idx], t_aux[b_idx])
            loss = criterion(logits, t_y[b_idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        calib_idx = np.linspace(0, n_tr - 1, min(1500, n_tr), dtype=int)
        a1_cal, a2_cal, _ = model.forward_features(t_x[calib_idx])

        s_in = max(float(np.percentile(np.abs(X_tr_cnn[calib_idx]), 99.95)) / 127.0, 1e-6)
        s_aux = max(float(np.percentile(np.abs(X_tr_aux_s[calib_idx]), 99.95)) / 127.0, 1e-6)
        s_act1 = max(float(np.percentile(a1_cal.numpy(), 99.95)) / 127.0, 1e-6)
        s_act2 = max(float(np.percentile(a2_cal.numpy(), 99.95)) / 127.0, 1e-6)

        w1_fp, b1_fp = model.conv1.weight.numpy(), model.conv1.bias.numpy()
        s_w1 = max(float(np.max(np.abs(w1_fp))) / 127.0, 1e-8)
        w1_q = np.clip(c_round_half_away(w1_fp / s_w1), -127, 127).astype(np.int8)
        b1_q = c_round_half_away(b1_fp / (s_in * s_w1)).astype(np.int32)
        conv1_eff = (s_in * s_w1) / s_act1

        w2_fp, b2_fp = model.conv2.weight.numpy(), model.conv2.bias.numpy()
        s_w2 = max(float(np.max(np.abs(w2_fp))) / 127.0, 1e-8)
        w2_q = np.clip(c_round_half_away(w2_fp / s_w2), -127, 127).astype(np.int8)
        b2_q = c_round_half_away(b2_fp / (s_act1 * s_w2)).astype(np.int32)
        conv2_eff = (s_act1 * s_w2) / s_act2

        wfc_fp, bfc_fp = model.fc_conv.weight.numpy(), model.fc_conv.bias.numpy()
        s_wfc = max(float(np.max(np.abs(wfc_fp))) / 127.0, 1e-8)
        wfc_q = np.clip(c_round_half_away(wfc_fp / s_wfc), -127, 127).astype(np.int8)
        fc_eff = s_act2 * s_wfc
        bfc_q = c_round_half_away(bfc_fp / fc_eff).astype(np.int32)

        waux_fp = model.fc_aux.weight.numpy()
        s_waux = max(float(np.max(np.abs(waux_fp))) / 127.0, 1e-8)
        waux_q = np.clip(c_round_half_away(waux_fp / s_waux), -127, 127).astype(np.int8)
        aux_fc_eff = (s_aux * s_waux) / fc_eff

    return QuantizedCnn1DArtifacts(
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
        conv1_eff_scale=float(conv1_eff),
        conv2_weights_q=w2_q.ravel(),
        conv2_bias_q=b2_q.ravel(),
        conv2_eff_scale=float(conv2_eff),
        fc_weights_q=wfc_q.ravel(),
        fc_bias_q=bfc_q.ravel(),
        fc_eff_scale=float(fc_eff),
        aux_fc_weights_q=waux_q.ravel(),
        aux_fc_eff_scale=float(aux_fc_eff),
        fp32_top1_acc=0.0,
        fp32_bal_acc=0.0,
        fp32_macro_f1=0.0,
        int8_top1_acc=0.0,
        int8_bal_acc=0.0,
        int8_macro_f1=0.0,
    )


def fit_lda_q7(X_tr: np.ndarray, tr_lbl: np.ndarray, num_classes: int) -> dict[str, Any]:
    """Fit FeatureScaler + Regularized Canonical LDA and quantize weights to Q7."""
    n_feat = X_tr.shape[1]
    scaler = FeatureScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_classes)
    trainer.fit(X_tr_s, tr_lbl)
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
    }


def extract_subblock_mav(win_f32: np.ndarray) -> np.ndarray:
    """Extract 40-step 5ms block MAV [N, 40, C] from float32 quantized windows [N, 400, C]."""
    n_win, _, n_ch = win_f32.shape
    blocks = np.abs(win_f32.reshape(n_win, DTW_ENV_LEN, DTW_SUBBLOCK_SAMPLES, n_ch))
    return np.mean(blocks, axis=2, dtype=np.float32).astype(np.float32)


def fit_dtw_from_mav(
    tr_mav: np.ndarray,
    tr_lbl: np.ndarray,
    te_mav: np.ndarray,
    num_classes: int,
    band_w: int = DTW_SAKOE_W,
) -> dict[str, Any]:
    """Fit per-channel scaling, int8 templates Q, and Keogh envelopes U, L directly from precision-p MAV."""
    env_mean = np.mean(tr_mav, axis=(0, 1), dtype=np.float64).astype(np.float32)
    env_std = np.std(tr_mav, axis=(0, 1), dtype=np.float64).astype(np.float32)
    env_std = np.where(env_std < 1e-8, 1.0, env_std).astype(np.float32)
    env_inv_std = (1.0 / env_std).astype(np.float32)

    z_tr = (tr_mav - env_mean[None, None, :]) * env_inv_std[None, None, :]
    max_99 = float(np.percentile(np.abs(z_tr), 99.9))
    q_scale = float(np.float32(max(max_99 / 127.0, 1e-6)))
    inv_q = np.float32(1.0 / q_scale)

    env_tr_q = np.clip(c_round_half_away(z_tr * inv_q), -127, 127).astype(np.int8)
    z_te = (te_mav - env_mean[None, None, :]) * env_inv_std[None, None, :]
    env_te_q = np.clip(c_round_half_away(z_te * inv_q), -127, 127).astype(np.int8)

    n_ch = tr_mav.shape[2]
    templates_q = np.zeros((num_classes, DTW_ENV_LEN, n_ch), dtype=np.int8)
    env_upper = np.zeros((num_classes, DTW_ENV_LEN, n_ch), dtype=np.int8)
    env_lower = np.zeros((num_classes, DTW_ENV_LEN, n_ch), dtype=np.int8)

    for k in range(num_classes):
        idx_k = np.where(tr_lbl == k)[0]
        if len(idx_k) == 0:
            continue
        q_k = np.clip(c_round_half_away(np.mean(env_tr_q[idx_k].astype(np.float64), axis=0)), -127, 127).astype(np.int8)
        templates_q[k] = q_k
        for i in range(DTW_ENV_LEN):
            lo, hi = max(0, i - band_w), min(DTW_ENV_LEN - 1, i + band_w)
            env_upper[k, i] = np.max(q_k[lo : hi + 1], axis=0)
            env_lower[k, i] = np.min(q_k[lo : hi + 1], axis=0)

    return {
        "env_mean": env_mean,
        "env_inv_std": env_inv_std,
        "env_quant_scale": q_scale,
        "templates_q": templates_q,
        "env_upper": env_upper,
        "env_lower": env_lower,
        "env_tr_q": env_tr_q,
        "env_te_q": env_te_q,
    }


def evaluate_scenario_10_models(
    tr_win_c: np.ndarray,
    tr_lbl: np.ndarray,
    te_win_c: np.ndarray,
    te_lbl: np.ndarray,
    tr_td_c: np.ndarray,
    te_td_c: np.ndarray,
    tr_mav_c: np.ndarray,
    te_mav_c: np.ndarray,
    cnn_art: QuantizedCnn1DArtifacts,
) -> dict[str, Any]:
    """Fit LDA/DTW/Hyb1/Fusion on training split and evaluate all 10 models on full LOTO test split."""
    n_ch = cnn_art.in_channels
    n_cls = cnn_art.num_classes
    n_te = len(te_lbl)

    # 1. M01 Micro-LDA Q7
    lda = fit_lda_q7(tr_td_c, tr_lbl, n_cls)
    pred_m01_tr, sc_m01_tr = predict_lda_q7_scores_batch(
        tr_td_c, lda["mean"], lda["inv_scale"], lda["zero_mask"], lda["coef_q7"], lda["weight_scale"], lda["intercept"], lda["classes"]
    )
    pred_m01_te, sc_m01_te = predict_lda_q7_scores_batch(
        te_td_c, lda["mean"], lda["inv_scale"], lda["zero_mask"], lda["coef_q7"], lda["weight_scale"], lda["intercept"], lda["classes"]
    )

    # 2. M02 CNN-1D INT8 + GAP features
    x_tr_q, aux_tr_q = quantize_windows_chunked(tr_win_c, tr_td_c, cnn_art)
    gap_tr_q, pred_m02_tr_idx, cnn_tr_logits = simulate_c99_cnn1d_backbone_and_logits(cnn_art, x_tr_q, aux_tr_q)
    pred_m02_tr = lda["classes"][pred_m02_tr_idx]

    x_te_q, aux_te_q = quantize_windows_chunked(te_win_c, te_td_c, cnn_art)
    gap_te_q, pred_m02_te_idx, cnn_te_logits = simulate_c99_cnn1d_backbone_and_logits(cnn_art, x_te_q, aux_te_q)
    pred_m02_te = lda["classes"][pred_m02_te_idx]

    # 3. DTW Templates & Test Scores (Classic M03, Sakoe-Chiba M04, LB_Keogh M05)
    dtw = fit_dtw_from_mav(tr_mav_c, tr_lbl, te_mav_c, n_cls, DTW_SAKOE_W)
    dtw_sc_te = compute_dtw_matrix_batch(dtw["env_te_q"], dtw["templates_q"], band_w=DTW_SAKOE_W)
    dtw_cl_te = compute_dtw_matrix_batch(dtw["env_te_q"], dtw["templates_q"], band_w=None)
    pred_m03_te = np.argmin(dtw_cl_te, axis=1).astype(np.int32)
    pred_m04_te = np.argmin(dtw_sc_te, axis=1).astype(np.int32)
    pred_m05_te = pred_m04_te.copy()

    lb_te = compute_lb_keogh_batch(dtw["env_te_q"], dtw["env_upper"], dtw["env_lower"])
    k_min_te = np.argmin(lb_te, axis=1)
    best_init_te = dtw_sc_te[np.arange(n_te), k_min_te]
    pruned_mask = lb_te >= best_init_te[:, None]
    pruned_mask[np.arange(n_te), k_min_te] = False
    lb_prune_rate = float(np.mean(np.sum(pruned_mask, axis=1) / float(n_cls)))

    # 4. M06 Hybrid 1 (CNN Backbone GAP + TD -> LDA Q7 Head)
    hyb1 = train_hybrid1_head(gap_tr_q, tr_td_c, tr_lbl, n_cls)
    x_hyb_tr = np.hstack([gap_tr_q.astype(np.float32), tr_td_c]).astype(np.float32)
    x_hyb_te = np.hstack([gap_te_q.astype(np.float32), te_td_c]).astype(np.float32)
    pred_m06_tr, sc_m06_tr = predict_lda_q7_scores_batch(
        x_hyb_tr, hyb1["mean"], hyb1["inv_scale"], hyb1["zero_mask"], hyb1["coef_q7"], hyb1["weight_scale"], hyb1["intercept"], hyb1["classes"]
    )
    pred_m06_te, sc_m06_te = predict_lda_q7_scores_batch(
        x_hyb_te, hyb1["mean"], hyb1["inv_scale"], hyb1["zero_mask"], hyb1["coef_q7"], hyb1["weight_scale"], hyb1["intercept"], hyb1["classes"]
    )

    # 5. Calibrate Fusion & Early-Exit on stratified training subset (up to 2000 samples)
    n_cal = min(2000, len(tr_lbl))
    c_idx = np.linspace(0, len(tr_lbl) - 1, n_cal, dtype=int)
    dtw_sc_cal = compute_dtw_matrix_batch(dtw["env_tr_q"][c_idx], dtw["templates_q"], band_w=DTW_SAKOE_W)
    cal_lbl = tr_lbl[c_idx]
    sc_m01_cal = sc_m01_tr[c_idx]
    pred_m01_cal = pred_m01_tr[c_idx]
    pred_m02_cal = pred_m02_tr[c_idx]
    pred_m06_cal = pred_m06_tr[c_idx]
    sc_m06_cal = sc_m06_tr[c_idx]
    cnn_cal_logits = cnn_tr_logits[c_idx]

    use_cnn_t2 = 1 if np.mean(pred_m02_cal == cal_lbl) >= np.mean(pred_m06_cal == cal_lbl) else 0
    tier2_cal = pred_m02_cal if use_cnn_t2 else pred_m06_cal

    s_cal = np.sort(sc_m01_cal, axis=1)
    m_cal = (s_cal[:, -1] - s_cal[:, -2]).astype(np.float32)
    best_tau_casc = float(np.percentile(m_cal, 50))
    best_sc_casc = -1.0
    for pct in (25, 35, 45, 55, 65):
        tau = float(np.percentile(m_cal, 100 - pct))
        acc_c = float(np.mean(np.where(m_cal >= tau, pred_m01_cal, tier2_cal) == cal_lbl)) + 0.005 * (pct / 100.0)
        if acc_c > best_sc_casc:
            best_sc_casc = acc_c
            best_tau_casc = tau

    inv_norm = np.float32(1.0 / float(DTW_ENV_LEN * n_ch))
    m_top = min(n_cls, DTW_TOPM_HYBRID)
    topm_lda_cal = np.argsort(-sc_m01_cal, axis=1, kind="stable")[:, :m_top]
    best_g_lda = 0.05
    best_acc_g = -1.0
    for g in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        corr = sum(
            int(lda["classes"][topm_lda_cal[i][int(np.argmax(sc_m01_cal[i, topm_lda_cal[i]] - g * (dtw_sc_cal[i, topm_lda_cal[i]] * inv_norm)))]] == cal_lbl[i])
            for i in range(n_cal)
        )
        if corr / float(n_cal) > best_acc_g:
            best_acc_g = corr / float(n_cal)
            best_g_lda = g

    l_scale = np.float32(cnn_art.fc_eff_scale)
    topm_cnn_cal = np.argsort(-cnn_cal_logits, axis=1, kind="stable")[:, :m_top]
    best_g_cnn = 0.05
    best_acc_c = -1.0
    for g in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        corr = sum(
            int(topm_cnn_cal[i][int(np.argmax((cnn_cal_logits[i, topm_cnn_cal[i]] * l_scale) - g * (dtw_sc_cal[i, topm_cnn_cal[i]] * inv_norm)))] == cal_lbl[i])
            for i in range(n_cal)
        )
        if corr / float(n_cal) > best_acc_c:
            best_acc_c = corr / float(n_cal)
            best_g_cnn = g

    best_tau_tri = float(np.percentile(m_cal, 55))
    m_tri = min(n_cls, DTW_TRI_TOPM)
    top3_cal = np.argsort(-sc_m01_cal, axis=1, kind="stable")[:, :m_tri]
    best_g_tri = 0.05
    best_acc_t = -1.0
    for g in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        corr = 0
        for i in range(n_cal):
            if m_cal[i] >= best_tau_tri:
                p = pred_m01_cal[i]
            else:
                cs = top3_cal[i]
                p = hyb1["classes"][cs[int(np.argmax(sc_m06_cal[i, cs] - g * (dtw_sc_cal[i, cs] * inv_norm)))]]
            if p == cal_lbl[i]:
                corr += 1
        if corr / float(n_cal) > best_acc_t:
            best_acc_t = corr / float(n_cal)
            best_g_tri = g

    # 6. Test split evaluation for Hybrids M07..M10
    s_te = np.sort(sc_m01_te, axis=1)
    m_te = (s_te[:, -1] - s_te[:, -2]).astype(np.float32)

    ee_m07 = m_te >= best_tau_casc
    pred_m07_te = np.where(ee_m07, pred_m01_te, pred_m02_te if use_cnn_t2 else pred_m06_te).astype(np.int32)

    topm_lda_te = np.argsort(-sc_m01_te, axis=1, kind="stable")[:, :m_top]
    pred_m08_te = np.array(
        [lda["classes"][topm_lda_te[i][int(np.argmax(sc_m01_te[i, topm_lda_te[i]] - np.float32(best_g_lda) * (dtw_sc_te[i, topm_lda_te[i]].astype(np.float32) * inv_norm)))]] for i in range(n_te)],
        dtype=np.int32,
    )

    topm_cnn_te = np.argsort(-cnn_te_logits, axis=1, kind="stable")[:, :m_top]
    pred_m09_te = np.array(
        [topm_cnn_te[i][int(np.argmax((cnn_te_logits[i, topm_cnn_te[i]].astype(np.float32) * l_scale) - np.float32(best_g_cnn) * (dtw_sc_te[i, topm_cnn_te[i]].astype(np.float32) * inv_norm)))] for i in range(n_te)],
        dtype=np.int32,
    )

    ee_m10 = m_te >= best_tau_tri
    top3_te = np.argsort(-sc_m01_te, axis=1, kind="stable")[:, :m_tri]
    pred_m10_te = np.zeros(n_te, dtype=np.int32)
    for i in range(n_te):
        if ee_m10[i]:
            pred_m10_te[i] = pred_m01_te[i]
        else:
            cs = top3_te[i]
            pred_m10_te[i] = hyb1["classes"][cs[int(np.argmax(sc_m06_te[i, cs] - np.float32(best_g_tri) * (dtw_sc_te[i, cs].astype(np.float32) * inv_norm)))]]

    preds_mat = np.column_stack(
        [pred_m01_te, pred_m02_te, pred_m03_te, pred_m04_te, pred_m05_te, pred_m06_te, pred_m07_te, pred_m08_te, pred_m09_te, pred_m10_te]
    ).astype(np.int32)

    metrics: dict[str, dict[str, Any]] = {}
    for idx, m_name in enumerate(MODEL_NAMES):
        acc, bal, f1 = compute_evaluation_metrics(te_lbl, preds_mat[:, idx])
        entry: dict[str, Any] = {
            "top1_acc": round(float(acc), 6),
            "bal_acc": round(float(bal), 6),
            "macro_f1": round(float(f1), 6),
        }
        if m_name == "M05_dtw_lb_keogh":
            entry["lb_prune_rate"] = round(lb_prune_rate, 6)
        elif m_name == "M07_hybrid_lda_cnn_cascade":
            entry["early_exit_rate"] = round(float(np.mean(ee_m07)), 6)
        elif m_name == "M10_hybrid_dtw_lda_cnn":
            entry["early_exit_rate"] = round(float(np.mean(ee_m10)), 6)
        metrics[m_name] = entry

    fusion_cfg = {
        "cascade_margin_threshold": float(best_tau_casc),
        "cascade_use_cnn1d_tier2": int(use_cnn_t2),
        "gamma_dtw_lda": float(best_g_lda),
        "gamma_dtw_cnn": float(best_g_cnn),
        "tri_margin_threshold": float(best_tau_tri),
        "gamma_tri_hybrid": float(best_g_tri),
    }

    return {
        "metrics": metrics,
        "qemu_payload_data": {
            "lda": lda,
            "cnn": cnn_art,
            "dtw": dtw,
            "hyb1": hyb1,
            "fusion": fusion_cfg,
            "te_preds": preds_mat,
            "te_x_q": x_te_q,
            "te_aux_q": aux_te_q,
            "te_td": te_td_c,
        },
    }


def write_qemu_channel_block(f: Any, dataset_code: int, qdata: dict[str, Any], num_vectors: int = 80) -> None:
    """Serialize one `(dataset, num_channels)` block for `main_channel_profile_qemu.c`."""
    cnn_art: QuantizedCnn1DArtifacts = qdata["cnn"]
    lda = qdata["lda"]
    dtw = qdata["dtw"]
    hyb1 = qdata["hyb1"]
    fus = qdata["fusion"]

    n_tot = len(qdata["te_preds"])
    idx = np.linspace(0, n_tot - 1, min(num_vectors, n_tot), dtype=int)
    n_vec = len(idx)

    bh = struct.pack(
        "<5I8fI4f",
        dataset_code,
        cnn_art.in_channels,
        cnn_art.num_classes,
        cnn_art.num_aux,
        n_vec,
        float(cnn_art.in_quant_scale),
        float(cnn_art.aux_quant_scale),
        float(cnn_art.conv1_eff_scale),
        float(cnn_art.conv2_eff_scale),
        float(cnn_art.fc_eff_scale),
        float(cnn_art.aux_fc_eff_scale),
        float(dtw["env_quant_scale"]),
        float(fus["cascade_margin_threshold"]),
        int(fus["cascade_use_cnn1d_tier2"]),
        float(fus["gamma_dtw_lda"]),
        float(fus["gamma_dtw_cnn"]),
        float(fus["tri_margin_threshold"]),
        float(fus["gamma_tri_hybrid"]),
    )
    f.write(bh)

    # Write model weights matching exact read_exact() order in main_channel_profile_qemu.c
    f.write(np.ascontiguousarray(lda["classes"], dtype="<i4").tobytes())
    f.write(np.ascontiguousarray(lda["mean"], dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(lda["inv_scale"], dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(lda["zero_mask"], dtype="<u1").tobytes())
    f.write(np.ascontiguousarray(lda["coef_q7"], dtype="<i1").tobytes())
    f.write(np.ascontiguousarray(lda["weight_scale"], dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(lda["intercept"], dtype="<f4").tobytes())

    f.write(np.ascontiguousarray(cnn_art.in_mean, dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(cnn_art.in_inv_std, dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(cnn_art.aux_mean, dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(cnn_art.aux_inv_std, dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(cnn_art.conv1_weights_q, dtype="<i1").tobytes())
    f.write(np.ascontiguousarray(cnn_art.conv1_bias_q, dtype="<i4").tobytes())
    f.write(np.ascontiguousarray(cnn_art.conv2_weights_q, dtype="<i1").tobytes())
    f.write(np.ascontiguousarray(cnn_art.conv2_bias_q, dtype="<i4").tobytes())
    f.write(np.ascontiguousarray(cnn_art.fc_weights_q, dtype="<i1").tobytes())
    f.write(np.ascontiguousarray(cnn_art.fc_bias_q, dtype="<i4").tobytes())
    f.write(np.ascontiguousarray(cnn_art.aux_fc_weights_q, dtype="<i1").tobytes())

    f.write(np.ascontiguousarray(dtw["env_mean"], dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(dtw["env_inv_std"], dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(dtw["templates_q"], dtype="<i1").tobytes())
    f.write(np.ascontiguousarray(dtw["env_upper"], dtype="<i1").tobytes())
    f.write(np.ascontiguousarray(dtw["env_lower"], dtype="<i1").tobytes())

    f.write(np.ascontiguousarray(hyb1["classes"], dtype="<i4").tobytes())
    f.write(np.ascontiguousarray(hyb1["mean"], dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(hyb1["inv_scale"], dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(hyb1["zero_mask"], dtype="<u1").tobytes())
    f.write(np.ascontiguousarray(hyb1["coef_q7"], dtype="<i1").tobytes())
    f.write(np.ascontiguousarray(hyb1["weight_scale"], dtype="<f4").tobytes())
    f.write(np.ascontiguousarray(hyb1["intercept"], dtype="<f4").tobytes())

    for i in idx:
        f.write(np.ascontiguousarray(qdata["te_preds"][i], dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(qdata["te_x_q"][i], dtype="<i1").tobytes())
        f.write(np.ascontiguousarray(qdata["te_aux_q"][i], dtype="<i1").tobytes())
        f.write(np.ascontiguousarray(dtw["env_te_q"][i], dtype="<i1").tobytes())
        f.write(np.ascontiguousarray(qdata["te_td"][i], dtype="<f4").tobytes())


def run_qemu_channel_profiler(qemu_blocks: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Compile `main_channel_profile_qemu.c` and collect cycle-exact QEMU telemetry across all (dataset, C)."""
    env = build_toolchain_env()
    build_dir = ROOT / "embedded" / "firmware" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    elf_path = build_dir / "channel_profile_esp32s3.elf"

    subprocess.run(
        [
            "xtensa-esp32s3-elf-gcc",
            "-std=c99",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "embedded" / "firmware" / "include"),
            "-I",
            str(ROOT / "firmware" / "esp32_qemu" / "include"),
            "-specs=board.elf.specs",
            "-specs=sys.qemu.specs",
            "-Wl,--defsym=entire_dram_seg=1",
            str(ROOT / "firmware" / "esp32_qemu" / "src" / "semg_pipeline.c"),
            str(ROOT / "embedded" / "firmware" / "src" / "cnn1d_engine.c"),
            str(ROOT / "embedded" / "firmware" / "src" / "dtw_hybrid_engine.c"),
            str(ROOT / "embedded" / "firmware" / "qemu_runner" / "main_channel_profile_qemu.c"),
            "-o",
            str(elf_path),
            "-lm",
        ],
        env=env,
        check=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "qemu_channel_profile.bin"
        with bin_path.open("wb") as f:
            f.write(struct.pack("<II", 0x43485052, len(qemu_blocks)))
            for ds_code, qdata in qemu_blocks:
                write_qemu_channel_block(f, ds_code, qdata, num_vectors=80)

        proc = subprocess.run(
            [
                "qemu-system-xtensa",
                "-nographic",
                "-machine",
                "esp32s3",
                "-semihosting",
                "-kernel",
                str(elf_path.resolve()),
            ],
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
            check=True,
        )
        for line in proc.stdout.splitlines():
            if "QEMU_CH_PROFILE_JSON:" in line:
                return json.loads(line.split("QEMU_CH_PROFILE_JSON:", 1)[1].strip())["configs"]
        raise RuntimeError(f"Missing QEMU_CH_PROFILE_JSON:\n{proc.stdout}\n{proc.stderr}")


def generate_markdown_report(results: dict[str, Any]) -> str:
    """Build consolidated Markdown report comparing all 10 models across channels & precisions."""
    lines: list[str] = [
        "# Benchmark Multi-Canal e Multi-Precisão dos 10 Modelos (DTW, Micro-LDA, CNN-1D e Híbridos)",
        "",
        "> **Escopo:** Avaliação cruzada dos **10 modelos (`M01`–`M10`)** sob redução progressiva de canais e 5 níveis de quantização numérica de entrada (`float32`, `float16`, `int32`, `int16`, `int8`), combinando **550 avaliações completas em Host LOTO** e perfilamento ciclo-exato **`CCOUNT` no Espressif QEMU (`esp32s3` @ 240 MHz)**.",
        f"> **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "---",
        "",
    ]

    for ds_idx, (ds_key, ds_title, ch_list, n_cls) in enumerate(
        (
            ("ozdemir", "1. Dataset Ozdemir et al. (Sujeito S1, 10 Classes, 2.340 Janelas LOTO — Canais: 4, 3, 2, 1)", OZ_CHANNELS, 10),
            ("ninapro", "2. Dataset NinaPro DB2 (Sujeito S4, 50 Classes, 15.824 Janelas LOTO — Canais: 12, 10, 8, 6, 4, 2, 1)", NI_CHANNELS, 50),
        ),
        start=1,
    ):
        scenarios = results["datasets"][ds_key]["scenarios"]
        qemu_prof = results["qemu_channel_profiles"][ds_key]

        lines.extend([
            f"## {ds_title}",
            "",
            f"### {ds_idx}A. Degradação por Redução de Canais (Precisão Padrão `int16`)",
            "",
            "| Canais | M01 LDA Q7 | M02 CNN-1D | M03 DTW Classic | M04 DTW Sakoe | M05 DTW Keogh | M06 Híbrido 1 | M07 Híbrido 2 (`EE`) | M08 DTW+LDA | M09 DTW+CNN | M10 Tri-Híbrido |",
            "| :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
        ])

        for ch in ch_list:
            sc = next(s for s in scenarios if s["num_channels"] == ch and s["precision"] == "int16")
            m = sc["metrics"]
            vals = [f"{m[mn]['top1_acc']*100:.2f}%" for mn in MODEL_NAMES]
            lines.append(f"| **`{ch}CH`** | " + " | ".join(vals) + " |")

        lines.extend([
            "",
            f"### {ds_idx}B. Impacto da Quantização numérica (`C = {ch_list[0]} canais` completos)",
            "",
            "| Precisão | M01 LDA Q7 | M02 CNN-1D | M03 DTW Classic | M04/M05 DTW SC/Keogh | M06 Híbrido 1 | M07 Híbrido 2 | M08 DTW+LDA | M09 DTW+CNN | M10 Tri-Híbrido |",
            "| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
        ])

        for prec in PRECISION_OPTIONS:
            sc = next(s for s in scenarios if s["num_channels"] == ch_list[0] and s["precision"] == prec)
            m = sc["metrics"]
            vals = [
                f"{m['M01_micro_lda_q7']['top1_acc']*100:.2f}%",
                f"{m['M02_cnn1d_int8']['top1_acc']*100:.2f}%",
                f"{m['M03_dtw_classic_on2']['top1_acc']*100:.2f}%",
                f"{m['M05_dtw_lb_keogh']['top1_acc']*100:.2f}%",
                f"{m['M06_hybrid_lda_cnn_head']['top1_acc']*100:.2f}%",
                f"{m['M07_hybrid_lda_cnn_cascade']['top1_acc']*100:.2f}%",
                f"{m['M08_hybrid_dtw_lda']['top1_acc']*100:.2f}%",
                f"{m['M09_hybrid_dtw_cnn']['top1_acc']*100:.2f}%",
                f"{m['M10_hybrid_dtw_lda_cnn']['top1_acc']*100:.2f}%",
            ]
            lines.append(f"| **`{prec}`** | " + " | ".join(vals) + " |")

        lines.extend([
            "",
            f"### {ds_idx}C. Cenário Prótese de Baixo Custo (`2 Canais` sob `int16` vs. `int8`)",
            "",
            "| Modelo | Acc (`2CH int16`) | Acc (`2CH int8`) | $\\Delta$ Quant (`int8` vs `int16`) | Latência QEMU `2CH` (ms) | SRAM `2CH` | **IRC (`2CH int16`)** | **IEE (`2CH int16`)** |",
            "| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
        ])

        sc_2_16 = next(s for s in scenarios if s["num_channels"] == 2 and s["precision"] == "int16")["metrics"]
        sc_2_8 = next(s for s in scenarios if s["num_channels"] == 2 and s["precision"] == "int8")["metrics"]
        q_2 = {m["name"]: m for m in qemu_prof[str(2)]["models"]}
        for mn in MODEL_NAMES:
            a16 = sc_2_16[mn]["top1_acc"] * 100.0
            a8 = sc_2_8[mn]["top1_acc"] * 100.0
            lat_ms = max(q_2[mn]["latency_mean_ms"], 1e-6)
            _, sram_kb = model_footprint_kb(mn, 2, n_cls)
            eff = compute_irc_iee(a16, n_cls, lat_ms, sram_kb, ESP32S3_ACTIVE_POWER_MW)
            lines.append(
                f"| **{mn}** | {a16:.2f}% | {a8:.2f}% | {a8-a16:+.2f} pp | **{lat_ms:.4f}** | {sram_kb:.2f} KB | {eff['irc']:,.1f} | {eff['iee']:,.1f} |"
            )

        lines.extend([
            "",
            f"### {ds_idx}D. Perfil Ciclo-Exato no Espressif QEMU (`esp32s3` @ 240 MHz) por Nº de Canais",
            "",
            "| Canais | Paridade Todos Modelos | M01 LDA (µs) | M02 CNN-1D (ms) | M03 DTW Clássico (ms) | M04 DTW Sakoe (ms) | M05 DTW Keogh (ms / Poda) | M07 Cascata (ms / EE) | M08 DTW+LDA (ms) | M09 DTW+CNN (ms) | M10 Tri (ms) |",
            "| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
        ])

        for ch in ch_list:
            qp = qemu_prof[str(ch)]
            qm = {m["name"]: m for m in qp["models"]}
            par_ok = all(abs(m["parity_ratio"] - 1.0) < 1e-9 for m in qp["models"])
            lines.append(
                f"| **`{ch}CH`** | {'100.0%' if par_ok else 'PARCIAL'} | "
                f"{qm['M01_micro_lda_q7']['latency_mean_ms']*1000:.2f} µs | "
                f"{qm['M02_cnn1d_int8']['latency_mean_ms']:.4f} | "
                f"{qm['M03_dtw_classic_on2']['latency_mean_ms']:.4f} | "
                f"{qm['M04_dtw_sakoe_chiba']['latency_mean_ms']:.4f} | "
                f"{qm['M05_dtw_lb_keogh']['latency_mean_ms']:.4f} ({qm['M05_dtw_lb_keogh']['pruned_rate']*100:.0f}%) | "
                f"{qm['M07_hybrid_lda_cnn_cascade']['latency_mean_ms']:.4f} ({qm['M07_hybrid_lda_cnn_cascade']['early_exit_rate']*100:.0f}%) | "
                f"{qm['M08_hybrid_dtw_lda']['latency_mean_ms']:.4f} | "
                f"{qm['M09_hybrid_dtw_cnn']['latency_mean_ms']:.4f} | "
                f"{qm['M10_hybrid_dtw_lda_cnn']['latency_mean_ms']:.4f} |"
            )

        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Channel & Multi-Precision Benchmark across 10 Models.")
    parser.add_argument("--ozdemir-subject", default="S1")
    parser.add_argument("--ninapro-subject", default="S4")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("=====================================================================", flush=True)
    print(" BENCHMARK MULTI-CANAL & MULTI-PRECISAO (55 CENARIOS x 10 MODELOS)", flush=True)
    print("=====================================================================", flush=True)

    # 1. Load Raw Continuous Signals ONCE
    raw_oz = load_ozdemir_recording(args.ozdemir_subject, ROOT / "data" / "raw" / "ozdemir", use_filtered=True)
    loader_ni = NinaProDB2Loader(raw_dir=ROOT / "data" / "raw" / "ninapro_db2")
    raw_cache_ni: dict[str, RecordingData] = {
        eid: loader_ni.load_recording(args.ninapro_subject, eid) for eid in ("E1", "E2", "E3")
    }
    runner_ni = MultiExerciseWithinSubjectRunner()

    # 2. Helper to build precision cache on demand (one precision at a time -> keeps RAM < 1 GB)
    def build_precision_data(ds_name: str, prec: str) -> dict[str, Any]:
        if ds_name == "ozdemir":
            q_sig = quantize_ozdemir(raw_oz, precision=prec, v_max=OZ_V_FS_MV)
            win, lbl, reps = segment_ozdemir_windows(q_sig, fs=2000)
            tr_m, te_m = np.isin(reps, [1, 3, 4]), np.isin(reps, [2, 5])
            tr_w, tr_l = win[tr_m], lbl[tr_m]
            te_w, te_l = win[te_m], lbl[te_m]
            cal = ThresholdCalibrator(default_r=0.5, num_channels=4).calibrate(tr_w, tr_l, 0, 0.5)
            ext = TimeDomainFeatureExtractor(epsilon=cal.epsilon_channels, num_channels=4)
        else:
            def qload(sid: str, eid: str, pr: str = prec) -> RecordingData:
                rec = raw_cache_ni[eid]
                return replace(rec, emg=quantize_ninapro(rec.emg, precision=pr, v_max=NI_V_FS_VOLTS))

            subj = MultiExerciseSubject(subject_id=args.ninapro_subject, load_recording=qload)
            tr_w, tr_l, _, _ = runner_ni.prepare_repetition_windows(subj, repetitions=runner_ni.TRAINING_REPETITIONS)
            te_w, te_l, _, _ = runner_ni.prepare_repetition_windows(subj, repetitions=runner_ni.TEST_REPETITIONS)
            cal = ThresholdCalibrator(default_r=0.5, num_channels=12).calibrate(tr_w, tr_l, 0, 0.5)
            ext = TimeDomainFeatureExtractor(epsilon=cal.epsilon_channels, num_channels=12)

        return {
            "tr_win": tr_w,
            "tr_lbl": tr_l.astype(np.int32),
            "te_win": te_w,
            "te_lbl": te_l.astype(np.int32),
            "X_tr_full": ext.extract(tr_w).astype(np.float32),
            "X_te_full": ext.extract(te_w).astype(np.float32),
            "mav_tr_full": extract_subblock_mav(tr_w),
            "mav_te_full": extract_subblock_mav(te_w),
        }

    # 3 & 4. Run Ozdemir (all precisions & channels) then NinaPro (all precisions & channels)
    qemu_blocks_oz: dict[int, dict[str, Any]] = {}
    oz_scenarios: list[dict[str, Any]] = []
    oz_cnns: dict[int, QuantizedCnn1DArtifacts] = {
        4: load_cnn_artifacts("oz", "Ozdemir_4CH", args.ozdemir_subject)
    }

    oz_int16 = build_precision_data("ozdemir", "int16")
    for ch in OZ_CHANNELS:
        if ch not in oz_cnns:
            idx_c = get_channel_feature_indices(ch, 4)
            oz_cnns[ch] = train_fast_compact_cnn1d(
                f"Ozdemir_{ch}CH", args.ozdemir_subject, oz_int16["tr_win"][:, :, :ch], oz_int16["tr_lbl"], oz_int16["X_tr_full"][:, idx_c], 10, epochs=35
            )
            print(f"  [CNN-1D Backbone][Ozdemir {ch}CH] params={oz_cnns[ch].total_params}", flush=True)

    for prec in PRECISION_OPTIONS:
        cp = oz_int16 if prec == "int16" else build_precision_data("ozdemir", prec)
        for ch in OZ_CHANNELS:
            idx_c = get_channel_feature_indices(ch, 4)
            res = evaluate_scenario_10_models(
                cp["tr_win"][:, :, :ch], cp["tr_lbl"], cp["te_win"][:, :, :ch], cp["te_lbl"],
                cp["X_tr_full"][:, idx_c], cp["X_te_full"][:, idx_c],
                cp["mav_tr_full"][:, :, :ch], cp["mav_te_full"][:, :, :ch],
                oz_cnns[ch],
            )
            oz_scenarios.append({"num_channels": ch, "precision": prec, "metrics": res["metrics"]})
            if prec == "int16":
                qemu_blocks_oz[ch] = res["qemu_payload_data"]
            print(f"  [Ozdemir {ch}CH / {prec:<7}] LDA={res['metrics']['M01_micro_lda_q7']['top1_acc']*100:.2f}% CNN={res['metrics']['M02_cnn1d_int8']['top1_acc']*100:.2f}% LB={res['metrics']['M05_dtw_lb_keogh']['top1_acc']*100:.2f}% M10={res['metrics']['M10_hybrid_dtw_lda_cnn']['top1_acc']*100:.2f}%", flush=True)
    del oz_int16

    qemu_blocks_ni: dict[int, dict[str, Any]] = {}
    ni_scenarios: list[dict[str, Any]] = []
    ni_cnns: dict[int, QuantizedCnn1DArtifacts] = {
        12: load_cnn_artifacts("ni", "NinaPro_DB2_12CH", args.ninapro_subject)
    }

    ni_int16 = build_precision_data("ninapro", "int16")
    for ch in NI_CHANNELS:
        if ch not in ni_cnns:
            idx_c = get_channel_feature_indices(ch, 12)
            ni_cnns[ch] = train_fast_compact_cnn1d(
                f"NinaPro_{ch}CH", args.ninapro_subject, ni_int16["tr_win"][:, :, :ch], ni_int16["tr_lbl"], ni_int16["X_tr_full"][:, idx_c], 50, epochs=20
            )
            print(f"  [CNN-1D Backbone][NinaPro {ch}CH] params={ni_cnns[ch].total_params}", flush=True)

    for prec in PRECISION_OPTIONS:
        cp = ni_int16 if prec == "int16" else build_precision_data("ninapro", prec)
        for ch in NI_CHANNELS:
            idx_c = get_channel_feature_indices(ch, 12)
            res = evaluate_scenario_10_models(
                cp["tr_win"][:, :, :ch], cp["tr_lbl"], cp["te_win"][:, :, :ch], cp["te_lbl"],
                cp["X_tr_full"][:, idx_c], cp["X_te_full"][:, idx_c],
                cp["mav_tr_full"][:, :, :ch], cp["mav_te_full"][:, :, :ch],
                ni_cnns[ch],
            )
            ni_scenarios.append({"num_channels": ch, "precision": prec, "metrics": res["metrics"]})
            if prec == "int16":
                qemu_blocks_ni[ch] = res["qemu_payload_data"]
            print(f"  [NinaPro {ch}CH / {prec:<7}] LDA={res['metrics']['M01_micro_lda_q7']['top1_acc']*100:.2f}% CNN={res['metrics']['M02_cnn1d_int8']['top1_acc']*100:.2f}% LB={res['metrics']['M05_dtw_lb_keogh']['top1_acc']*100:.2f}% M09={res['metrics']['M09_hybrid_dtw_cnn']['top1_acc']*100:.2f}%", flush=True)
    del ni_int16

    qemu_blocks = [(0, qemu_blocks_oz[ch]) for ch in OZ_CHANNELS] + [(1, qemu_blocks_ni[ch]) for ch in NI_CHANNELS]

    # 5. Profile cycles in QEMU ESP32-S3 across all 11 `(dataset, C)` configurations
    print("Running multi-channel QEMU CCOUNT cycle profiler across all 11 (dataset, C) configs...", flush=True)
    qemu_configs = run_qemu_channel_profiler(qemu_blocks)
    qemu_map: dict[str, dict[str, Any]] = {"ozdemir": {}, "ninapro": {}}
    for qc in qemu_configs:
        dk = "ozdemir" if qc["dataset_code"] == 0 else "ninapro"
        qemu_map[dk][str(qc["num_channels"])] = qc

    results = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "datasets": {
            "ozdemir": {"subject": args.ozdemir_subject, "scenarios": oz_scenarios},
            "ninapro": {"subject": args.ninapro_subject, "scenarios": ni_scenarios},
        },
        "qemu_channel_profiles": qemu_map,
    }
    res_json = args.out_dir / "channel_and_quantization_benchmark_results.json"
    res_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = generate_markdown_report(results)
    res_md = args.out_dir / "channel_and_quantization_master_report.md"
    res_md.write_text(md, encoding="utf-8")

    print(f"Completed 550 evaluations + QEMU channel scaling in {time.time() - t0:.2f}s!\n  JSON: {res_json}\n  Report: {res_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
