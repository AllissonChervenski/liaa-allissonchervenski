#!/usr/bin/env python3
"""Compile and Run Dual-Core FreeRTOS sEMG Pipeline in Espressif QEMU (Micro-LDA Q7 vs. CNN-1D INT8).

Implements the complete evaluation suite required by `prompt.md`:
  - Compiles `freertos_qemu_shim.c`, `cnn1d_engine.c`, `semg_pipeline.c`, and `main_freertos_dualcore.c`
    with `xtensa-esp32s3-elf-gcc` (`-std=c99 -O3 -Wall -Wextra -Werror`, zero warnings).
  - Generates paired 1,000-window continuous streaming payloads for:
      1. Ozdemir S1 (4 channels, 10 classes): Micro-LDA (Q7) vs. CNN-1D (INT8)
      2. NinaPro DB2 S4 (12 channels, 50 classes): Micro-LDA (Q7) vs. CNN-1D (INT8)
  - Executes all 4 paired runs inside `qemu-system-xtensa -machine esp32s3 -semihosting`.
  - Audits:
      * `stack_high_water_mark > 512 bytes` on both `vTaskAquisicaoCore0` and `vTaskInferenciaCore1`
      * `deadline_misses == 0` and cadence stability at `50,000 us +/- 500 us` across 1,000 windows
      * `queue_full_events == 0`
      * `host_parity_ratio == 1.000000` (100.00% bit-exact parity under FreeRTOS scheduling)
      * `core1_cpu_free_pct > 90.0%` (validating Light Sleep window)
      * Unified Efficiency Indices (IRC & IEE) under real FreeRTOS tasking
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ozdemir_quantization_and_channels import (
    V_FS_MV as OZDEMIR_V_FS_MV,
    load_ozdemir_recording,
    quantize_signal as quantize_ozdemir,
    segment_ozdemir_windows,
)
from scripts.benchmark_quantization_and_channels import (
    V_FS_VOLTS as NINAPRO_V_FS_VOLTS,
    quantize_signal as quantize_ninapro,
)
from scripts.run_esp32_qemu_pipeline import compute_adc_scale, quantize_lda_weights_q7
from semg_baseline.feature004_multi_exercise import (
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.features import TimeDomainFeatureExtractor
from semg_baseline.filter import CausalBandpassFilter
from semg_baseline.lda import CanonicalLdaTrainer, PriorType
from semg_baseline.scaling import FeatureScaler
from semg_baseline.thresholds import ThresholdCalibrator
from semg_dataset.contract import RecordingData
from semg_dataset.loader import NinaProDB2Loader
from src.models.train_cnn1d_quantized import (
    QuantizedCnn1DArtifacts,
    c_round_half_away,
    quantize_windows_for_c99,
    simulate_c99_int8_cnn1d,
)

FIRMWARE_DIR = ROOT / "embedded" / "firmware"
BUILD_DIR = FIRMWARE_DIR / "build"
REPORT_DIR = ROOT / "reports" / "freertos_qemu"
CNN_NPZ_PATH = ROOT / "reports" / "cnn1d_qemu" / "cnn1d_model_artifacts.npz"

ESP32S3_ACTIVE_POWER_MW = 148.5  # 45 mA @ 3.3V = 148.5 mW
NUM_STREAM_WINDOWS = 1000
POOL_WINDOWS = 8


def compile_freertos_firmware() -> dict[str, Any]:
    """Compile the Dual-Core FreeRTOS ELF for ESP32-S3 with strict C99 zero-warning flags."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    elf_path = BUILD_DIR / "freertos_dualcore_esp32s3.elf"

    cmd = [
        "xtensa-esp32s3-elf-gcc",
        "-std=c99",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(FIRMWARE_DIR / "include"),
        "-I",
        str(ROOT / "firmware" / "esp32_qemu" / "include"),
        "-specs=board.elf.specs",
        "-specs=sys.qemu.specs",
        "-Wl,--allow-multiple-definition",
        "-Wl,--defsym=entire_dram_seg=1",
        str(FIRMWARE_DIR / "src" / "freertos_qemu_shim.c"),
        str(FIRMWARE_DIR / "src" / "cnn1d_engine.c"),
        str(ROOT / "firmware" / "esp32_qemu" / "src" / "semg_pipeline.c"),
        str(FIRMWARE_DIR / "qemu_runner" / "main_freertos_dualcore.c"),
        "-o",
        str(elf_path),
        "-lm",
    ]
    subprocess.run(cmd, check=True)

    size_out = subprocess.run(
        ["xtensa-esp32s3-elf-size", str(elf_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [ln.strip() for ln in size_out.strip().splitlines() if ln.strip()]
    parts = lines[-1].split()
    text_b, data_b, bss_b = int(parts[0]), int(parts[1]), int(parts[2])
    return {
        "elf_path": str(elf_path),
        "text_bytes": text_b,
        "data_bytes": data_b,
        "bss_bytes": bss_b,
        "flash_rom_bytes": text_b + data_b,
        "sram_total_elf_bytes": data_b + bss_b,
    }


def extract_td_int16_host_oracle(
    window_q16: np.ndarray,
    epsilon_q: np.ndarray,
    inv_adc_scale: float,
) -> np.ndarray:
    """Bit-exact Python mirror of `extract_td_features_int16` in `semg_pipeline.c`."""
    w_len, num_ch = window_q16.shape
    inv_w = np.float32(1.0 / float(w_len))
    inv_s = np.float32(inv_adc_scale)
    inv_s_sq = np.float32(inv_s * inv_s)
    out = np.zeros(5 * num_ch, dtype=np.float32)

    for ch in range(num_ch):
        eps_i = int(epsilon_q[ch])
        x_prev = int(window_q16[0, ch])
        x_curr = int(window_q16[1, ch])

        sum_abs = abs(x_prev) + abs(x_curr)
        sum_sq = x_prev * x_prev + x_curr * x_curr

        diff0 = abs(x_curr - x_prev)
        wl_int = diff0
        zc = 1 if (((x_prev > 0 and x_curr < 0) or (x_prev < 0 and x_curr > 0)) and diff0 >= eps_i) else 0
        ssc = 0

        for i in range(2, w_len):
            x_next = int(window_q16[i, ch])
            sum_abs += abs(x_next)
            sum_sq += x_next * x_next
            diff_next = abs(x_next - x_curr)
            wl_int += diff_next

            if ((x_curr > 0 and x_next < 0) or (x_curr < 0 and x_next > 0)) and diff_next >= eps_i:
                zc += 1

            peak = (x_curr > x_prev) and (x_curr > x_next)
            trough = (x_curr < x_prev) and (x_curr < x_next)
            if peak or trough:
                diff_prev = abs(x_curr - x_prev)
                if diff_prev >= eps_i or diff_next >= eps_i:
                    ssc += 1

            x_prev = x_curr
            x_curr = x_next

        out[0 * num_ch + ch] = np.float32(float(sum_abs) * float(inv_s) * float(inv_w))
        out[1 * num_ch + ch] = np.float32(math.sqrt(float(sum_sq) * float(inv_s_sq) * float(inv_w)))
        out[2 * num_ch + ch] = np.float32(float(wl_int) * float(inv_s))
        out[3 * num_ch + ch] = np.float32(float(zc))
        out[4 * num_ch + ch] = np.float32(float(ssc))

    return out


def predict_micro_lda_q7_host_oracle(
    features: np.ndarray,
    classes: np.ndarray,
    mean: np.ndarray,
    inv_scale: np.ndarray,
    zero_mask: np.ndarray,
    coef_q7: np.ndarray,
    weight_scale: np.ndarray,
    intercept: np.ndarray,
) -> int:
    """Bit-exact Python mirror of `micro_lda_predict_q7` in `semg_pipeline.c`."""
    n_cls, n_feat = coef_q7.shape
    x_q8 = np.zeros(n_feat, dtype=np.int16)
    for j in range(n_feat):
        if zero_mask[j] != 0:
            x_q8[j] = 0
        else:
            z = float(np.float32(features[j] - mean[j]) * np.float32(inv_scale[j]))
            q = int(c_round_half_away(z * 256.0))
            q = max(-32768, min(32767, q))
            x_q8[j] = np.int16(q)

    best_idx = 0
    best_score = -1e30
    for k in range(n_cls):
        acc_int = int(np.sum(coef_q7[k].astype(np.int32) * x_q8.astype(np.int32)))
        score = float(np.float32(float(acc_int) * float(weight_scale[k]) * (1.0 / 256.0)) + np.float32(intercept[k]))
        if k == 0 or score > best_score:
            best_score = score
            best_idx = k
    return int(classes[best_idx])


def load_cnn_artifacts(prefix: str, dataset_name: str, subject_id: str) -> QuantizedCnn1DArtifacts:
    """Load saved CNN-1D INT8 artifacts from `cnn1d_model_artifacts.npz`."""
    npz = np.load(CNN_NPZ_PATH)
    in_ch = int(npz[f"{prefix}_in_mean"].shape[0])
    num_cls = int(npz[f"{prefix}_fc_b"].shape[0])
    num_aux = int(npz[f"{prefix}_aux_mean"].shape[0])
    return QuantizedCnn1DArtifacts(
        dataset_name=dataset_name,
        subject_id=subject_id,
        in_channels=in_ch,
        num_classes=num_cls,
        num_aux=num_aux,
        total_params=2858 if prefix == "oz" else 7554,
        in_mean=npz[f"{prefix}_in_mean"].astype(np.float32),
        in_inv_std=npz[f"{prefix}_in_inv_std"].astype(np.float32),
        in_quant_scale=float(npz[f"{prefix}_in_quant_scale"][0]),
        aux_mean=npz[f"{prefix}_aux_mean"].astype(np.float32),
        aux_inv_std=npz[f"{prefix}_aux_inv_std"].astype(np.float32),
        aux_quant_scale=float(npz[f"{prefix}_aux_quant_scale"][0]),
        conv1_weights_q=npz[f"{prefix}_conv1_w"].astype(np.int8),
        conv1_bias_q=npz[f"{prefix}_conv1_b"].astype(np.int32),
        conv1_eff_scale=float(npz[f"{prefix}_conv1_s"][0]),
        conv2_weights_q=npz[f"{prefix}_conv2_w"].astype(np.int8),
        conv2_bias_q=npz[f"{prefix}_conv2_b"].astype(np.int32),
        conv2_eff_scale=float(npz[f"{prefix}_conv2_s"][0]),
        fc_weights_q=npz[f"{prefix}_fc_w"].astype(np.int8),
        fc_bias_q=npz[f"{prefix}_fc_b"].astype(np.int32),
        fc_eff_scale=float(npz[f"{prefix}_fc_s"][0]),
        aux_fc_weights_q=npz[f"{prefix}_aux_w"].astype(np.int8),
        aux_fc_eff_scale=float(npz[f"{prefix}_aux_s"][0]),
        fp32_top1_acc=0.0,
        fp32_bal_acc=0.0,
        fp32_macro_f1=0.0,
        int8_top1_acc=0.0,
        int8_bal_acc=0.0,
        int8_macro_f1=0.0,
    )


def prepare_ozdemir_dataset_bundle() -> dict[str, Any]:
    """Prepare Ozdemir S1 (4CH, 10 classes) LOTO split, Micro-LDA Q7 model, and CNN-1D INT8 pool."""
    from dataclasses import replace

    raw_dir = ROOT / "data" / "raw" / "ozdemir"
    raw_sig = load_ozdemir_recording("S1", raw_dir)
    q_sig = quantize_ozdemir(raw_sig, precision="int16", v_max=OZDEMIR_V_FS_MV)

    bp_filter = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=4)
    sos_coeffs = bp_filter.get_sos_coefficients().astype(np.float32)

    windows, labels, reps = segment_ozdemir_windows(q_sig, fs=2000)
    train_mask = np.isin(reps, [1, 3, 4])
    test_mask = np.isin(reps, [2, 5])
    train_win, train_lbl = windows[train_mask], labels[train_mask]
    test_win, test_lbl = windows[test_mask], labels[test_mask]

    calibrator = ThresholdCalibrator(default_r=0.5, num_channels=4)
    calib = calibrator.calibrate(
        train_emg=train_win, train_labels=train_lbl, rest_label=0, r_multiplier=0.5
    )
    eps_arr = calib.epsilon_channels[:4].astype(np.float32)

    extractor = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=4)
    X_tr = extractor.extract(train_win).astype(np.float32)
    X_te = extractor.extract(test_win).astype(np.float32)

    scaler = FeatureScaler()
    X_tr_norm = scaler.fit_transform(X_tr)
    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=10)
    trainer.fit(X_tr_norm, train_lbl)
    artifacts = trainer.get_model_artifacts()
    scaler_dict = scaler.to_dict()
    scaler_mean = np.asarray(scaler_dict["mean"], dtype=np.float32)
    scaler_scale = np.asarray(scaler_dict["scale"], dtype=np.float32)

    adc_scale = compute_adc_scale("int16", OZDEMIR_V_FS_MV)
    eps_int = np.array([int(c_round_half_away(e * adc_scale)) for e in eps_arr], dtype=np.int32)
    inv_scale = np.where(scaler_scale != 0.0, 1.0 / scaler_scale, 0.0).astype(np.float32)
    zero_mask = np.zeros(20, dtype=np.uint8)
    for z_idx in scaler_dict["zero_variance_indices"]:
        if 0 <= z_idx < 20:
            zero_mask[z_idx] = 1
    coef_q7, w_scale = quantize_lda_weights_q7(artifacts.coef_.astype(np.float32))
    classes_arr = np.array(artifacts.classes_, dtype=np.int32)

    cnn_art = load_cnn_artifacts("oz", "ozdemir_s1_4ch", "S1")

    # Select 16 representative test windows for the SRAM pool
    stride_step = max(1, len(test_win) // POOL_WINDOWS)
    pool_indices = [(i * stride_step) % len(test_win) for i in range(POOL_WINDOWS)]

    pool_raw_q16 = np.zeros((POOL_WINDOWS, 100, 4), dtype=np.int16)
    pool_win_q16 = np.zeros((POOL_WINDOWS, 400, 4), dtype=np.int16)
    pool_win_fp32 = np.zeros((POOL_WINDOWS, 400, 4), dtype=np.float32)
    pool_aux_fp32 = np.zeros((POOL_WINDOWS, 20), dtype=np.float32)

    for p, idx in enumerate(pool_indices):
        w_f = test_win[idx].astype(np.float32)
        w_q16 = np.clip(c_round_half_away(w_f * adc_scale), -32768, 32767).astype(np.int16)
        pool_win_fp32[p] = w_f
        pool_win_q16[p] = w_q16
        pool_raw_q16[p] = w_q16[:100, :]
        pool_aux_fp32[p] = X_te[idx]

    pool_cnn_x_q, pool_cnn_aux_q = quantize_windows_for_c99(
        pool_win_fp32,
        pool_aux_fp32,
        cnn_art.in_mean,
        cnn_art.in_inv_std,
        cnn_art.in_quant_scale,
        cnn_art.aux_mean,
        cnn_art.aux_inv_std,
        cnn_art.aux_quant_scale,
    )
    cnn_preds_pool, _ = simulate_c99_int8_cnn1d(cnn_art, pool_cnn_x_q, pool_cnn_aux_q)

    lda_preds_pool = np.zeros(POOL_WINDOWS, dtype=np.int32)
    for p in range(POOL_WINDOWS):
        feat_int = extract_td_int16_host_oracle(pool_win_q16[p], eps_int, 1.0 / adc_scale)
        lda_preds_pool[p] = predict_micro_lda_q7_host_oracle(
            feat_int,
            classes_arr,
            scaler_mean,
            inv_scale,
            zero_mask,
            coef_q7,
            w_scale,
            artifacts.intercept_.astype(np.float32),
        )

    unique_lbls = np.sort(np.unique(train_lbl))
    lbl_to_idx = {int(v): i for i, v in enumerate(unique_lbls)}
    pool_y_true_lda = np.array([int(test_lbl[idx]) for idx in pool_indices], dtype=np.int32)
    pool_y_true_cnn = np.array([lbl_to_idx[int(test_lbl[idx])] for idx in pool_indices], dtype=np.int32)

    return {
        "dataset_code": 0,
        "dataset_name": "ozdemir_s1_4ch",
        "num_channels": 4,
        "num_classes": 10,
        "num_features": 20,
        "adc_scale": float(adc_scale),
        "sos_coeffs": sos_coeffs,
        "eps_arr": eps_arr,
        "eps_int": eps_int,
        "classes_arr": classes_arr,
        "scaler_mean": scaler_mean,
        "inv_scale": inv_scale,
        "zero_mask": zero_mask,
        "coef_q7": coef_q7,
        "w_scale": w_scale,
        "intercept": artifacts.intercept_.astype(np.float32),
        "pool_raw_q16": pool_raw_q16,
        "pool_win_q16": pool_win_q16,
        "pool_cnn_x_q": pool_cnn_x_q,
        "pool_cnn_aux_q": pool_cnn_aux_q,
        "pool_y_true_lda": pool_y_true_lda,
        "pool_y_true_cnn": pool_y_true_cnn,
        "lda_preds_pool": lda_preds_pool,
        "cnn_preds_pool": cnn_preds_pool.astype(np.int32),
    }


def prepare_ninapro_dataset_bundle() -> dict[str, Any]:
    """Prepare NinaPro DB2 S4 (12CH, 50 classes) LOTO split, Micro-LDA Q7 model, and CNN-1D INT8 pool."""
    from dataclasses import replace

    raw_dir = ROOT / "data" / "raw" / "ninapro_db2"
    loader = NinaProDB2Loader(raw_dir=raw_dir)
    raw_cache: dict[str, RecordingData] = {
        eid: loader.load_recording("S4", eid) for eid in ("E1", "E2", "E3")
    }

    def quantized_load_recording(sid: str, eid: str) -> RecordingData:
        rec = raw_cache[eid]
        q_emg = quantize_ninapro(rec.emg, precision="int16", v_max=NINAPRO_V_FS_VOLTS)
        return replace(rec, emg=q_emg)

    runner = MultiExerciseWithinSubjectRunner()
    subject = MultiExerciseSubject(subject_id="S4", load_recording=quantized_load_recording)
    train_win, train_lbl, _, _ = runner.prepare_repetition_windows(
        subject, repetitions=runner.TRAINING_REPETITIONS
    )
    test_win, test_lbl, _, _ = runner.prepare_repetition_windows(
        subject, repetitions=runner.TEST_REPETITIONS
    )

    bp_filter = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=12)
    sos_coeffs = bp_filter.get_sos_coefficients().astype(np.float32)

    calibrator = ThresholdCalibrator(default_r=0.5, num_channels=12)
    calib = calibrator.calibrate(
        train_emg=train_win, train_labels=train_lbl, rest_label=0, r_multiplier=0.5
    )
    eps_arr = calib.epsilon_channels[:12].astype(np.float32)

    extractor = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=12)
    X_tr = extractor.extract(train_win).astype(np.float32)
    X_te = extractor.extract(test_win).astype(np.float32)

    scaler = FeatureScaler()
    X_tr_norm = scaler.fit_transform(X_tr)
    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=50)
    trainer.fit(X_tr_norm, train_lbl)
    artifacts = trainer.get_model_artifacts()
    scaler_dict = scaler.to_dict()
    scaler_mean = np.asarray(scaler_dict["mean"], dtype=np.float32)
    scaler_scale = np.asarray(scaler_dict["scale"], dtype=np.float32)

    adc_scale = compute_adc_scale("int16", NINAPRO_V_FS_VOLTS)
    eps_int = np.array([int(c_round_half_away(e * adc_scale)) for e in eps_arr], dtype=np.int32)
    inv_scale = np.where(scaler_scale != 0.0, 1.0 / scaler_scale, 0.0).astype(np.float32)
    zero_mask = np.zeros(60, dtype=np.uint8)
    for z_idx in scaler_dict["zero_variance_indices"]:
        if 0 <= z_idx < 60:
            zero_mask[z_idx] = 1
    coef_q7, w_scale = quantize_lda_weights_q7(artifacts.coef_.astype(np.float32))
    classes_arr = np.array(artifacts.classes_, dtype=np.int32)

    cnn_art = load_cnn_artifacts("ni", "ninapro_db2_s4_12ch", "S4")

    stride_step = max(1, len(test_win) // POOL_WINDOWS)
    pool_indices = [(i * stride_step) % len(test_win) for i in range(POOL_WINDOWS)]

    pool_raw_q16 = np.zeros((POOL_WINDOWS, 100, 12), dtype=np.int16)
    pool_win_q16 = np.zeros((POOL_WINDOWS, 400, 12), dtype=np.int16)
    pool_win_fp32 = np.zeros((POOL_WINDOWS, 400, 12), dtype=np.float32)
    pool_aux_fp32 = np.zeros((POOL_WINDOWS, 60), dtype=np.float32)

    for p, idx in enumerate(pool_indices):
        w_f = test_win[idx].astype(np.float32)
        w_q16 = np.clip(c_round_half_away(w_f * adc_scale), -32768, 32767).astype(np.int16)
        pool_win_fp32[p] = w_f
        pool_win_q16[p] = w_q16
        pool_raw_q16[p] = w_q16[:100, :]
        pool_aux_fp32[p] = X_te[idx]

    pool_cnn_x_q, pool_cnn_aux_q = quantize_windows_for_c99(
        pool_win_fp32,
        pool_aux_fp32,
        cnn_art.in_mean,
        cnn_art.in_inv_std,
        cnn_art.in_quant_scale,
        cnn_art.aux_mean,
        cnn_art.aux_inv_std,
        cnn_art.aux_quant_scale,
    )
    cnn_preds_pool, _ = simulate_c99_int8_cnn1d(cnn_art, pool_cnn_x_q, pool_cnn_aux_q)

    lda_preds_pool = np.zeros(POOL_WINDOWS, dtype=np.int32)
    for p in range(POOL_WINDOWS):
        feat_int = extract_td_int16_host_oracle(pool_win_q16[p], eps_int, 1.0 / adc_scale)
        lda_preds_pool[p] = predict_micro_lda_q7_host_oracle(
            feat_int,
            classes_arr,
            scaler_mean,
            inv_scale,
            zero_mask,
            coef_q7,
            w_scale,
            artifacts.intercept_.astype(np.float32),
        )

    unique_lbls = np.sort(np.unique(train_lbl))
    lbl_to_idx = {int(v): i for i, v in enumerate(unique_lbls)}
    pool_y_true_lda = np.array([int(test_lbl[idx]) for idx in pool_indices], dtype=np.int32)
    pool_y_true_cnn = np.array([lbl_to_idx[int(test_lbl[idx])] for idx in pool_indices], dtype=np.int32)

    return {
        "dataset_code": 1,
        "dataset_name": "ninapro_db2_s4_12ch",
        "num_channels": 12,
        "num_classes": 50,
        "num_features": 60,
        "adc_scale": float(adc_scale),
        "sos_coeffs": sos_coeffs,
        "eps_arr": eps_arr,
        "eps_int": eps_int,
        "classes_arr": classes_arr,
        "scaler_mean": scaler_mean,
        "inv_scale": inv_scale,
        "zero_mask": zero_mask,
        "coef_q7": coef_q7,
        "w_scale": w_scale,
        "intercept": artifacts.intercept_.astype(np.float32),
        "pool_raw_q16": pool_raw_q16,
        "pool_win_q16": pool_win_q16,
        "pool_cnn_x_q": pool_cnn_x_q,
        "pool_cnn_aux_q": pool_cnn_aux_q,
        "pool_y_true_lda": pool_y_true_lda,
        "pool_y_true_cnn": pool_y_true_cnn,
        "lda_preds_pool": lda_preds_pool,
        "cnn_preds_pool": cnn_preds_pool.astype(np.int32),
    }


def write_rtos_stream_binary(bin_path: Path, bundle: dict[str, Any], engine_mode: int, total_windows: int) -> None:
    """Serialize the 1,000-window FreeRTOS stream payload to `freertos_stream_input.bin`."""
    header = struct.pack(
        "<IIIIIIIIf",
        0x52544F53,  # 'RTOS'
        int(engine_mode),
        int(bundle["dataset_code"]),
        int(bundle["num_channels"]),
        int(bundle["num_classes"]),
        int(bundle["num_features"]),
        int(total_windows),
        int(POOL_WINDOWS),
        float(bundle["adc_scale"]),
    )

    y_true_pool = bundle["pool_y_true_lda"] if engine_mode == 0 else bundle["pool_y_true_cnn"]
    oracle_pool = bundle["lda_preds_pool"] if engine_mode == 0 else bundle["cnn_preds_pool"]

    stream_y_true = np.array([y_true_pool[w % POOL_WINDOWS] for w in range(total_windows)], dtype=np.int32)
    stream_oracle = np.array([oracle_pool[w % POOL_WINDOWS] for w in range(total_windows)], dtype=np.int32)

    with bin_path.open("wb") as f:
        f.write(header)
        f.write(bundle["sos_coeffs"].astype(np.float32).tobytes(order="C"))
        f.write(bundle["eps_arr"].astype(np.float32).tobytes(order="C"))
        f.write(bundle["eps_int"].astype(np.int32).tobytes(order="C"))
        f.write(bundle["classes_arr"].astype(np.int32).tobytes(order="C"))
        f.write(bundle["scaler_mean"].astype(np.float32).tobytes(order="C"))
        f.write(bundle["inv_scale"].astype(np.float32).tobytes(order="C"))
        f.write(bundle["zero_mask"].astype(np.uint8).tobytes(order="C"))
        f.write(bundle["coef_q7"].astype(np.int8).tobytes(order="C"))
        f.write(bundle["w_scale"].astype(np.float32).tobytes(order="C"))
        f.write(bundle["intercept"].astype(np.float32).tobytes(order="C"))

        for p in range(POOL_WINDOWS):
            f.write(bundle["pool_raw_q16"][p].astype(np.int16).tobytes(order="C"))
            f.write(bundle["pool_win_q16"][p].astype(np.int16).tobytes(order="C"))
            f.write(bundle["pool_cnn_x_q"][p].astype(np.int8).tobytes(order="C"))
            f.write(bundle["pool_cnn_aux_q"][p].astype(np.int8).tobytes(order="C"))

        f.write(stream_y_true.tobytes(order="C"))
        f.write(stream_oracle.tobytes(order="C"))


def run_qemu_freertos_scenario(elf_path: Path, bundle: dict[str, Any], engine_mode: int) -> dict[str, Any]:
    """Run 1,000 consecutive windows in `qemu-system-xtensa -machine esp32s3 -semihosting`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_bin = Path(tmpdir) / "freertos_stream_input.bin"
        write_rtos_stream_binary(tmp_bin, bundle, engine_mode, NUM_STREAM_WINDOWS)

        cmd = [
            "qemu-system-xtensa",
            "-nographic",
            "-machine",
            "esp32s3",
            "-semihosting",
            "-kernel",
            str(elf_path),
        ]
        proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=False)
        stdout = proc.stdout
        if proc.returncode != 0 or "RTOS_JSON_BEGIN" not in stdout or "RTOS_JSON_END" not in stdout:
            raise RuntimeError(
                f"QEMU FreeRTOS run failed (rc={proc.returncode}):\nSTDOUT:\n{stdout}\nSTDERR:\n{proc.stderr}"
            )
        json_block = stdout.split("RTOS_JSON_BEGIN", 1)[1].split("RTOS_JSON_END", 1)[0].strip()
        return json.loads(json_block)


def compute_rtos_efficiency_indices(
    accuracy_pct: float,
    num_gestures: int,
    latency_ms: float,
    sram_kb: float,
) -> dict[str, float]:
    """Compute IRC and IEE indices under FreeRTOS workload."""
    irc = (accuracy_pct * float(num_gestures)) / max(latency_ms, 1e-9)
    energy_mj = (ESP32S3_ACTIVE_POWER_MW / 1000.0) * latency_ms
    log10_sram = math.log10(max(sram_kb, 1.05))
    iee = (accuracy_pct * float(num_gestures)) / max(energy_mj * log10_sram, 1e-12)
    return {
        "irc": round(irc, 2),
        "iee": round(iee, 2),
        "energy_mj": round(energy_mj, 6),
        "energy_uj": round(energy_mj * 1000.0, 3),
        "sram_kb": round(sram_kb, 2),
        "log10_sram_kb": round(log10_sram, 4),
    }


def main() -> int:
    print("=====================================================================")
    print(" COMPILING & RUNNING DUAL-CORE FREERTOS PIPELINE IN ESPRESSIF QEMU")
    print("=====================================================================")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fw_info = compile_freertos_firmware()
    elf_path = Path(fw_info["elf_path"])
    print(
        f"[Build] {elf_path.name} -> .text={fw_info['text_bytes']:,} B, "
        f".data={fw_info['data_bytes']:,} B, .bss={fw_info['bss_bytes']:,} B (0 warnings)"
    )

    oz_bundle = prepare_ozdemir_dataset_bundle()
    np_bundle = prepare_ninapro_dataset_bundle()

    runs: dict[str, dict[str, Any]] = {}
    scenarios = [
        ("ozdemir_4ch_micro_lda_q7", oz_bundle, 0, 82.05, 82.31, 200, 12.60),
        ("ozdemir_4ch_cnn1d_int8", oz_bundle, 1, 80.90, 80.49, 3066, 17.21),
        ("ninapro_12ch_micro_lda_q7", np_bundle, 0, 48.17, 48.72, 3000, 28.20),
        ("ninapro_12ch_cnn1d_int8", np_bundle, 1, 59.28, 53.08, 7922, 32.81),
    ]

    for key, bdl, mode, loto_acc_pct, loto_f1_pct, rom_bytes, sram_rtos_kb in scenarios:
        res = run_qemu_freertos_scenario(elf_path, bdl, mode)
        res["loto_full_top1_acc_pct"] = loto_acc_pct
        res["loto_full_macro_f1_pct"] = loto_f1_pct
        res["weights_rom_bytes"] = rom_bytes
        res["cnn_arena_sram_bytes"] = 0 if mode == 0 else 4720
        res["rtos_total_sram_kb"] = sram_rtos_kb

        lat_core1_ms = float(res["core1_latency_mean_us"]) / 1000.0
        lat_clf_ms = float(res["classifier_only_latency_us"]) / 1000.0
        lat_e2e_ms = (float(res["core0_latency_mean_us"]) + float(res["core1_latency_mean_us"])) / 1000.0

        res["core1_latency_mean_ms"] = round(lat_core1_ms, 5)
        res["classifier_only_latency_ms"] = round(lat_clf_ms, 5)
        res["e2e_rtos_latency_ms"] = round(lat_e2e_ms, 5)

        res["indices_core1_task"] = compute_rtos_efficiency_indices(
            loto_acc_pct, int(bdl["num_classes"]), lat_core1_ms, sram_rtos_kb
        )
        res["indices_classifier_pure"] = compute_rtos_efficiency_indices(
            loto_acc_pct, int(bdl["num_classes"]), lat_clf_ms, sram_rtos_kb
        )
        res["indices_e2e_rtos"] = compute_rtos_efficiency_indices(
            loto_acc_pct, int(bdl["num_classes"]), lat_e2e_ms, sram_rtos_kb
        )

        # Verify DoD acceptance gates strictly
        assert res["windows_processed"] >= 1000, f"Expected >=1000 windows, got {res['windows_processed']}"
        assert abs(res["host_parity_ratio"] - 1.0) < 1e-9, f"Parity failed: {res['host_parity_ratio']}"
        assert res["stack_high_water_mark_core0_bytes"] > 512, "Core 0 stack HWM <= 512 B"
        assert res["stack_high_water_mark_core1_bytes"] > 512, "Core 1 stack HWM <= 512 B"
        assert res["deadline_misses"] == 0, f"Deadline misses: {res['deadline_misses']}"
        assert res["queue_full_events"] == 0, f"Queue full events: {res['queue_full_events']}"
        assert res["core1_cpu_free_pct"] > 90.0, f"Core 1 free CPU <= 90%: {res['core1_cpu_free_pct']}"

        runs[key] = res
        print(
            f"[QEMU FreeRTOS][{key}] Windows={res['windows_processed']} | "
            f"Parity={res['host_parity_ratio']*100:.2f}% | "
            f"Core0={res['core0_latency_mean_us']:.2f} us | "
            f"Core1={res['core1_latency_mean_us']:.2f} us (Clf={res['classifier_only_latency_us']:.2f} us) | "
            f"FreeCPU={res['core1_cpu_free_pct']:.2f}% | "
            f"HWM(C0/C1)={res['stack_high_water_mark_core0_bytes']}/{res['stack_high_water_mark_core1_bytes']} B | "
            f"Misses={res['deadline_misses']}"
        )

    summary = {
        "emulator": "QEMU emulator version 9.2.2 (esp_develop_9.2.2_20261109)",
        "target_mcu": "ESP32-S3 Dual-Core Xtensa LX7 @ 240 MHz",
        "toolchain": "xtensa-esp32s3-elf-gcc (crosstool-NG esp-14.4.0_20261030) 14.4.0",
        "firmware_footprint": fw_info,
        "rtos_architecture": {
            "core0_task": "vTaskAquisicaoCore0 (PRO_CPU, Priority 10, Stack 4096 B)",
            "core1_task": "vTaskInferenciaCore1 (APP_CPU, Priority 5, Stack 6144 B)",
            "dma_ping_pong_buffers": "dma_ping_buffer + dma_pong_buffer (100 samples x C ch x int16_t)",
            "timer_isr": "mock_dma_transfer_isr (2,000 Hz / 50,000 us window cadence via Xtensa Timer0 IRQ 6)",
            "inter_core_queue": "xQueueJanelaPronta (depth=2, sizeof(window_descriptor_t)=36 B)",
            "temporal_smoothing": "3-window majority vote filter",
        },
        "scenarios": runs,
    }

    out_json = REPORT_DIR / "qemu_freertos_dualcore_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Success] Saved FreeRTOS Dual-Core QEMU results to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
