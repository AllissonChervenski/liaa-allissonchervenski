#!/usr/bin/env python3
"""Export Static Micro-LDA (Q7), SOS Bandpass Filter, and Self-Test Oracles to C99 Header.

Generates `firmware/esp32s3_wroom/components/semg_core/include/micro_lda_weights.h`
so the ESP32-S3-WROOM firmware runs completely standalone from internal Flash (`.rodata`)
with zero semihosting dependencies.

Includes:
  1. Butterworth 4th-order (20-450 Hz @ 2 kHz) Causal SOS filter coefficients.
  2. Ozdemir et al. S1 (4 channels, 10 classes) Micro-LDA Q7 parameters (200 B weights).
  3. Optional NinaPro DB2 S4 (12 channels, 50 classes) Micro-LDA Q7 parameters (3,000 B weights).
  4. Built-in 800-sample continuous self-test stream (5 overlapping 400-sample windows)
     with host-verified Micro-LDA Q7 and CNN-1D INT8 oracle predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ozdemir_quantization_and_channels import (  # noqa: E402
    GESTURE_TRIAL_TOTAL_SEC,
    REP_COEFFS_SEC,
    V_FS_MV as OZDEMIR_V_FS_MV,
    load_ozdemir_recording,
    quantize_signal as quantize_ozdemir,
    segment_ozdemir_windows,
)
from scripts.benchmark_quantization_and_channels import (  # noqa: E402
    V_FS_VOLTS as NINAPRO_V_FS_VOLTS,
    quantize_signal as quantize_ninapro,
)
from scripts.run_esp32_qemu_pipeline import (  # noqa: E402
    compute_adc_scale,
    quantize_lda_weights_q7,
)
from scripts.run_qemu_freertos_dualcore_benchmark import (  # noqa: E402
    extract_td_int16_host_oracle,
    load_cnn_artifacts,
    predict_micro_lda_q7_host_oracle,
)
from semg_baseline.feature004_multi_exercise import (  # noqa: E402
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.features import TimeDomainFeatureExtractor  # noqa: E402
from semg_baseline.filter import CausalBandpassFilter  # noqa: E402
from semg_baseline.lda import CanonicalLdaTrainer, PriorType  # noqa: E402
from semg_baseline.scaling import FeatureScaler  # noqa: E402
from semg_baseline.thresholds import ThresholdCalibrator  # noqa: E402
from semg_dataset.contract import RecordingData  # noqa: E402
from semg_dataset.loader import NinaProDB2Loader  # noqa: E402
from src.export.export_cnn1d_c_header import format_float_array, format_int_array  # noqa: E402
from src.models.train_cnn1d_quantized import (  # noqa: E402
    c_round_half_away,
    quantize_windows_for_c99,
    simulate_c99_int8_cnn1d,
)

DEFAULT_OUT_HEADER = (
    ROOT
    / "firmware"
    / "esp32s3_wroom"
    / "components"
    / "semg_core"
    / "include"
    / "micro_lda_weights.h"
)
CNN_NPZ_PATH = ROOT / "reports" / "cnn1d_qemu" / "cnn1d_model_artifacts.npz"


def train_ozdemir_micro_lda_bundle(subject_id: str = "S1") -> dict[str, Any]:
    """Train Ozdemir 4CH Micro-LDA Q7 model on LOTO split (reps [1,3,4] -> [2,5])."""
    raw_dir = ROOT / "data" / "raw" / "ozdemir"
    raw_sig = load_ozdemir_recording(subject_id=subject_id, raw_dir=raw_dir, use_filtered=True)
    q_sig = quantize_ozdemir(raw_sig, precision="int16", v_max=OZDEMIR_V_FS_MV)

    bp_filter = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=4)
    sos_coeffs = bp_filter.get_sos_coefficients().astype(np.float32)

    windows, labels, reps = segment_ozdemir_windows(q_sig, fs=2000)
    tr_mask = np.isin(reps, [1, 3, 4])
    train_win, train_lbl = windows[tr_mask], labels[tr_mask]

    calibrator = ThresholdCalibrator(default_r=0.5, num_channels=4)
    calib = calibrator.calibrate(
        train_emg=train_win, train_labels=train_lbl, rest_label=0, r_multiplier=0.5
    )
    eps_arr = calib.epsilon_channels[:4].astype(np.float32)

    extractor = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=4)
    X_tr = extractor.extract(train_win).astype(np.float32)

    scaler = FeatureScaler()
    X_tr_norm = scaler.fit_transform(X_tr)
    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=10)
    trainer.fit(X_tr_norm, train_lbl)
    artifacts = trainer.get_model_artifacts()
    scaler_dict = scaler.to_dict()

    scaler_mean = np.asarray(scaler_dict["mean"], dtype=np.float32)
    scaler_scale = np.asarray(scaler_dict["scale"], dtype=np.float32)
    inv_scale = np.where(scaler_scale != 0.0, 1.0 / scaler_scale, 0.0).astype(np.float32)

    zero_mask = np.zeros(20, dtype=np.uint8)
    for z_idx in scaler_dict["zero_variance_indices"]:
        if 0 <= z_idx < 20:
            zero_mask[z_idx] = 1

    adc_scale = compute_adc_scale("int16", OZDEMIR_V_FS_MV)
    eps_int = np.array([int(c_round_half_away(e * adc_scale)) for e in eps_arr], dtype=np.int32)
    coef_q7, w_scale = quantize_lda_weights_q7(artifacts.coef_.astype(np.float32))
    classes_arr = np.asarray(artifacts.classes_, dtype=np.int32)
    intercept_arr = artifacts.intercept_.astype(np.float32)

    # Extract an 800-sample continuous stream (5 overlapping 400-sample windows, stride 100)
    # from test repetition 2, gesture 2 (middle active region) for built-in Flash self-test.
    selftest_gesture = 2
    rep2_sec = REP_COEFFS_SEC[1]
    start_samp = int((rep2_sec + selftest_gesture * GESTURE_TRIAL_TOTAL_SEC) * 2000) + 1000
    stream_fp32 = q_sig[start_samp : start_samp + 800, :4].astype(np.float32)
    stream_q16 = np.clip(c_round_half_away(stream_fp32 * adc_scale), -32768, 32767).astype(np.int16)

    # Compute exact host oracles for all 5 sliding windows inside the 800-sample stream
    cnn_art = load_cnn_artifacts("oz", "ozdemir_s1_4ch", subject_id) if CNN_NPZ_PATH.exists() else None
    lda_oracle = np.zeros(5, dtype=np.int32)
    cnn_oracle = np.zeros(5, dtype=np.int32)
    y_true_arr = np.full(5, selftest_gesture, dtype=np.int32)

    win_fp32_batch = np.zeros((5, 400, 4), dtype=np.float32)
    aux_fp32_batch = np.zeros((5, 20), dtype=np.float32)
    for w_idx in range(5):
        s0 = w_idx * 100
        w_q16 = stream_q16[s0 : s0 + 400]
        w_f32 = (w_q16.astype(np.float32) * np.float32(1.0 / adc_scale)).astype(np.float32)
        feat_int = extract_td_int16_host_oracle(w_q16, eps_int, 1.0 / adc_scale)
        lda_oracle[w_idx] = predict_micro_lda_q7_host_oracle(
            feat_int,
            classes_arr,
            scaler_mean,
            inv_scale,
            zero_mask,
            coef_q7,
            w_scale,
            intercept_arr,
        )
        win_fp32_batch[w_idx] = w_f32
        aux_fp32_batch[w_idx] = feat_int

    if cnn_art is not None:
        # Note: cnn1d_quantize_input in C99 uses lrintf(), match rounding on host for oracle check
        x_q_selftest, aux_q_selftest = quantize_windows_for_c99(
            win_fp32_batch,
            aux_fp32_batch,
            cnn_art.in_mean,
            cnn_art.in_inv_std,
            cnn_art.in_quant_scale,
            cnn_art.aux_mean,
            cnn_art.aux_inv_std,
            cnn_art.aux_quant_scale,
        )
        cnn_preds, _ = simulate_c99_int8_cnn1d(cnn_art, x_q_selftest, aux_q_selftest)
        cnn_oracle = cnn_preds.astype(np.int32)

    return {
        "sos_coeffs": sos_coeffs,
        "adc_scale": float(adc_scale),
        "v_max": float(OZDEMIR_V_FS_MV),
        "eps_arr": eps_arr,
        "eps_int": eps_int,
        "classes": classes_arr,
        "mean": scaler_mean,
        "inv_scale": inv_scale,
        "zero_mask": zero_mask,
        "coef_q7": coef_q7,
        "weight_scale": w_scale,
        "intercept": intercept_arr,
        "selftest_stream_q16": stream_q16,
        "selftest_y_true": y_true_arr,
        "selftest_lda_oracle": lda_oracle,
        "selftest_cnn_oracle": cnn_oracle,
    }


def train_ninapro_micro_lda_bundle(subject_id: str = "S4") -> dict[str, Any]:
    """Train NinaPro DB2 12CH Micro-LDA Q7 model on LOTO split."""
    raw_dir = ROOT / "data" / "raw" / "ninapro_db2"
    loader = NinaProDB2Loader(raw_dir=raw_dir)
    raw_cache: dict[str, RecordingData] = {
        eid: loader.load_recording(subject_id, eid) for eid in ("E1", "E2", "E3")
    }

    def quantized_load_recording(sid: str, eid: str) -> RecordingData:
        rec = raw_cache[eid]
        q_emg = quantize_ninapro(rec.emg, precision="int16", v_max=NINAPRO_V_FS_VOLTS)
        return replace(rec, emg=q_emg)

    runner = MultiExerciseWithinSubjectRunner()
    subject = MultiExerciseSubject(subject_id=subject_id, load_recording=quantized_load_recording)
    train_win, train_lbl, _, _ = runner.prepare_repetition_windows(
        subject, repetitions=runner.TRAINING_REPETITIONS
    )

    calibrator = ThresholdCalibrator(default_r=0.5, num_channels=12)
    calib = calibrator.calibrate(
        train_emg=train_win, train_labels=train_lbl, rest_label=0, r_multiplier=0.5
    )
    eps_arr = calib.epsilon_channels[:12].astype(np.float32)

    extractor = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=12)
    X_tr = extractor.extract(train_win).astype(np.float32)

    scaler = FeatureScaler()
    X_tr_norm = scaler.fit_transform(X_tr)
    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=50)
    trainer.fit(X_tr_norm, train_lbl)
    artifacts = trainer.get_model_artifacts()
    scaler_dict = scaler.to_dict()

    scaler_mean = np.asarray(scaler_dict["mean"], dtype=np.float32)
    scaler_scale = np.asarray(scaler_dict["scale"], dtype=np.float32)
    inv_scale = np.where(scaler_scale != 0.0, 1.0 / scaler_scale, 0.0).astype(np.float32)

    zero_mask = np.zeros(60, dtype=np.uint8)
    for z_idx in scaler_dict["zero_variance_indices"]:
        if 0 <= z_idx < 60:
            zero_mask[z_idx] = 1

    adc_scale = compute_adc_scale("int16", NINAPRO_V_FS_VOLTS)
    eps_int = np.array([int(c_round_half_away(e * adc_scale)) for e in eps_arr], dtype=np.int32)
    coef_q7, w_scale = quantize_lda_weights_q7(artifacts.coef_.astype(np.float32))
    classes_arr = np.asarray(artifacts.classes_, dtype=np.int32)
    intercept_arr = artifacts.intercept_.astype(np.float32)

    return {
        "adc_scale": float(adc_scale),
        "v_max": float(NINAPRO_V_FS_VOLTS),
        "eps_arr": eps_arr,
        "eps_int": eps_int,
        "classes": classes_arr,
        "mean": scaler_mean,
        "inv_scale": inv_scale,
        "zero_mask": zero_mask,
        "coef_q7": coef_q7,
        "weight_scale": w_scale,
        "intercept": intercept_arr,
    }


def render_micro_lda_header(
    oz: dict[str, Any],
    ni: dict[str, Any] | None = None,
) -> str:
    """Render static C99 header containing Micro-LDA Q7 weights, SOS filter, and self-test stream."""
    chunks: list[str] = [
        "/*",
        " * micro_lda_weights.h — Auto-generated Static Flash (.rodata) Weights for ESP32-S3-WROOM",
        " *",
        " * Generated by: src/export/export_micro_lda_c_header.py",
        " * Contains:",
        " *   - Butterworth 4th-order (20-450 Hz @ 2 kHz) Causal SOS Filter Coefficients",
        " *   - Ozdemir S1 (4CH, 10 classes) Micro-LDA Q7 parameters (200 B weights)",
        " *   - NinaPro DB2 S4 (12CH, 50 classes) Micro-LDA Q7 parameters (3,000 B weights)" if ni else " *   - (NinaPro DB2 skipped)",
        " *   - 800-sample built-in self-test stream + host oracle predictions",
        " */",
        "",
        "#ifndef MICRO_LDA_WEIGHTS_H",
        "#define MICRO_LDA_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "#include <string.h>",
        '#include "semg_pipeline.h"',
        "",
        "/* Butterworth 4th-Order Bandpass (20-450 Hz @ Fs=2000 Hz, 4 SOS biquads x 6 coeffs) */",
        format_float_array("g_semg_sos_20_450hz", oz["sos_coeffs"], cols=6),
        "",
        "/* ==========================================================================",
        " * 1. OZDEMIR ET AL. S1 (4 CHANNELS, 10 CLASSES, Q7 WEIGHTS = 200 BYTES)",
        " * ========================================================================== */",
        f"#define OZDEMIR_LDA_NUM_CHANNELS  (4)",
        f"#define OZDEMIR_LDA_NUM_FEATURES  (20)",
        f"#define OZDEMIR_LDA_NUM_CLASSES   (10)",
        f"#define OZDEMIR_LDA_V_MAX         ({oz['v_max']:.9e}f)",
        f"#define OZDEMIR_LDA_ADC_SCALE     ({oz['adc_scale']:.9e}f)",
        "",
        format_float_array("g_ozdemir_lda_epsilon", oz["eps_arr"], cols=4),
        format_int_array("g_ozdemir_lda_epsilon_int", "int32_t", oz["eps_int"], cols=4),
        format_int_array("g_ozdemir_lda_classes", "int32_t", oz["classes"], cols=10),
        format_float_array("g_ozdemir_lda_mean", oz["mean"], cols=5),
        format_float_array("g_ozdemir_lda_inv_scale", oz["inv_scale"], cols=5),
        format_int_array("g_ozdemir_lda_zero_var_mask", "uint8_t", oz["zero_mask"], cols=20),
        format_int_array("g_ozdemir_lda_coef_q7", "int8_t", oz["coef_q7"], cols=20),
        format_float_array("g_ozdemir_lda_weight_scale", oz["weight_scale"], cols=5),
        format_float_array("g_ozdemir_lda_intercept", oz["intercept"], cols=5),
        "",
        "/* Built-in Self-Test Stream: 800 samples x 4 channels (5 overlapping 400-sample windows) */",
        "#define OZDEMIR_SELFTEST_SAMPLES  (800)",
        "#define OZDEMIR_SELFTEST_WINDOWS  (5)",
        format_int_array("g_ozdemir_selftest_stream_q16", "int16_t", oz["selftest_stream_q16"], cols=16),
        format_int_array("g_ozdemir_selftest_y_true", "int32_t", oz["selftest_y_true"], cols=5),
        format_int_array("g_ozdemir_selftest_lda_oracle", "int32_t", oz["selftest_lda_oracle"], cols=5),
        format_int_array("g_ozdemir_selftest_cnn_oracle", "int32_t", oz["selftest_cnn_oracle"], cols=5),
        "",
        "static inline void semg_load_ozdemir_micro_lda(",
        "    micro_lda_model_t *m,",
        "    td_feature_extractor_t *td",
        ") {",
        "    memset(m, 0, sizeof(*m));",
        "    m->num_channels = OZDEMIR_LDA_NUM_CHANNELS;",
        "    m->num_features = OZDEMIR_LDA_NUM_FEATURES;",
        "    m->num_classes  = OZDEMIR_LDA_NUM_CLASSES;",
        "    memcpy(m->classes, g_ozdemir_lda_classes, sizeof(g_ozdemir_lda_classes));",
        "    memcpy(m->mean, g_ozdemir_lda_mean, sizeof(g_ozdemir_lda_mean));",
        "    memcpy(m->inv_scale, g_ozdemir_lda_inv_scale, sizeof(g_ozdemir_lda_inv_scale));",
        "    memcpy(m->zero_var_mask, g_ozdemir_lda_zero_var_mask, sizeof(g_ozdemir_lda_zero_var_mask));",
        "    memcpy(m->coef_q7, g_ozdemir_lda_coef_q7, sizeof(g_ozdemir_lda_coef_q7));",
        "    memcpy(m->weight_scale, g_ozdemir_lda_weight_scale, sizeof(g_ozdemir_lda_weight_scale));",
        "    memcpy(m->intercept, g_ozdemir_lda_intercept, sizeof(g_ozdemir_lda_intercept));",
        "    if (td != NULL) {",
        "        memset(td, 0, sizeof(*td));",
        "        td->num_channels = OZDEMIR_LDA_NUM_CHANNELS;",
        "        td->inv_scale = 1.0f / OZDEMIR_LDA_ADC_SCALE;",
        "        memcpy(td->epsilon, g_ozdemir_lda_epsilon, sizeof(g_ozdemir_lda_epsilon));",
        "        memcpy(td->epsilon_int, g_ozdemir_lda_epsilon_int, sizeof(g_ozdemir_lda_epsilon_int));",
        "    }",
        "}",
        "",
    ]

    if ni is not None:
        chunks.extend([
            "/* ==========================================================================",
            " * 2. NINAPRO DB2 S4 (12 CHANNELS, 50 CLASSES, Q7 WEIGHTS = 3,000 BYTES)",
            " * ========================================================================== */",
            "#define NINAPRO_LDA_NUM_CHANNELS  (12)",
            "#define NINAPRO_LDA_NUM_FEATURES  (60)",
            "#define NINAPRO_LDA_NUM_CLASSES   (50)",
            f"#define NINAPRO_LDA_V_MAX         ({ni['v_max']:.9e}f)",
            f"#define NINAPRO_LDA_ADC_SCALE     ({ni['adc_scale']:.9e}f)",
            "",
            format_float_array("g_ninapro_lda_epsilon", ni["eps_arr"], cols=6),
            format_int_array("g_ninapro_lda_epsilon_int", "int32_t", ni["eps_int"], cols=6),
            format_int_array("g_ninapro_lda_classes", "int32_t", ni["classes"], cols=10),
            format_float_array("g_ninapro_lda_mean", ni["mean"], cols=6),
            format_float_array("g_ninapro_lda_inv_scale", ni["inv_scale"], cols=6),
            format_int_array("g_ninapro_lda_zero_var_mask", "uint8_t", ni["zero_mask"], cols=20),
            format_int_array("g_ninapro_lda_coef_q7", "int8_t", ni["coef_q7"], cols=20),
            format_float_array("g_ninapro_lda_weight_scale", ni["weight_scale"], cols=5),
            format_float_array("g_ninapro_lda_intercept", ni["intercept"], cols=5),
            "",
            "static inline void semg_load_ninapro_micro_lda(",
            "    micro_lda_model_t *m,",
            "    td_feature_extractor_t *td",
            ") {",
            "    memset(m, 0, sizeof(*m));",
            "    m->num_channels = NINAPRO_LDA_NUM_CHANNELS;",
            "    m->num_features = NINAPRO_LDA_NUM_FEATURES;",
            "    m->num_classes  = NINAPRO_LDA_NUM_CLASSES;",
            "    memcpy(m->classes, g_ninapro_lda_classes, sizeof(g_ninapro_lda_classes));",
            "    memcpy(m->mean, g_ninapro_lda_mean, sizeof(g_ninapro_lda_mean));",
            "    memcpy(m->inv_scale, g_ninapro_lda_inv_scale, sizeof(g_ninapro_lda_inv_scale));",
            "    memcpy(m->zero_var_mask, g_ninapro_lda_zero_var_mask, sizeof(g_ninapro_lda_zero_var_mask));",
            "    memcpy(m->coef_q7, g_ninapro_lda_coef_q7, sizeof(g_ninapro_lda_coef_q7));",
            "    memcpy(m->weight_scale, g_ninapro_lda_weight_scale, sizeof(g_ninapro_lda_weight_scale));",
            "    memcpy(m->intercept, g_ninapro_lda_intercept, sizeof(g_ninapro_lda_intercept));",
            "    if (td != NULL) {",
            "        memset(td, 0, sizeof(*td));",
            "        td->num_channels = NINAPRO_LDA_NUM_CHANNELS;",
            "        td->inv_scale = 1.0f / NINAPRO_LDA_ADC_SCALE;",
            "        memcpy(td->epsilon, g_ninapro_lda_epsilon, sizeof(g_ninapro_lda_epsilon));",
            "        memcpy(td->epsilon_int, g_ninapro_lda_epsilon_int, sizeof(g_ninapro_lda_epsilon_int));",
            "    }",
            "}",
            "",
        ])

    chunks.extend(["#endif /* MICRO_LDA_WEIGHTS_H */", ""])
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export static Micro-LDA Q7, SOS coefficients, and self-test oracle to C99 header."
    )
    parser.add_argument("--ozdemir-subject", default="S1")
    parser.add_argument("--ninapro-subject", default="S4")
    parser.add_argument("--skip-ninapro", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_HEADER)
    args = parser.parse_args()

    print(f"Training Ozdemir {args.ozdemir_subject} Micro-LDA Q7...")
    oz = train_ozdemir_micro_lda_bundle(args.ozdemir_subject)
    ni = None
    if not args.skip_ninapro and (ROOT / "data" / "raw" / "ninapro_db2").exists():
        print(f"Training NinaPro DB2 {args.ninapro_subject} Micro-LDA Q7...")
        ni = train_ninapro_micro_lda_bundle(args.ninapro_subject)

    content = render_micro_lda_header(oz, ni)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Exported {args.out} ({len(content):,} bytes)")
    print(f"Self-test y_true:     {oz['selftest_y_true'].tolist()}")
    print(f"Self-test LDA oracle: {oz['selftest_lda_oracle'].tolist()}")
    print(f"Self-test CNN oracle: {oz['selftest_cnn_oracle'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
