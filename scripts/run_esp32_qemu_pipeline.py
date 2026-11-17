#!/usr/bin/env python3
"""Run real ESP32 and ESP32-S3 QEMU hardware emulation on quantized sEMG datasets.

Executes the full documented pipeline inside Espressif's QEMU Xtensa emulator
(`qemu-system-xtensa -machine esp32` and `-machine esp32s3`):
  1. ADC Quantization (float32, float16, int32, int16, int8)
  2. CausalSosFilter (DF2T 4th-order Butterworth 20-450 Hz)
  3. StatefulWindowBuffer (W = 200 ms / 400 samples, S = 50 ms / 100 samples)
  4. Hudgins Time-Domain Feature Extraction (Float32 & Zero-FPU Integer ALU)
  5. Train-only Z-Score Scaler + Canonical Micro-LDA (Float32 & 200-byte Q7 INT8)

Evaluates both datasets:
  - Ozdemir et al. (4 channels, 10 gesture classes, Fs = 2000 Hz, V_FS = 10.0 mV)
  - NinaPro DB2 (12..1 channels, 50 gesture classes, Fs = 2000 Hz, V_FS = 5.5 mV)

Produces JSON and Markdown reports with real emulated hardware cycle counts (CCOUNT),
SRAM/Flash footprints, latencies at 240 MHz, and classification accuracy parity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ozdemir_quantization_and_channels import (  # noqa: E402
    BYTES_PER_SAMPLE_1CH,
    V_FS_MV as OZDEMIR_V_FS_MV,
    compute_evaluation_metrics,
    get_channel_feature_indices as ozdemir_feat_indices,
    load_ozdemir_recording,
    quantize_signal as quantize_ozdemir,
    segment_ozdemir_windows,
)
from scripts.benchmark_quantization_and_channels import (  # noqa: E402
    V_FS_VOLTS as NINAPRO_V_FS_VOLTS,
    get_channel_feature_indices as ninapro_feat_indices,
    quantize_signal as quantize_ninapro,
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
from semg_dsp.filter import CausalSosFilter  # noqa: E402

PRECISION_CODE_MAP = {
    "float32": 0,
    "float16": 1,
    "int32": 2,
    "int16": 3,
    "int8": 4,
}


@dataclass(frozen=True)
class QemuEmulationScenarioResult:
    """Complete telemetry and accuracy record from a real QEMU ESP32/ESP32-S3 run."""

    dataset: str
    subject_id: str
    machine: str
    precision: str
    num_channels: int
    num_features: int
    num_classes: int
    bytes_per_sample: int
    data_rate_kbps: float
    data_reduction_pct: float
    num_test_windows: int
    host_top1_acc: float
    host_bal_acc: float
    host_macro_f1: float
    esp32_fp32_top1_acc: float
    esp32_fp32_bal_acc: float
    esp32_fp32_macro_f1: float
    esp32_q7_top1_acc: float
    esp32_q7_bal_acc: float
    esp32_q7_macro_f1: float
    esp32_int_feat_top1_acc: float
    esp32_int_feat_bal_acc: float
    esp32_int_feat_macro_f1: float
    host_parity_ratio: float
    max_filter_diff: float
    max_feat_diff: float
    emitted_stream_windows: int
    cycles_filter_per_stride: int
    cycles_winbuf_per_stride: int
    cycles_feat_fp32: int
    cycles_feat_int16: int
    cycles_lda_fp32: int
    cycles_lda_q7: int
    latency_filter_us_240mhz: float
    latency_feat_fp32_us_240mhz: float
    latency_feat_int16_us_240mhz: float
    latency_lda_fp32_us_240mhz: float
    latency_lda_q7_us_240mhz: float
    latency_total_fp32_ms_240mhz: float
    latency_total_tinyml_ms_240mhz: float
    cpu_duty_cycle_pct_50ms: float
    model_weights_bytes_fp32: int
    model_weights_bytes_q7: int
    window_buffer_sram_bytes: int


def compute_adc_scale(precision: str, v_max: float) -> float:
    """Return integer quantization scale factor for ADC representation."""
    if precision == "int8":
        return float((2**7 - 1) / v_max)
    return float((2**15 - 1) / v_max)


def quantize_lda_weights_q7(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize LDA weight matrix (n_classes, n_features) to symmetric int8 Q7 per class row."""
    n_classes, n_features = coef.shape
    coef_q7 = np.zeros((n_classes, n_features), dtype=np.int8)
    weight_scale = np.zeros(n_classes, dtype=np.float32)

    for k in range(n_classes):
        row = coef[k]
        max_abs = float(np.max(np.abs(row)))
        if max_abs < 1e-12:
            weight_scale[k] = 1.0
            coef_q7[k, :] = 0
        else:
            scale = max_abs / 127.0
            weight_scale[k] = np.float32(scale)
            q_row = np.clip(np.round(row / scale), -127, 127).astype(np.int8)
            coef_q7[k, :] = q_row

    return coef_q7, weight_scale


def write_esp32_eval_bundle(
    bundle_path: Path,
    precision: str,
    num_channels: int,
    num_classes: int,
    v_max: float,
    sos_coeffs: np.ndarray,
    epsilon_ch: np.ndarray,
    classes: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    zero_var_indices: Sequence[int],
    lda_coef: np.ndarray,
    lda_intercept: np.ndarray,
    stream_in: np.ndarray,
    stream_expected: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    host_preds: np.ndarray,
    host_features: np.ndarray,
) -> None:
    """Serialize model parameters, streaming snippet, and test windows into binary bundle."""
    num_features = 5 * num_channels
    num_test_windows = int(test_windows.shape[0])
    num_stream_samples = int(stream_in.shape[0])
    adc_scale = compute_adc_scale(precision, v_max)

    inv_scale = np.where(scaler_scale != 0.0, 1.0 / scaler_scale, 0.0).astype(np.float32)
    zero_mask = np.zeros(num_features, dtype=np.uint8)
    for idx in zero_var_indices:
        if 0 <= idx < num_features:
            zero_mask[idx] = 1

    coef_q7, weight_scale = quantize_lda_weights_q7(lda_coef)

    header_bytes = struct.pack(
        "<IIIIIIIIIff",
        0x53454D47,
        PRECISION_CODE_MAP[precision],
        num_channels,
        num_features,
        num_classes,
        400,
        100,
        num_test_windows,
        num_stream_samples,
        float(v_max),
        float(adc_scale),
    )

    with bundle_path.open("wb") as f:
        f.write(header_bytes)
        f.write(np.asarray(sos_coeffs, dtype="<f4").tobytes())
        f.write(np.asarray(epsilon_ch[:num_channels], dtype="<f4").tobytes())
        f.write(np.asarray(classes, dtype="<i4").tobytes())
        f.write(np.asarray(scaler_mean, dtype="<f4").tobytes())
        f.write(np.asarray(inv_scale, dtype="<f4").tobytes())
        f.write(zero_mask.tobytes())
        f.write(np.asarray(lda_coef, dtype="<f4").tobytes())
        f.write(np.asarray(lda_intercept, dtype="<f4").tobytes())
        f.write(coef_q7.tobytes())
        f.write(np.asarray(weight_scale, dtype="<f4").tobytes())

        if num_stream_samples > 0:
            f.write(np.ascontiguousarray(stream_in[:, :num_channels], dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(stream_expected[:, :num_channels], dtype="<f4").tobytes())

        for w in range(num_test_windows):
            f.write(struct.pack("<ii", int(test_labels[w]), int(host_preds[w])))
            f.write(np.ascontiguousarray(host_features[w], dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(test_windows[w, :, :num_channels], dtype="<f4").tobytes())


def invoke_qemu_esp32(
    bundle_path: Path,
    machine: str,
    elf_path: Path,
) -> dict[str, Any]:
    """Run qemu-system-xtensa on the compiled ELF with semihosting in the bundle directory."""
    cmd = [
        "qemu-system-xtensa",
        "-nographic",
        "-machine",
        machine,
        "-semihosting",
        "-kernel",
        str(elf_path.resolve()),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(bundle_path.parent),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"QEMU exited with code {proc.returncode}.\nStdout:\n{proc.stdout}\nStderr:\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if "QEMU_ESP32_RESULT_JSON:" in line:
            json_str = line.split("QEMU_ESP32_RESULT_JSON:", 1)[1].strip()
            return json.loads(json_str)
    raise RuntimeError(f"QEMU did not emit result JSON. Stdout:\n{proc.stdout}\nStderr:\n{proc.stderr}")


def build_scenario_result(
    dataset: str,
    subject_id: str,
    machine: str,
    precision: str,
    num_channels: int,
    baseline_channels: int,
    host_metrics: tuple[float, float, float],
    qemu_json: dict[str, Any],
) -> QemuEmulationScenarioResult:
    """Combine host oracle metrics and real QEMU ESP32 hardware telemetry."""
    bytes_per_sample = num_channels * BYTES_PER_SAMPLE_1CH[precision]
    baseline_bytes = baseline_channels * 4
    data_reduction_pct = (1.0 - (bytes_per_sample / baseline_bytes)) * 100.0
    data_rate_kbps = (bytes_per_sample * 2000.0) / 1000.0

    cyc_filt = int(qemu_json["cycles_filter_per_stride"])
    cyc_win = int(qemu_json["cycles_winbuf_per_stride"])
    cyc_feat_fp32 = int(qemu_json["cycles_feat_fp32"])
    cyc_feat_int16 = int(qemu_json["cycles_feat_int16"])
    cyc_lda_fp32 = int(qemu_json["cycles_lda_fp32"])
    cyc_lda_q7 = int(qemu_json["cycles_lda_q7"])

    # 240 MHz clock -> 240 cycles per microsecond
    lat_filt_us = cyc_filt / 240.0
    lat_win_us = cyc_win / 240.0
    lat_feat_fp32_us = cyc_feat_fp32 / 240.0
    lat_feat_int16_us = cyc_feat_int16 / 240.0
    lat_lda_fp32_us = cyc_lda_fp32 / 240.0
    lat_lda_q7_us = cyc_lda_q7 / 240.0

    total_fp32_ms = (lat_filt_us + lat_win_us + lat_feat_fp32_us + lat_lda_fp32_us) / 1000.0
    total_tinyml_ms = (lat_filt_us + lat_win_us + lat_feat_int16_us + lat_lda_q7_us) / 1000.0
    duty_cycle_pct = (total_tinyml_ms / 50.0) * 100.0

    n_feat = int(qemu_json["num_features"])
    n_cls = int(qemu_json["num_classes"])

    return QemuEmulationScenarioResult(
        dataset=dataset,
        subject_id=subject_id,
        machine=machine,
        precision=precision,
        num_channels=num_channels,
        num_features=n_feat,
        num_classes=n_cls,
        bytes_per_sample=bytes_per_sample,
        data_rate_kbps=round(data_rate_kbps, 2),
        data_reduction_pct=round(data_reduction_pct, 2),
        num_test_windows=int(qemu_json["num_test_windows"]),
        host_top1_acc=round(host_metrics[0], 6),
        host_bal_acc=round(host_metrics[1], 6),
        host_macro_f1=round(host_metrics[2], 6),
        esp32_fp32_top1_acc=round(float(qemu_json["top1_accuracy"]), 6),
        esp32_fp32_bal_acc=round(float(qemu_json["balanced_accuracy"]), 6),
        esp32_fp32_macro_f1=round(float(qemu_json["macro_f1"]), 6),
        esp32_q7_top1_acc=round(float(qemu_json["q7_top1_accuracy"]), 6),
        esp32_q7_bal_acc=round(float(qemu_json["q7_balanced_accuracy"]), 6),
        esp32_q7_macro_f1=round(float(qemu_json["q7_macro_f1"]), 6),
        esp32_int_feat_top1_acc=round(float(qemu_json["int_feat_top1_accuracy"]), 6),
        esp32_int_feat_bal_acc=round(float(qemu_json["int_feat_balanced_accuracy"]), 6),
        esp32_int_feat_macro_f1=round(float(qemu_json["int_feat_macro_f1"]), 6),
        host_parity_ratio=round(float(qemu_json["host_parity_ratio"]), 6),
        max_filter_diff=float(qemu_json["max_filter_diff"]),
        max_feat_diff=float(qemu_json["max_feat_diff"]),
        emitted_stream_windows=int(qemu_json["emitted_stream_windows"]),
        cycles_filter_per_stride=cyc_filt,
        cycles_winbuf_per_stride=cyc_win,
        cycles_feat_fp32=cyc_feat_fp32,
        cycles_feat_int16=cyc_feat_int16,
        cycles_lda_fp32=cyc_lda_fp32,
        cycles_lda_q7=cyc_lda_q7,
        latency_filter_us_240mhz=round(lat_filt_us, 2),
        latency_feat_fp32_us_240mhz=round(lat_feat_fp32_us, 2),
        latency_feat_int16_us_240mhz=round(lat_feat_int16_us, 2),
        latency_lda_fp32_us_240mhz=round(lat_lda_fp32_us, 2),
        latency_lda_q7_us_240mhz=round(lat_lda_q7_us, 2),
        latency_total_fp32_ms_240mhz=round(total_fp32_ms, 4),
        latency_total_tinyml_ms_240mhz=round(total_tinyml_ms, 4),
        cpu_duty_cycle_pct_50ms=round(duty_cycle_pct, 4),
        model_weights_bytes_fp32=n_cls * n_feat * 4,
        model_weights_bytes_q7=n_cls * n_feat * 1,
        window_buffer_sram_bytes=400 * num_channels * BYTES_PER_SAMPLE_1CH[precision] * 2,
    )


def run_ozdemir_on_qemu(
    subject_id: str,
    raw_dir: Path,
    precisions: Sequence[str],
    channel_counts: Sequence[int],
    machines: Sequence[tuple[str, Path]],
) -> list[QemuEmulationScenarioResult]:
    """Run Ozdemir quantized benchmark on QEMU ESP32/ESP32-S3."""
    results: list[QemuEmulationScenarioResult] = []
    raw_sig = load_ozdemir_recording(subject_id=subject_id, raw_dir=raw_dir, use_filtered=True)

    bp_filter = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=4)
    sos_coeffs = bp_filter.get_sos_coefficients()

    for prec in precisions:
        q_sig = quantize_ozdemir(raw_sig, precision=prec, v_max=OZDEMIR_V_FS_MV)

        # Prepare 1000-sample streaming snippet for CausalSosFilter + StatefulWindowBuffer check
        stream_in = q_sig[8000:9000].astype(np.float32)
        ref_sos = CausalSosFilter(sos_coefficients=sos_coeffs, num_channels=4)
        stream_expected = ref_sos.process_chunk(stream_in)

        windows, labels, reps = segment_ozdemir_windows(q_sig, fs=2000)
        train_mask = np.isin(reps, [1, 3, 4])
        test_mask = np.isin(reps, [2, 5])
        train_win, train_lbl = windows[train_mask], labels[train_mask]
        test_win, test_lbl = windows[test_mask], labels[test_mask]

        calibrator = ThresholdCalibrator(default_r=0.5, num_channels=4)
        calib = calibrator.calibrate(
            train_emg=train_win, train_labels=train_lbl, rest_label=0, r_multiplier=0.5
        )
        extractor = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=4)
        X_train_full = extractor.extract(train_win)
        X_test_full = extractor.extract(test_win)

        for k in channel_counts:
            feat_indices = ozdemir_feat_indices(num_channels=k, total_channels=4)
            X_tr_k = X_train_full[:, feat_indices]
            X_te_k = X_test_full[:, feat_indices]

            scaler = FeatureScaler()
            X_tr_s = scaler.fit_transform(X_tr_k)
            X_te_s = scaler.transform(X_te_k)

            trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=10)
            trainer.fit(X_tr_s, train_lbl)
            host_preds = trainer.predict(X_te_s)
            host_metrics = compute_evaluation_metrics(test_lbl, host_preds)

            artifacts = trainer.get_model_artifacts()
            scaler_dict = scaler.to_dict()

            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_path = Path(tmpdir) / "esp32_eval_bundle.bin"
                write_esp32_eval_bundle(
                    bundle_path=bundle_path,
                    precision=prec,
                    num_channels=k,
                    num_classes=10,
                    v_max=OZDEMIR_V_FS_MV,
                    sos_coeffs=sos_coeffs,
                    epsilon_ch=calib.epsilon_channels[:k],
                    classes=artifacts.classes_,
                    scaler_mean=np.asarray(scaler_dict["mean"], dtype=np.float32),
                    scaler_scale=np.asarray(scaler_dict["scale"], dtype=np.float32),
                    zero_var_indices=scaler_dict["zero_variance_indices"],
                    lda_coef=artifacts.coef_.astype(np.float32),
                    lda_intercept=artifacts.intercept_.astype(np.float32),
                    stream_in=stream_in[:, :k],
                    stream_expected=stream_expected[:, :k],
                    test_windows=test_win[:, :, :k],
                    test_labels=test_lbl,
                    host_preds=host_preds,
                    host_features=X_te_k.astype(np.float32),
                )

                for machine_name, elf_path in machines:
                    qemu_json = invoke_qemu_esp32(bundle_path, machine_name, elf_path)
                    res = build_scenario_result(
                        dataset="Ozdemir",
                        subject_id=subject_id,
                        machine=machine_name,
                        precision=prec,
                        num_channels=k,
                        baseline_channels=4,
                        host_metrics=host_metrics,
                        qemu_json=qemu_json,
                    )
                    results.append(res)
                    print(
                        f"[QEMU {machine_name:7s}][Ozdemir {subject_id}] {prec:7s} | {k:2d} ch | "
                        f"Host Acc: {res.host_top1_acc*100:5.2f}% | "
                        f"ESP32 FP32: {res.esp32_fp32_top1_acc*100:5.2f}% (Parity: {res.host_parity_ratio*100:5.1f}%) | "
                        f"ESP32 Q7: {res.esp32_q7_top1_acc*100:5.2f}% | "
                        f"Feat Cyc (FP32/INT16): {res.cycles_feat_fp32}/{res.cycles_feat_int16} | "
                        f"LDA Cyc (FP32/Q7): {res.cycles_lda_fp32}/{res.cycles_lda_q7}",
                        flush=True,
                    )

    return results


def run_ninapro_on_qemu(
    subject_id: str,
    raw_dir: Path,
    precisions: Sequence[str],
    channel_counts: Sequence[int],
    machines: Sequence[tuple[str, Path]],
) -> list[QemuEmulationScenarioResult]:
    """Run NinaPro DB2 quantized benchmark on QEMU ESP32/ESP32-S3."""
    results: list[QemuEmulationScenarioResult] = []
    loader = NinaProDB2Loader(raw_dir=raw_dir)
    raw_cache: dict[str, RecordingData] = {
        eid: loader.load_recording(subject_id, eid) for eid in ("E1", "E2", "E3")
    }

    bp_filter = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=12)
    sos_coeffs = bp_filter.get_sos_coefficients()

    for prec in precisions:
        def quantized_load_recording(sid: str, eid: str) -> RecordingData:
            rec = raw_cache[eid]
            q_emg = quantize_ninapro(rec.emg, precision=prec, v_max=NINAPRO_V_FS_VOLTS)
            return replace(rec, emg=q_emg)

        runner = MultiExerciseWithinSubjectRunner()
        subject = MultiExerciseSubject(subject_id=subject_id, load_recording=quantized_load_recording)

        train_win, train_lbl, _, _ = runner.prepare_repetition_windows(
            subject, repetitions=runner.TRAINING_REPETITIONS
        )
        test_win, test_lbl, _, _ = runner.prepare_repetition_windows(
            subject, repetitions=runner.TEST_REPETITIONS
        )

        stream_in = quantize_ninapro(raw_cache["E1"].emg[10000:11000], precision=prec, v_max=NINAPRO_V_FS_VOLTS)
        ref_sos = CausalSosFilter(sos_coefficients=sos_coeffs, num_channels=12)
        stream_expected = ref_sos.process_chunk(stream_in)

        calibrator = ThresholdCalibrator(default_r=0.5, num_channels=12)
        calib = calibrator.calibrate(
            train_emg=train_win, train_labels=train_lbl, rest_label=0, r_multiplier=0.5
        )
        extractor = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=12)
        X_train_full = extractor.extract(train_win)
        X_test_full = extractor.extract(test_win)

        del train_win

        for k in channel_counts:
            feat_indices = ninapro_feat_indices(num_channels=k, total_channels=12)
            X_tr_k = X_train_full[:, feat_indices]
            X_te_k = X_test_full[:, feat_indices]

            scaler = FeatureScaler()
            X_tr_s = scaler.fit_transform(X_tr_k)
            X_te_s = scaler.transform(X_te_k)

            trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=50)
            trainer.fit(X_tr_s, train_lbl)
            host_preds = trainer.predict(X_te_s)
            host_metrics = compute_evaluation_metrics(test_lbl, host_preds)

            artifacts = trainer.get_model_artifacts()
            scaler_dict = scaler.to_dict()

            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_path = Path(tmpdir) / "esp32_eval_bundle.bin"
                write_esp32_eval_bundle(
                    bundle_path=bundle_path,
                    precision=prec,
                    num_channels=k,
                    num_classes=50,
                    v_max=NINAPRO_V_FS_VOLTS,
                    sos_coeffs=sos_coeffs,
                    epsilon_ch=calib.epsilon_channels[:k],
                    classes=artifacts.classes_,
                    scaler_mean=np.asarray(scaler_dict["mean"], dtype=np.float32),
                    scaler_scale=np.asarray(scaler_dict["scale"], dtype=np.float32),
                    zero_var_indices=scaler_dict["zero_variance_indices"],
                    lda_coef=artifacts.coef_.astype(np.float32),
                    lda_intercept=artifacts.intercept_.astype(np.float32),
                    stream_in=stream_in[:, :k],
                    stream_expected=stream_expected[:, :k],
                    test_windows=test_win[:, :, :k],
                    test_labels=test_lbl,
                    host_preds=host_preds,
                    host_features=X_te_k.astype(np.float32),
                )

                for machine_name, elf_path in machines:
                    qemu_json = invoke_qemu_esp32(bundle_path, machine_name, elf_path)
                    res = build_scenario_result(
                        dataset="NinaPro_DB2",
                        subject_id=subject_id,
                        machine=machine_name,
                        precision=prec,
                        num_channels=k,
                        baseline_channels=12,
                        host_metrics=host_metrics,
                        qemu_json=qemu_json,
                    )
                    results.append(res)
                    print(
                        f"[QEMU {machine_name:7s}][NinaPro {subject_id}] {prec:7s} | {k:2d} ch | "
                        f"Host Acc: {res.host_top1_acc*100:5.2f}% | "
                        f"ESP32 FP32: {res.esp32_fp32_top1_acc*100:5.2f}% (Parity: {res.host_parity_ratio*100:5.1f}%) | "
                        f"ESP32 Q7: {res.esp32_q7_top1_acc*100:5.2f}% | "
                        f"Feat Cyc (FP32/INT16): {res.cycles_feat_fp32}/{res.cycles_feat_int16} | "
                        f"LDA Cyc (FP32/Q7): {res.cycles_lda_fp32}/{res.cycles_lda_q7}",
                        flush=True,
                    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quantized sEMG pipeline on QEMU ESP32/ESP32-S3.")
    parser.add_argument("--ozdemir-subject", default="S1", help="Ozdemir subject ID (default: S1)")
    parser.add_argument("--ninapro-subject", default="S4", help="NinaPro DB2 subject ID (default: S4)")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports" / "esp32_qemu" / "esp32_qemu_emulation_results.json",
    )
    args = parser.parse_args()

    elf_esp32 = ROOT / "firmware" / "esp32_qemu" / "build" / "semg_esp32.elf"
    elf_esp32s3 = ROOT / "firmware" / "esp32_qemu" / "build" / "semg_esp32s3.elf"

    machines_ozdemir = [
        ("esp32", elf_esp32),
        ("esp32s3", elf_esp32s3),
    ]
    machines_ninapro = [
        ("esp32", elf_esp32),
        ("esp32s3", elf_esp32s3),
    ]

    t0 = time.time()
    print("=====================================================================")
    print(" EXECUTING REAL QEMU ESP32 & ESP32-S3 EMULATION ON QUANTIZED DATASETS")
    print("=====================================================================")

    ozdemir_results = run_ozdemir_on_qemu(
        subject_id=args.ozdemir_subject,
        raw_dir=ROOT / "data" / "raw" / "ozdemir",
        precisions=("float32", "float16", "int32", "int16", "int8"),
        channel_counts=(4, 3, 2, 1),
        machines=machines_ozdemir,
    )

    ninapro_results = run_ninapro_on_qemu(
        subject_id=args.ninapro_subject,
        raw_dir=ROOT / "data" / "raw" / "ninapro_db2",
        precisions=("float32", "float16", "int32", "int16", "int8"),
        channel_counts=(12, 10, 8, 6, 4, 2, 1),
        machines=machines_ninapro,
    )

    elapsed = time.time() - t0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "emulator_version": "QEMU emulator version 9.2.2 (esp_develop_9.2.2_20261109)",
        "toolchain_version": "xtensa-esp-elf-gcc (crosstool-NG esp-14.4.0_20261030) 14.4.0",
        "elapsed_seconds": round(elapsed, 2),
        "ozdemir_subject": args.ozdemir_subject,
        "ninapro_subject": args.ninapro_subject,
        "ozdemir_results": [asdict(r) for r in ozdemir_results],
        "ninapro_results": [asdict(r) for r in ninapro_results],
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved complete QEMU emulation results to {args.output_json} in {elapsed:.2f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
