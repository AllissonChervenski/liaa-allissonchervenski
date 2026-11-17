#!/usr/bin/env python3
"""Run Full Benchmark for M14 Feature-Rich SupCon Residual MLP (636 Features).

Evaluates:
  1. `M14a_enhanced_lda`: Canonical LDA directly on the 636/180/86 multi-domain features
  2. `M14_rich_supcon_mlp_int8`: SupCon-pretrained Residual MLP (BatchNorm-folded INT8)
  Combined with Passo 1 (Causal Majority Vote & Soft Voting K in [1, 3, 5, 7, 9]).

Datasets & Channels:
  - Ozdemir et al. (Subject S1, 10 classes): 4CH (180 feats) and 2CH (86 feats)
  - NinaPro DB2 (Subject S4, 50 classes): 12CH (636 feats) and 2CH (86 feats)

Outputs:
  - JSON results: reports/dtw_hybrid_benchmark/rich_features_m14_results.json
  - Markdown report: reports/dtw_hybrid_benchmark/rich_features_m14_master_report.md
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ozdemir_quantization_and_channels import (  # noqa: E402
    V_FS_MV as OZDEMIR_V_FS_MV,
    load_ozdemir_recording,
    quantize_signal as quantize_ozdemir,
    segment_ozdemir_windows,
)
from scripts.benchmark_quantization_and_channels import (  # noqa: E402
    V_FS_VOLTS as NINAPRO_V_FS_VOLTS,
    quantize_signal as quantize_ninapro,
)
from scripts.run_qemu_cnn1d_benchmark import (  # noqa: E402
    ESP32S3_ACTIVE_POWER_MW,
    compute_irc_iee,
)
from scripts.run_supcon_cnn2d_benchmark import _evaluate_smoothing_sweeps  # noqa: E402
from semg_baseline.feature004_multi_exercise import (  # noqa: E402
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.lda import CanonicalLdaTrainer, PriorType  # noqa: E402
from semg_baseline.scaling import FeatureScaler  # noqa: E402
from semg_baseline.thresholds import ThresholdCalibrator  # noqa: E402
from semg_dataset.contract import RecordingData  # noqa: E402
from semg_dataset.loader import NinaProDB2Loader  # noqa: E402
from src.features.enhanced_features import EnhancedSemgFeatureExtractor  # noqa: E402
from src.models.temporal_smoothing import causal_soft_voting  # noqa: E402
from src.models.train_rich_supcon_mlp import (  # noqa: E402
    QuantizedRichMlpArtifacts,
    train_and_quantize_rich_supcon_mlp,
)

OUT_DIR = ROOT / "reports" / "dtw_hybrid_benchmark"
OUT_JSON = OUT_DIR / "rich_features_m14_results.json"
OUT_MD = OUT_DIR / "rich_features_m14_master_report.md"


def compute_m14_hardware_metrics(
    art: QuantizedRichMlpArtifacts,
    acc_pct: float,
) -> dict[str, float]:
    """Compute ESP32-S3 @ 240 MHz DSP + MLP cycles, latency, Flash, SRAM, IRC, IEE."""
    c = art.num_channels
    # DSP feature extraction cycles on ESP32-S3 (Xtensa LX7 FPU + esp-dsp 512-pt FFT):
    # Per channel: TD/HOS (~1500 cyc) + AR-4 Yule-Walker (~900 cyc) + Haar DWT (~700 cyc)
    # + esp-dsp 512-pt FFT & bands (~2400 cyc) = ~5,500 cycles / channel
    # Cross-channel synergy (C*(C-1)/2 correlations): ~40 cycles / pair
    pair_dim = (c * (c - 1)) // 2
    dsp_cycles = (5500.0 * c) + (40.0 * pair_dim)

    # MLP INT8 MACs
    macs_fc1 = art.in_features * art.hidden1
    macs_fc2 = art.hidden1 * art.hidden2
    macs_head = art.hidden2 * art.num_classes
    macs_skip = art.in_features * art.num_classes
    total_macs = macs_fc1 + macs_fc2 + macs_head + macs_skip

    # ~0.18 cycles/MAC with Xtensa PIE/unrolled int8 dot product
    cls_cycles = total_macs * 0.18
    total_cycles = dsp_cycles + cls_cycles

    latency_ms = total_cycles / 240000.0
    energy_mj = latency_ms * (ESP32S3_ACTIVE_POWER_MW / 1000.0)

    # Flash weights footprint (1B int8 weights + 4B int32 biases)
    w_bytes = art.fc1_w_q.size + art.fc2_w_q.size + art.head_w_q.size + art.skip_w_q.size
    b_bytes = 4 * (art.fc1_b_q.size + art.fc2_b_q.size + art.head_b_q.size)
    flash_kb = (w_bytes + b_bytes) / 1024.0

    # Peak Activation SRAM Arena:
    # Input features (in_features) + hidden1 + hidden2 + FFT 512-pt scratch (2048 B) + window slice (c*400)
    sram_bytes = (c * 400) + art.in_features + art.hidden1 + art.hidden2 + 2048
    sram_kb = sram_bytes / 1024.0

    metrics = compute_irc_iee(
        acc_pct=acc_pct,
        num_gestures=art.num_classes,
        latency_ms=latency_ms,
        sram_kb=sram_kb,
    )

    return {
        "total_params": art.total_params,
        "total_macs": total_macs,
        "dsp_cycles": round(dsp_cycles, 1),
        "cls_cycles": round(cls_cycles, 1),
        "cycles_est": round(total_cycles, 1),
        "latency_ms": round(latency_ms, 4),
        "energy_mj": round(energy_mj, 5),
        "flash_weights_kb": round(flash_kb, 2),
        "sram_arena_kb": round(sram_kb, 2),
        "irc": round(metrics["irc"], 2),
        "iee": round(metrics["iee"], 2),
    }


def _evaluate_enhanced_lda(
    X_tr: np.ndarray,
    tr_lbl: np.ndarray,
    X_te: np.ndarray,
    te_lbl: np.ndarray,
    boundaries: list[int],
    num_classes: int,
) -> dict[str, float]:
    """Evaluate Canonical LDA directly on the multi-domain features."""
    scaler = FeatureScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    lda = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_classes)
    lda.fit(X_tr_s, tr_lbl)
    art = lda.get_model_artifacts()
    scores = X_te_s @ art.coef_.T + art.intercept_[None, :]
    p1 = np.argmax(scores, axis=1)
    p5 = causal_soft_voting(scores, boundaries=boundaries, window_size=5)
    p9 = causal_soft_voting(scores, boundaries=boundaries, window_size=9)
    return {
        "raw_top1_pct": round(float(accuracy_score(te_lbl, p1)) * 100.0, 2),
        "raw_macro_f1_pct": round(float(f1_score(te_lbl, p1, average="macro", zero_division=0)) * 100.0, 2),
        "soft_k5_pct": round(float(accuracy_score(te_lbl, p5)) * 100.0, 2),
        "soft_k9_pct": round(float(accuracy_score(te_lbl, p9)) * 100.0, 2),
    }


def run_benchmark() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    results: dict[str, Any] = {
        "benchmark": "M14 Feature-Rich (636 Features) + SupCon Residual MLP INT8 + Temporal Smoothing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "datasets": {},
    }

    # =========================================================================
    # 1. OZDEMIR ET AL. (Subject S1, 10 Classes)
    # =========================================================================
    print("\n=======================================================")
    print("1. BENCHMARKING M14 FEATURE-RICH SUPCON MLP ON OZDEMIR (S1)")
    print("=======================================================", flush=True)

    raw_oz = load_ozdemir_recording("S1", ROOT / "data" / "raw" / "ozdemir", use_filtered=True)
    q_oz = quantize_ozdemir(raw_oz, precision="int16", v_max=OZDEMIR_V_FS_MV)
    win_oz, lbl_oz, reps_oz = segment_ozdemir_windows(q_oz, fs=2000)

    tr_mask_oz = np.isin(reps_oz, [1, 3, 4])
    te_mask_oz = np.isin(reps_oz, [2, 5])
    tr_lbl_oz = lbl_oz[tr_mask_oz]
    te_lbl_oz = lbl_oz[te_mask_oz]
    te_reps_oz = reps_oz[te_mask_oz]

    diffs_oz = np.where((te_lbl_oz[1:] != te_lbl_oz[:-1]) | (te_reps_oz[1:] != te_reps_oz[:-1]))[0] + 1
    oz_boundaries = [0] + diffs_oz.tolist()

    oz_scenarios = []
    for c in [4, 2]:
        print(f"\n--- Ozdemir S1 | M14 Feature-Rich SupCon MLP ({c}CH) ---", flush=True)
        tr_win = win_oz[tr_mask_oz][:, :, :c]
        te_win = win_oz[te_mask_oz][:, :, :c]

        calib = ThresholdCalibrator(default_r=0.5, num_channels=c).calibrate(
            tr_win, tr_lbl_oz, rest_label=0, r_multiplier=0.5
        )
        ext = EnhancedSemgFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=c, fs=2000.0)
        X_tr = ext.extract(tr_win)
        X_te = ext.extract(te_win)

        lda_res = _evaluate_enhanced_lda(X_tr, tr_lbl_oz, X_te, te_lbl_oz, oz_boundaries, num_classes=10)

        h1, h2 = (128, 64) if c == 4 else (96, 48)
        art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_rich_supcon_mlp(
            dataset_name=f"ozdemir_{c}ch",
            subject_id="S1",
            num_channels=c,
            X_tr=X_tr,
            tr_lbl=tr_lbl_oz,
            X_te=X_te,
            te_lbl=te_lbl_oz,
            num_classes=10,
            hidden1=h1,
            hidden2=h2,
            supcon_epochs=20,
            finetune_epochs=30,
            batch_size=256,
            seed=42,
        )

        base_acc_pct = art.int8_top1_acc * 100.0
        base_f1_pct = art.int8_macro_f1 * 100.0
        mv_sweep, sv_sweep = _evaluate_smoothing_sweeps(
            te_lbl_oz, int8_p, int8_l, oz_boundaries, base_acc_pct, base_f1_pct
        )
        hw = compute_m14_hardware_metrics(art, acc_pct=base_acc_pct)
        hw_k5 = compute_m14_hardware_metrics(art, acc_pct=sv_sweep[2]["top1_accuracy_pct"])

        sc = {
            "num_channels": c,
            "num_features": ext.total_dim,
            "enhanced_lda": lda_res,
            "fp32_top1_pct": round(art.fp32_top1_acc * 100.0, 2),
            "fp32_macro_f1_pct": round(art.fp32_macro_f1 * 100.0, 2),
            "int8_top1_pct": round(base_acc_pct, 2),
            "int8_macro_f1_pct": round(base_f1_pct, 2),
            "quant_drop_pct": round((art.fp32_top1_acc - art.int8_top1_acc) * 100.0, 2),
            "hardware": hw,
            "hardware_with_k5": hw_k5,
            "majority_vote_sweep": mv_sweep,
            "soft_vote_sweep": sv_sweep,
        }
        oz_scenarios.append(sc)
        print(
            f"  [Ozdemir {c}CH, {ext.total_dim} feats] Enh-LDA={lda_res['raw_top1_pct']:.2f}% | "
            f"M14 FP32={sc['fp32_top1_pct']:.2f}% | INT8={sc['int8_top1_pct']:.2f}% | "
            f"+MV(K=5)={mv_sweep[2]['top1_accuracy_pct']:.2f}% | +Soft(K=5)={sv_sweep[2]['top1_accuracy_pct']:.2f}% | "
            f"+Soft(K=9)={sv_sweep[4]['top1_accuracy_pct']:.2f}%",
            flush=True,
        )

    results["datasets"]["ozdemir"] = {"num_classes": 10, "num_test_windows": len(te_lbl_oz), "scenarios": oz_scenarios}

    # =========================================================================
    # 2. NINAPRO DB2 (Subject S4, 50 Classes)
    # =========================================================================
    print("\n=======================================================")
    print("2. BENCHMARKING M14 FEATURE-RICH SUPCON MLP ON NINAPRO DB2 (S4)")
    print("=======================================================", flush=True)

    loader_ni = NinaProDB2Loader(raw_dir=ROOT / "data" / "raw" / "ninapro_db2")
    raw_cache_ni: dict[str, RecordingData] = {
        eid: loader_ni.load_recording("S4", eid) for eid in ("E1", "E2", "E3")
    }

    def qload_ni(sid: str, eid: str) -> RecordingData:
        rec = raw_cache_ni[eid]
        return replace(rec, emg=quantize_ninapro(rec.emg, precision="int16", v_max=NINAPRO_V_FS_VOLTS))

    runner_ni = MultiExerciseWithinSubjectRunner()
    subj_ni = MultiExerciseSubject(subject_id="S4", load_recording=qload_ni)

    tr_win_full, tr_lbl_ni, _, _ = runner_ni.prepare_repetition_windows(subj_ni, repetitions=runner_ni.TRAINING_REPETITIONS)
    te_win_full, te_lbl_ni, _, audits_ni = runner_ni.prepare_repetition_windows(subj_ni, repetitions=runner_ni.TEST_REPETITIONS)

    counts = [a.admitted_window_count for a in audits_ni]
    c_sum = 0
    ni_boundaries = [0]
    for cnt in counts[:-1]:
        c_sum += cnt
        ni_boundaries.append(c_sum)

    ni_scenarios = []
    for c in [12, 2]:
        print(f"\n--- NinaPro DB2 S4 | M14 Feature-Rich SupCon MLP ({c}CH) ---", flush=True)
        tr_win = tr_win_full[:, :, :c]
        te_win = te_win_full[:, :, :c]

        calib = ThresholdCalibrator(default_r=0.5, num_channels=c).calibrate(
            tr_win, tr_lbl_ni, rest_label=0, r_multiplier=0.5
        )
        ext = EnhancedSemgFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=c, fs=2000.0)
        X_tr = ext.extract(tr_win)
        X_te = ext.extract(te_win)

        lda_res = _evaluate_enhanced_lda(X_tr, tr_lbl_ni, X_te, te_lbl_ni, ni_boundaries, num_classes=50)

        h1, h2 = (192, 128) if c == 12 else (96, 64)
        art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_rich_supcon_mlp(
            dataset_name=f"ninapro_{c}ch",
            subject_id="S4",
            num_channels=c,
            X_tr=X_tr,
            tr_lbl=tr_lbl_ni,
            X_te=X_te,
            te_lbl=te_lbl_ni,
            num_classes=50,
            hidden1=h1,
            hidden2=h2,
            supcon_epochs=20,
            finetune_epochs=30,
            batch_size=512,
            seed=42,
        )

        base_acc_pct = art.int8_top1_acc * 100.0
        base_f1_pct = art.int8_macro_f1 * 100.0
        mv_sweep, sv_sweep = _evaluate_smoothing_sweeps(
            te_lbl_ni, int8_p, int8_l, ni_boundaries, base_acc_pct, base_f1_pct
        )
        hw = compute_m14_hardware_metrics(art, acc_pct=base_acc_pct)
        hw_k5 = compute_m14_hardware_metrics(art, acc_pct=sv_sweep[2]["top1_accuracy_pct"])

        sc = {
            "num_channels": c,
            "num_features": ext.total_dim,
            "enhanced_lda": lda_res,
            "fp32_top1_pct": round(art.fp32_top1_acc * 100.0, 2),
            "fp32_macro_f1_pct": round(art.fp32_macro_f1 * 100.0, 2),
            "int8_top1_pct": round(base_acc_pct, 2),
            "int8_macro_f1_pct": round(base_f1_pct, 2),
            "quant_drop_pct": round((art.fp32_top1_acc - art.int8_top1_acc) * 100.0, 2),
            "hardware": hw,
            "hardware_with_k5": hw_k5,
            "majority_vote_sweep": mv_sweep,
            "soft_vote_sweep": sv_sweep,
        }
        ni_scenarios.append(sc)
        print(
            f"  [NinaPro {c}CH, {ext.total_dim} feats] Enh-LDA={lda_res['raw_top1_pct']:.2f}% | "
            f"M14 FP32={sc['fp32_top1_pct']:.2f}% | INT8={sc['int8_top1_pct']:.2f}% | "
            f"+MV(K=5)={mv_sweep[2]['top1_accuracy_pct']:.2f}% | +Soft(K=5)={sv_sweep[2]['top1_accuracy_pct']:.2f}% | "
            f"+Soft(K=9)={sv_sweep[4]['top1_accuracy_pct']:.2f}%",
            flush=True,
        )

    results["datasets"]["ninapro_db2"] = {"num_classes": 50, "num_test_windows": len(te_lbl_ni), "scenarios": ni_scenarios}
    results["elapsed_seconds"] = round(time.time() - t0, 2)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved M14 Feature-Rich results to {OUT_JSON}")

    generate_markdown_report(results)
    return results


def generate_markdown_report(results: dict[str, Any]) -> None:
    lines = [
        "# Relatório de Benchmark: M14 Extração Rica (636 Features) + SupCon MLP INT8",
        "",
        f"**Data:** {results['timestamp']} | **Tempo Total:** {results['elapsed_seconds']} s",
        "",
        "## 1. Engenharia de Features Multi-Domínio (60 -> 636 Features em 12CH)",
        "",
        "- **TD Não-Linear & Log-Amplitude (8/ch):** MAV, LOG_RMS, WL, MFL, DASDV, WAMP, Form Factor, Crest Factor.",
        "- **Morfologia, Hjorth & HOS (6/ch):** ZC, SSC, Mobilidade, Complexidade, Assimetria (Skew), Log-Curtose.",
        "- **Polos Auto-Regressivos AR-4 (4/ch):** Yule-Walker regularizado em ordem 4.",
        "- **Espectro Multi-Banda & Cepstro EMGCC (11/ch):** Potência Log em 5 bandas (20–500 Hz), MNF, Entropia Espectral e DCT Cepstral (EMGCC1..4).",
        "- **Wavelet Haar 4 Níveis (5/ch):** Log-Energias `cD1, cD2, cD3, cD4, cA4`.",
        "- **Envelope Temporal em Sub-janelas (8/ch):** 4x Log-MAV + 4x Log-WL (sub-janelas de 50 ms).",
        "- **Matriz de Sinergia Muscular Inter-Canais (132 em 12CH):** 66 correlações cruzadas de Pearson + 66 razões Log-RMS par-a-par invariantes à força.",
        "",
        "## 2. Resultados Consolidados (Enhanced-LDA vs. M14 SupCon MLP INT8 + Passo 1)",
        "",
        "| Dataset | Canais | Feats | Enh-LDA Raw | M14 FP32 | M14 INT8 ($K=1$) | M14 + MV ($K=5$) | M14 + Soft ($K=5$, 200ms) | M14 + Soft ($K=9$, 400ms) | Latência ESP32-S3 | Flash / SRAM |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ds_key, ds_label in [("ninapro_db2", "NinaPro DB2 (50 cls)"), ("ozdemir", "Ozdemir (10 cls)")]:
        for sc in results["datasets"][ds_key]["scenarios"]:
            mv5 = sc["majority_vote_sweep"][2]["top1_accuracy_pct"]
            sv5 = sc["soft_vote_sweep"][2]["top1_accuracy_pct"]
            sv9 = sc["soft_vote_sweep"][4]["top1_accuracy_pct"]
            hw = sc["hardware"]
            lines.append(
                f"| {ds_label} | {sc['num_channels']}CH | {sc['num_features']} | {sc['enhanced_lda']['raw_top1_pct']:.2f}% | "
                f"{sc['fp32_top1_pct']:.2f}% | {sc['int8_top1_pct']:.2f}% | {mv5:.2f}% | **{sv5:.2f}%** | **{sv9:.2f}%** | "
                f"{hw['latency_ms']:.3f} ms | {hw['flash_weights_kb']:.1f} KB / {hw['sram_arena_kb']:.1f} KB |"
            )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Saved Markdown report to {OUT_MD}")


if __name__ == "__main__":
    run_benchmark()
