#!/usr/bin/env python3
"""Run Full Benchmark for Spatial-Temporal Circular CNN-2D (M12) — Passo 2.

Evaluates M12 (`CompactCircularSemgCnn2D`, FP32 + Folded INT8 Q24 simulation)
against M01 (Micro-LDA) and M02 (CNN-1D INT8), AND combines M12 with Passo 1
(Causal Temporal Majority Voting & Soft Voting K in [1, 3, 5, 7, 9]).

Datasets & Channels:
  - Ozdemir et al. (Subject S1, 10 classes): 4CH and 2CH (int16 input)
  - NinaPro DB2 (Subject S4, 50 classes): 12CH (8 ring + 4 anat) and 2CH (int16 input)

Outputs:
  - JSON results: reports/dtw_hybrid_benchmark/cnn2d_circular_results.json
  - Markdown report: reports/dtw_hybrid_benchmark/cnn2d_circular_master_report.md
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

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
from semg_baseline.feature004_multi_exercise import (  # noqa: E402
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.features import TimeDomainFeatureExtractor  # noqa: E402
from semg_baseline.thresholds import ThresholdCalibrator  # noqa: E402
from semg_dataset.contract import RecordingData  # noqa: E402
from semg_dataset.loader import NinaProDB2Loader  # noqa: E402
from src.models.temporal_smoothing import (  # noqa: E402
    causal_soft_voting,
    compute_smoothing_metrics,
    stream_causal_majority_vote,
)
from src.models.train_cnn2d_circular import (  # noqa: E402
    QuantizedCnn2DCircularArtifacts,
    infer_ring_topology,
    train_and_quantize_cnn2d_circular,
)

OUT_DIR = ROOT / "reports" / "dtw_hybrid_benchmark"
OUT_JSON = OUT_DIR / "cnn2d_circular_results.json"
OUT_MD = OUT_DIR / "cnn2d_circular_master_report.md"

K_VALUES = [1, 3, 5, 7, 9]


def compute_m12_hardware_metrics(
    art: QuantizedCnn2DCircularArtifacts,
    acc_pct: float,
) -> dict[str, float]:
    """Compute MACs, ESP32-S3 cycles, latency, SRAM, Flash, energy, IRC, and IEE for M12."""
    c_ring, c_anat = infer_ring_topology(art.in_channels)
    k_h1 = 3 if c_ring >= 3 else min(2, c_ring)
    k_h2 = 3 if c_ring >= 3 else 1

    # MAC count calculation
    # Ring Conv1: (1, 12, k_h1, 9) stride=(1,3) -> (H1, 134)
    h1 = c_ring if c_ring >= 3 else 1
    macs_r1 = 1 * 12 * k_h1 * 9 * h1 * 134
    # Ring Conv2: (12, 24, k_h2, 5) stride=(1,2) over Pool1 (H1, 67) -> (H2, 34)
    macs_r2 = 12 * 24 * k_h2 * 5 * h1 * 34

    macs_anat = 0
    if c_anat > 0:
        macs_anat = (c_anat * 12 * 9 * 134) + (12 * 16 * 5 * 34)

    # Classifier head MACs
    ring_feat = 24 * 4 if c_ring == 8 else (24 * c_ring if c_ring >= 3 else 24)
    anat_feat = 16 if c_anat > 0 else 0
    macs_fc = (ring_feat + anat_feat + art.num_aux) * art.num_classes
    total_macs = macs_r1 + macs_r2 + macs_anat + macs_fc

    # Calibrated against ESP32-S3 QEMU (M02: 359,208 MACs -> 51,924 classifier cycles = 0.1445 cycles/MAC)
    # Using conservative 0.18 cycles/MAC for 2D circular indexing + DSP feature cycles (7980 * C/12)
    dsp_cycles = 2500.0 + (450.0 * art.in_channels)
    cls_cycles = total_macs * 0.18
    total_cycles = dsp_cycles + cls_cycles

    latency_ms = total_cycles / 240000.0
    energy_mj = latency_ms * (ESP32S3_ACTIVE_POWER_MW / 1000.0)

    # Memory footprint
    # Flash: INT8 weights + INT32 biases (1 byte per weight, 4 bytes per bias)
    w_bytes = (
        art.r_conv1_w_q.size + art.r_conv2_w_q.size +
        art.a_conv1_w_q.size + art.a_conv2_w_q.size +
        art.fc_w_q.size + art.aux_fc_w_q.size
    )
    b_bytes = 4 * (
        art.r_conv1_b_q.size + art.r_conv2_b_q.size +
        art.a_conv1_b_q.size + art.a_conv2_b_q.size +
        art.fc_b_q.size
    )
    flash_kb = (w_bytes + b_bytes) / 1024.0

    # Peak Activation SRAM: Input (C * 400) + Ring Conv1 out (12 * h1 * 134) + Pool1 (12 * h1 * 67)
    sram_bytes = (art.in_channels * 400) + (12 * h1 * 134) + (12 * h1 * 67) + (24 * h1 * 34) + 1024
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
        "cycles_est": round(total_cycles, 1),
        "latency_ms": round(latency_ms, 4),
        "energy_mj": round(energy_mj, 5),
        "flash_weights_kb": round(flash_kb, 2),
        "sram_arena_kb": round(sram_kb, 2),
        "irc": round(metrics["irc"], 2),
        "iee": round(metrics["iee"], 2),
    }


def run_benchmark() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    results: dict[str, Any] = {
        "benchmark": "M12 Spatial-Temporal Circular CNN-2D + Temporal Smoothing (Passo 2 + Passo 1)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "datasets": {},
    }

    # =========================================================================
    # 1. OZDEMIR ET AL. (Subject S1, 10 Classes)
    # =========================================================================
    print("\n=======================================================")
    print("1. BENCHMARKING M12 CIRCULAR CNN-2D ON OZDEMIR (S1)")
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
        print(f"\n--- Ozdemir S1 | M12 Circular CNN-2D ({c}CH) ---", flush=True)
        tr_win = win_oz[tr_mask_oz][:, :, :c]
        te_win = win_oz[te_mask_oz][:, :, :c]

        calib = ThresholdCalibrator(default_r=0.5, num_channels=c).calibrate(tr_win, tr_lbl_oz, rest_label=0, r_multiplier=0.5)
        ext = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=c)
        X_tr_aux = ext.extract(tr_win)
        X_te_aux = ext.extract(te_win)

        art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_cnn2d_circular(
            dataset_name=f"ozdemir_{c}ch",
            subject_id="S1",
            train_win=tr_win,
            train_lbl=tr_lbl_oz,
            test_win=te_win,
            test_lbl=te_lbl_oz,
            X_tr_aux=X_tr_aux,
            X_te_aux=X_te_aux,
            num_classes=10,
            epochs=35,
            batch_size=256,
            seed=42,
            augment=True,
        )

        # Temporal Smoothing on INT8 M12 (Majority Vote & Soft Vote)
        base_acc_pct = art.int8_top1_acc * 100.0
        base_f1_pct = art.int8_macro_f1 * 100.0
        mv_sweep = []
        sv_sweep = []
        for k in K_VALUES:
            sm_mv = int8_p.copy() if k == 1 else stream_causal_majority_vote(int8_p, boundaries=oz_boundaries, window_size=k, tie_breaker="persistence")
            sm_sv = int8_p.copy() if k == 1 else causal_soft_voting(int8_l, boundaries=oz_boundaries, window_size=k)
            mv_sweep.append(asdict(compute_smoothing_metrics(te_lbl_oz, sm_mv, base_acc_pct, base_f1_pct, k)))
            sv_sweep.append(asdict(compute_smoothing_metrics(te_lbl_oz, sm_sv, base_acc_pct, base_f1_pct, k)))

        hw = compute_m12_hardware_metrics(art, acc_pct=base_acc_pct)
        hw_k5 = compute_m12_hardware_metrics(art, acc_pct=mv_sweep[2]["top1_accuracy_pct"])

        sc = {
            "num_channels": c,
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
            f"  [Ozdemir {c}CH] FP32={sc['fp32_top1_pct']:.2f}% | INT8={sc['int8_top1_pct']:.2f}% | "
            f"+MV(K=5)={mv_sweep[2]['top1_accuracy_pct']:.2f}% | +Soft(K=5)={sv_sweep[2]['top1_accuracy_pct']:.2f}% | "
            f"+Soft(K=9)={sv_sweep[4]['top1_accuracy_pct']:.2f}%"
        )

    results["datasets"]["ozdemir"] = {"num_classes": 10, "num_test_windows": len(te_lbl_oz), "scenarios": oz_scenarios}

    # =========================================================================
    # 2. NINAPRO DB2 (Subject S4, 50 Classes)
    # =========================================================================
    print("\n=======================================================")
    print("2. BENCHMARKING M12 CIRCULAR CNN-2D ON NINAPRO DB2 (S4)")
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
        print(f"\n--- NinaPro DB2 S4 | M12 Circular CNN-2D ({c}CH) ---", flush=True)
        tr_win = tr_win_full[:, :, :c]
        te_win = te_win_full[:, :, :c]

        calib = ThresholdCalibrator(default_r=0.5, num_channels=c).calibrate(tr_win, tr_lbl_ni, rest_label=0, r_multiplier=0.5)
        ext = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=c)
        X_tr_aux = ext.extract(tr_win)
        X_te_aux = ext.extract(te_win)

        art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_cnn2d_circular(
            dataset_name=f"ninapro_{c}ch",
            subject_id="S4",
            train_win=tr_win,
            train_lbl=tr_lbl_ni,
            test_win=te_win,
            test_lbl=te_lbl_ni,
            X_tr_aux=X_tr_aux,
            X_te_aux=X_te_aux,
            num_classes=50,
            epochs=25,
            batch_size=1024,
            seed=42,
            augment=True,
        )

        # Temporal Smoothing on INT8 M12
        base_acc_pct = art.int8_top1_acc * 100.0
        base_f1_pct = art.int8_macro_f1 * 100.0
        mv_sweep = []
        sv_sweep = []
        for k in K_VALUES:
            sm_mv = int8_p.copy() if k == 1 else stream_causal_majority_vote(int8_p, boundaries=ni_boundaries, window_size=k, tie_breaker="persistence")
            sm_sv = int8_p.copy() if k == 1 else causal_soft_voting(int8_l, boundaries=ni_boundaries, window_size=k)
            mv_sweep.append(asdict(compute_smoothing_metrics(te_lbl_ni, sm_mv, base_acc_pct, base_f1_pct, k)))
            sv_sweep.append(asdict(compute_smoothing_metrics(te_lbl_ni, sm_sv, base_acc_pct, base_f1_pct, k)))

        hw = compute_m12_hardware_metrics(art, acc_pct=base_acc_pct)
        hw_k5 = compute_m12_hardware_metrics(art, acc_pct=mv_sweep[2]["top1_accuracy_pct"])

        sc = {
            "num_channels": c,
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
            f"  [NinaPro {c}CH] FP32={sc['fp32_top1_pct']:.2f}% | INT8={sc['int8_top1_pct']:.2f}% | "
            f"+MV(K=5)={mv_sweep[2]['top1_accuracy_pct']:.2f}% | +Soft(K=5)={sv_sweep[2]['top1_accuracy_pct']:.2f}% | "
            f"+Soft(K=9)={sv_sweep[4]['top1_accuracy_pct']:.2f}%"
        )

    results["datasets"]["ninapro_db2"] = {"num_classes": 50, "num_test_windows": len(te_lbl_ni), "scenarios": ni_scenarios}
    results["elapsed_seconds"] = round(time.time() - t0, 2)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved M12 Circular CNN-2D results to {OUT_JSON}")

    generate_markdown_report(results)
    return results


def generate_markdown_report(results: dict[str, Any]) -> None:
    lines = [
        "# Relatório de Benchmark: Spatial-Temporal Circular CNN-2D (M12) — Passo 2",
        "",
        f"**Data:** {results['timestamp']} | **Tempo Total:** {results['elapsed_seconds']} s",
        "",
        "## 1. Resumo Arquitetural (M12 Circular CNN-2D)",
        "",
        "- **Ramo Circumferencial em Anel (Circular Conv2D):** Modela os 8 eletrodos do anel do antebraço (NinaPro Ch 0..7) ou 4 eletrodos (Ozdemir) com *padding espacial circular* (`ch[C-1] <-> ch[0]`), capturando propagação transversal de MUAPs.",
        "- **Preservação de Quadrantes Azimutais:** Ao invés de colapsar todos os eletrodos via GAP global único, retém **4 quadrantes anatômicos espaciais** (flexores vs. extensores vs. radial vs. ulnar).",
        "- **Ramo Anatômico Dedicado (NinaPro 12CH):** Processa isoladamente os eletrodos Ch 8..11 (FDS, EDC, Bíceps, Tríceps).",
        "- **Quantização INT8 com Folding de BatchNorm:** Todo `Conv + BatchNorm` é fundido antes da quantização simétrica INT8/Q24 (custo zero em runtime no ESP32-S3).",
        "",
        "## 2. Resultados: Precisão (FP32 vs. INT8) + Suavização Temporal (Passo 1)",
        "",
        "| Dataset | Canais | FP32 Raw | INT8 Raw ($K=1$) | INT8 + MV ($K=5$, 200ms) | INT8 + Soft ($K=5$, 200ms) | INT8 + Soft ($K=9$, 400ms) | Parâmetros | Flash (KB) | Latência ESP32-S3 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ds_key, ds_label in [("ninapro_db2", "NinaPro DB2 (50 cls)"), ("ozdemir", "Ozdemir (10 cls)")]:
        for sc in results["datasets"][ds_key]["scenarios"]:
            mv5 = sc["majority_vote_sweep"][2]["top1_accuracy_pct"]
            sv5 = sc["soft_vote_sweep"][2]["top1_accuracy_pct"]
            sv9 = sc["soft_vote_sweep"][4]["top1_accuracy_pct"]
            hw = sc["hardware"]
            lines.append(
                f"| {ds_label} | {sc['num_channels']}CH | {sc['fp32_top1_pct']:.2f}% | "
                f"{sc['int8_top1_pct']:.2f}% | **{mv5:.2f}%** | **{sv5:.2f}%** | **{sv9:.2f}%** | "
                f"{hw['total_params']:,} | {hw['flash_weights_kb']:.2f} KB | {hw['latency_ms']:.3f} ms |"
            )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Saved Markdown report to {OUT_MD}")


if __name__ == "__main__":
    run_benchmark()
