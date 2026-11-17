#!/usr/bin/env python3
"""Run Full Benchmark for M13 Supervised Contrastive Circular CNN-2D (Passo 3).

Evaluates M13 (`train_and_quantize_supcon_cnn2d`: Two-stage Class-Balanced SupCon +
Prototype Initialization + Circular CNN-2D + Folded INT8 Q24 simulation) combined
with Passo 1 (Causal Temporal Majority Voting & Soft Voting K in [1, 3, 5, 7, 9]).

Datasets & Channels:
  - Ozdemir et al. (Subject S1, 10 classes): 4CH and 2CH (int16 input)
  - NinaPro DB2 (Subject S4, 50 classes): 12CH (8 ring + 4 anat) and 2CH (int16 input)

Outputs:
  - JSON results: reports/dtw_hybrid_benchmark/supcon_cnn2d_results.json
  - Markdown report: reports/dtw_hybrid_benchmark/supcon_cnn2d_master_report.md
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
from scripts.run_cnn2d_circular_benchmark import (  # noqa: E402
    K_VALUES,
    compute_m12_hardware_metrics,
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
from src.models.train_supcon_cnn2d import train_and_quantize_supcon_cnn2d  # noqa: E402

OUT_DIR = ROOT / "reports" / "dtw_hybrid_benchmark"
OUT_JSON = OUT_DIR / "supcon_cnn2d_results.json"
OUT_MD = OUT_DIR / "supcon_cnn2d_master_report.md"


def _evaluate_smoothing_sweeps(
    y_true: np.ndarray,
    int8_preds: np.ndarray,
    int8_logits: np.ndarray,
    boundaries: list[int],
    base_acc_pct: float,
    base_f1_pct: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run causal majority voting and soft voting across K_VALUES."""
    mv_sweep: list[dict[str, Any]] = []
    sv_sweep: list[dict[str, Any]] = []
    for k in K_VALUES:
        sm_mv = (
            int8_preds.copy()
            if k == 1
            else stream_causal_majority_vote(int8_preds, boundaries=boundaries, window_size=k, tie_breaker="persistence")
        )
        sm_sv = (
            int8_preds.copy()
            if k == 1
            else causal_soft_voting(int8_logits, boundaries=boundaries, window_size=k)
        )
        mv_sweep.append(asdict(compute_smoothing_metrics(y_true, sm_mv, base_acc_pct, base_f1_pct, k)))
        sv_sweep.append(asdict(compute_smoothing_metrics(y_true, sm_sv, base_acc_pct, base_f1_pct, k)))
    return mv_sweep, sv_sweep


def run_benchmark() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    results: dict[str, Any] = {
        "benchmark": "M13 Supervised Contrastive Circular CNN-2D + Temporal Smoothing (Passos 3 + 2 + 1)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "temperature": 0.10,
        "class_balanced_anchors": True,
        "datasets": {},
    }

    # =========================================================================
    # 1. OZDEMIR ET AL. (Subject S1, 10 Classes)
    # =========================================================================
    print("\n=======================================================")
    print("1. BENCHMARKING M13 SUPCON CIRCULAR CNN-2D ON OZDEMIR (S1)")
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
        print(f"\n--- Ozdemir S1 | M13 SupCon Circular CNN-2D ({c}CH) ---", flush=True)
        tr_win = win_oz[tr_mask_oz][:, :, :c]
        te_win = win_oz[te_mask_oz][:, :, :c]

        calib = ThresholdCalibrator(default_r=0.5, num_channels=c).calibrate(
            tr_win, tr_lbl_oz, rest_label=0, r_multiplier=0.5
        )
        ext = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=c)
        X_tr_aux = ext.extract(tr_win)
        X_te_aux = ext.extract(te_win)

        art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_supcon_cnn2d(
            dataset_name=f"ozdemir_{c}ch",
            subject_id="S1",
            train_win=tr_win,
            train_lbl=tr_lbl_oz,
            test_win=te_win,
            test_lbl=te_lbl_oz,
            X_tr_aux=X_tr_aux,
            X_te_aux=X_te_aux,
            num_classes=10,
            supcon_epochs=15,
            finetune_epochs=20,
            batch_size=256,
            temperature=0.10,
            seed=42,
        )

        base_acc_pct = art.int8_top1_acc * 100.0
        base_f1_pct = art.int8_macro_f1 * 100.0
        mv_sweep, sv_sweep = _evaluate_smoothing_sweeps(
            te_lbl_oz, int8_p, int8_l, oz_boundaries, base_acc_pct, base_f1_pct
        )

        hw = compute_m12_hardware_metrics(art, acc_pct=base_acc_pct)
        hw_k5 = compute_m12_hardware_metrics(art, acc_pct=sv_sweep[2]["top1_accuracy_pct"])

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
            f"+Soft(K=9)={sv_sweep[4]['top1_accuracy_pct']:.2f}% (MacroF1 K=5: {sv_sweep[2]['macro_f1_pct']:.2f}%)",
            flush=True,
        )

    results["datasets"]["ozdemir"] = {"num_classes": 10, "num_test_windows": len(te_lbl_oz), "scenarios": oz_scenarios}

    # =========================================================================
    # 2. NINAPRO DB2 (Subject S4, 50 Classes)
    # =========================================================================
    print("\n=======================================================")
    print("2. BENCHMARKING M13 SUPCON CIRCULAR CNN-2D ON NINAPRO DB2 (S4)")
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
        print(f"\n--- NinaPro DB2 S4 | M13 SupCon Circular CNN-2D ({c}CH) ---", flush=True)
        tr_win = tr_win_full[:, :, :c]
        te_win = te_win_full[:, :, :c]

        calib = ThresholdCalibrator(default_r=0.5, num_channels=c).calibrate(
            tr_win, tr_lbl_ni, rest_label=0, r_multiplier=0.5
        )
        ext = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=c)
        X_tr_aux = ext.extract(tr_win)
        X_te_aux = ext.extract(te_win)

        art, fp32_p, fp32_l, int8_p, int8_l = train_and_quantize_supcon_cnn2d(
            dataset_name=f"ninapro_{c}ch",
            subject_id="S4",
            train_win=tr_win,
            train_lbl=tr_lbl_ni,
            test_win=te_win,
            test_lbl=te_lbl_ni,
            X_tr_aux=X_tr_aux,
            X_te_aux=X_te_aux,
            num_classes=50,
            supcon_epochs=12,
            finetune_epochs=15,
            batch_size=1024,
            temperature=0.10,
            seed=42,
        )

        base_acc_pct = art.int8_top1_acc * 100.0
        base_f1_pct = art.int8_macro_f1 * 100.0
        mv_sweep, sv_sweep = _evaluate_smoothing_sweeps(
            te_lbl_ni, int8_p, int8_l, ni_boundaries, base_acc_pct, base_f1_pct
        )

        hw = compute_m12_hardware_metrics(art, acc_pct=base_acc_pct)
        hw_k5 = compute_m12_hardware_metrics(art, acc_pct=sv_sweep[2]["top1_accuracy_pct"])

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
            f"+Soft(K=9)={sv_sweep[4]['top1_accuracy_pct']:.2f}% (MacroF1 K=5: {sv_sweep[2]['macro_f1_pct']:.2f}%)",
            flush=True,
        )

    results["datasets"]["ninapro_db2"] = {"num_classes": 50, "num_test_windows": len(te_lbl_ni), "scenarios": ni_scenarios}
    results["elapsed_seconds"] = round(time.time() - t0, 2)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved M13 SupCon results to {OUT_JSON}")

    generate_markdown_report(results)
    return results


def generate_markdown_report(results: dict[str, Any]) -> None:
    lines = [
        "# Relatório de Benchmark: M13 SupCon Circular CNN-2D (Passo 3)",
        "",
        f"**Data:** {results['timestamp']} | **Temperatura SupCon:** $\\tau={results['temperature']}$ | **Tempo Total:** {results['elapsed_seconds']} s",
        "",
        "## 1. Resumo da Abordagem (Passo 3)",
        "",
        "- **Stage 1 (Class-Balanced SupCon):** Treina o backbone `CompactCircularSemgCnn2D` + *Projection Head* MLP (`112 -> 128 -> 64`) aproximando repetições do mesmo gesto e repelindo as demais classes. Âncoras ponderadas inversamente pela frequência no lote evitam que o repouso (36,3% do NinaPro DB2) domine os gradientes.",
        "- **Stage 2 (Prototype Init & Fine-Tuning):** Descarta a *Projection Head*, inicializa `fc_conv` com os centróides L2-normalizados dos clusters contrastivos e refina ponta-a-ponta.",
        "- **Custo Zero em TinyML:** O modelo exportado em INT8 possui exatamente os mesmos **14.750 parâmetros**, **14,74 KB de Flash** e **1,249 ms** de latência do M12 no ESP32-S3.",
        "",
        "## 2. Resultados Consolidados (INT8 + Passo 1: Votação Majoritária & Soft Voting)",
        "",
        "| Dataset | Canais | FP32 Raw | INT8 Raw ($K=1$) | INT8 + MV ($K=5$) | INT8 + Soft ($K=5$, 200ms) | INT8 + Soft ($K=9$, 400ms) | Macro F1 ($K=5$) | Latência ESP32-S3 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ds_key, ds_label in [("ninapro_db2", "NinaPro DB2 (50 cls)"), ("ozdemir", "Ozdemir (10 cls)")]:
        for sc in results["datasets"][ds_key]["scenarios"]:
            mv5 = sc["majority_vote_sweep"][2]["top1_accuracy_pct"]
            sv5 = sc["soft_vote_sweep"][2]["top1_accuracy_pct"]
            sv5_f1 = sc["soft_vote_sweep"][2]["macro_f1_pct"]
            sv9 = sc["soft_vote_sweep"][4]["top1_accuracy_pct"]
            hw = sc["hardware"]
            lines.append(
                f"| {ds_label} | {sc['num_channels']}CH | {sc['fp32_top1_pct']:.2f}% | "
                f"{sc['int8_top1_pct']:.2f}% | {mv5:.2f}% | **{sv5:.2f}%** | **{sv9:.2f}%** | "
                f"**{sv5_f1:.2f}%** | {hw['latency_ms']:.3f} ms |"
            )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Saved Markdown report to {OUT_MD}")


if __name__ == "__main__":
    run_benchmark()
