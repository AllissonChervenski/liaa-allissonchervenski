"""Unit and dry-run tests for ESP32-S3 Micro-LDA evaluation and metrics generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_esp32s3_lda import (
    OZDEMIR_GESTURE_NAMES,
    compute_classification_metrics,
    generate_markdown_report,
    load_ozdemir_gesture_stream,
)


def test_ozdemir_gesture_names_coverage() -> None:
    assert len(OZDEMIR_GESTURE_NAMES) == 10
    assert OZDEMIR_GESTURE_NAMES[0] == "Rest"
    assert OZDEMIR_GESTURE_NAMES[1] == "Extension"
    assert OZDEMIR_GESTURE_NAMES[5] == "Grip"


def test_load_ozdemir_gesture_stream() -> None:
    raw_dir = Path("data/raw/ozdemir")
    if not (raw_dir / "sEMG-dataset" / "filtered" / "mat" / "1_filtered.mat").exists():
        pytest.skip("Ozdemir raw data not present")

    gestures_data, bundle = load_ozdemir_gesture_stream(subject_id="S1", rep_idx=1, raw_dir=raw_dir)
    assert len(gestures_data) == 10
    assert len(bundle["classes"]) == 10
    assert len(bundle["eps_arr"]) == 4
    for g in range(10):
        assert gestures_data[g].ndim == 2
        assert gestures_data[g].shape[1] == 4
        # 6 seconds @ 2000 Hz = 12000 samples
        assert gestures_data[g].shape[0] == 12000


def test_compute_classification_metrics_synthetic() -> None:
    decisions = []
    # 2 gestures, 10 windows each
    for g in range(2):
        for w in range(10):
            decisions.append({
                "window_seq": g * 10 + w,
                "window_in_gesture": w,
                "gesture_true": g,
                "gesture_name": OZDEMIR_GESTURE_NAMES[g],
                "host_pred": g,
                "esp32_raw_pred": g,
                "esp32_vote3_pred": g,
                "parity_match": True,
                "raw_correct": True,
                "vote3_correct": True,
                "cycles_c0": 11000,
                "cycles_c1": 79000,
                "clf_cycles": 3800,
                "us_c1": 329.0,
                "us_clf": 16.0,
                "stack_c0_free": 2400,
                "stack_c1_free": 3300,
            })

    metrics = compute_classification_metrics(decisions)
    assert metrics["total_windows"] == 20
    assert metrics["parity"]["percent"] == 100.0
    assert metrics["accuracy"]["raw_top1_percent"] == 100.0
    assert metrics["timing_and_resources"]["core1_latency_us"]["mean"] == pytest.approx(329.0)

    md = generate_markdown_report(metrics, {"target": "esp32s3-wroom", "adc_scale": 3276.7})
    assert "# Relatório Técnico: Classificação de Sinais sEMG via Micro-LDA" in md
    assert "100.00%" in md


def test_real_classification_artifacts_exist_and_valid() -> None:
    json_path = Path("reports/esp32s3_lda_classification_results.json")
    md_path = Path("reports/esp32s3_lda_classification_report.md")
    assert json_path.exists(), "JSON report missing"
    assert md_path.exists(), "Markdown report missing"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["metrics"]["total_windows"] == 1170
    assert data["metrics"]["parity"]["matches"] == 1170
    assert data["metrics"]["accuracy"]["raw_top1_percent"] > 75.0
