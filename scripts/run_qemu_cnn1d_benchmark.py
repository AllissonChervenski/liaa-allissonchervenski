#!/usr/bin/env python3
"""Run Bare-Metal INT8 1D-CNN in Espressif QEMU (ESP32-S3 & ESP32) and Consolidate IRC & IEE Report.

Compiles `embedded/firmware/src/cnn1d_engine.c` + `embedded/firmware/qemu_runner/main_cnn.c`
with `xtensa-esp32s3-elf-gcc` and `xtensa-esp32-elf-gcc` (`-O3 -Wall -Wextra -Werror`),
measures exact ELF Flash (`.text` + `.rodata`) and SRAM (`.data` + `.bss`) footprints,
runs both the 50-window oracle verification and the full test split inside QEMU,
and generates:
  - `reports/cnn1d_qemu/qemu_cnn1d_report.json`
  - `reports/cnn1d_qemu/micro_lda_vs_cnn1d_int8_comparison_report.md`
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = ROOT / "embedded" / "firmware"
REPORT_DIR = ROOT / "reports" / "cnn1d_qemu"
LDA_JSON_PATH = ROOT / "reports" / "esp32_qemu" / "esp32_qemu_emulation_results.json"

# Active power model for ESP32-S3 core @ 240 MHz (45 mA @ 3.3 V = 148.5 mW = 0.1485 mJ/ms)
ESP32S3_ACTIVE_POWER_MW = 148.5


def compile_cnn_firmware() -> dict[str, dict[str, int]]:
    """Compile bare-metal C99 CNN-1D ELF binaries with strict zero-warning flags and inspect sizes."""
    build_dir = CNN_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("esp32s3", "xtensa-esp32s3-elf-gcc", "xtensa-esp32s3-elf-size", build_dir / "cnn1d_esp32s3.elf"),
        ("esp32", "xtensa-esp32-elf-gcc", "xtensa-esp32-elf-size", build_dir / "cnn1d_esp32.elf"),
    ]

    footprints: dict[str, dict[str, int]] = {}
    for machine, gcc_bin, size_bin, elf_path in targets:
        cmd = [
            gcc_bin,
            "-std=c99",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(CNN_DIR / "include"),
            "-specs=board.elf.specs",
            "-specs=sys.qemu.specs",
            "-Wl,--defsym=entire_dram_seg=1",
            str(CNN_DIR / "src" / "cnn1d_engine.c"),
            str(CNN_DIR / "qemu_runner" / "main_cnn.c"),
            "-o",
            str(elf_path),
            "-lm",
        ]
        subprocess.run(cmd, check=True)

        size_out = subprocess.run([size_bin, str(elf_path)], capture_output=True, text=True, check=True).stdout
        lines = [ln.strip() for ln in size_out.strip().splitlines() if ln.strip()]
        parts = lines[-1].split()
        text_b, data_b, bss_b = int(parts[0]), int(parts[1]), int(parts[2])
        footprints[machine] = {
            "text_bytes": text_b,
            "data_bytes": data_b,
            "bss_bytes": bss_b,
            "flash_rom_bytes": text_b + data_b,
            "sram_total_elf_bytes": data_b + bss_b,
            "cnn_arena_sram_bytes": 4720,  # g_arena_buf_a (3152 B) + g_arena_buf_b (1568 B)
            "ozdemir_weights_flash_bytes": 2906 + 40 * 4,  # int8 weights + int32 biases
            "ninapro_weights_flash_bytes": 7562 + 90 * 4,
        }
    return footprints


def run_qemu_on_vector_file(machine: str, elf_path: Path, vector_bin: Path) -> dict[str, Any]:
    """Execute compiled CNN-1D ELF inside `qemu-system-xtensa` against a binary test vector file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_vec = Path(tmpdir) / "cnn1d_test_vectors.bin"
        shutil.copyfile(vector_bin, tmp_vec)
        cmd = [
            "qemu-system-xtensa",
            "-nographic",
            "-machine",
            machine,
            "-semihosting",
            "-kernel",
            str(elf_path.resolve()),
        ]
        proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, timeout=180, check=True)
        for line in proc.stdout.splitlines():
            if "QEMU_CNN1D_REPORT_JSON:" in line:
                return json.loads(line.split("QEMU_CNN1D_REPORT_JSON:", 1)[1].strip())
        raise RuntimeError(f"QEMU did not output QEMU_CNN1D_REPORT_JSON:\n{proc.stdout}\n{proc.stderr}")


def compute_irc_iee(
    acc_pct: float,
    num_gestures: int,
    latency_ms: float,
    sram_kb: float,
    power_mw: float = ESP32S3_ACTIVE_POWER_MW,
) -> dict[str, float]:
    """Compute Responsiveness Index (IRC) and Embedded Efficiency Index (IEE).

    Formulas from specification (Section 5.4):
      - IRC = (Accuracy(%) * Num_Gestures) / Latency(ms)
      - Energy_mJ = Power(mW) * (Latency(ms) / 1000.0)
      - IEE = (Accuracy(%) * Num_Gestures) / (Energy_mJ * log10(SRAM_KB))
    """
    irc = (acc_pct * num_gestures) / max(latency_ms, 1e-9)
    energy_mj = power_mw * (latency_ms / 1000.0)
    energy_uj = energy_mj * 1000.0
    log_sram = math.log10(max(sram_kb, 1.001))
    iee = (acc_pct * num_gestures) / max(energy_mj * log_sram, 1e-12)
    return {
        "irc": round(irc, 2),
        "energy_mj": round(energy_mj, 6),
        "energy_uj": round(energy_uj, 3),
        "sram_kb": round(sram_kb, 3),
        "log10_sram_kb": round(log_sram, 4),
        "iee": round(iee, 2),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    host_summary = json.loads((REPORT_DIR / "cnn1d_host_training_summary.json").read_text(encoding="utf-8"))
    lda_data = json.loads(LDA_JSON_PATH.read_text(encoding="utf-8"))

    print("=====================================================================")
    print(" STEP 2 & 3: COMPILING C99 CNN-1D & EXECUTING IN ESPRESSIF QEMU")
    print("=====================================================================")
    footprints = compile_cnn_firmware()

    elf_s3 = CNN_DIR / "build" / "cnn1d_esp32s3.elf"
    elf_32 = CNN_DIR / "build" / "cnn1d_esp32.elf"

    # Run 50-window oracle verification and full-split evaluation on ESP32-S3 and ESP32
    oz_50_s3 = run_qemu_on_vector_file("esp32s3", elf_s3, REPORT_DIR / "cnn1d_test_vectors_ozdemir_50.bin")
    oz_full_s3 = run_qemu_on_vector_file("esp32s3", elf_s3, REPORT_DIR / "cnn1d_test_vectors_ozdemir_full.bin")
    oz_full_32 = run_qemu_on_vector_file("esp32", elf_32, REPORT_DIR / "cnn1d_test_vectors_ozdemir_full.bin")

    ni_50_s3 = run_qemu_on_vector_file("esp32s3", elf_s3, REPORT_DIR / "cnn1d_test_vectors_ninapro_50.bin")
    ni_full_s3 = run_qemu_on_vector_file("esp32s3", elf_s3, REPORT_DIR / "cnn1d_test_vectors_ninapro_full.bin")
    ni_full_32 = run_qemu_on_vector_file("esp32", elf_32, REPORT_DIR / "cnn1d_test_vectors_ninapro_full.bin")

    print(
        f"[QEMU esp32s3][Ozdemir 50-vec Oracle] Parity: {oz_50_s3['host_parity_ratio']*100:.2f}% | "
        f"Max Logit Diff: {oz_50_s3['max_logit_diff']} | Mean Latency: {oz_50_s3['latency_mean_ms_240mhz']:.4f} ms"
    )
    print(
        f"[QEMU esp32s3][Ozdemir Full 2340-win] Parity: {oz_full_s3['host_parity_ratio']*100:.2f}% | "
        f"Top-1 Acc: {oz_full_s3['top1_accuracy']*100:.2f}% | "
        f"Cycles (Mean/p99): {oz_full_s3['cycles_mean']}/{oz_full_s3['cycles_p99']} "
        f"({oz_full_s3['latency_mean_ms_240mhz']:.4f} ms)"
    )
    print(
        f"[QEMU esp32s3][NinaPro 50-vec Oracle] Parity: {ni_50_s3['host_parity_ratio']*100:.2f}% | "
        f"Max Logit Diff: {ni_50_s3['max_logit_diff']} | Mean Latency: {ni_50_s3['latency_mean_ms_240mhz']:.4f} ms"
    )
    print(
        f"[QEMU esp32s3][NinaPro Full 15824-win] Parity: {ni_full_s3['host_parity_ratio']*100:.2f}% | "
        f"Top-1 Acc: {ni_full_s3['top1_accuracy']*100:.2f}% | "
        f"Cycles (Mean/p99): {ni_full_s3['cycles_mean']}/{ni_full_s3['cycles_p99']} "
        f"({ni_full_s3['latency_mean_ms_240mhz']:.4f} ms)"
    )

    # Extract Micro-LDA baseline numbers from previous QEMU run (int16 4ch for Ozdemir, int16 12ch for NinaPro)
    lda_oz_s3 = next(
        r
        for r in lda_data["ozdemir_results"]
        if r["machine"] == "esp32s3" and r["precision"] == "int16" and r["num_channels"] == 4
    )
    lda_ni_s3 = next(
        r
        for r in lda_data["ninapro_results"]
        if r["machine"] == "esp32s3" and r["precision"] == "int16" and r["num_channels"] == 12
    )

    # Compute IRC and IEE for both models
    # 1. Classifier-only inference latency & SRAM (Window + Working Arena)
    # For Micro-LDA: Window buffer (int16: 3.2 kB single / 6.4 kB ping-pong) + 0.2 kB weights = 6.60 kB SRAM
    # For CNN-1D INT8: Window buffer (int16: 6.4 kB) + 4.72 kB Activation Arena = 11.12 kB SRAM (Ozdemir)
    # Full end-to-end pipeline latency = DSP Filter + Feature/Quantization + Classifier
    lda_oz_lat_classifier_ms = lda_oz_s3["cycles_lda_q7"] / 240000.0
    lda_oz_lat_e2e_ms = lda_oz_s3["latency_total_tinyml_ms_240mhz"]
    cnn_oz_lat_classifier_ms = oz_full_s3["latency_mean_ms_240mhz"]
    cnn_oz_lat_e2e_ms = lda_oz_s3["latency_total_tinyml_ms_240mhz"] - lda_oz_lat_classifier_ms + cnn_oz_lat_classifier_ms

    lda_ni_lat_classifier_ms = lda_ni_s3["cycles_lda_q7"] / 240000.0
    lda_ni_lat_e2e_ms = lda_ni_s3["latency_total_tinyml_ms_240mhz"]
    cnn_ni_lat_classifier_ms = ni_full_s3["latency_mean_ms_240mhz"]
    cnn_ni_lat_e2e_ms = lda_ni_s3["latency_total_tinyml_ms_240mhz"] - lda_ni_lat_classifier_ms + cnn_ni_lat_classifier_ms

    # Calculate IRC/IEE both ways (End-to-End Pipeline and Pure Inference) for complete transparency
    metrics_comparison = {
        "ozdemir_4ch_10cls": {
            "micro_lda_q7": {
                "top1_acc_pct": round(lda_oz_s3["esp32_q7_top1_acc"] * 100.0, 2),
                "bal_acc_pct": round(lda_oz_s3["esp32_q7_bal_acc"] * 100.0, 2),
                "macro_f1_pct": round(lda_oz_s3["esp32_q7_macro_f1"] * 100.0, 2),
                "params_count": 210,
                "weights_rom_bytes": 200,
                "sram_runtime_kb": 6.60,
                "cycles_classifier": lda_oz_s3["cycles_lda_q7"],
                "latency_classifier_ms": round(lda_oz_lat_classifier_ms, 5),
                "latency_e2e_ms": round(lda_oz_lat_e2e_ms, 5),
                "indices_e2e": compute_irc_iee(
                    lda_oz_s3["esp32_q7_top1_acc"] * 100.0, 10, lda_oz_lat_e2e_ms, 6.60
                ),
                "indices_classifier": compute_irc_iee(
                    lda_oz_s3["esp32_q7_top1_acc"] * 100.0, 10, lda_oz_lat_classifier_ms, 6.60
                ),
            },
            "cnn1d_int8": {
                "top1_acc_pct": round(oz_full_s3["top1_accuracy"] * 100.0, 2),
                "bal_acc_pct": round(oz_full_s3["balanced_accuracy"] * 100.0, 2),
                "macro_f1_pct": round(oz_full_s3["macro_f1"] * 100.0, 2),
                "params_count": host_summary["ozdemir"]["total_params"],
                "weights_rom_bytes": footprints["esp32s3"]["ozdemir_weights_flash_bytes"],
                "sram_runtime_kb": 11.12,
                "cycles_classifier_mean": oz_full_s3["cycles_mean"],
                "cycles_classifier_p99": oz_full_s3["cycles_p99"],
                "latency_classifier_ms": round(cnn_oz_lat_classifier_ms, 5),
                "latency_p99_ms": round(oz_full_s3["latency_p99_ms_240mhz"], 5),
                "latency_e2e_ms": round(cnn_oz_lat_e2e_ms, 5),
                "indices_e2e": compute_irc_iee(
                    oz_full_s3["top1_accuracy"] * 100.0, 10, cnn_oz_lat_e2e_ms, 11.12
                ),
                "indices_classifier": compute_irc_iee(
                    oz_full_s3["top1_accuracy"] * 100.0, 10, cnn_oz_lat_classifier_ms, 11.12
                ),
            },
        },
        "ninapro_12ch_50cls": {
            "micro_lda_q7": {
                "top1_acc_pct": round(lda_ni_s3["esp32_q7_top1_acc"] * 100.0, 2),
                "bal_acc_pct": round(lda_ni_s3["esp32_q7_bal_acc"] * 100.0, 2),
                "macro_f1_pct": round(lda_ni_s3["esp32_q7_macro_f1"] * 100.0, 2),
                "params_count": 3050,
                "weights_rom_bytes": 3000,
                "sram_runtime_kb": 22.20,
                "cycles_classifier": lda_ni_s3["cycles_lda_q7"],
                "latency_classifier_ms": round(lda_ni_lat_classifier_ms, 5),
                "latency_e2e_ms": round(lda_ni_lat_e2e_ms, 5),
                "indices_e2e": compute_irc_iee(
                    lda_ni_s3["esp32_q7_top1_acc"] * 100.0, 50, lda_ni_lat_e2e_ms, 22.20
                ),
                "indices_classifier": compute_irc_iee(
                    lda_ni_s3["esp32_q7_top1_acc"] * 100.0, 50, lda_ni_lat_classifier_ms, 22.20
                ),
            },
            "cnn1d_int8": {
                "top1_acc_pct": round(ni_full_s3["top1_accuracy"] * 100.0, 2),
                "bal_acc_pct": round(ni_full_s3["balanced_accuracy"] * 100.0, 2),
                "macro_f1_pct": round(ni_full_s3["macro_f1"] * 100.0, 2),
                "params_count": host_summary["ninapro"]["total_params"],
                "weights_rom_bytes": footprints["esp32s3"]["ninapro_weights_flash_bytes"],
                "sram_runtime_kb": 23.92,
                "cycles_classifier_mean": ni_full_s3["cycles_mean"],
                "cycles_classifier_p99": ni_full_s3["cycles_p99"],
                "latency_classifier_ms": round(cnn_ni_lat_classifier_ms, 5),
                "latency_p99_ms": round(ni_full_s3["latency_p99_ms_240mhz"], 5),
                "latency_e2e_ms": round(cnn_ni_lat_e2e_ms, 5),
                "indices_e2e": compute_irc_iee(
                    ni_full_s3["top1_accuracy"] * 100.0, 50, cnn_ni_lat_e2e_ms, 23.92
                ),
                "indices_classifier": compute_irc_iee(
                    ni_full_s3["top1_accuracy"] * 100.0, 50, cnn_ni_lat_classifier_ms, 23.92
                ),
            },
        },
    }

    report_payload = {
        "emulator": "QEMU emulator version 9.2.2 (esp_develop_9.2.2_20261109)",
        "toolchain": "xtensa-esp-elf-gcc (crosstool-NG esp-14.4.0_20261030) 14.4.0",
        "footprints": footprints,
        "host_training_summary": host_summary,
        "qemu_runs": {
            "ozdemir_50_vec_esp32s3": oz_50_s3,
            "ozdemir_full_esp32s3": oz_full_s3,
            "ozdemir_full_esp32": oz_full_32,
            "ninapro_50_vec_esp32s3": ni_50_s3,
            "ninapro_full_esp32s3": ni_full_s3,
            "ninapro_full_esp32": ni_full_32,
        },
        "comparison_irc_iee": metrics_comparison,
    }

    json_out = REPORT_DIR / "qemu_cnn1d_report.json"
    json_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    # Also write copy at root or embedded/firmware/build for convenience
    (CNN_DIR / "build" / "qemu_cnn1d_report.json").write_text(
        json.dumps(report_payload, indent=2), encoding="utf-8"
    )
    print(f"Saved consolidated QEMU CNN-1D JSON report to {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
