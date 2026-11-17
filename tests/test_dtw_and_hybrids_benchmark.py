"""Unit, Mathematical Property, and C99 Parity Tests for DTW Variants and Hybrid (DTW+LDA+CNN) Models."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from src.models.train_cnn1d_quantized import c_round_half_away
from src.models.train_dtw_and_hybrids import (
    DTW_ENV_LEN,
    DTW_SAKOE_W,
    compute_dtw_matrix_batch,
    compute_lb_keogh_batch,
    compute_ram_model_asymptotic_scaling,
)

ROOT = Path(__file__).resolve().parents[1]
EMB_FW_DIR = ROOT / "embedded" / "firmware"
QEMU_FW_DIR = ROOT / "firmware" / "esp32_qemu"
XTENSA_GCC_PATH = Path("/home/allisson/.espressif/tools/xtensa-esp-elf/xtensa-esp-elf/bin/xtensa-esp32s3-elf-gcc")


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


class _DtwTelemetryC(ctypes.Structure):
    _fields_ = [
        ("cycles_total", ctypes.c_uint32),
        ("pruned_lb_count", ctypes.c_uint32),
        ("dp_calls_count", ctypes.c_uint32),
        ("early_exit", ctypes.c_uint8),
        ("best_cost", ctypes.c_int32),
    ]


def _compile_host_c99_lib(tmp_path: Path) -> ctypes.CDLL:
    gcc_bin = shutil.which("gcc")
    if gcc_bin is None:
        pytest.skip("Host gcc unavailable")
    assert gcc_bin is not None
    so_path = tmp_path / "libdtw_hyb_test.so"
    cmd = [
        gcc_bin,
        "-std=c99",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-shared",
        "-fPIC",
        str(EMB_FW_DIR / "src" / "dtw_hybrid_engine.c"),
        str(EMB_FW_DIR / "src" / "cnn1d_engine.c"),
        str(QEMU_FW_DIR / "src" / "semg_pipeline.c"),
        "-I",
        str(EMB_FW_DIR / "include"),
        "-I",
        str(QEMU_FW_DIR / "include"),
        "-o",
        str(so_path),
        "-lm",
    ]
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so_path))
    lib.dtw_classic_dist_one.restype = ctypes.c_int32
    lib.dtw_sakoe_chiba_dist_one.restype = ctypes.c_int32
    lib.dtw_lb_keogh_dist_one.restype = ctypes.c_int32
    lib.dtw_classic_on2_predict.restype = ctypes.c_int32
    lib.dtw_sakoe_chiba_predict.restype = ctypes.c_int32
    lib.dtw_lb_keogh_predict.restype = ctypes.c_int32
    return lib


def test_c99_dtw_inequalities_and_lossless_lb_keogh_pruning(tmp_path: Path) -> None:
    """Verify mathematical bounds: Classic <= SakoeChiba and LB_Keogh <= SakoeChiba, plus exact pruning parity."""
    lib = _compile_host_c99_lib(tmp_path)
    rng = np.random.default_rng(123)
    num_cls = 10
    num_ch = 4
    w = DTW_SAKOE_W

    # Generate smooth class-structured sEMG-like envelopes (distinct spatial/temporal signatures)
    t_ax = np.linspace(0.0, 2.0 * np.pi, DTW_ENV_LEN)
    templates_q = np.zeros((num_cls, DTW_ENV_LEN, num_ch), dtype=np.int8)
    for k in range(num_cls):
        base = np.zeros((DTW_ENV_LEN, num_ch), dtype=np.float64)
        base[:, k % num_ch] = 75.0 * (0.6 + 0.4 * np.sin(t_ax + 0.5 * k))
        base[:, (k + 1) % num_ch] = -55.0 + 15.0 * np.cos(t_ax)
        templates_q[k] = np.clip(np.round(base + rng.normal(0, 5, size=base.shape)), -127, 127).astype(np.int8)

    env_upper = np.zeros_like(templates_q)
    env_lower = np.zeros_like(templates_q)
    for k in range(num_cls):
        for i in range(DTW_ENV_LEN):
            lo = max(0, i - w)
            hi = min(DTW_ENV_LEN - 1, i + w)
            env_upper[k, i] = np.max(templates_q[k, lo : hi + 1], axis=0)
            env_lower[k, i] = np.min(templates_q[k, lo : hi + 1], axis=0)

    env_mean = np.zeros(num_ch, dtype=np.float32)
    env_inv_std = np.ones(num_ch, dtype=np.float32)
    cfg = _DtwModelConfigC(
        num_channels=num_ch,
        num_classes=num_cls,
        env_len=DTW_ENV_LEN,
        band_w=w,
        env_mean=env_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        env_inv_std=env_inv_std.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        env_quant_scale=ctypes.c_float(1.0),
        templates_q=templates_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        env_upper=env_upper.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        env_lower=env_lower.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )

    # Queries: warped/noisy class realizations + adversarial random noise queries
    queries = np.zeros((25, DTW_ENV_LEN, num_ch), dtype=np.int8)
    for q_idx in range(20):
        shifted = np.roll(templates_q[q_idx % num_cls], shift=(q_idx % 3) - 1, axis=0)
        queries[q_idx] = np.clip(shifted.astype(np.int32) + rng.integers(-8, 9, size=shifted.shape), -127, 127).astype(
            np.int8
        )
    queries[20:] = rng.integers(-100, 101, size=(5, DTW_ENV_LEN, num_ch), dtype=np.int8)
    py_sc = compute_dtw_matrix_batch(queries, templates_q, band_w=w)
    py_cl = compute_dtw_matrix_batch(queries, templates_q, band_w=None)
    py_lb = compute_lb_keogh_batch(queries, env_upper, env_lower)

    assert np.all(py_lb <= py_sc), "Lower bound LB_Keogh must never exceed Sakoe-Chiba DTW"
    assert np.all(py_cl <= py_sc), "Unconstrained Classic DTW must always be <= Sakoe-Chiba DTW"

    total_pruned = 0
    for q_idx in range(len(queries)):
        q_row = np.ascontiguousarray(queries[q_idx], dtype=np.int8)
        q_ptr = q_row.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        dists_sc = (ctypes.c_int32 * num_cls)()
        dists_cl = (ctypes.c_int32 * num_cls)()
        tel_sc = _DtwTelemetryC()
        tel_lb = _DtwTelemetryC()

        pred_cl = lib.dtw_classic_on2_predict(ctypes.byref(cfg), q_ptr, dists_cl, None)
        pred_sc = lib.dtw_sakoe_chiba_predict(ctypes.byref(cfg), q_ptr, dists_sc, ctypes.byref(tel_sc))
        pred_lb = lib.dtw_lb_keogh_predict(ctypes.byref(cfg), q_ptr, ctypes.byref(tel_lb))

        assert list(dists_cl) == py_cl[q_idx].tolist()
        assert list(dists_sc) == py_sc[q_idx].tolist()
        assert pred_cl == int(np.argmin(py_cl[q_idx]))
        # Lossless pruning guarantees identical class and minimum cost between M04 and M05
        assert pred_lb == pred_sc
        assert tel_lb.best_cost == tel_sc.best_cost
        total_pruned += int(tel_lb.pruned_lb_count)

        for k in range(num_cls):
            lb_c = lib.dtw_lb_keogh_dist_one(ctypes.byref(cfg), q_ptr, k)
            assert lb_c == int(py_lb[q_idx, k])
            assert lb_c <= dists_sc[k]

    assert total_pruned > 0, "Expected LB_Keogh to prune at least some uncompetitive templates"


def test_c99_envelope_extractor_parity(tmp_path: Path) -> None:
    """Verify bit-exact match between C99 `extract_dtw_envelope_int8` and Python MAV block oracle."""
    lib = _compile_host_c99_lib(tmp_path)
    rng = np.random.default_rng(7)
    n_ch = 4
    win_q16 = rng.integers(-4000, 4001, size=(400, n_ch), dtype=np.int16)
    inv_adc = np.float32(1.0 / 3276.7)
    env_mean = np.array([0.1, 0.2, 0.15, 0.05], dtype=np.float32)
    env_inv_std = np.array([2.0, 1.5, 2.5, 3.0], dtype=np.float32)
    env_q_scale = np.float32(0.025)

    dummy_i8 = np.zeros(DTW_ENV_LEN * n_ch, dtype=np.int8)
    cfg = _DtwModelConfigC(
        num_channels=n_ch,
        num_classes=1,
        env_len=DTW_ENV_LEN,
        band_w=DTW_SAKOE_W,
        env_mean=env_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        env_inv_std=env_inv_std.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        env_quant_scale=ctypes.c_float(float(env_q_scale)),
        templates_q=dummy_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        env_upper=dummy_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        env_lower=dummy_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )

    c_env = np.zeros((DTW_ENV_LEN, n_ch), dtype=np.int8)
    lib.extract_dtw_envelope_int8(
        ctypes.byref(cfg),
        win_q16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ctypes.c_float(float(inv_adc)),
        c_env.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )

    blocks = np.abs(win_q16.reshape(DTW_ENV_LEN, 10, n_ch).astype(np.int32))
    mav = (np.sum(blocks, axis=1).astype(np.float32) * inv_adc) * np.float32(0.1)
    z = (mav - env_mean[None, :]) * env_inv_std[None, :]
    py_env = np.clip(c_round_half_away(z * np.float32(1.0 / float(env_q_scale))), -127, 127).astype(np.int8)

    assert np.array_equal(c_env, py_env)


def test_ram_asymptotic_scaling_overflow_proof() -> None:
    """Confirm Classic O(N^2) exceeds 512 KB SRAM at N=400 while Sakoe-Chiba fits easily."""
    rows = compute_ram_model_asymptotic_scaling()
    n400 = [r for r in rows if r["N"] == 400][0]
    assert n400["classic_sram_bytes"] == 640_000
    assert n400["exceeds_512kb_sram"] is True
    assert n400["sakoe_sram_bytes"] == 3_200


def test_xtensa_zero_warning_compilation(tmp_path: Path) -> None:
    """All DTW + Hybrid C99 engines compile on Xtensa LX7 with -Werror and zero warnings."""
    if not XTENSA_GCC_PATH.exists():
        pytest.skip("Xtensa GCC not found on disk")
    out_obj_dir = tmp_path / "objs"
    out_obj_dir.mkdir(parents=True, exist_ok=True)
    for src_name in (
        EMB_FW_DIR / "src" / "dtw_hybrid_engine.c",
        EMB_FW_DIR / "src" / "cnn1d_engine.c",
        QEMU_FW_DIR / "src" / "semg_pipeline.c",
    ):
        cmd = [
            str(XTENSA_GCC_PATH),
            "-std=c99",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            str(src_name),
            "-I",
            str(EMB_FW_DIR / "include"),
            "-I",
            str(QEMU_FW_DIR / "include"),
            "-o",
            str(out_obj_dir / f"{src_name.stem}.o"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"Xtensa -Werror build failed for {src_name.name}:\n{proc.stderr}"


def test_master_benchmark_artifacts_valid_if_generated() -> None:
    """Validate final consolidated JSON and Markdown reports match schema and 100% QEMU parity."""
    json_path = ROOT / "reports" / "dtw_hybrid_benchmark" / "dtw_and_hybrids_benchmark_results.json"
    md_path = ROOT / "reports" / "dtw_hybrid_benchmark" / "dtw_and_hybrids_master_report.md"
    if not json_path.exists():
        pytest.skip("Benchmark results JSON not generated yet")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("esp32s3_ozdemir", "esp32s3_ninapro"):
        models = data["qemu_results"][key]["models"]
        assert len(models) == 10
        for m in models:
            assert m["parity_ratio"] == pytest.approx(1.0)
    assert data["espidf_build"]["status"] == "SUCCESS"
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "M10_hybrid_dtw_lda_cnn" in md_text
    assert "M05_dtw_lb_keogh" in md_text


def test_channel_and_quantization_benchmark_artifacts() -> None:
    """Validate 55 scenarios x 10 models (550 evaluations) and QEMU channel-scaling parity."""
    json_path = ROOT / "reports" / "dtw_hybrid_benchmark" / "channel_and_quantization_benchmark_results.json"
    md_path = ROOT / "reports" / "dtw_hybrid_benchmark" / "channel_and_quantization_master_report.md"
    if not json_path.exists():
        pytest.skip("Channel/quantization benchmark results not generated yet")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    oz_scenarios = data["datasets"]["ozdemir"]["scenarios"]
    ni_scenarios = data["datasets"]["ninapro"]["scenarios"]
    assert len(oz_scenarios) == 20  # 4 channel counts x 5 precisions
    assert len(ni_scenarios) == 35  # 7 channel counts x 5 precisions
    for sc in oz_scenarios + ni_scenarios:
        assert len(sc["metrics"]) == 10

    for ds_key, expected_ch in (("ozdemir", ("4", "3", "2", "1")), ("ninapro", ("12", "10", "8", "6", "4", "2", "1"))):
        profs = data["qemu_channel_profiles"][ds_key]
        for ch_k in expected_ch:
            assert ch_k in profs
            assert len(profs[ch_k]["models"]) == 10
            for m in profs[ch_k]["models"]:
                assert m["parity_ratio"] == pytest.approx(1.0)

    assert md_path.exists()
    assert "Cenário Prótese de Baixo Custo" in md_path.read_text(encoding="utf-8")
