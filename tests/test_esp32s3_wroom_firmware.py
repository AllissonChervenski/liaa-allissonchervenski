"""Unit and integration tests for ESP32-S3-WROOM ESP-IDF firmware and HIL protocol."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest

from scripts.validate_esp32s3_hil import (
    HIL_CMD_PING,
    HIL_CMD_PUSH_SAMPLES,
    HIL_CMD_RESET,
    HIL_FLAG_BYPASS_FILTER,
    crc16_ccitt,
    encode_hil_frame,
    parse_tagged_json_line,
)
from semg_dsp.filter import CausalSosFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
FW_DIR = REPO_ROOT / "firmware" / "esp32s3_wroom"
QEMU_FW_DIR = REPO_ROOT / "firmware" / "esp32_qemu"
CNN_FW_DIR = REPO_ROOT / "embedded" / "firmware"


class HilFrameC(ctypes.Structure):
    _fields_ = [
        ("cmd", ctypes.c_uint8),
        ("payload_len", ctypes.c_uint16),
        ("payload", ctypes.c_uint8 * 4096),
        ("crc_received", ctypes.c_uint16),
        ("crc_computed", ctypes.c_uint16),
    ]


class HilParserC(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int),
        ("cmd", ctypes.c_uint8),
        ("payload_len", ctypes.c_uint16),
        ("payload_pos", ctypes.c_uint16),
        ("crc_rx", ctypes.c_uint16),
        ("crc_calc", ctypes.c_uint16),
        ("payload", ctypes.c_uint8 * 4096),
        ("frames_ok", ctypes.c_uint32),
        ("frames_crc_err", ctypes.c_uint32),
        ("frames_len_err", ctypes.c_uint32),
    ]


class SelfTestOutC(ctypes.Structure):
    _fields_ = [
        ("windows_total", ctypes.c_uint32),
        ("lda_matches", ctypes.c_uint32),
        ("cnn_matches", ctypes.c_uint32),
        ("lda_preds", ctypes.c_int32 * 5),
        ("cnn_preds", ctypes.c_int32 * 5),
    ]


C_HARNESS_SOURCE = r"""
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "semg_pipeline.h"
#include "cnn1d_inference.h"
#include "cnn1d_weights_int8.h"
#include "micro_lda_weights.h"

typedef struct {
    uint32_t windows_total;
    uint32_t lda_matches;
    uint32_t cnn_matches;
    int32_t  lda_preds[5];
    int32_t  cnn_preds[5];
} host_selftest_out_t;

static int16_t clamp_i16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

void run_flash_selftest_c99(host_selftest_out_t *out) {
    memset(out, 0, sizeof(*out));
    micro_lda_model_t lda_m;
    td_feature_extractor_t td_cfg;
    stateful_window_buffer_t wb;
    semg_load_ozdemir_micro_lda(&lda_m, &td_cfg);
    window_buffer_init(&wb, WINDOW_LEN, STRIDE_LEN, OZDEMIR_LDA_NUM_CHANNELS);

    float sample_fp32[OZDEMIR_LDA_NUM_CHANNELS];
    float win_fp32[WINDOW_LEN * OZDEMIR_LDA_NUM_CHANNELS];
    int16_t win_q16[WINDOW_LEN * OZDEMIR_LDA_NUM_CHANNELS];
    float td_feat[OZDEMIR_LDA_NUM_FEATURES];
    float lda_scores[OZDEMIR_LDA_NUM_CLASSES];
    int8_t x_q[OZDEMIR_LDA_NUM_CHANNELS * WINDOW_LEN];
    int8_t aux_q[OZDEMIR_LDA_NUM_FEATURES];
    int32_t cnn_logits[CNN1D_MAX_CLASSES];
    cnn1d_cycle_profile_t prof;

    uint32_t w_idx = 0;
    for (int s = 0; s < OZDEMIR_SELFTEST_SAMPLES; s++) {
        for (int c = 0; c < OZDEMIR_LDA_NUM_CHANNELS; c++) {
            sample_fp32[c] = (float)g_ozdemir_selftest_stream_q16[s * OZDEMIR_LDA_NUM_CHANNELS + c] * td_cfg.inv_scale;
        }
        if (window_buffer_push_sample(&wb, sample_fp32, win_fp32)) {
            if (w_idx < OZDEMIR_SELFTEST_WINDOWS) {
                for (int i = 0; i < WINDOW_LEN * OZDEMIR_LDA_NUM_CHANNELS; i++) {
                    win_q16[i] = clamp_i16((int32_t)lrintf(win_fp32[i] * OZDEMIR_LDA_ADC_SCALE));
                }
                extract_td_features_int16(
                    win_q16, WINDOW_LEN, OZDEMIR_LDA_NUM_CHANNELS,
                    td_cfg.epsilon_int, td_cfg.inv_scale, td_feat
                );
                int32_t p_lda = micro_lda_predict_q7(&lda_m, td_feat, lda_scores);
                cnn1d_quantize_input(&g_cnn1d_ozdemir_config, win_fp32, td_feat, x_q, aux_q);
                int32_t p_cnn = cnn1d_forward_int8(&g_cnn1d_ozdemir_config, x_q, aux_q, cnn_logits, &prof);

                out->lda_preds[w_idx] = p_lda;
                out->cnn_preds[w_idx] = p_cnn;
                if (p_lda == g_ozdemir_selftest_lda_oracle[w_idx]) out->lda_matches++;
                if (p_cnn == g_ozdemir_selftest_cnn_oracle[w_idx]) out->cnn_matches++;
                out->windows_total++;
                w_idx++;
            }
        }
    }
}

void run_sos_filter_4ch_c99(const float *in_buf, float *out_buf, int num_samples) {
    causal_sos_filter_t f;
    sos_filter_init(&f, MAX_SOS_SECTIONS, 4, g_semg_sos_20_450hz);
    sos_filter_process_chunk(&f, in_buf, out_buf, num_samples);
}

void get_sos_coeffs_c99(float *out_24) {
    memcpy(out_24, g_semg_sos_20_450hz, 24 * sizeof(float));
}
"""


def test_esp32s3_project_layout_exists() -> None:
    """All required ESP-IDF v5.x project files and generated headers exist."""
    required = [
        FW_DIR / "CMakeLists.txt",
        FW_DIR / "partitions.csv",
        FW_DIR / "sdkconfig.defaults.esp32s3",
        FW_DIR / "README.md",
        FW_DIR / "components" / "semg_core" / "CMakeLists.txt",
        FW_DIR / "components" / "semg_core" / "include" / "micro_lda_weights.h",
        FW_DIR / "main" / "CMakeLists.txt",
        FW_DIR / "main" / "Kconfig.projbuild",
        FW_DIR / "main" / "include" / "semg_app_config.h",
        FW_DIR / "main" / "include" / "semg_hil_protocol.h",
        FW_DIR / "main" / "include" / "semg_hal_acq.h",
        FW_DIR / "main" / "include" / "semg_rtos_app.h",
        FW_DIR / "main" / "src" / "main.c",
        FW_DIR / "main" / "src" / "semg_hil_protocol.c",
        FW_DIR / "main" / "src" / "semg_hal_acq.c",
        FW_DIR / "main" / "src" / "semg_rtos_app.c",
    ]
    for path in required:
        assert path.exists(), f"Missing ESP-IDF project artifact: {path}"


def test_xtensa_esp32s3_zero_warning_compilation(tmp_path: Path) -> None:
    """Pure C99 firmware engines, weights, and HIL protocol compile on Xtensa LX7 with -Werror."""
    xtensa_gcc = shutil.which("xtensa-esp32s3-elf-gcc")
    if xtensa_gcc is None:
        pytest.skip("xtensa-esp32s3-elf-gcc not available")
    assert xtensa_gcc is not None

    harness_c = tmp_path / "harness.c"
    harness_c.write_text(C_HARNESS_SOURCE, encoding="utf-8")

    cmd = [
        xtensa_gcc,
        "-std=c99",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-c",
        str(QEMU_FW_DIR / "src" / "semg_pipeline.c"),
        str(CNN_FW_DIR / "src" / "cnn1d_engine.c"),
        str(FW_DIR / "main" / "src" / "semg_hil_protocol.c"),
        str(harness_c),
        "-I",
        str(QEMU_FW_DIR / "include"),
        "-I",
        str(CNN_FW_DIR / "include"),
        "-I",
        str(FW_DIR / "components" / "semg_core" / "include"),
        "-I",
        str(FW_DIR / "main" / "include"),
    ]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"Xtensa build failed:\n{proc.stderr}"


def _build_host_shared_lib(tmp_path: Path) -> ctypes.CDLL:
    gcc_bin = shutil.which("gcc")
    if gcc_bin is None:
        pytest.skip("Host gcc not available")
    assert gcc_bin is not None

    harness_c = tmp_path / "harness.c"
    harness_c.write_text(C_HARNESS_SOURCE, encoding="utf-8")
    so_path = tmp_path / "libsemg_esp32s3_test.so"

    cmd = [
        gcc_bin,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-shared",
        "-fPIC",
        str(QEMU_FW_DIR / "src" / "semg_pipeline.c"),
        str(CNN_FW_DIR / "src" / "cnn1d_engine.c"),
        str(FW_DIR / "main" / "src" / "semg_hil_protocol.c"),
        str(harness_c),
        "-I",
        str(QEMU_FW_DIR / "include"),
        "-I",
        str(CNN_FW_DIR / "include"),
        "-I",
        str(FW_DIR / "components" / "semg_core" / "include"),
        "-I",
        str(FW_DIR / "main" / "include"),
        "-o",
        str(so_path),
        "-lm",
    ]
    subprocess.run(cmd, check=True)
    return ctypes.CDLL(str(so_path))


def test_c99_hil_protocol_framing_and_crc(tmp_path: Path) -> None:
    """Python encoder and C99 state-machine parser agree byte-for-byte and reject errors."""
    lib = _build_host_shared_lib(tmp_path)
    lib.semg_hil_crc16_ccitt.restype = ctypes.c_uint16
    lib.semg_hil_parser_feed_byte.restype = ctypes.c_int
    lib.semg_hil_encode_frame.restype = ctypes.c_size_t

    sample_payload = struct.pack("<HH4h", 1, 4, -120, 340, 512, -999)
    py_frame = encode_hil_frame(HIL_CMD_PUSH_SAMPLES, sample_payload)

    # Verify C99 encoder matches Python encoder identically
    c_buf = (ctypes.c_uint8 * 128)()
    c_payload = (ctypes.c_uint8 * len(sample_payload)).from_buffer_copy(sample_payload)
    c_len = lib.semg_hil_encode_frame(
        HIL_CMD_PUSH_SAMPLES, c_payload, len(sample_payload), c_buf, len(c_buf)
    )
    assert bytes(c_buf[:c_len]) == py_frame

    # Verify CRC16 helper matches Python
    py_crc = crc16_ccitt(b"\x01\x00\x00")
    c_crc = lib.semg_hil_crc16_ccitt(b"\x01\x00\x00", 3, 0xFFFF)
    assert c_crc == py_crc

    # Feed noisy bytes + PING + RESET + PUSH_SAMPLES byte-by-byte
    parser = HilParserC()
    frame_out = HilFrameC()
    lib.semg_hil_parser_init(ctypes.byref(parser))

    stream = b"\x00\xff\xaa\x12" + encode_hil_frame(HIL_CMD_PING) + encode_hil_frame(
        HIL_CMD_RESET, bytes([HIL_FLAG_BYPASS_FILTER])
    ) + py_frame

    decoded_cmds = []
    for b in stream:
        st = lib.semg_hil_parser_feed_byte(ctypes.byref(parser), b, ctypes.byref(frame_out))
        if st == 1:
            decoded_cmds.append((frame_out.cmd, bytes(frame_out.payload[: frame_out.payload_len])))

    assert decoded_cmds == [
        (HIL_CMD_PING, b""),
        (HIL_CMD_RESET, bytes([HIL_FLAG_BYPASS_FILTER])),
        (HIL_CMD_PUSH_SAMPLES, sample_payload),
    ]
    assert parser.frames_ok == 3
    assert parser.frames_crc_err == 0

    # Corrupt payload byte and assert CRC error (-1)
    bad = bytearray(py_frame)
    bad[6] ^= 0x5A
    statuses = [
        lib.semg_hil_parser_feed_byte(ctypes.byref(parser), b, ctypes.byref(frame_out))
        for b in bad
    ]
    assert statuses[-1] == -1
    assert parser.frames_crc_err == 1

    # Oversized length (> 4096) triggers LEN error (-2)
    oversized = bytes([0xAA, 0x55, HIL_CMD_PUSH_SAMPLES, 0x01, 0x10])
    statuses_len = [
        lib.semg_hil_parser_feed_byte(ctypes.byref(parser), b, ctypes.byref(frame_out))
        for b in oversized
    ]
    assert statuses_len[-1] == -2
    assert parser.frames_len_err == 1


def test_c99_flash_selftest_and_sos_parity(tmp_path: Path) -> None:
    """Embedded Flash self-test stream achieves 5/5 parity for Micro-LDA Q7 and CNN-1D INT8."""
    lib = _build_host_shared_lib(tmp_path)
    out = SelfTestOutC()
    lib.run_flash_selftest_c99(ctypes.byref(out))

    assert out.windows_total == 5
    assert out.lda_matches == 5
    assert out.cnn_matches == 5

    # Verify C99 CausalSosFilter against Python CausalSosFilter with exported g_semg_sos_20_450hz
    sos_arr = (ctypes.c_float * 24)()
    lib.get_sos_coeffs_c99(sos_arr)
    sos_np = np.ctypeslib.as_array(sos_arr).reshape(4, 6).astype(np.float32)

    rng = np.random.default_rng(7)
    x_in = rng.normal(0.0, 0.5, size=(200, 4)).astype(np.float32)
    y_c = np.zeros_like(x_in)

    lib.run_sos_filter_4ch_c99(
        x_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        200,
    )
    py_sos = CausalSosFilter(sos_coefficients=sos_np, num_channels=4)
    y_py = py_sos.process_chunk(x_in)

    assert np.max(np.abs(y_c - y_py)) < 1e-5


def test_parse_tagged_json_line() -> None:
    """Telemetry parser extracts structured JSON payloads cleanly."""
    parsed = parse_tagged_json_line(
        'SEMG_DECISION: {"win":0,"engine":"micro_lda_q7","raw_pred":2,"us_c1":8.50}'
    )
    assert parsed is not None
    tag, payload = parsed
    assert tag == "SEMG_DECISION"
    assert payload["win"] == 0
    assert payload["raw_pred"] == 2
    assert parse_tagged_json_line("I (123) boot: esp-idf") is None
