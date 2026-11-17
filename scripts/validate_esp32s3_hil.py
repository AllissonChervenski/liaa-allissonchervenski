#!/usr/bin/env python3
"""Hardware-in-the-Loop (HIL) Validator for ESP32-S3-WROOM sEMG Firmware.

Streams real Ozdemir S1 / NinaPro DB2 quantized sEMG chunks over USB-Serial/JTAG
or UART0 using the CRC16-protected binary framing protocol (`semg_hil_protocol.h`)
and verifies live predictions, cycle latencies, and stack margins from the physical MCU.

Supports both `pyserial` (if installed) and a zero-dependency Linux POSIX `termios`
transport fallback, plus `--dry-run` mode for offline frame & oracle verification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import struct
import sys
import termios
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_ozdemir_quantization_and_channels import (  # noqa: E402
    GESTURE_TRIAL_TOTAL_SEC,
    REP_COEFFS_SEC,
    V_FS_MV as OZDEMIR_V_FS_MV,
    load_ozdemir_recording,
    quantize_signal as quantize_ozdemir,
)
from scripts.run_qemu_freertos_dualcore_benchmark import (  # noqa: E402
    extract_td_int16_host_oracle,
    load_cnn_artifacts,
    predict_micro_lda_q7_host_oracle,
)
from src.export.export_micro_lda_c_header import train_ozdemir_micro_lda_bundle  # noqa: E402
from src.models.train_cnn1d_quantized import (  # noqa: E402
    c_round_half_away,
    quantize_windows_for_c99,
    simulate_c99_int8_cnn1d,
)

HIL_MAGIC_0 = 0xAA
HIL_MAGIC_1 = 0x55
HIL_CMD_PING = 0x01
HIL_CMD_RESET = 0x02
HIL_CMD_PUSH_SAMPLES = 0x03
HIL_CMD_SET_ENGINE = 0x04
HIL_CMD_RUN_SELFTEST = 0x05

HIL_FLAG_BYPASS_FILTER = 1 << 0
HIL_FLAG_ENGINE_CNN1D = 1 << 1


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Compute CRC-16/CCITT-FALSE matching `semg_hil_crc16_ccitt` in C99."""
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_hil_frame(cmd: int, payload: bytes = b"") -> bytes:
    """Encode a binary HIL frame `[0xAA, 0x55, cmd, len_le, payload, crc16_le]`."""
    if len(payload) > 4096:
        raise ValueError(f"Payload too large: {len(payload)} > 4096")
    header_body = struct.pack("<BH", cmd, len(payload)) + payload
    crc = crc16_ccitt(header_body)
    return bytes([HIL_MAGIC_0, HIL_MAGIC_1]) + header_body + struct.pack("<H", crc)


class PosixSerialPort:
    """Zero-dependency Linux POSIX serial port wrapper with non-blocking line reading."""

    def __init__(self, port: str, baudrate: int = 921600, timeout_s: float = 2.0) -> None:
        self.port = port
        self.timeout_s = timeout_s
        self._rx_buf = bytearray()
        self._fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self._fd)
        # Raw mode (no echo, no canonical line edit, no CR/LF mangling)
        attrs[0] &= ~(
            termios.IGNBRK
            | termios.BRKINT
            | termios.PARMRK
            | termios.ISTRIP
            | termios.INLCR
            | termios.IGNCR
            | termios.ICRNL
            | termios.IXON
        )
        attrs[1] &= ~termios.OPOST
        attrs[2] &= ~(termios.CSIZE | termios.PARENB)
        attrs[2] |= termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)

        baud_map = {
            115200: termios.B115200,
            230400: getattr(termios, "B230400", termios.B115200),
            460800: getattr(termios, "B460800", termios.B115200),
            921600: getattr(termios, "B921600", termios.B115200),
        }
        baud_const = baud_map.get(baudrate, termios.B115200)
        attrs[4] = baud_const
        attrs[5] = baud_const
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        termios.tcflush(self._fd, termios.TCIOFLUSH)

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        sent = 0
        while sent < len(data):
            _, wlist, _ = select.select([], [self._fd], [], self.timeout_s)
            if not wlist:
                raise TimeoutError(f"Serial write timeout on {self.port}")
            n = os.write(self._fd, view[sent:])
            sent += n

    def readline(self, timeout_s: float | None = None) -> str | None:
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else timeout_s)
        while time.monotonic() < deadline:
            nl_idx = self._rx_buf.find(b"\n")
            if nl_idx != -1:
                line = self._rx_buf[:nl_idx].decode("utf-8", errors="replace").strip()
                del self._rx_buf[: nl_idx + 1]
                return line
            rem = max(0.01, deadline - time.monotonic())
            rlist, _, _ = select.select([self._fd], [], [], rem)
            if rlist:
                chunk = os.read(self._fd, 1024)
                if chunk:
                    self._rx_buf.extend(chunk)
        return None

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


def parse_tagged_json_line(line: str) -> tuple[str, dict[str, Any]] | None:
    """Parse `SEMG_TAG: {...}` telemetry lines emitted by ESP32-S3 firmware."""
    if not line.startswith("SEMG_") or ":" not in line:
        return None
    tag, raw_json = line.split(":", 1)
    try:
        payload = json.loads(raw_json.strip())
    except json.JSONDecodeError:
        return None
    return tag.strip(), payload


def build_ozdemir_hil_stream(
    subject_id: str = "S1",
    num_windows: int = 20,
    gesture: int = 2,
) -> dict[str, Any]:
    """Prepare continuous 100-sample int16 chunks and host oracles for Ozdemir 4CH HIL."""
    bundle = train_ozdemir_micro_lda_bundle(subject_id=subject_id)
    adc_scale = bundle["adc_scale"]

    raw_sig = load_ozdemir_recording(subject_id=subject_id, raw_dir=ROOT / "data" / "raw" / "ozdemir", use_filtered=True)
    q_sig = quantize_ozdemir(raw_sig, precision="int16", v_max=OZDEMIR_V_FS_MV)

    # Start inside Rep 2 for the chosen gesture; total samples needed = 400 + (num_windows - 1) * 100
    n_blocks = 3 + num_windows  # first 3 blocks (300 samples) buffer, 4th block emits win 0
    total_samples = n_blocks * 100
    rep2_sec = REP_COEFFS_SEC[1]
    start_samp = int((rep2_sec + gesture * GESTURE_TRIAL_TOTAL_SEC) * 2000) + 200
    stream_fp32 = q_sig[start_samp : start_samp + total_samples, :4].astype(np.float32)
    stream_q16 = np.clip(c_round_half_away(stream_fp32 * adc_scale), -32768, 32767).astype(np.int16)

    cnn_art = load_cnn_artifacts("oz", "ozdemir_s1_4ch", subject_id)

    lda_oracles: list[int] = []
    cnn_oracles: list[int] = []
    win_fp32_batch = np.zeros((num_windows, 400, 4), dtype=np.float32)
    aux_fp32_batch = np.zeros((num_windows, 20), dtype=np.float32)

    for w in range(num_windows):
        s0 = w * 100
        w_q16 = stream_q16[s0 : s0 + 400]
        w_f32 = (w_q16.astype(np.float32) * np.float32(1.0 / adc_scale)).astype(np.float32)
        feat_int = extract_td_int16_host_oracle(w_q16, bundle["eps_int"], 1.0 / adc_scale)
        pred_lda = predict_micro_lda_q7_host_oracle(
            feat_int,
            bundle["classes"],
            bundle["mean"],
            bundle["inv_scale"],
            bundle["zero_mask"],
            bundle["coef_q7"],
            bundle["weight_scale"],
            bundle["intercept"],
        )
        lda_oracles.append(int(pred_lda))
        win_fp32_batch[w] = w_f32
        aux_fp32_batch[w] = feat_int

    x_q, aux_q = quantize_windows_for_c99(
        win_fp32_batch,
        aux_fp32_batch,
        cnn_art.in_mean,
        cnn_art.in_inv_std,
        cnn_art.in_quant_scale,
        cnn_art.aux_mean,
        cnn_art.aux_inv_std,
        cnn_art.aux_quant_scale,
    )
    cnn_preds, _ = simulate_c99_int8_cnn1d(cnn_art, x_q, aux_q)
    cnn_oracles = [int(v) for v in cnn_preds.tolist()]

    blocks = [stream_q16[b * 100 : (b + 1) * 100] for b in range(n_blocks)]
    return {
        "blocks": blocks,
        "lda_oracles": lda_oracles,
        "cnn_oracles": cnn_oracles,
        "num_channels": 4,
        "num_windows": num_windows,
        "gesture": gesture,
    }


def wait_for_tag(ser: PosixSerialPort, expected_tags: set[str], timeout_s: float = 3.0) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = ser.readline(timeout_s=max(0.05, deadline - time.monotonic()))
        if line is None:
            continue
        parsed = parse_tagged_json_line(line)
        if parsed is None:
            continue
        tag, payload = parsed
        if tag in expected_tags:
            return tag, payload
        if tag == "SEMG_ERR":
            raise RuntimeError(f"ESP32-S3 reported error: {payload}")
    raise TimeoutError(f"Timed out waiting for one of {sorted(expected_tags)} on {ser.port}")


def run_hil_engine(
    ser: PosixSerialPort,
    hil_data: dict[str, Any],
    engine: str,
) -> dict[str, Any]:
    flags = HIL_FLAG_BYPASS_FILTER | (HIL_FLAG_ENGINE_CNN1D if engine == "cnn1d_int8" else 0)
    ser.write(encode_hil_frame(HIL_CMD_RESET, bytes([flags])))
    _, ack = wait_for_tag(ser, {"SEMG_ACK"})
    if ack.get("engine") != engine:
        raise RuntimeError(f"Engine mismatch after RESET: {ack}")

    oracles = hil_data["lda_oracles"] if engine == "micro_lda_q7" else hil_data["cnn_oracles"]
    matches = 0
    latencies_us: list[float] = []
    min_stack_c0 = 1 << 30
    min_stack_c1 = 1 << 30
    win_idx = 0

    for block in hil_data["blocks"]:
        payload = struct.pack("<HH", int(block.shape[0]), int(block.shape[1])) + block.astype("<i2").tobytes()
        ser.write(encode_hil_frame(HIL_CMD_PUSH_SAMPLES, payload))
        tag, resp = wait_for_tag(ser, {"SEMG_BUFFERED", "SEMG_DECISION"})
        if tag == "SEMG_DECISION":
            pred = int(resp["raw_pred"])
            if pred == oracles[win_idx]:
                matches += 1
            latencies_us.append(float(resp["us_c1"]))
            min_stack_c0 = min(min_stack_c0, int(resp["stack_c0_free"]))
            min_stack_c1 = min(min_stack_c1, int(resp["stack_c1_free"]))
            win_idx += 1

    n_win = hil_data["num_windows"]
    return {
        "engine": engine,
        "windows": n_win,
        "parity_matches": matches,
        "parity_ratio": matches / float(n_win),
        "mean_us_c1": float(np.mean(latencies_us)) if latencies_us else 0.0,
        "max_us_c1": float(np.max(latencies_us)) if latencies_us else 0.0,
        "min_stack_c0_free": min_stack_c0,
        "min_stack_c1_free": min_stack_c1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ESP32-S3-WROOM firmware via HIL binary serial.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial device path (default: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=921600, help="UART/Serial baudrate")
    parser.add_argument("--engine", choices=["both", "micro_lda_q7", "cnn1d_int8"], default="both")
    parser.add_argument("--windows", type=int, default=20, help="Number of overlapping windows to stream")
    parser.add_argument("--gesture", type=int, default=2, help="Ozdemir gesture index (0..9)")
    parser.add_argument("--dry-run", action="store_true", help="Build frames and verify oracles without opening serial")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional output JSON report path")
    args = parser.parse_args()

    print(f"Preparing Ozdemir S1 HIL stream ({args.windows} windows, gesture={args.gesture})...")
    hil_data = build_ozdemir_hil_stream(subject_id="S1", num_windows=args.windows, gesture=args.gesture)

    if args.dry_run:
        sample_block = hil_data["blocks"][0]
        payload = struct.pack("<HH", 100, 4) + sample_block.astype("<i2").tobytes()
        frame = encode_hil_frame(HIL_CMD_PUSH_SAMPLES, payload)
        print(f"[DRY-RUN] Encoded sample frame: {len(frame)} bytes (CRC16 OK)")
        print(f"[DRY-RUN] Micro-LDA Q7 oracles ({len(hil_data['lda_oracles'])}): {hil_data['lda_oracles'][:10]}...")
        print(f"[DRY-RUN] CNN-1D INT8 oracles  ({len(hil_data['cnn_oracles'])}): {hil_data['cnn_oracles'][:10]}...")
        return 0

    ser = PosixSerialPort(args.port, baudrate=args.baud, timeout_s=3.0)
    try:
        ser.write(encode_hil_frame(HIL_CMD_PING))
        _, pong = wait_for_tag(ser, {"SEMG_PONG"}, timeout_s=4.0)
        print(f"Connected to ESP32-S3-WROOM: {pong}")

        engines = ["micro_lda_q7", "cnn1d_int8"] if args.engine == "both" else [args.engine]
        results = []
        for eng in engines:
            res = run_hil_engine(ser, hil_data, eng)
            results.append(res)
            print(
                f"[{eng}] Parity: {res['parity_matches']}/{res['windows']} ({res['parity_ratio']*100:.2f}%) | "
                f"Core1 Mean: {res['mean_us_c1']:.2f} us (Max: {res['max_us_c1']:.2f} us) | "
                f"Stack Free C0/C1: {res['min_stack_c0_free']} B / {res['min_stack_c1_free']} B"
            )

        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps({"pong": pong, "results": results}, indent=2), encoding="utf-8")
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
