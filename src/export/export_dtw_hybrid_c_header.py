#!/usr/bin/env python3
"""Export Static DTW Templates, Keogh Envelopes, and Hybrid (LDA+CNN+DTW) Weights to C99.

Generates:
  - `embedded/firmware/include/dtw_hybrid_weights.h`
  - `firmware/esp32s3_wroom/components/semg_core/include/dtw_hybrid_weights.h`
  - `reports/dtw_hybrid_benchmark/test_vectors_ozdemir.bin`
  - `reports/dtw_hybrid_benchmark/test_vectors_ninapro.bin`

Every binary vector file stores the exact host-computed predictions for all 10 models
(`M01..M10`), `win_q16`, pre-quantized `x_q`, `aux_q`, `env_q`, and `td_features`, enabling
deterministic cycle-exact benchmarking and bit-exact verification inside Espressif QEMU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_qemu_freertos_dualcore_benchmark import load_cnn_artifacts  # noqa: E402
from src.export.export_cnn1d_c_header import format_float_array, format_int_array  # noqa: E402
from src.export.export_micro_lda_c_header import (  # noqa: E402
    train_ninapro_micro_lda_bundle,
    train_ozdemir_micro_lda_bundle,
)
from src.models.train_cnn1d_quantized import quantize_windows_for_c99  # noqa: E402
from src.models.train_dtw_and_hybrids import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DTW_ENV_LEN,
    DTW_SAKOE_W,
    evaluate_all_10_models_host,
    extract_dtw_envelope_int8_batch,
    extract_td_int16_batch,
    load_dataset_splits,
    windows_f32_to_q16,
)

HEADER_MAGIC_DTWH = 0x44545748  # "DTWH"


def _dataset_c_block(
    prefix_macro: str,
    prefix_var: str,
    in_channels: int,
    num_classes: int,
    data: np.lib.npyio.NpzFile,
    key_prefix: str,
) -> list[str]:
    num_td = 5 * in_channels
    num_hyb_feat = 24 + num_td
    env_quant_s = float(data[f"{key_prefix}_env_quant_scale"][0])
    casc_tau = float(data[f"{key_prefix}_cascade_tau"][0])
    casc_use_cnn = int(data[f"{key_prefix}_cascade_use_cnn"][0])
    gamma_lda = float(data[f"{key_prefix}_gamma_dtw_lda"][0])
    gamma_cnn = float(data[f"{key_prefix}_gamma_dtw_cnn"][0])
    tri_tau = float(data[f"{key_prefix}_tri_tau"][0])
    gamma_tri = float(data[f"{key_prefix}_gamma_tri"][0])

    lines: list[str] = [
        f"#define {prefix_macro}_DTW_QUANT_SCALE ({env_quant_s:.9e}f)",
        "",
        format_float_array(f"g_{prefix_var}_dtw_env_mean", data[f"{key_prefix}_env_mean"]),
        format_float_array(f"g_{prefix_var}_dtw_env_inv_std", data[f"{key_prefix}_env_inv_std"]),
        format_int_array(f"g_{prefix_var}_dtw_templates_q", "int8_t", data[f"{key_prefix}_dtw_templates_q"]),
        format_int_array(f"g_{prefix_var}_dtw_env_upper", "int8_t", data[f"{key_prefix}_dtw_env_upper"]),
        format_int_array(f"g_{prefix_var}_dtw_env_lower", "int8_t", data[f"{key_prefix}_dtw_env_lower"]),
        "",
        f"static const dtw_model_config_t g_dtw_{prefix_var}_config = {{",
        f"    .num_channels    = {in_channels},",
        f"    .num_classes     = {num_classes},",
        f"    .env_len         = {DTW_ENV_LEN},",
        f"    .band_w          = {DTW_SAKOE_W},",
        f"    .env_mean        = g_{prefix_var}_dtw_env_mean,",
        f"    .env_inv_std     = g_{prefix_var}_dtw_env_inv_std,",
        f"    .env_quant_scale = {prefix_macro}_DTW_QUANT_SCALE,",
        f"    .templates_q     = g_{prefix_var}_dtw_templates_q,",
        f"    .env_upper       = g_{prefix_var}_dtw_env_upper,",
        f"    .env_lower       = g_{prefix_var}_dtw_env_lower",
        "};",
        "",
        format_int_array(f"g_{prefix_var}_hyb1_classes", "int32_t", data[f"{key_prefix}_hyb1_classes"]),
        format_float_array(f"g_{prefix_var}_hyb1_mean", data[f"{key_prefix}_hyb1_mean"]),
        format_float_array(f"g_{prefix_var}_hyb1_inv_scale", data[f"{key_prefix}_hyb1_inv_scale"]),
        format_int_array(f"g_{prefix_var}_hyb1_zero_mask", "uint8_t", data[f"{key_prefix}_hyb1_zero_mask"]),
        format_int_array(f"g_{prefix_var}_hyb1_coef_q7", "int8_t", data[f"{key_prefix}_hyb1_coef_q7"]),
        format_float_array(f"g_{prefix_var}_hyb1_w_scale", data[f"{key_prefix}_hyb1_w_scale"]),
        format_float_array(f"g_{prefix_var}_hyb1_intercept", data[f"{key_prefix}_hyb1_intercept"]),
        "",
        f"static const hybrid_lda_head_t g_hyb1_{prefix_var}_head = {{",
        f"    .num_channels  = {in_channels},",
        f"    .num_features  = {num_hyb_feat},",
        f"    .num_classes   = {num_classes},",
        f"    .classes       = g_{prefix_var}_hyb1_classes,",
        f"    .mean          = g_{prefix_var}_hyb1_mean,",
        f"    .inv_scale     = g_{prefix_var}_hyb1_inv_scale,",
        f"    .zero_var_mask = g_{prefix_var}_hyb1_zero_mask,",
        f"    .coef_q7       = g_{prefix_var}_hyb1_coef_q7,",
        f"    .weight_scale  = g_{prefix_var}_hyb1_w_scale,",
        f"    .intercept     = g_{prefix_var}_hyb1_intercept",
        "};",
        "",
        f"static const hybrid_fusion_config_t g_fusion_{prefix_var}_config = {{",
        f"    .cascade_margin_threshold = {casc_tau:.9e}f,",
        f"    .cascade_use_cnn1d_tier2  = {casc_use_cnn}U,",
        f"    .gamma_dtw_lda            = {gamma_lda:.9e}f,",
        f"    .gamma_dtw_cnn            = {gamma_cnn:.9e}f,",
        f"    .tri_margin_threshold     = {tri_tau:.9e}f,",
        f"    .gamma_tri_hybrid         = {gamma_tri:.9e}f",
        "};",
        "",
    ]
    return lines


def generate_dtw_hybrid_header(npz_path: Path, out_headers: list[Path]) -> None:
    """Render static C99 weight header from `dtw_hybrid_model_artifacts.npz`."""
    data = np.load(npz_path)
    chunks: list[str] = [
        "/*",
        " * dtw_hybrid_weights.h — Auto-generated Static DTW Templates, Keogh Envelopes,",
        " *                        Hybrid 1 LDA Heads & Multi-Modal Cascade Configs",
        " *",
        " * Generated by: src/export/export_dtw_hybrid_c_header.py",
        " */",
        "",
        "#ifndef DTW_HYBRID_WEIGHTS_H",
        "#define DTW_HYBRID_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        '#include "dtw_hybrid_engine.h"',
        "",
        "/* ==========================================================================",
        " * 1. OZDEMIR ET AL. (4 CHANNELS, 10 CLASSES)",
        " * ========================================================================== */",
    ]
    chunks.extend(_dataset_c_block("OZDEMIR", "ozdemir", 4, 10, data, "oz"))
    chunks.extend([
        "/* ==========================================================================",
        " * 2. NINAPRO DB2 (12 CHANNELS, 50 CLASSES)",
        " * ========================================================================== */",
    ])
    chunks.extend(_dataset_c_block("NINAPRO", "ninapro", 12, 50, data, "ni"))
    chunks.extend(["#endif /* DTW_HYBRID_WEIGHTS_H */", ""])

    content = "\n".join(chunks)
    for out_p in out_headers:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(content, encoding="utf-8")


def write_dtw_hybrid_test_vectors(
    bin_path: Path,
    dataset_code: int,
    bundle: dict[str, Any],
    win_q16: np.ndarray,
    labels: np.ndarray,
    preds_mat: np.ndarray,
    num_vectors: int | None = None,
) -> None:
    """Write binary QEMU test vector file containing inputs and M01..M10 host oracles."""
    n_tot = len(labels)
    if num_vectors is not None and num_vectors < n_tot:
        # Stratified uniform sample preserves all classes across repetitions
        idx = np.linspace(0, n_tot - 1, num_vectors, dtype=int)
    else:
        idx = np.arange(n_tot)

    sub_q16 = win_q16[idx]
    sub_lbl = labels[idx]
    sub_preds = preds_mat[idx]
    n_vec, w_len, n_ch = sub_q16.shape
    lda_b = bundle["lda"]
    cnn_art = bundle["cnn"]
    dtw_b = bundle["dtw"]
    inv_adc = 1.0 / float(lda_b["adc_scale"])

    win_fp32 = (sub_q16.astype(np.float32) * np.float32(inv_adc)).astype(np.float32)
    td_feat = extract_td_int16_batch(sub_q16, lda_b["eps_int"], inv_adc)
    x_q, aux_q = quantize_windows_for_c99(
        win_fp32,
        td_feat,
        cnn_art.in_mean,
        cnn_art.in_inv_std,
        cnn_art.in_quant_scale,
        cnn_art.aux_mean,
        cnn_art.aux_inv_std,
        cnn_art.aux_quant_scale,
    )
    env_q = extract_dtw_envelope_int8_batch(
        sub_q16, inv_adc, dtw_b["env_mean"], dtw_b["env_inv_std"], dtw_b["env_quant_scale"]
    )

    header = struct.pack(
        "<IIIIIIf",
        HEADER_MAGIC_DTWH,
        dataset_code,
        n_ch,
        cnn_art.num_classes,
        cnn_art.num_aux,
        n_vec,
        float(inv_adc),
    )
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open("wb") as f:
        f.write(header)
        for i in range(n_vec):
            f.write(struct.pack("<i", int(sub_lbl[i])))
            f.write(np.ascontiguousarray(sub_preds[i], dtype="<i4").tobytes())
            f.write(np.ascontiguousarray(sub_q16[i], dtype="<i2").tobytes())
            f.write(np.ascontiguousarray(x_q[i], dtype="<i1").tobytes())
            f.write(np.ascontiguousarray(aux_q[i], dtype="<i1").tobytes())
            f.write(np.ascontiguousarray(env_q[i], dtype="<i1").tobytes())
            f.write(np.ascontiguousarray(td_feat[i], dtype="<f4").tobytes())


def load_bundle_from_npz(
    npz_path: Path,
    key_prefix: str,
    lda_bundle: dict[str, Any],
    cnn_art: Any,
) -> dict[str, Any]:
    """Reconstruct complete Python oracle bundle from saved NPZ artifacts."""
    data = np.load(npz_path)
    return {
        "lda": lda_bundle,
        "cnn": cnn_art,
        "dtw": {
            "env_mean": data[f"{key_prefix}_env_mean"].astype(np.float32),
            "env_inv_std": data[f"{key_prefix}_env_inv_std"].astype(np.float32),
            "env_quant_scale": float(data[f"{key_prefix}_env_quant_scale"][0]),
            "templates_q": data[f"{key_prefix}_dtw_templates_q"].astype(np.int8),
            "env_upper": data[f"{key_prefix}_dtw_env_upper"].astype(np.int8),
            "env_lower": data[f"{key_prefix}_dtw_env_lower"].astype(np.int8),
        },
        "hyb1": {
            "classes": data[f"{key_prefix}_hyb1_classes"].astype(np.int32),
            "mean": data[f"{key_prefix}_hyb1_mean"].astype(np.float32),
            "inv_scale": data[f"{key_prefix}_hyb1_inv_scale"].astype(np.float32),
            "zero_mask": data[f"{key_prefix}_hyb1_zero_mask"].astype(np.uint8),
            "coef_q7": data[f"{key_prefix}_hyb1_coef_q7"].astype(np.int8),
            "weight_scale": data[f"{key_prefix}_hyb1_w_scale"].astype(np.float32),
            "intercept": data[f"{key_prefix}_hyb1_intercept"].astype(np.float32),
        },
        "fusion": {
            "cascade_margin_threshold": float(data[f"{key_prefix}_cascade_tau"][0]),
            "cascade_use_cnn1d_tier2": int(data[f"{key_prefix}_cascade_use_cnn"][0]),
            "gamma_dtw_lda": float(data[f"{key_prefix}_gamma_dtw_lda"][0]),
            "gamma_dtw_cnn": float(data[f"{key_prefix}_gamma_dtw_cnn"][0]),
            "tri_margin_threshold": float(data[f"{key_prefix}_tri_tau"][0]),
            "gamma_tri_hybrid": float(data[f"{key_prefix}_gamma_tri"][0]),
        },
    }


def _subsample_stratified(
    win_q16: np.ndarray,
    labels: np.ndarray,
    num_vectors: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_vectors is not None and 0 < num_vectors < len(labels):
        idx = np.linspace(0, len(labels) - 1, num_vectors, dtype=int)
        return win_q16[idx], labels[idx]
    return win_q16, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Export DTW + Hybrid C99 headers & QEMU test vectors.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_OUT_DIR / "dtw_hybrid_model_artifacts.npz")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ozdemir-subject", default="S1")
    parser.add_argument("--ninapro-subject", default="S4")
    parser.add_argument("--ozdemir-qemu-windows", type=int, default=500)
    parser.add_argument("--ninapro-qemu-windows", type=int, default=500)
    args = parser.parse_args()

    out_headers = [
        ROOT / "embedded" / "firmware" / "include" / "dtw_hybrid_weights.h",
        ROOT / "firmware" / "esp32s3_wroom" / "components" / "semg_core" / "include" / "dtw_hybrid_weights.h",
    ]
    generate_dtw_hybrid_header(args.npz, out_headers)
    print(f"Generated C99 headers: {[str(p) for p in out_headers]}")

    oz_ds, ni_ds = load_dataset_splits(args.ozdemir_subject, args.ninapro_subject)
    oz_lda = train_ozdemir_micro_lda_bundle(args.ozdemir_subject)
    ni_lda = train_ninapro_micro_lda_bundle(args.ninapro_subject)
    oz_cnn = load_cnn_artifacts("oz", "Ozdemir_4CH", args.ozdemir_subject)
    ni_cnn = load_cnn_artifacts("ni", "NinaPro_DB2_12CH", args.ninapro_subject)

    oz_bundle = load_bundle_from_npz(args.npz, "oz", oz_lda, oz_cnn)
    oz_q16_all = windows_f32_to_q16(oz_ds["te_win"], float(oz_lda["adc_scale"]))
    oz_q16, oz_lbl = _subsample_stratified(oz_q16_all, oz_ds["te_lbl"], args.ozdemir_qemu_windows)
    _, oz_preds = evaluate_all_10_models_host(oz_bundle, oz_q16, oz_lbl, include_classic_on2=True)
    write_dtw_hybrid_test_vectors(
        args.out_dir / "test_vectors_ozdemir.bin", 0, oz_bundle, oz_q16, oz_lbl, oz_preds, num_vectors=None
    )

    ni_bundle = load_bundle_from_npz(args.npz, "ni", ni_lda, ni_cnn)
    ni_q16_all = windows_f32_to_q16(ni_ds["te_win"], float(ni_lda["adc_scale"]))
    ni_q16, ni_lbl = _subsample_stratified(ni_q16_all, ni_ds["te_lbl"], args.ninapro_qemu_windows)
    _, ni_preds = evaluate_all_10_models_host(ni_bundle, ni_q16, ni_lbl, include_classic_on2=True)
    write_dtw_hybrid_test_vectors(
        args.out_dir / "test_vectors_ninapro.bin",
        1,
        ni_bundle,
        ni_q16,
        ni_lbl,
        ni_preds,
        num_vectors=None,
    )
    print(f"Exported QEMU binary test vectors ({len(oz_lbl)} Ozdemir, {len(ni_lbl)} NinaPro) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
