/*
 * cnn1d_inference.h — Pure C99 Bare-Metal INT8 1D-CNN Inference Engine
 *
 * Implements zero-dependency symmetric INT8 quantized 1D Convolutional Neural
 * Network inference for ESP32 / ESP32-S3 (Xtensa LX6 / LX7):
 *   - 8x8 -> 32-bit integer multiply-accumulate (MAC)
 *   - Symmetric per-layer activation requantization with [-127, 127] saturation
 *   - Conv1D with configurable kernel_size, stride, and ReLU activation
 *   - MaxPool1D / AvgPool1D temporal downsampling
 *   - Global Average Pooling (GAP) across the temporal dimension
 *   - Fully Connected (Dense) classification layer with Argmax output
 *   - Strictly bounded static SRAM arena (< 16 kB)
 */

#ifndef CNN1D_INFERENCE_H
#define CNN1D_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

#define CNN1D_WINDOW_LEN   400
#define CNN1D_C1_FILTERS   16
#define CNN1D_C1_KERNEL    7
#define CNN1D_C1_STRIDE    2
#define CNN1D_C1_OUT_LEN   197  /* (400 - 7) / 2 + 1 */

#define CNN1D_POOL_SIZE    2
#define CNN1D_P1_OUT_LEN   98   /* 197 / 2 */

#define CNN1D_C2_FILTERS   24
#define CNN1D_C2_KERNEL    5
#define CNN1D_C2_STRIDE    2
#define CNN1D_C2_OUT_LEN   47   /* (98 - 5) / 2 + 1 */

#define CNN1D_MAX_CLASSES  50
#define CNN1D_ARENA_BYTES  6400 /* Static working buffer <= 16 kB (6.25 kB) */

/* Model descriptor referencing static const int8_t weight tables in Flash/RODATA */
typedef struct {
    int in_channels;       /* 4 (Ozdemir) or 12 (NinaPro DB2) */
    int num_classes;       /* 10 (Ozdemir) or 50 (NinaPro DB2) */

    /* Input channel normalization & quantization to int8 [-127, 127] */
    const float *in_mean;  /* [in_channels] */
    const float *in_inv_std; /* [in_channels] */
    float in_quant_scale;  /* s_in: q_in = clip(round(x_norm / s_in), -127, 127) */

    /* Layer 1: Conv1D (in_channels -> 16, kernel=7, stride=2, ReLU) */
    const int8_t  *conv1_weights; /* [16 * in_channels * 7] */
    const int32_t *conv1_bias;    /* [16] */
    float          conv1_eff_scale; /* (s_in * s_w1) / s_act1 */

    /* Layer 2: Conv1D (16 -> 24, kernel=5, stride=2, ReLU) */
    const int8_t  *conv2_weights; /* [24 * 16 * 5] */
    const int32_t *conv2_bias;    /* [24] */
    float          conv2_eff_scale; /* (s_act1 * s_w2) / s_act2 */

    /* Layer 3: Dense / Fully Connected (24 -> num_classes) */
    const int8_t  *fc_weights;    /* [num_classes * 24] */
    const int32_t *fc_bias;       /* [num_classes] */
    float          fc_eff_scale;  /* (s_act2 * s_wfc) */

    /* Optional pre-integrated TD auxiliary skip projection (24 + 5*C -> num_classes) */
    int            num_aux_features; /* 5 * in_channels (if hybrid skip enabled, else 0) */
    const float   *aux_mean;
    const float   *aux_inv_std;
    float          aux_quant_scale;
    const int8_t  *aux_fc_weights; /* [num_classes * num_aux_features] */
    float          aux_fc_eff_scale;
} cnn1d_model_config_t;

/* Cycle breakdown per layer inside the CNN-1D engine */
typedef struct {
    uint32_t cycles_quant_in;
    uint32_t cycles_conv1;
    uint32_t cycles_pool1;
    uint32_t cycles_conv2;
    uint32_t cycles_gap;
    uint32_t cycles_fc;
    uint32_t cycles_total;
} cnn1d_cycle_profile_t;

/* Read hardware cycle counter (CCOUNT) on Xtensa LX6 / LX7 */
static inline uint32_t cnn1d_read_ccount(void) {
#ifdef __XTENSA__
    uint32_t ccount;
    __asm__ __volatile__("rsr %0, ccount" : "=a"(ccount));
    return ccount;
#else
    return 0U;
#endif
}

/* Quantize normalized float32 window [400, C] + auxiliary vector into int8 tensors */
void cnn1d_quantize_input(
    const cnn1d_model_config_t *cfg,
    const float *raw_window,
    const float *aux_features,
    int8_t *out_x_q,
    int8_t *out_aux_q
);

/* Execute INT8 convolutional backbone (Conv1 -> Pool1 -> Conv2 -> GAP) into out_gap[24] */
void cnn1d_backbone_gap_int8(
    const cnn1d_model_config_t *cfg,
    const int8_t *x_q,
    int8_t *out_gap,
    cnn1d_cycle_profile_t *prof
);

/* Execute pure INT8 1D-CNN forward pass on pre-quantized int8 input [in_channels, 400] */
int32_t cnn1d_forward_int8(
    const cnn1d_model_config_t *cfg,
    const int8_t *x_q,
    const int8_t *aux_q,
    int32_t *out_raw_logits,
    cnn1d_cycle_profile_t *prof
);

#endif /* CNN1D_INFERENCE_H */
