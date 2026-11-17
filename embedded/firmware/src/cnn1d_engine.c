/*
 * cnn1d_engine.c — Pure C99 Bare-Metal INT8 1D-CNN Inference Implementation
 *
 * Implements zero-dependency symmetric INT8 quantized 1D Convolutional Neural
 * Network inference for ESP32 / ESP32-S3 (Xtensa LX6 / LX7):
 *   - 8x8 -> 32-bit integer multiply-accumulate (MAC)
 *   - Fixed-point Q24 multiplier requantization (zero runtime FPU dependency inside conv/pool/dense)
 *   - Conv1D Layer 1 + ReLU + MaxPool1D (factor 2)
 *   - Conv1D Layer 2 + ReLU + Global Average Pooling (GAP)
 *   - Dense / Fully-Connected Classifier + Argmax
 *   - Static SRAM working tensor arena: 4,720 bytes (<= 16 kB budget)
 */

#include "../include/cnn1d_inference.h"
#include <math.h>
#include <string.h>

/*
 * Static SRAM Working Arena (4,720 bytes total < 16 kB limit):
 *   - g_arena_buf_a: [16 * 197] = 3,152 bytes (Conv1 output)
 *   - g_arena_buf_b: [16 * 98]  = 1,568 bytes (MaxPool1 output / Conv2 output [24 * 47 = 1,128 B])
 */
static int8_t g_arena_buf_a[CNN1D_C1_FILTERS * CNN1D_C1_OUT_LEN];
static int8_t g_arena_buf_b[CNN1D_C1_FILTERS * CNN1D_P1_OUT_LEN];

static inline int8_t clip_int8_relu(int32_t v) {
    if (v <= 0) return 0;
    if (v > 127) return 127;
    return (int8_t)v;
}

static inline int8_t clip_int8_sym(int32_t v) {
    if (v < -127) return -127;
    if (v > 127) return 127;
    return (int8_t)v;
}

/*
 * Fixed-point rounding scale: round(acc * mult_q24 / 2^24)
 * Uses 64-bit integer product with symmetric half-way rounding, matching
 * integer quantized neural network specifications deterministically across Host and Xtensa.
 */
static inline int32_t requantize_q24(int32_t acc, int32_t mult_q24) {
    int64_t prod = (int64_t)acc * (int64_t)mult_q24;
    if (prod >= 0) {
        return (int32_t)((prod + (1LL << 23)) >> 24);
    } else {
        return (int32_t)(-(((-prod) + (1LL << 23)) >> 24));
    }
}

void cnn1d_quantize_input(
    const cnn1d_model_config_t *cfg,
    const float *raw_window,
    const float *aux_features,
    int8_t *out_x_q,
    int8_t *out_aux_q
) {
    const int n_ch = cfg->in_channels;
    const float inv_s_in = (cfg->in_quant_scale > 0.0f) ? (1.0f / cfg->in_quant_scale) : 1.0f;

    /* Transpose & quantize [400, C] -> [C, 400] for contiguous temporal Conv1D */
    for (int c = 0; c < n_ch; c++) {
        const float mean_c = cfg->in_mean[c];
        const float inv_std_c = cfg->in_inv_std[c];
        int8_t *dst_c = &out_x_q[c * CNN1D_WINDOW_LEN];

        for (int t = 0; t < CNN1D_WINDOW_LEN; t++) {
            float x_norm = (raw_window[t * n_ch + c] - mean_c) * inv_std_c;
            int32_t q = (int32_t)lrintf(x_norm * inv_s_in);
            dst_c[t] = clip_int8_sym(q);
        }
    }

    if (cfg->num_aux_features > 0 && aux_features != NULL && out_aux_q != NULL) {
        const float inv_s_aux = (cfg->aux_quant_scale > 0.0f) ? (1.0f / cfg->aux_quant_scale) : 1.0f;
        for (int j = 0; j < cfg->num_aux_features; j++) {
            float z = (aux_features[j] - cfg->aux_mean[j]) * cfg->aux_inv_std[j];
            int32_t q = (int32_t)lrintf(z * inv_s_aux);
            out_aux_q[j] = clip_int8_sym(q);
        }
    }
}

/*
 * Layer 1: Conv1D (in_channels -> 16, kernel=7, stride=2, ReLU)
 * Input:  x_q [in_channels, 400]
 * Output: out [16, 197]
 */
static void cnn1d_layer_conv1_relu(
    int in_channels,
    const int8_t *x_q,
    const int8_t *weights,
    const int32_t *bias,
    int32_t mult_q24,
    int8_t *out
) {
    for (int f = 0; f < CNN1D_C1_FILTERS; f++) {
        const int32_t b_f = bias[f];
        const int8_t *w_f = &weights[f * in_channels * CNN1D_C1_KERNEL];
        int8_t *out_f = &out[f * CNN1D_C1_OUT_LEN];

        for (int t_out = 0; t_out < CNN1D_C1_OUT_LEN; t_out++) {
            const int t_in_base = t_out * CNN1D_C1_STRIDE;
            int32_t acc = b_f;

            for (int c = 0; c < in_channels; c++) {
                const int8_t *x_c = &x_q[c * CNN1D_WINDOW_LEN + t_in_base];
                const int8_t *w_fc = &w_f[c * CNN1D_C1_KERNEL];
                acc += (int32_t)x_c[0] * (int32_t)w_fc[0]
                     + (int32_t)x_c[1] * (int32_t)w_fc[1]
                     + (int32_t)x_c[2] * (int32_t)w_fc[2]
                     + (int32_t)x_c[3] * (int32_t)w_fc[3]
                     + (int32_t)x_c[4] * (int32_t)w_fc[4]
                     + (int32_t)x_c[5] * (int32_t)w_fc[5]
                     + (int32_t)x_c[6] * (int32_t)w_fc[6];
            }

            int32_t rq = requantize_q24(acc, mult_q24);
            out_f[t_out] = clip_int8_relu(rq);
        }
    }
}

/*
 * Pooling Layer: MaxPool1D (kernel=2, stride=2)
 * Input:  in  [16, 197]
 * Output: out [16, 98]
 */
static void cnn1d_layer_maxpool1d(const int8_t *in, int8_t *out) {
    for (int f = 0; f < CNN1D_C1_FILTERS; f++) {
        const int8_t *in_f = &in[f * CNN1D_C1_OUT_LEN];
        int8_t *out_f = &out[f * CNN1D_P1_OUT_LEN];
        for (int t = 0; t < CNN1D_P1_OUT_LEN; t++) {
            int8_t v0 = in_f[t * 2];
            int8_t v1 = in_f[t * 2 + 1];
            out_f[t] = (v0 >= v1) ? v0 : v1;
        }
    }
}

/*
 * Layer 2: Conv1D (16 -> 24, kernel=5, stride=2, ReLU)
 * Input:  in  [16, 98]
 * Output: out [24, 47]
 */
static void cnn1d_layer_conv2_relu(
    const int8_t *in,
    const int8_t *weights,
    const int32_t *bias,
    int32_t mult_q24,
    int8_t *out
) {
    for (int f = 0; f < CNN1D_C2_FILTERS; f++) {
        const int32_t b_f = bias[f];
        const int8_t *w_f = &weights[f * CNN1D_C1_FILTERS * CNN1D_C2_KERNEL];
        int8_t *out_f = &out[f * CNN1D_C2_OUT_LEN];

        for (int t_out = 0; t_out < CNN1D_C2_OUT_LEN; t_out++) {
            const int t_in_base = t_out * CNN1D_C2_STRIDE;
            int32_t acc = b_f;

            for (int c = 0; c < CNN1D_C1_FILTERS; c++) {
                const int8_t *in_c = &in[c * CNN1D_P1_OUT_LEN + t_in_base];
                const int8_t *w_fc = &w_f[c * CNN1D_C2_KERNEL];
                acc += (int32_t)in_c[0] * (int32_t)w_fc[0]
                     + (int32_t)in_c[1] * (int32_t)w_fc[1]
                     + (int32_t)in_c[2] * (int32_t)w_fc[2]
                     + (int32_t)in_c[3] * (int32_t)w_fc[3]
                     + (int32_t)in_c[4] * (int32_t)w_fc[4];
            }

            int32_t rq = requantize_q24(acc, mult_q24);
            out_f[t_out] = clip_int8_relu(rq);
        }
    }
}

/*
 * Global Average Pooling (GAP) across T=47 steps
 * Input:  in  [24, 47]
 * Output: gap [24] in int8_t (rounded integer average)
 */
static void cnn1d_layer_gap(const int8_t *in, int8_t *gap_out) {
    for (int f = 0; f < CNN1D_C2_FILTERS; f++) {
        const int8_t *in_f = &in[f * CNN1D_C2_OUT_LEN];
        int32_t sum = 0;
        for (int t = 0; t < CNN1D_C2_OUT_LEN; t++) {
            sum += (int32_t)in_f[t];
        }
        /* Symmetric integer rounding division by 47 */
        int32_t avg = (sum + (CNN1D_C2_OUT_LEN / 2)) / CNN1D_C2_OUT_LEN;
        gap_out[f] = clip_int8_relu(avg);
    }
}

/*
 * Execute INT8 convolutional backbone (Conv1 -> MaxPool1 -> Conv2 -> GAP)
 * Output: out_gap [24] in int8_t
 */
void cnn1d_backbone_gap_int8(
    const cnn1d_model_config_t *cfg,
    const int8_t *x_q,
    int8_t *out_gap,
    cnn1d_cycle_profile_t *prof
) {
    uint32_t t0 = cnn1d_read_ccount();

    const int32_t mult1_q24 = (int32_t)lrintf(cfg->conv1_eff_scale * (float)(1 << 24));
    const int32_t mult2_q24 = (int32_t)lrintf(cfg->conv2_eff_scale * (float)(1 << 24));

    /* 1. Conv1D + ReLU -> g_arena_buf_a [16, 197] */
    uint32_t tc1_0 = cnn1d_read_ccount();
    cnn1d_layer_conv1_relu(
        cfg->in_channels,
        x_q,
        cfg->conv1_weights,
        cfg->conv1_bias,
        mult1_q24,
        g_arena_buf_a
    );
    uint32_t tc1_1 = cnn1d_read_ccount();

    /* 2. MaxPool1D (2) -> g_arena_buf_b [16, 98] */
    uint32_t tp1_0 = cnn1d_read_ccount();
    cnn1d_layer_maxpool1d(g_arena_buf_a, g_arena_buf_b);
    uint32_t tp1_1 = cnn1d_read_ccount();

    /* 3. Conv1D Layer 2 + ReLU -> g_arena_buf_a [24, 47] */
    uint32_t tc2_0 = cnn1d_read_ccount();
    cnn1d_layer_conv2_relu(
        g_arena_buf_b,
        cfg->conv2_weights,
        cfg->conv2_bias,
        mult2_q24,
        g_arena_buf_a
    );
    uint32_t tc2_1 = cnn1d_read_ccount();

    /* 4. Global Average Pooling -> out_gap [24] */
    uint32_t tgap_0 = cnn1d_read_ccount();
    cnn1d_layer_gap(g_arena_buf_a, out_gap);
    uint32_t tgap_1 = cnn1d_read_ccount();

    if (prof != NULL) {
        prof->cycles_quant_in = 0U;
        prof->cycles_conv1 = tc1_1 - tc1_0;
        prof->cycles_pool1 = tp1_1 - tp1_0;
        prof->cycles_conv2 = tc2_1 - tc2_0;
        prof->cycles_gap   = tgap_1 - tgap_0;
        prof->cycles_fc    = 0U;
        prof->cycles_total = tgap_1 - t0;
    }
}

/*
 * Execute complete INT8 1D-CNN forward pass and return predicted class index (Argmax)
 */
int32_t cnn1d_forward_int8(
    const cnn1d_model_config_t *cfg,
    const int8_t *x_q,
    const int8_t *aux_q,
    int32_t *out_raw_logits,
    cnn1d_cycle_profile_t *prof
) {
    uint32_t t0 = cnn1d_read_ccount();

    int8_t gap_feat[CNN1D_C2_FILTERS];
    cnn1d_backbone_gap_int8(cfg, x_q, gap_feat, prof);

    const int32_t mult_aux_q16 = (int32_t)lrintf(cfg->aux_fc_eff_scale * 65536.0f);

    /* 5. Dense / Fully-Connected Layer + Argmax */
    uint32_t tfc_0 = cnn1d_read_ccount();
    int32_t best_cls = 0;
    int32_t best_logit = -2147483647;
    const int n_cls = cfg->num_classes;
    const int n_aux = cfg->num_aux_features;

    for (int k = 0; k < n_cls; k++) {
        int32_t acc = cfg->fc_bias[k];
        const int8_t *w_fc_row = &cfg->fc_weights[k * CNN1D_C2_FILTERS];
        for (int f = 0; f < CNN1D_C2_FILTERS; f++) {
            acc += (int32_t)gap_feat[f] * (int32_t)w_fc_row[f];
        }

        if (n_aux > 0 && aux_q != NULL && cfg->aux_fc_weights != NULL) {
            int32_t aux_acc = 0;
            const int8_t *w_aux_row = &cfg->aux_fc_weights[k * n_aux];
            for (int j = 0; j < n_aux; j++) {
                aux_acc += (int32_t)aux_q[j] * (int32_t)w_aux_row[j];
            }
            /* Scale auxiliary branch into same fixed-point domain as main FC branch */
            acc += (int32_t)(((int64_t)aux_acc * (int64_t)mult_aux_q16 + 32768LL) >> 16);
        }

        if (out_raw_logits != NULL) {
            out_raw_logits[k] = acc;
        }
        if (k == 0 || acc > best_logit) {
            best_logit = acc;
            best_cls = k;
        }
    }
    uint32_t tfc_1 = cnn1d_read_ccount();

    if (prof != NULL) {
        prof->cycles_fc    = tfc_1 - tfc_0;
        prof->cycles_total = tfc_1 - t0;
    }

    return best_cls;
}
