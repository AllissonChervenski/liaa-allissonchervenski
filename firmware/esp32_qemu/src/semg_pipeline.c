/*
 * semg_pipeline.c — ESP32 / ESP32-S3 Bare-Metal Pipeline Implementation
 *
 * Implements strict causal DF2T biquad filtering, sliding window buffering,
 * Hudgins time-domain feature extraction (both Float32 and Zero-FPU Integer/Fixed-Point),
 * and Canonical Micro-LDA inference (both Float32 and Q7 200-byte quantized).
 */

#include "semg_pipeline.h"
#include <string.h>
#include <math.h>

void sos_filter_init(causal_sos_filter_t *f, int n_sections, int num_channels, const float *sos_coeffs) {
    f->n_sections = n_sections;
    f->num_channels = num_channels;
    for (int s = 0; s < n_sections; s++) {
        for (int k = 0; k < 6; k++) {
            f->sos[s][k] = sos_coeffs[s * 6 + k];
        }
    }
    sos_filter_reset(f);
}

void sos_filter_reset(causal_sos_filter_t *f) {
    memset(f->state, 0, sizeof(f->state));
}

void sos_filter_process_chunk(causal_sos_filter_t *f, const float *in_chunk, float *out_chunk, int num_samples) {
    const int n_sec = f->n_sections;
    const int n_ch = f->num_channels;

    for (int ch = 0; ch < n_ch; ch++) {
        /* Copy input channel to output buffer for in-place section cascade */
        for (int n = 0; n < num_samples; n++) {
            out_chunk[n * n_ch + ch] = in_chunk[n * n_ch + ch];
        }

        for (int s = 0; s < n_sec; s++) {
            const float b0 = f->sos[s][0];
            const float b1 = f->sos[s][1];
            const float b2 = f->sos[s][2];
            const float a1 = f->sos[s][4];
            const float a2 = f->sos[s][5];

            float z1 = f->state[s][0][ch];
            float z2 = f->state[s][1][ch];

            for (int n = 0; n < num_samples; n++) {
                float xn = out_chunk[n * n_ch + ch];
                float yn = b0 * xn + z1;
                z1 = b1 * xn - a1 * yn + z2;
                z2 = b2 * xn - a2 * yn;
                out_chunk[n * n_ch + ch] = yn;
            }

            f->state[s][0][ch] = z1;
            f->state[s][1][ch] = z2;
        }
    }
}

void window_buffer_init(stateful_window_buffer_t *wb, int window_length, int stride, int num_channels) {
    wb->window_length = window_length;
    wb->stride = stride;
    wb->num_channels = num_channels;
    wb->buffered_samples = 0;
    memset(wb->buffer, 0, sizeof(wb->buffer));
}

void window_buffer_reset(stateful_window_buffer_t *wb) {
    wb->buffered_samples = 0;
}

int window_buffer_push_sample(stateful_window_buffer_t *wb, const float *sample, float *out_window) {
    const int n_ch = wb->num_channels;
    const int idx = wb->buffered_samples;

    for (int ch = 0; ch < n_ch; ch++) {
        wb->buffer[idx][ch] = sample[ch];
    }
    wb->buffered_samples++;

    if (wb->buffered_samples == wb->window_length) {
        /* Emit full window (window_length x num_channels) */
        for (int w = 0; w < wb->window_length; w++) {
            for (int ch = 0; ch < n_ch; ch++) {
                out_window[w * n_ch + ch] = wb->buffer[w][ch];
            }
        }
        /* Advance buffer by stride */
        const int remain = wb->window_length - wb->stride;
        for (int w = 0; w < remain; w++) {
            for (int ch = 0; ch < n_ch; ch++) {
                wb->buffer[w][ch] = wb->buffer[w + wb->stride][ch];
            }
        }
        wb->buffered_samples = remain;
        return 1;
    }
    return 0;
}

/*
 * Hudgins 5 Time-Domain Features (Float32 Reference Engine)
 * Feature ordering matches TimeDomainFeatureExtractor.extract():
 *   [MAV (0..C-1), RMS (C..2C-1), WL (2C..3C-1), ZC (3C..4C-1), SSC (4C..5C-1)]
 */
void extract_td_features_fp32(
    const float *window,
    int window_len,
    int num_channels,
    const float *epsilon,
    float *out_features
) {
    const float inv_w = 1.0f / (float)window_len;

    for (int ch = 0; ch < num_channels; ch++) {
        float sum_abs = 0.0f;
        float sum_sq = 0.0f;
        float wl = 0.0f;
        int zc = 0;
        int ssc = 0;
        const float eps = epsilon[ch];

        float x_prev = window[0 * num_channels + ch];
        sum_abs += fabsf(x_prev);
        sum_sq += x_prev * x_prev;

        float x_curr = window[1 * num_channels + ch];
        sum_abs += fabsf(x_curr);
        sum_sq += x_curr * x_curr;

        float diff0 = fabsf(x_curr - x_prev);
        wl += diff0;
        if (((x_prev > 0.0f && x_curr < 0.0f) || (x_prev < 0.0f && x_curr > 0.0f)) && (diff0 >= eps)) {
            zc++;
        }

        for (int i = 2; i < window_len; i++) {
            float x_next = window[i * num_channels + ch];
            sum_abs += fabsf(x_next);
            sum_sq += x_next * x_next;

            float diff_next = fabsf(x_next - x_curr);
            wl += diff_next;

            /* Zero Crossing (ZC) */
            if (((x_curr > 0.0f && x_next < 0.0f) || (x_curr < 0.0f && x_next > 0.0f)) && (diff_next >= eps)) {
                zc++;
            }

            /* Slope Sign Change (SSC) at x_curr */
            int peak = (x_curr > x_prev) && (x_curr > x_next);
            int trough = (x_curr < x_prev) && (x_curr < x_next);
            if (peak || trough) {
                float diff_prev = fabsf(x_curr - x_prev);
                if (diff_prev >= eps || diff_next >= eps) {
                    ssc++;
                }
            }

            x_prev = x_curr;
            x_curr = x_next;
        }

        out_features[0 * num_channels + ch] = sum_abs * inv_w;
        out_features[1 * num_channels + ch] = sqrtf(sum_sq * inv_w);
        out_features[2 * num_channels + ch] = wl;
        out_features[3 * num_channels + ch] = (float)zc;
        out_features[4 * num_channels + ch] = (float)ssc;
    }
}

/*
 * Zero-FPU Integer Feature Extraction Engine (INT16 / INT8 Quantized ADC)
 * Computes MAV, WL, ZC, SSC and Sum-of-Squares strictly using integer ALU
 * operations across the 400-sample window, with a single scale conversion at the end.
 */
void extract_td_features_int16(
    const int16_t *window_q,
    int window_len,
    int num_channels,
    const int32_t *epsilon_q,
    float inv_adc_scale,
    float *out_features
) {
    const float scale_mav = inv_adc_scale / (float)window_len;
    const float inv_w = 1.0f / (float)window_len;

    for (int ch = 0; ch < num_channels; ch++) {
        int32_t sum_abs_q = 0;
        int64_t sum_sq_q = 0;
        int32_t wl_q = 0;
        int32_t zc = 0;
        int32_t ssc = 0;
        const int32_t eps_q = epsilon_q[ch];

        int32_t x_prev = (int32_t)window_q[0 * num_channels + ch];
        sum_abs_q += (x_prev < 0) ? -x_prev : x_prev;
        sum_sq_q += (int64_t)x_prev * (int64_t)x_prev;

        int32_t x_curr = (int32_t)window_q[1 * num_channels + ch];
        sum_abs_q += (x_curr < 0) ? -x_curr : x_curr;
        sum_sq_q += (int64_t)x_curr * (int64_t)x_curr;

        int32_t d0 = x_curr - x_prev;
        int32_t abs_d0 = (d0 < 0) ? -d0 : d0;
        wl_q += abs_d0;
        if (((x_prev > 0 && x_curr < 0) || (x_prev < 0 && x_curr > 0)) && (abs_d0 >= eps_q)) {
            zc++;
        }

        for (int i = 2; i < window_len; i++) {
            int32_t x_next = (int32_t)window_q[i * num_channels + ch];
            sum_abs_q += (x_next < 0) ? -x_next : x_next;
            sum_sq_q += (int64_t)x_next * (int64_t)x_next;

            int32_t dn = x_next - x_curr;
            int32_t abs_dn = (dn < 0) ? -dn : dn;
            wl_q += abs_dn;

            if (((x_curr > 0 && x_next < 0) || (x_curr < 0 && x_next > 0)) && (abs_dn >= eps_q)) {
                zc++;
            }

            if ((x_curr > x_prev && x_curr > x_next) || (x_curr < x_prev && x_curr < x_next)) {
                int32_t dp = x_curr - x_prev;
                int32_t abs_dp = (dp < 0) ? -dp : dp;
                if (abs_dp >= eps_q || abs_dn >= eps_q) {
                    ssc++;
                }
            }

            x_prev = x_curr;
            x_curr = x_next;
        }

        out_features[0 * num_channels + ch] = (float)sum_abs_q * scale_mav;
        out_features[1 * num_channels + ch] = sqrtf((float)sum_sq_q * inv_w) * inv_adc_scale;
        out_features[2 * num_channels + ch] = (float)wl_q * inv_adc_scale;
        out_features[3 * num_channels + ch] = (float)zc;
        out_features[4 * num_channels + ch] = (float)ssc;
    }
}

/*
 * Canonical Micro-LDA Classifier (Float32 Reference)
 * Performs train-only Z-score normalization followed by y = W * x_norm + b
 */
int32_t micro_lda_predict_fp32(
    const micro_lda_model_t *model,
    const float *features,
    float *out_scores
) {
    const int n_feat = model->num_features;
    const int n_cls = model->num_classes;
    float x_norm[MAX_FEATURES];

    for (int j = 0; j < n_feat; j++) {
        if (model->zero_var_mask[j]) {
            x_norm[j] = 0.0f;
        } else {
            x_norm[j] = (features[j] - model->mean[j]) * model->inv_scale[j];
        }
    }

    int best_idx = 0;
    float best_score = -1e30f;

    for (int k = 0; k < n_cls; k++) {
        float acc = model->intercept[k];
        const float *w_row = &model->coef[k * n_feat];
        for (int j = 0; j < n_feat; j++) {
            acc += w_row[j] * x_norm[j];
        }
        if (out_scores) {
            out_scores[k] = acc;
        }
        if (k == 0 || acc > best_score) {
            best_score = acc;
            best_idx = k;
        }
    }

    return model->classes[best_idx];
}

/*
 * Quantized Q7/INT8 Micro-LDA Classifier (200-byte weight matrix for Ozdemir 4ch)
 * Uses int8_t weights in L1 data cache with integer multiply-accumulate.
 */
int32_t micro_lda_predict_q7(
    const micro_lda_model_t *model,
    const float *features,
    float *out_scores
) {
    const int n_feat = model->num_features;
    const int n_cls = model->num_classes;
    int16_t x_q8[MAX_FEATURES];

    /* Normalize and quantize Z-score features to Q8 fixed-point (scale = 256) */
    for (int j = 0; j < n_feat; j++) {
        if (model->zero_var_mask[j]) {
            x_q8[j] = 0;
        } else {
            float z = (features[j] - model->mean[j]) * model->inv_scale[j];
            int32_t q = (int32_t)lrintf(z * 256.0f);
            if (q > 32767) q = 32767;
            if (q < -32768) q = -32768;
            x_q8[j] = (int16_t)q;
        }
    }

    int best_idx = 0;
    float best_score = -1e30f;

    for (int k = 0; k < n_cls; k++) {
        int32_t acc_int = 0;
        const int8_t *w_row = &model->coef_q7[k * n_feat];
        for (int j = 0; j < n_feat; j++) {
            acc_int += (int32_t)w_row[j] * (int32_t)x_q8[j];
        }
        float score = ((float)acc_int * model->weight_scale[k] * (1.0f / 256.0f)) + model->intercept[k];
        if (out_scores) {
            out_scores[k] = score;
        }
        if (k == 0 || score > best_score) {
            best_score = score;
            best_idx = k;
        }
    }

    return model->classes[best_idx];
}
