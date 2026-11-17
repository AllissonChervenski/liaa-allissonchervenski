/*
 * main.c — Bare-Metal ESP32 / ESP32-S3 QEMU Runner for Quantized sEMG Pipeline
 *
 * Reads model parameters and quantized test windows from the host filesystem
 * via Xtensa Semihosting, executes:
 *   1. CausalSosFilter (DF2T Biquad Cascade) + StatefulWindowBuffer streaming verification
 *   2. Full evaluation over all test windows (Float32 and Zero-FPU INT16/INT8 paths)
 *   3. Canonical Micro-LDA inference (Float32 and Q7 200-byte quantized weights)
 *   4. Hardware cycle counting (CCOUNT) per pipeline stage
 * Emits structured JSON telemetry to stdout for host verification and reporting.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include "semg_pipeline.h"

/* Enable semihosting _read syscall in newlib/libgloss */
extern int __sim_call(int id, int a1, int a2, int a3, int a4);
int _read(int fd, void *buf, size_t cnt) {
    return __sim_call(3, fd, (int)buf, (int)cnt, 0);
}

/* Binary bundle header written by Python exporter */
typedef struct {
    uint32_t magic;          /* 0x53454D47 ("SEMG") */
    uint32_t precision_code; /* 0=fp32, 1=fp16, 2=int32, 3=int16, 4=int8 */
    uint32_t num_channels;   /* 1..12 */
    uint32_t num_features;   /* 5 * num_channels */
    uint32_t num_classes;    /* 10 (Ozdemir) or 50 (NinaPro) */
    uint32_t window_len;     /* 400 */
    uint32_t stride_len;     /* 100 */
    uint32_t num_test_windows;
    uint32_t num_stream_samples; /* Continuous stream snippet for filter+window check */
    float    v_max;          /* Full-scale voltage (10.0 mV or 0.0055 V) */
    float    adc_scale;      /* Integer ADC quantization scale */
} bundle_header_t;

/* Static allocations in ESP32 DRAM to avoid heap fragmentation */
static micro_lda_model_t g_model;
static causal_sos_filter_t g_filter;
static stateful_window_buffer_t g_winbuf;

static float g_sos_coeffs[MAX_SOS_SECTIONS * 6];
static float g_epsilon[MAX_CHANNELS];
static int32_t g_epsilon_q[MAX_CHANNELS];

static float g_window_fp32[WINDOW_LEN * MAX_CHANNELS];
static int16_t g_window_int16[WINDOW_LEN * MAX_CHANNELS];
static float g_features_fp32[MAX_FEATURES];
static float g_features_int[MAX_FEATURES];

/* Continuous streaming test buffers (1000 samples = 7 windows @ W=400, S=100) */
#define STREAM_TEST_SAMPLES 1000
static float g_stream_in[STREAM_TEST_SAMPLES * MAX_CHANNELS];
static float g_stream_filtered[STREAM_TEST_SAMPLES * MAX_CHANNELS];
static float g_stream_expected[STREAM_TEST_SAMPLES * MAX_CHANNELS];

/* Confusion matrix and per-class counters for Balanced Accuracy and Macro F1 */
static uint32_t g_tp[MAX_CLASSES];
static uint32_t g_fp[MAX_CLASSES];
static uint32_t g_fn[MAX_CLASSES];

static uint32_t g_tp_q7[MAX_CLASSES];
static uint32_t g_fp_q7[MAX_CLASSES];
static uint32_t g_fn_q7[MAX_CLASSES];

static uint32_t g_tp_int_feat[MAX_CLASSES];
static uint32_t g_fp_int_feat[MAX_CLASSES];
static uint32_t g_fn_int_feat[MAX_CLASSES];

static void read_exact(int fd, void *dst, size_t nbytes) {
    uint8_t *p = (uint8_t *)dst;
    size_t rem = nbytes;
    while (rem > 0) {
        int r = _read(fd, p, rem);
        if (r <= 0) {
            printf("{\"error\": \"EOF or error (%d) while reading bundle (%u bytes remaining)\"}\n", r, (unsigned)rem);
            exit(2);
        }
        p += (size_t)r;
        rem -= (size_t)r;
    }
}

static void compute_metrics_from_counts(
    int num_classes,
    uint32_t total_correct,
    uint32_t total_windows,
    const uint32_t *tp,
    const uint32_t *fp,
    const uint32_t *fn,
    float *out_acc,
    float *out_bal_acc,
    float *out_macro_f1
) {
    *out_acc = (total_windows > 0) ? ((float)total_correct / (float)total_windows) : 0.0f;
    float sum_recall = 0.0f;
    float sum_f1 = 0.0f;
    int active_classes = 0;

    for (int c = 0; c < num_classes; c++) {
        uint32_t c_tp = tp[c];
        uint32_t c_fn = fn[c];
        uint32_t c_fp = fp[c];
        if (c_tp + c_fn > 0) {
            active_classes++;
            float recall = (float)c_tp / (float)(c_tp + c_fn);
            float prec = (c_tp + c_fp > 0) ? ((float)c_tp / (float)(c_tp + c_fp)) : 0.0f;
            float f1 = (prec + recall > 0.0f) ? (2.0f * prec * recall / (prec + recall)) : 0.0f;
            sum_recall += recall;
            sum_f1 += f1;
        }
    }
    *out_bal_acc = (active_classes > 0) ? (sum_recall / (float)active_classes) : 0.0f;
    *out_macro_f1 = (active_classes > 0) ? (sum_f1 / (float)active_classes) : 0.0f;
}

int main(void) {
    extern int _open(const char *name, int flags, int mode);
    extern int _close(int fd);
    int f = _open("esp32_eval_bundle.bin", 0, 0);
    if (f < 0) {
        printf("{\"error\": \"Could not open esp32_eval_bundle.bin via semihosting\"}\n");
        return 1;
    }

    bundle_header_t hdr;
    read_exact(f, &hdr, sizeof(bundle_header_t));
    if (hdr.magic != 0x53454D47U) {
        printf("{\"error\": \"Invalid magic 0x%08x\"}\n", (unsigned)hdr.magic);
        _close(f);
        return 1;
    }

    const int n_ch = (int)hdr.num_channels;
    const int n_feat = (int)hdr.num_features;
    const int n_cls = (int)hdr.num_classes;
    const int w_len = (int)hdr.window_len;
    const int s_len = (int)hdr.stride_len;
    const int n_win = (int)hdr.num_test_windows;

    g_model.num_channels = n_ch;
    g_model.num_features = n_feat;
    g_model.num_classes = n_cls;

    /* Read SOS filter coefficients (4 sections x 6 floats) */
    read_exact(f, g_sos_coeffs, MAX_SOS_SECTIONS * 6 * sizeof(float));
    /* Read epsilon array (n_ch floats) */
    read_exact(f, g_epsilon, n_ch * sizeof(float));
    /* Read model classes, scaler mean, inv_scale, zero_var_mask */
    read_exact(f, g_model.classes, n_cls * sizeof(int32_t));
    read_exact(f, g_model.mean, n_feat * sizeof(float));
    read_exact(f, g_model.inv_scale, n_feat * sizeof(float));
    read_exact(f, g_model.zero_var_mask, n_feat * sizeof(uint8_t));
    /* Read float32 LDA coef and intercept */
    read_exact(f, g_model.coef, n_cls * n_feat * sizeof(float));
    read_exact(f, g_model.intercept, n_cls * sizeof(float));
    /* Read Q7 quantized LDA coef_q7 and weight_scale */
    read_exact(f, g_model.coef_q7, n_cls * n_feat * sizeof(int8_t));
    read_exact(f, g_model.weight_scale, n_cls * sizeof(float));

    /* Prepare integer deadzone thresholds */
    const float inv_adc_scale = (hdr.adc_scale > 0.0f) ? (1.0f / hdr.adc_scale) : 1.0f;
    for (int ch = 0; ch < n_ch; ch++) {
        g_epsilon_q[ch] = (int32_t)lrintf(g_epsilon[ch] * hdr.adc_scale);
        if (g_epsilon_q[ch] < 1) g_epsilon_q[ch] = 1;
    }

    /* Stage 1: Verify CausalSosFilter + StatefulWindowBuffer on streaming snippet */
    uint32_t cycles_filter_total = 0;
    uint32_t cycles_winbuf_total = 0;
    int emitted_stream_windows = 0;
    float max_filter_diff = 0.0f;

    if (hdr.num_stream_samples > 0 && hdr.num_stream_samples <= STREAM_TEST_SAMPLES) {
        const int n_s = (int)hdr.num_stream_samples;
        read_exact(f, g_stream_in, n_s * n_ch * sizeof(float));
        read_exact(f, g_stream_expected, n_s * n_ch * sizeof(float));

        sos_filter_init(&g_filter, MAX_SOS_SECTIONS, n_ch, g_sos_coeffs);
        window_buffer_init(&g_winbuf, w_len, s_len, n_ch);

        /* Process in causal streaming chunks of S=100 samples */
        int cursor = 0;
        while (cursor < n_s) {
            int chunk = (n_s - cursor >= s_len) ? s_len : (n_s - cursor);
            uint32_t t0 = read_ccount();
            sos_filter_process_chunk(
                &g_filter,
                &g_stream_in[cursor * n_ch],
                &g_stream_filtered[cursor * n_ch],
                chunk
            );
            uint32_t t1 = read_ccount();
            cycles_filter_total += (t1 - t0);

            uint32_t tw0 = read_ccount();
            for (int i = 0; i < chunk; i++) {
                if (window_buffer_push_sample(&g_winbuf, &g_stream_filtered[(cursor + i) * n_ch], g_window_fp32)) {
                    emitted_stream_windows++;
                }
            }
            uint32_t tw1 = read_ccount();
            cycles_winbuf_total += (tw1 - tw0);

            cursor += chunk;
        }

        for (int i = 0; i < n_s * n_ch; i++) {
            float d = fabsf(g_stream_filtered[i] - g_stream_expected[i]);
            if (d > max_filter_diff) max_filter_diff = d;
        }
    }

    /* Stage 2: Evaluate all test windows on ESP32 CPU */
    memset(g_tp, 0, sizeof(g_tp));
    memset(g_fp, 0, sizeof(g_fp));
    memset(g_fn, 0, sizeof(g_fn));
    memset(g_tp_q7, 0, sizeof(g_tp_q7));
    memset(g_fp_q7, 0, sizeof(g_fp_q7));
    memset(g_fn_q7, 0, sizeof(g_fn_q7));
    memset(g_tp_int_feat, 0, sizeof(g_tp_int_feat));
    memset(g_fp_int_feat, 0, sizeof(g_fp_int_feat));
    memset(g_fn_int_feat, 0, sizeof(g_fn_int_feat));

    uint32_t correct_fp32 = 0;
    uint32_t correct_q7 = 0;
    uint32_t correct_int_feat = 0;
    uint32_t exact_match_host_preds = 0;

    uint64_t total_cycles_feat_fp32 = 0;
    uint64_t total_cycles_feat_int16 = 0;
    uint64_t total_cycles_lda_fp32 = 0;
    uint64_t total_cycles_lda_q7 = 0;
    float max_feat_diff_vs_host = 0.0f;

    for (int w = 0; w < n_win; w++) {
        int32_t y_true = 0;
        int32_t y_host_pred = 0;
        float host_feats[MAX_FEATURES];

        read_exact(f, &y_true, sizeof(int32_t));
        read_exact(f, &y_host_pred, sizeof(int32_t));
        read_exact(f, host_feats, n_feat * sizeof(float));
        read_exact(f, g_window_fp32, w_len * n_ch * sizeof(float));

        if (y_true < 0 || y_true >= n_cls) {
            printf("{\"error\": \"Corrupt y_true=%d at window %d (n_cls=%d)\"}\n", (int)y_true, w, n_cls);
            return 3;
        }

        /* Quantize window to int16 representation for Zero-FPU integer feature extractor benchmark */
        for (int i = 0; i < w_len * n_ch; i++) {
            int32_t q = (int32_t)lrintf(g_window_fp32[i] * hdr.adc_scale);
            if (q > 32767) q = 32767;
            if (q < -32768) q = -32768;
            g_window_int16[i] = (int16_t)q;
        }

        /* 2a. Float32 Feature Extraction on ESP32 */
        uint32_t tf0 = read_ccount();
        extract_td_features_fp32(g_window_fp32, w_len, n_ch, g_epsilon, g_features_fp32);
        uint32_t tf1 = read_ccount();
        total_cycles_feat_fp32 += (uint64_t)(tf1 - tf0);

        for (int j = 0; j < n_feat; j++) {
            float df = fabsf(g_features_fp32[j] - host_feats[j]);
            if (df > max_feat_diff_vs_host) max_feat_diff_vs_host = df;
        }

        /* 2b. Zero-FPU INT16/INT8 Feature Extraction on ESP32 */
        uint32_t ti0 = read_ccount();
        extract_td_features_int16(g_window_int16, w_len, n_ch, g_epsilon_q, inv_adc_scale, g_features_int);
        uint32_t ti1 = read_ccount();
        total_cycles_feat_int16 += (uint64_t)(ti1 - ti0);

        /* 2c. Canonical Micro-LDA Inference (Float32) */
        uint32_t tl0 = read_ccount();
        int32_t pred_fp32 = micro_lda_predict_fp32(&g_model, g_features_fp32, NULL);
        uint32_t tl1 = read_ccount();
        total_cycles_lda_fp32 += (uint64_t)(tl1 - tl0);

        /* 2d. Quantized Q7 Micro-LDA Inference (200 bytes weights) */
        uint32_t tq0 = read_ccount();
        int32_t pred_q7 = micro_lda_predict_q7(&g_model, g_features_fp32, NULL);
        uint32_t tq1 = read_ccount();
        total_cycles_lda_q7 += (uint64_t)(tq1 - tq0);

        /* 2e. Zero-FPU Feature + Micro-LDA Inference */
        int32_t pred_int_feat = micro_lda_predict_fp32(&g_model, g_features_int, NULL);

        if (pred_fp32 == y_host_pred) {
            exact_match_host_preds++;
        }

        /* Update confusion counters for Float32 pipeline */
        if (pred_fp32 == y_true) {
            correct_fp32++;
            g_tp[y_true]++;
        } else {
            g_fn[y_true]++;
            g_fp[pred_fp32]++;
        }

        /* Update confusion counters for Q7 Micro-LDA */
        if (pred_q7 == y_true) {
            correct_q7++;
            g_tp_q7[y_true]++;
        } else {
            g_fn_q7[y_true]++;
            g_fp_q7[pred_q7]++;
        }

        /* Update confusion counters for Zero-FPU Integer Feature Extractor */
        if (pred_int_feat == y_true) {
            correct_int_feat++;
            g_tp_int_feat[y_true]++;
        } else {
            g_fn_int_feat[y_true]++;
            g_fp_int_feat[pred_int_feat]++;
        }
    }

    _close(f);

    float acc_fp32, bal_fp32, f1_fp32;
    float acc_q7, bal_q7, f1_q7;
    float acc_int_feat, bal_int_feat, f1_int_feat;

    compute_metrics_from_counts(n_cls, correct_fp32, n_win, g_tp, g_fp, g_fn, &acc_fp32, &bal_fp32, &f1_fp32);
    compute_metrics_from_counts(n_cls, correct_q7, n_win, g_tp_q7, g_fp_q7, g_fn_q7, &acc_q7, &bal_q7, &f1_q7);
    compute_metrics_from_counts(n_cls, correct_int_feat, n_win, g_tp_int_feat, g_fp_int_feat, g_fn_int_feat, &acc_int_feat, &bal_int_feat, &f1_int_feat);

    uint32_t avg_cycles_feat_fp32 = (n_win > 0) ? (uint32_t)(total_cycles_feat_fp32 / (uint64_t)n_win) : 0;
    uint32_t avg_cycles_feat_int16 = (n_win > 0) ? (uint32_t)(total_cycles_feat_int16 / (uint64_t)n_win) : 0;
    uint32_t avg_cycles_lda_fp32 = (n_win > 0) ? (uint32_t)(total_cycles_lda_fp32 / (uint64_t)n_win) : 0;
    uint32_t avg_cycles_lda_q7 = (n_win > 0) ? (uint32_t)(total_cycles_lda_q7 / (uint64_t)n_win) : 0;
    /* Filter + Window cycles per 100-sample stride (50 ms stride) */
    uint32_t num_strides = (hdr.num_stream_samples > 0) ? (hdr.num_stream_samples / s_len) : 1;
    uint32_t avg_cycles_filter_stride = cycles_filter_total / num_strides;
    uint32_t avg_cycles_winbuf_stride = cycles_winbuf_total / num_strides;

    printf("QEMU_ESP32_RESULT_JSON: {");
    printf("\"precision_code\": %u, ", (unsigned)hdr.precision_code);
    printf("\"num_channels\": %d, ", n_ch);
    printf("\"num_features\": %d, ", n_feat);
    printf("\"num_classes\": %d, ", n_cls);
    printf("\"num_test_windows\": %d, ", n_win);
    printf("\"top1_accuracy\": %.6f, ", (double)acc_fp32);
    printf("\"balanced_accuracy\": %.6f, ", (double)bal_fp32);
    printf("\"macro_f1\": %.6f, ", (double)f1_fp32);
    printf("\"q7_top1_accuracy\": %.6f, ", (double)acc_q7);
    printf("\"q7_balanced_accuracy\": %.6f, ", (double)bal_q7);
    printf("\"q7_macro_f1\": %.6f, ", (double)f1_q7);
    printf("\"int_feat_top1_accuracy\": %.6f, ", (double)acc_int_feat);
    printf("\"int_feat_balanced_accuracy\": %.6f, ", (double)bal_int_feat);
    printf("\"int_feat_macro_f1\": %.6f, ", (double)f1_int_feat);
    printf("\"host_parity_ratio\": %.6f, ", (double)exact_match_host_preds / (double)n_win);
    printf("\"max_filter_diff\": %.9e, ", (double)max_filter_diff);
    printf("\"max_feat_diff\": %.9e, ", (double)max_feat_diff_vs_host);
    printf("\"emitted_stream_windows\": %d, ", emitted_stream_windows);
    printf("\"cycles_filter_per_stride\": %u, ", (unsigned)avg_cycles_filter_stride);
    printf("\"cycles_winbuf_per_stride\": %u, ", (unsigned)avg_cycles_winbuf_stride);
    printf("\"cycles_feat_fp32\": %u, ", (unsigned)avg_cycles_feat_fp32);
    printf("\"cycles_feat_int16\": %u, ", (unsigned)avg_cycles_feat_int16);
    printf("\"cycles_lda_fp32\": %u, ", (unsigned)avg_cycles_lda_fp32);
    printf("\"cycles_lda_q7\": %u", (unsigned)avg_cycles_lda_q7);
    printf("}\n");

    return 0;
}
