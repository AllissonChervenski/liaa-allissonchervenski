/*
 * semg_pipeline.h — ESP32 / ESP32-S3 Bare-Metal Pipeline Header
 *
 * Implements the full sEMG Processing & TinyML Pipeline for QEMU ESP32/ESP32-S3:
 *   1. Quantized ADC Ingestion (float32, float16, int32, int16, int8)
 *   2. CausalSosFilter (Direct Form II Transposed Biquad Cascade)
 *   3. StatefulWindowBuffer (Sliding Window W=400, Stride S=100)
 *   4. TimeDomainFeatureExtractor (Hudgins TD: MAV, RMS, WL, ZC, SSC)
 *      - Floating-Point Reference Engine
 *      - Zero-FPU Integer / Fixed-Point Engine (int16 / int8)
 *   5. Train-Only Z-Score FeatureScaler + Canonical Micro-LDA Classifier
 *      - Floating-Point Reference (float32)
 *      - Quantized Q7/INT8 Micro-LDA (200 bytes weights in L1 Cache)
 */

#ifndef SEMG_PIPELINE_H
#define SEMG_PIPELINE_H

#include <stdint.h>
#include <stddef.h>

#define MAX_CHANNELS    12
#define MAX_FEATURES    60
#define MAX_CLASSES     50
#define MAX_SOS_SECTIONS 4
#define WINDOW_LEN      400
#define STRIDE_LEN      100

/* Precision quantization identifiers */
typedef enum {
    PREC_FLOAT32 = 0,
    PREC_FLOAT16 = 1,
    PREC_INT32   = 2,
    PREC_INT16   = 3,
    PREC_INT8    = 4
} semg_precision_t;

/* 1. Causal SOS Biquad Filter (DF2T) */
typedef struct {
    int n_sections;
    int num_channels;
    float sos[MAX_SOS_SECTIONS][6]; /* [b0, b1, b2, 1.0, a1, a2] */
    float state[MAX_SOS_SECTIONS][2][MAX_CHANNELS];
} causal_sos_filter_t;

/* 2. Stateful Sliding Window Buffer */
typedef struct {
    int window_length;
    int stride;
    int num_channels;
    int buffered_samples;
    float buffer[WINDOW_LEN][MAX_CHANNELS];
} stateful_window_buffer_t;

/* 3. Time-Domain Feature Extractor Configuration */
typedef struct {
    int num_channels;
    float epsilon[MAX_CHANNELS];
    int32_t epsilon_int[MAX_CHANNELS]; /* Quantized deadzone for integer path */
    float inv_scale;                   /* 1.0 / adc_scale for integer-to-physical conversion */
} td_feature_extractor_t;

/* 4. Canonical Micro-LDA & Z-Score Scaler Model */
typedef struct {
    int num_channels;
    int num_features;
    int num_classes;
    int32_t classes[MAX_CLASSES];
    float mean[MAX_FEATURES];
    float inv_scale[MAX_FEATURES];
    uint8_t zero_var_mask[MAX_FEATURES];
    /* Float32 LDA parameters */
    float coef[MAX_CLASSES * MAX_FEATURES];
    float intercept[MAX_CLASSES];
    /* Fused & Quantized INT8 (Q7) Micro-LDA parameters */
    int8_t coef_q7[MAX_CLASSES * MAX_FEATURES];
    float weight_scale[MAX_CLASSES];
    float intercept_fused[MAX_CLASSES];
} micro_lda_model_t;

/* Cycle profiling telemetry per stage */
typedef struct {
    uint32_t cycles_filter;
    uint32_t cycles_window;
    uint32_t cycles_features_fp32;
    uint32_t cycles_features_int;
    uint32_t cycles_lda_fp32;
    uint32_t cycles_lda_q7;
    uint32_t cycles_total_window;
    uint32_t windows_evaluated;
} pipeline_telemetry_t;

/* Read hardware cycle counter (CCOUNT) on Xtensa LX6 / LX7 */
static inline uint32_t read_ccount(void) {
#ifdef __XTENSA__
    uint32_t ccount;
    __asm__ __volatile__("rsr %0, ccount" : "=a"(ccount));
    return ccount;
#else
    return 0U;
#endif
}

void sos_filter_init(causal_sos_filter_t *f, int n_sections, int num_channels, const float *sos_coeffs);
void sos_filter_reset(causal_sos_filter_t *f);
void sos_filter_process_chunk(causal_sos_filter_t *f, const float *in_chunk, float *out_chunk, int num_samples);

void window_buffer_init(stateful_window_buffer_t *wb, int window_length, int stride, int num_channels);
void window_buffer_reset(stateful_window_buffer_t *wb);
int window_buffer_push_sample(stateful_window_buffer_t *wb, const float *sample, float *out_window);

void extract_td_features_fp32(
    const float *window,
    int window_len,
    int num_channels,
    const float *epsilon,
    float *out_features
);

void extract_td_features_int16(
    const int16_t *window_q,
    int window_len,
    int num_channels,
    const int32_t *epsilon_q,
    float inv_adc_scale,
    float *out_features
);

int32_t micro_lda_predict_fp32(
    const micro_lda_model_t *model,
    const float *features,
    float *out_scores
);

int32_t micro_lda_predict_q7(
    const micro_lda_model_t *model,
    const float *features,
    float *out_scores
);

#endif /* SEMG_PIPELINE_H */
