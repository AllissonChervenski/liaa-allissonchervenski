/*
 * main_dtw_hybrid_qemu.c — Bare-Metal ESP32 / ESP32-S3 QEMU Harness for 10 sEMG Models
 *
 * Benchmarks all 10 models head-to-head on exact Xtensa hardware cycle counters (CCOUNT):
 *   M01: Micro-LDA (Q7)
 *   M02: CNN-1D (Symmetric INT8)
 *   M03: Classic DTW O(N^2) time & space
 *   M04: Sakoe-Chiba Band DTW O(w*N) time, O(w) space
 *   M05: LB_Keogh Lower-Bound Pruned DTW O(N) best-case + Early Abandoning
 *   M06: Hybrid 1 — CNN-1D Backbone (24D) + Hudgins TD -> Canonical LDA Q7
 *   M07: Hybrid 2 — Early-Exit Margin Cascade (Micro-LDA Q7 -> Tier 2 CNN/Hyb1)
 *   M08: Hybrid DTW + LDA
 *   M09: Hybrid DTW + CNN
 *   M10: Tri-Hybrid DTW + LDA + CNN
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "semg_pipeline.h"
#include "micro_lda_weights.h"
#include "cnn1d_inference.h"
#include "cnn1d_weights_int8.h"
#include "dtw_hybrid_engine.h"
#include "dtw_hybrid_weights.h"

/* Semihosting raw syscall interface from libsys_qemu.a */
extern int __sim_call(int id, int a1, int a2, int a3, int a4);
extern int _open(const char *name, int flags, int mode);
extern int _close(int fd);

int _read(int fd, void *buf, size_t cnt) {
    return __sim_call(3, fd, (int)buf, (int)cnt, 0);
}

typedef struct {
    uint32_t magic;         /* 0x44545748 ("DTWH") */
    uint32_t dataset_code;  /* 0 = Ozdemir 4CH, 1 = NinaPro DB2 12CH */
    uint32_t in_channels;
    uint32_t num_classes;
    uint32_t num_aux;
    uint32_t num_vectors;
    float    inv_adc_scale;
} dtw_hybrid_vec_header_t;

#define MAX_CYCLE_SAMPLES 1000
static uint32_t g_cycles[MODEL_COUNT][MAX_CYCLE_SAMPLES];
static uint64_t g_sum_cycles[MODEL_COUNT];
static uint32_t g_max_cycles[MODEL_COUNT];
static uint32_t g_oracle_matches[MODEL_COUNT];
static uint32_t g_correct[MODEL_COUNT];
static uint32_t g_early_exits[MODEL_COUNT];
static uint64_t g_sum_pruned_lb[MODEL_COUNT];
static uint64_t g_sum_dp_calls[MODEL_COUNT];

static int32_t g_oracle_preds[MODEL_COUNT];
static int16_t g_win_q16[WINDOW_LEN * MAX_CHANNELS];
static int8_t  g_x_q[MAX_CHANNELS * WINDOW_LEN];
static int8_t  g_aux_q[MAX_FEATURES];
static int8_t  g_env_q[DTW_ENV_LEN * MAX_CHANNELS];
static int8_t  g_env_calc[DTW_ENV_LEN * MAX_CHANNELS];
static float   g_td_feat[MAX_FEATURES];
static float   g_td_calc[MAX_FEATURES];

static micro_lda_model_t      g_lda_model;
static td_feature_extractor_t g_td_cfg;
static float                  g_scores[MAX_CLASSES];
static int32_t                g_logits[CNN1D_MAX_CLASSES];
static int32_t                g_dtw_dists[DTW_MAX_CLASSES];

static const char *MODEL_NAMES[MODEL_COUNT] = {
    "M01_micro_lda_q7",
    "M02_cnn1d_int8",
    "M03_dtw_classic_on2",
    "M04_dtw_sakoe_chiba",
    "M05_dtw_lb_keogh",
    "M06_hybrid_lda_cnn_head",
    "M07_hybrid_lda_cnn_cascade",
    "M08_hybrid_dtw_lda",
    "M09_hybrid_dtw_cnn",
    "M10_hybrid_dtw_lda_cnn"
};

static void read_exact(int fd, void *dst, size_t nbytes) {
    uint8_t *p = (uint8_t *)dst;
    size_t rem = nbytes;
    while (rem > 0) {
        int r = _read(fd, p, rem);
        if (r <= 0) {
            printf("{\"error\": \"EOF reading dtw_hybrid_test_vectors.bin (%u left)\"}\n", (unsigned)rem);
            exit(2);
        }
        p += (size_t)r;
        rem -= (size_t)r;
    }
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t va = *(const uint32_t *)a;
    uint32_t vb = *(const uint32_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

static uint32_t percentile_sorted(const uint32_t *sorted, int count, int pct) {
    if (count <= 0) return 0U;
    int idx = (count * pct) / 100;
    if (idx >= count) idx = count - 1;
    return sorted[idx];
}

int main(void) {
    int fd = _open("dtw_hybrid_test_vectors.bin", 0, 0);
    if (fd < 0) {
        printf("{\"error\": \"Could not open dtw_hybrid_test_vectors.bin\"}\n");
        return 1;
    }

    dtw_hybrid_vec_header_t hdr;
    read_exact(fd, &hdr, sizeof(hdr));
    if (hdr.magic != 0x44545748U) {
        printf("{\"error\": \"Invalid magic 0x%08x\"}\n", (unsigned)hdr.magic);
        _close(fd);
        return 1;
    }

    const cnn1d_model_config_t   *cnn_cfg;
    const dtw_model_config_t     *dtw_cfg;
    const hybrid_lda_head_t      *hyb_head;
    const hybrid_fusion_config_t *fus_cfg;

    if (hdr.dataset_code == 0U) {
        semg_load_ozdemir_micro_lda(&g_lda_model, &g_td_cfg);
        cnn_cfg  = &g_cnn1d_ozdemir_config;
        dtw_cfg  = &g_dtw_ozdemir_config;
        hyb_head = &g_hyb1_ozdemir_head;
        fus_cfg  = &g_fusion_ozdemir_config;
    } else {
        semg_load_ninapro_micro_lda(&g_lda_model, &g_td_cfg);
        cnn_cfg  = &g_cnn1d_ninapro_config;
        dtw_cfg  = &g_dtw_ninapro_config;
        hyb_head = &g_hyb1_ninapro_head;
        fus_cfg  = &g_fusion_ninapro_config;
    }

    const int n_ch  = (int)hdr.in_channels;
    const int n_cls = (int)hdr.num_classes;
    const int n_aux = (int)hdr.num_aux;
    const int n_vec = (int)hdr.num_vectors;
    const float inv_adc = hdr.inv_adc_scale;

    memset(g_sum_cycles, 0, sizeof(g_sum_cycles));
    memset(g_max_cycles, 0, sizeof(g_max_cycles));
    memset(g_oracle_matches, 0, sizeof(g_oracle_matches));
    memset(g_correct, 0, sizeof(g_correct));
    memset(g_early_exits, 0, sizeof(g_early_exits));
    memset(g_sum_pruned_lb, 0, sizeof(g_sum_pruned_lb));
    memset(g_sum_dp_calls, 0, sizeof(g_sum_dp_calls));

    uint64_t sum_cyc_td = 0U;
    uint64_t sum_cyc_env = 0U;
    int32_t  env_max_diff = 0;
    uint32_t n_cyc_recorded = 0U;

    for (int i = 0; i < n_vec; i++) {
        int32_t y_true = 0;
        read_exact(fd, &y_true, sizeof(int32_t));
        read_exact(fd, g_oracle_preds, MODEL_COUNT * sizeof(int32_t));
        read_exact(fd, g_win_q16, (size_t)(WINDOW_LEN * n_ch) * sizeof(int16_t));
        read_exact(fd, g_x_q, (size_t)(n_ch * WINDOW_LEN) * sizeof(int8_t));
        read_exact(fd, g_aux_q, (size_t)n_aux * sizeof(int8_t));
        read_exact(fd, g_env_q, (size_t)(DTW_ENV_LEN * n_ch) * sizeof(int8_t));
        read_exact(fd, g_td_feat, (size_t)n_aux * sizeof(float));

        /* Measure TD and Envelope feature extraction latencies and verify parity */
        uint32_t t_a = cnn1d_read_ccount();
        extract_td_features_int16(g_win_q16, WINDOW_LEN, n_ch, g_td_cfg.epsilon_int, inv_adc, g_td_calc);
        uint32_t t_b = cnn1d_read_ccount();
        sum_cyc_td += (uint64_t)(t_b - t_a);

        t_a = cnn1d_read_ccount();
        extract_dtw_envelope_int8(dtw_cfg, g_win_q16, inv_adc, g_env_calc);
        t_b = cnn1d_read_ccount();
        sum_cyc_env += (uint64_t)(t_b - t_a);

        for (int e = 0; e < DTW_ENV_LEN * n_ch; e++) {
            int32_t d = (int32_t)g_env_calc[e] - (int32_t)g_env_q[e];
            if (d < 0) d = -d;
            if (d > env_max_diff) env_max_diff = d;
        }

        int32_t preds[MODEL_COUNT];
        uint32_t cyc[MODEL_COUNT];
        dtw_hybrid_telemetry_t tel;
        cnn1d_cycle_profile_t cnn_prof;

        /* M01: Micro-LDA Q7 */
        t_a = cnn1d_read_ccount();
        preds[0] = micro_lda_predict_q7(&g_lda_model, g_td_calc, g_scores);
        t_b = cnn1d_read_ccount();
        cyc[0] = t_b - t_a;

        /* M02: CNN-1D INT8 */
        t_a = cnn1d_read_ccount();
        int32_t p_cnn = cnn1d_forward_int8(cnn_cfg, g_x_q, g_aux_q, g_logits, &cnn_prof);
        t_b = cnn1d_read_ccount();
        preds[1] = (p_cnn >= 0 && p_cnn < n_cls) ? g_lda_model.classes[p_cnn] : p_cnn;
        cyc[1] = t_b - t_a;

        /* M03: DTW Classic O(N^2) */
        preds[2] = dtw_classic_on2_predict(dtw_cfg, g_env_calc, g_dtw_dists, &tel);
        cyc[2] = tel.cycles_total;
        g_sum_dp_calls[2] += tel.dp_calls_count;

        /* M04: DTW Sakoe-Chiba O(w*N) */
        preds[3] = dtw_sakoe_chiba_predict(dtw_cfg, g_env_calc, g_dtw_dists, &tel);
        cyc[3] = tel.cycles_total;
        g_sum_dp_calls[3] += tel.dp_calls_count;

        /* M05: DTW LB_Keogh O(N) best-case */
        preds[4] = dtw_lb_keogh_predict(dtw_cfg, g_env_calc, &tel);
        cyc[4] = tel.cycles_total;
        g_sum_pruned_lb[4] += tel.pruned_lb_count;
        g_sum_dp_calls[4] += tel.dp_calls_count;

        /* M06: Hybrid 1 (CNN Backbone + TD + LDA Head Q7) */
        preds[5] = hybrid_lda_cnn_head_predict_q7(cnn_cfg, hyb_head, g_x_q, g_td_calc, g_scores, &tel);
        cyc[5] = tel.cycles_total;

        /* M07: Hybrid 2 (Early-Exit Margin Cascade) */
        preds[6] = hybrid_lda_cnn_cascade_predict(
            &g_lda_model, cnn_cfg, hyb_head, fus_cfg, NULL, g_x_q, g_aux_q, g_td_calc, &tel
        );
        cyc[6] = tel.cycles_total;
        g_early_exits[6] += (uint32_t)tel.early_exit;

        /* M08: Hybrid DTW + LDA */
        preds[7] = hybrid_dtw_lda_predict(&g_lda_model, dtw_cfg, fus_cfg, g_td_calc, g_env_calc, &tel);
        cyc[7] = tel.cycles_total;
        g_sum_pruned_lb[7] += tel.pruned_lb_count;
        g_sum_dp_calls[7] += tel.dp_calls_count;

        /* M09: Hybrid DTW + CNN */
        preds[8] = hybrid_dtw_cnn_predict(cnn_cfg, dtw_cfg, fus_cfg, g_x_q, g_aux_q, g_env_calc, &tel);
        cyc[8] = tel.cycles_total;
        g_sum_pruned_lb[8] += tel.pruned_lb_count;
        g_sum_dp_calls[8] += tel.dp_calls_count;

        /* M10: Tri-Hybrid DTW + LDA + CNN */
        preds[9] = hybrid_dtw_lda_cnn_predict(
            &g_lda_model, cnn_cfg, hyb_head, dtw_cfg, fus_cfg, NULL, g_x_q, g_env_calc, NULL, inv_adc, g_td_calc, &tel
        );
        cyc[9] = tel.cycles_total;
        g_early_exits[9] += (uint32_t)tel.early_exit;
        g_sum_pruned_lb[9] += tel.pruned_lb_count;
        g_sum_dp_calls[9] += tel.dp_calls_count;

        for (int m = 0; m < MODEL_COUNT; m++) {
            g_sum_cycles[m] += (uint64_t)cyc[m];
            if (cyc[m] > g_max_cycles[m]) {
                g_max_cycles[m] = cyc[m];
            }
            if (i < MAX_CYCLE_SAMPLES) {
                g_cycles[m][i] = cyc[m];
            }
            if (preds[m] == g_oracle_preds[m]) {
                g_oracle_matches[m]++;
            }
            if (preds[m] == y_true) {
                g_correct[m]++;
            }
        }
        if (i < MAX_CYCLE_SAMPLES) {
            n_cyc_recorded++;
        }
    }
    _close(fd);

    float avg_td_cyc  = (n_vec > 0) ? ((float)sum_cyc_td / (float)n_vec) : 0.0f;
    float avg_env_cyc = (n_vec > 0) ? ((float)sum_cyc_env / (float)n_vec) : 0.0f;

    printf(
        "QEMU_DTW_HYBRID_REPORT_JSON: {\"dataset_code\":%lu,\"in_channels\":%d,\"num_classes\":%d,"
        "\"num_vectors\":%d,\"env_max_diff\":%ld,\"td_cycles_mean\":%.2f,\"env_cycles_mean\":%.2f,"
        "\"models\":[",
        (unsigned long)hdr.dataset_code,
        n_ch,
        n_cls,
        n_vec,
        (long)env_max_diff,
        (double)avg_td_cyc,
        (double)avg_env_cyc
    );

    for (int m = 0; m < MODEL_COUNT; m++) {
        qsort(g_cycles[m], (size_t)n_cyc_recorded, sizeof(uint32_t), cmp_u32);
        uint32_t p50 = percentile_sorted(g_cycles[m], (int)n_cyc_recorded, 50);
        uint32_t p95 = percentile_sorted(g_cycles[m], (int)n_cyc_recorded, 95);
        uint32_t p99 = percentile_sorted(g_cycles[m], (int)n_cyc_recorded, 99);
        double mean_cyc = (n_vec > 0) ? ((double)g_sum_cycles[m] / (double)n_vec) : 0.0;
        double parity = (n_vec > 0) ? ((double)g_oracle_matches[m] / (double)n_vec) : 0.0;
        double acc = (n_vec > 0) ? ((double)g_correct[m] / (double)n_vec) : 0.0;
        double ee_rate = (n_vec > 0) ? ((double)g_early_exits[m] / (double)n_vec) : 0.0;
        double prune_rate = (n_vec > 0 && n_cls > 0)
            ? ((double)g_sum_pruned_lb[m] / (double)((uint64_t)n_vec * (uint64_t)n_cls))
            : 0.0;
        double mean_dp = (n_vec > 0) ? ((double)g_sum_dp_calls[m] / (double)n_vec) : 0.0;

        printf(
            "%s{\"id\":%d,\"name\":\"%s\",\"parity_ratio\":%.6f,\"top1_accuracy\":%.6f,"
            "\"cycles_mean\":%.2f,\"cycles_p50\":%lu,\"cycles_p95\":%lu,\"cycles_p99\":%lu,"
            "\"cycles_max\":%lu,\"latency_mean_ms_240mhz\":%.6f,\"latency_p99_ms_240mhz\":%.6f,"
            "\"early_exit_rate\":%.6f,\"pruned_rate\":%.6f,\"mean_dp_calls\":%.2f}",
            (m > 0) ? "," : "",
            m,
            MODEL_NAMES[m],
            parity,
            acc,
            mean_cyc,
            (unsigned long)p50,
            (unsigned long)p95,
            (unsigned long)p99,
            (unsigned long)g_max_cycles[m],
            mean_cyc / 240000.0,
            (double)p99 / 240000.0,
            ee_rate,
            prune_rate,
            mean_dp
        );
    }
    printf("]}\n");
    return 0;
}
