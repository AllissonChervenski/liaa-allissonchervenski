/*
 * main_channel_profile_qemu.c — Multi-Channel QEMU Cycle Profiler (CCOUNT) for 10 Models
 *
 * Reads serialized `(dataset, num_channels)` blocks from `qemu_channel_profile.bin`
 * via Xtensa semihosting, executes all 10 engines across real test windows inside
 * `qemu-system-xtensa -machine esp32s3`, verifies 100% host parity, and emits
 * JSON cycle telemetry per channel configuration (`QEMU_CH_PROFILE_JSON: {...}`).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "semg_pipeline.h"
#include "cnn1d_inference.h"
#include "dtw_hybrid_engine.h"

extern int __sim_call(int id, int a1, int a2, int a3, int a4);
extern int _open(const char *name, int flags, int mode);
extern int _close(int fd);

int _read(int fd, void *buf, size_t cnt) {
    return __sim_call(3, fd, (int)buf, (int)cnt, 0);
}

#define PROFILE_MAGIC 0x43485052U /* "CHPR" */

typedef struct {
    uint32_t magic;
    uint32_t num_configs;
} ch_file_header_t;

typedef struct {
    uint32_t dataset_code;
    uint32_t num_channels;
    uint32_t num_classes;
    uint32_t num_aux;
    uint32_t num_vectors;
    float    in_quant_scale;
    float    aux_quant_scale;
    float    conv1_eff_scale;
    float    conv2_eff_scale;
    float    fc_eff_scale;
    float    aux_fc_eff_scale;
    float    env_quant_scale;
    float    cascade_margin_threshold;
    uint32_t cascade_use_cnn1d_tier2;
    float    gamma_dtw_lda;
    float    gamma_dtw_cnn;
    float    tri_margin_threshold;
    float    gamma_tri_hybrid;
} ch_block_header_t;

/* Dynamic weight storage inside static BSS (reloaded per channel block) */
static micro_lda_model_t g_lda;
static float   g_in_mean[MAX_CHANNELS];
static float   g_in_inv_std[MAX_CHANNELS];
static float   g_aux_mean[MAX_FEATURES];
static float   g_aux_inv_std[MAX_FEATURES];
static int8_t  g_conv1_w[16 * MAX_CHANNELS * 7];
static int32_t g_conv1_b[16];
static int8_t  g_conv2_w[24 * 16 * 5];
static int32_t g_conv2_b[24];
static int8_t  g_fc_w[MAX_CLASSES * 24];
static int32_t g_fc_b[MAX_CLASSES];
static int8_t  g_aux_w[MAX_CLASSES * MAX_FEATURES];

static float   g_env_mean[MAX_CHANNELS];
static float   g_env_inv_std[MAX_CHANNELS];
static int8_t  g_dtw_tmpl[MAX_CLASSES * DTW_ENV_LEN * MAX_CHANNELS];
static int8_t  g_dtw_u[MAX_CLASSES * DTW_ENV_LEN * MAX_CHANNELS];
static int8_t  g_dtw_l[MAX_CLASSES * DTW_ENV_LEN * MAX_CHANNELS];

static int32_t g_hyb1_classes[MAX_CLASSES];
static float   g_hyb1_mean[MAX_HYBRID_FEATURES];
static float   g_hyb1_inv_scale[MAX_HYBRID_FEATURES];
static uint8_t g_hyb1_zmask[MAX_HYBRID_FEATURES];
static int8_t  g_hyb1_coef[MAX_CLASSES * MAX_HYBRID_FEATURES];
static float   g_hyb1_wscale[MAX_CLASSES];
static float   g_hyb1_inter[MAX_CLASSES];

static cnn1d_model_config_t   g_cnn_cfg;
static dtw_model_config_t     g_dtw_cfg;
static hybrid_lda_head_t      g_hyb_head;
static hybrid_fusion_config_t g_fus_cfg;

/* Per-window scratch & telemetry */
static int32_t g_oracle_preds[MODEL_COUNT];
static int8_t  g_x_q[MAX_CHANNELS * WINDOW_LEN];
static int8_t  g_aux_q[MAX_FEATURES];
static int8_t  g_env_q[DTW_ENV_LEN * MAX_CHANNELS];
static float   g_td_feat[MAX_FEATURES];
static float   g_scores[MAX_CLASSES];
static int32_t g_logits[CNN1D_MAX_CLASSES];
static int32_t g_dtw_dists[DTW_MAX_CLASSES];

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
    while (rem > 0U) {
        int r = _read(fd, p, rem);
        if (r <= 0) {
            printf("{\"error\":\"EOF reading qemu_channel_profile.bin\"}\n");
            exit(2);
        }
        p += (size_t)r;
        rem -= (size_t)r;
    }
}

int main(void) {
    int fd = _open("qemu_channel_profile.bin", 0, 0);
    if (fd < 0) {
        printf("{\"error\":\"Could not open qemu_channel_profile.bin\"}\n");
        return 1;
    }

    ch_file_header_t fhdr;
    read_exact(fd, &fhdr, sizeof(fhdr));
    if (fhdr.magic != PROFILE_MAGIC) {
        printf("{\"error\":\"Invalid magic 0x%08x\"}\n", (unsigned)fhdr.magic);
        _close(fd);
        return 1;
    }

    printf("QEMU_CH_PROFILE_JSON: {\"configs\":[");

    for (uint32_t cfg_idx = 0; cfg_idx < fhdr.num_configs; cfg_idx++) {
        ch_block_header_t bh;
        read_exact(fd, &bh, sizeof(bh));

        const int n_ch  = (int)bh.num_channels;
        const int n_cls = (int)bh.num_classes;
        const int n_aux = (int)bh.num_aux;
        const int n_hyb = DTW_GAP_DIM + n_aux;
        const int n_vec = (int)bh.num_vectors;

        memset(&g_lda, 0, sizeof(g_lda));
        g_lda.num_channels = n_ch;
        g_lda.num_features = n_aux;
        g_lda.num_classes  = n_cls;
        read_exact(fd, g_lda.classes, (size_t)n_cls * sizeof(int32_t));
        read_exact(fd, g_lda.mean, (size_t)n_aux * sizeof(float));
        read_exact(fd, g_lda.inv_scale, (size_t)n_aux * sizeof(float));
        read_exact(fd, g_lda.zero_var_mask, (size_t)n_aux * sizeof(uint8_t));
        read_exact(fd, g_lda.coef_q7, (size_t)(n_cls * n_aux) * sizeof(int8_t));
        read_exact(fd, g_lda.weight_scale, (size_t)n_cls * sizeof(float));
        read_exact(fd, g_lda.intercept, (size_t)n_cls * sizeof(float));

        read_exact(fd, g_in_mean, (size_t)n_ch * sizeof(float));
        read_exact(fd, g_in_inv_std, (size_t)n_ch * sizeof(float));
        read_exact(fd, g_aux_mean, (size_t)n_aux * sizeof(float));
        read_exact(fd, g_aux_inv_std, (size_t)n_aux * sizeof(float));
        read_exact(fd, g_conv1_w, (size_t)(16 * n_ch * 7) * sizeof(int8_t));
        read_exact(fd, g_conv1_b, 16U * sizeof(int32_t));
        read_exact(fd, g_conv2_w, (size_t)(24 * 16 * 5) * sizeof(int8_t));
        read_exact(fd, g_conv2_b, 24U * sizeof(int32_t));
        read_exact(fd, g_fc_w, (size_t)(n_cls * 24) * sizeof(int8_t));
        read_exact(fd, g_fc_b, (size_t)n_cls * sizeof(int32_t));
        read_exact(fd, g_aux_w, (size_t)(n_cls * n_aux) * sizeof(int8_t));

        g_cnn_cfg.in_channels      = n_ch;
        g_cnn_cfg.num_classes      = n_cls;
        g_cnn_cfg.in_mean          = g_in_mean;
        g_cnn_cfg.in_inv_std       = g_in_inv_std;
        g_cnn_cfg.in_quant_scale   = bh.in_quant_scale;
        g_cnn_cfg.conv1_weights    = g_conv1_w;
        g_cnn_cfg.conv1_bias       = g_conv1_b;
        g_cnn_cfg.conv1_eff_scale  = bh.conv1_eff_scale;
        g_cnn_cfg.conv2_weights    = g_conv2_w;
        g_cnn_cfg.conv2_bias       = g_conv2_b;
        g_cnn_cfg.conv2_eff_scale  = bh.conv2_eff_scale;
        g_cnn_cfg.fc_weights       = g_fc_w;
        g_cnn_cfg.fc_bias          = g_fc_b;
        g_cnn_cfg.fc_eff_scale     = bh.fc_eff_scale;
        g_cnn_cfg.num_aux_features = n_aux;
        g_cnn_cfg.aux_mean         = g_aux_mean;
        g_cnn_cfg.aux_inv_std      = g_aux_inv_std;
        g_cnn_cfg.aux_quant_scale  = bh.aux_quant_scale;
        g_cnn_cfg.aux_fc_weights   = g_aux_w;
        g_cnn_cfg.aux_fc_eff_scale = bh.aux_fc_eff_scale;

        read_exact(fd, g_env_mean, (size_t)n_ch * sizeof(float));
        read_exact(fd, g_env_inv_std, (size_t)n_ch * sizeof(float));
        const size_t tmpl_bytes = (size_t)(n_cls * DTW_ENV_LEN * n_ch);
        read_exact(fd, g_dtw_tmpl, tmpl_bytes);
        read_exact(fd, g_dtw_u, tmpl_bytes);
        read_exact(fd, g_dtw_l, tmpl_bytes);

        g_dtw_cfg.num_channels    = n_ch;
        g_dtw_cfg.num_classes     = n_cls;
        g_dtw_cfg.env_len         = DTW_ENV_LEN;
        g_dtw_cfg.band_w          = DTW_SAKOE_W;
        g_dtw_cfg.env_mean        = g_env_mean;
        g_dtw_cfg.env_inv_std     = g_env_inv_std;
        g_dtw_cfg.env_quant_scale = bh.env_quant_scale;
        g_dtw_cfg.templates_q     = g_dtw_tmpl;
        g_dtw_cfg.env_upper       = g_dtw_u;
        g_dtw_cfg.env_lower       = g_dtw_l;

        read_exact(fd, g_hyb1_classes, (size_t)n_cls * sizeof(int32_t));
        read_exact(fd, g_hyb1_mean, (size_t)n_hyb * sizeof(float));
        read_exact(fd, g_hyb1_inv_scale, (size_t)n_hyb * sizeof(float));
        read_exact(fd, g_hyb1_zmask, (size_t)n_hyb * sizeof(uint8_t));
        read_exact(fd, g_hyb1_coef, (size_t)(n_cls * n_hyb) * sizeof(int8_t));
        read_exact(fd, g_hyb1_wscale, (size_t)n_cls * sizeof(float));
        read_exact(fd, g_hyb1_inter, (size_t)n_cls * sizeof(float));

        g_hyb_head.num_channels  = n_ch;
        g_hyb_head.num_features  = n_hyb;
        g_hyb_head.num_classes   = n_cls;
        g_hyb_head.classes       = g_hyb1_classes;
        g_hyb_head.mean          = g_hyb1_mean;
        g_hyb_head.inv_scale     = g_hyb1_inv_scale;
        g_hyb_head.zero_var_mask = g_hyb1_zmask;
        g_hyb_head.coef_q7       = g_hyb1_coef;
        g_hyb_head.weight_scale  = g_hyb1_wscale;
        g_hyb_head.intercept     = g_hyb1_inter;

        g_fus_cfg.cascade_margin_threshold = bh.cascade_margin_threshold;
        g_fus_cfg.cascade_use_cnn1d_tier2  = (uint8_t)bh.cascade_use_cnn1d_tier2;
        g_fus_cfg.gamma_dtw_lda            = bh.gamma_dtw_lda;
        g_fus_cfg.gamma_dtw_cnn            = bh.gamma_dtw_cnn;
        g_fus_cfg.tri_margin_threshold     = bh.tri_margin_threshold;
        g_fus_cfg.gamma_tri_hybrid         = bh.gamma_tri_hybrid;

        uint64_t sum_cyc[MODEL_COUNT];
        uint32_t matches[MODEL_COUNT];
        uint32_t ee_cnt[MODEL_COUNT];
        uint64_t prune_sum[MODEL_COUNT];
        memset(sum_cyc, 0, sizeof(sum_cyc));
        memset(matches, 0, sizeof(matches));
        memset(ee_cnt, 0, sizeof(ee_cnt));
        memset(prune_sum, 0, sizeof(prune_sum));

        for (int i = 0; i < n_vec; i++) {
            read_exact(fd, g_oracle_preds, MODEL_COUNT * sizeof(int32_t));
            read_exact(fd, g_x_q, (size_t)(n_ch * WINDOW_LEN) * sizeof(int8_t));
            read_exact(fd, g_aux_q, (size_t)n_aux * sizeof(int8_t));
            read_exact(fd, g_env_q, (size_t)(DTW_ENV_LEN * n_ch) * sizeof(int8_t));
            read_exact(fd, g_td_feat, (size_t)n_aux * sizeof(float));

            int32_t preds[MODEL_COUNT];
            uint32_t cyc[MODEL_COUNT];
            dtw_hybrid_telemetry_t tel;
            cnn1d_cycle_profile_t cnn_prof;
            uint32_t t_a, t_b;

            t_a = cnn1d_read_ccount();
            preds[0] = micro_lda_predict_q7(&g_lda, g_td_feat, g_scores);
            t_b = cnn1d_read_ccount();
            cyc[0] = t_b - t_a;

            t_a = cnn1d_read_ccount();
            int32_t p_cnn = cnn1d_forward_int8(&g_cnn_cfg, g_x_q, g_aux_q, g_logits, &cnn_prof);
            t_b = cnn1d_read_ccount();
            preds[1] = (p_cnn >= 0 && p_cnn < n_cls) ? g_lda.classes[p_cnn] : p_cnn;
            cyc[1] = t_b - t_a;

            preds[2] = dtw_classic_on2_predict(&g_dtw_cfg, g_env_q, g_dtw_dists, &tel);
            cyc[2] = tel.cycles_total;

            preds[3] = dtw_sakoe_chiba_predict(&g_dtw_cfg, g_env_q, g_dtw_dists, &tel);
            cyc[3] = tel.cycles_total;

            preds[4] = dtw_lb_keogh_predict(&g_dtw_cfg, g_env_q, &tel);
            cyc[4] = tel.cycles_total;
            prune_sum[4] += tel.pruned_lb_count;

            preds[5] = hybrid_lda_cnn_head_predict_q7(&g_cnn_cfg, &g_hyb_head, g_x_q, g_td_feat, g_scores, &tel);
            cyc[5] = tel.cycles_total;

            preds[6] = hybrid_lda_cnn_cascade_predict(
                &g_lda, &g_cnn_cfg, &g_hyb_head, &g_fus_cfg, NULL, g_x_q, g_aux_q, g_td_feat, &tel
            );
            cyc[6] = tel.cycles_total;
            ee_cnt[6] += tel.early_exit;

            preds[7] = hybrid_dtw_lda_predict(&g_lda, &g_dtw_cfg, &g_fus_cfg, g_td_feat, g_env_q, &tel);
            cyc[7] = tel.cycles_total;

            preds[8] = hybrid_dtw_cnn_predict(&g_cnn_cfg, &g_dtw_cfg, &g_fus_cfg, g_x_q, g_aux_q, g_env_q, &tel);
            cyc[8] = tel.cycles_total;

            preds[9] = hybrid_dtw_lda_cnn_predict(
                &g_lda, &g_cnn_cfg, &g_hyb_head, &g_dtw_cfg, &g_fus_cfg,
                NULL, g_x_q, g_env_q, NULL, 1.0f, g_td_feat, &tel
            );
            cyc[9] = tel.cycles_total;
            ee_cnt[9] += tel.early_exit;
            prune_sum[9] += tel.pruned_lb_count;

            for (int m = 0; m < MODEL_COUNT; m++) {
                sum_cyc[m] += cyc[m];
                if (preds[m] == g_oracle_preds[m]) {
                    matches[m]++;
                }
            }
        }

        printf("%s{\"dataset_code\":%lu,\"num_channels\":%d,\"num_classes\":%d,\"num_vectors\":%d,\"models\":[",
            (cfg_idx > 0U) ? "," : "",
            (unsigned long)bh.dataset_code, n_ch, n_cls, n_vec);

        for (int m = 0; m < MODEL_COUNT; m++) {
            double mean_c = (n_vec > 0) ? ((double)sum_cyc[m] / (double)n_vec) : 0.0;
            double par    = (n_vec > 0) ? ((double)matches[m] / (double)n_vec) : 0.0;
            double ee     = (n_vec > 0) ? ((double)ee_cnt[m] / (double)n_vec) : 0.0;
            double pr     = (n_vec > 0 && n_cls > 0)
                ? ((double)prune_sum[m] / (double)((uint64_t)n_vec * (uint64_t)n_cls)) : 0.0;
            printf("%s{\"name\":\"%s\",\"parity_ratio\":%.6f,\"cycles_mean\":%.2f,\"latency_mean_ms\":%.6f,\"early_exit_rate\":%.6f,\"pruned_rate\":%.6f}",
                (m > 0) ? "," : "", MODEL_NAMES[m], par, mean_c, mean_c / 240000.0, ee, pr);
        }
        printf("]}");
    }

    printf("]}\n");
    _close(fd);
    return 0;
}
