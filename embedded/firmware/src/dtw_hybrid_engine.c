/*
 * dtw_hybrid_engine.c — Pure C99 DTW Variants & Multi-Modal Hybrid Classifiers
 *
 * Implements strictly deterministic static-SRAM inference for:
 *   - M03: Classic DTW O(N^2) time & space (full static DP matrix)
 *   - M04: Sakoe-Chiba Constrained DTW O(w*N) time & two-row static space
 *   - M05: LB_Keogh Pruned DTW O(N) best-case with Early Abandoning
 *   - M06: Hybrid 1 (CNN-1D INT8 GAP Backbone + Hudgins TD -> Canonical LDA Q7)
 *   - M07: Hybrid 2 (Early-Exit Gatekeeper Cascade: Micro-LDA Q7 -> Tier 2 CNN/Hyb1)
 *   - M08: Hybrid DTW + LDA (Top-M LDA candidates reranked with Sakoe-Chiba DTW)
 *   - M09: Hybrid DTW + CNN (Top-M CNN-1D candidates reranked with Sakoe-Chiba DTW)
 *   - M10: Tri-Hybrid DTW + LDA + CNN (Margin Gate -> Top-3 + LB_Keogh + Hybrid 1)
 */

#include "../include/dtw_hybrid_engine.h"
#include <math.h>
#include <string.h>

#define DTW_INF 1000000000

/* M03 O(N^2) Full 2D Dynamic Programming Matrix (static SRAM allocation) */
static int32_t g_dtw_full_matrix[DTW_ENV_LEN][DTW_ENV_LEN];

/* Shared internal working buffers (kept static in BSS to protect RTOS task stacks) */
static int8_t g_hyb_x_q[DTW_MAX_CHANNELS * WINDOW_LEN];
static int8_t g_hyb_aux_q[MAX_FEATURES];
static int8_t g_hyb_env_q[DTW_ENV_LEN * DTW_MAX_CHANNELS];

static inline int32_t min3_i32(int32_t a, int32_t b, int32_t c) {
    int32_t m = (a < b) ? a : b;
    return (m < c) ? m : c;
}

static inline int8_t clip_int8_dtw(int32_t v) {
    if (v > 127) return 127;
    if (v < -127) return -127;
    return (int8_t)v;
}

static inline int32_t point_sq_dist_i8(const int8_t *a, const int8_t *b, int n_ch) {
    int32_t sum = 0;
    for (int c = 0; c < n_ch; c++) {
        int32_t diff = (int32_t)a[c] - (int32_t)b[c];
        sum += diff * diff;
    }
    return sum;
}

/* Compute Top-2 margin from an array of float scores */
static void compute_top2_f32(
    const float *scores,
    int n_cls,
    int *out_top1_idx,
    float *out_top1_val,
    float *out_top2_val
) {
    int b1 = 0;
    float v1 = -1e30f;
    float v2 = -1e30f;
    for (int k = 0; k < n_cls; k++) {
        float s = scores[k];
        if (k == 0 || s > v1) {
            v2 = v1;
            v1 = s;
            b1 = k;
        } else if (s > v2) {
            v2 = s;
        }
    }
    *out_top1_idx = b1;
    *out_top1_val = v1;
    *out_top2_val = v2;
}

/* Select top_m class indices sorted descending by float score */
static int select_topm_f32(const float *scores, int n_cls, int max_m, int *out_indices) {
    int m = (n_cls < max_m) ? n_cls : max_m;
    uint8_t used[DTW_MAX_CLASSES];
    memset(used, 0, sizeof(used));

    for (int r = 0; r < m; r++) {
        int best_k = -1;
        float best_s = -1e30f;
        for (int k = 0; k < n_cls; k++) {
            if (!used[k] && (best_k < 0 || scores[k] > best_s)) {
                best_k = k;
                best_s = scores[k];
            }
        }
        if (best_k >= 0) {
            used[best_k] = 1U;
            out_indices[r] = best_k;
        }
    }
    return m;
}

/* Select top_m class indices sorted descending by int32 logit */
static int select_topm_i32(const int32_t *logits, int n_cls, int max_m, int *out_indices) {
    int m = (n_cls < max_m) ? n_cls : max_m;
    uint8_t used[DTW_MAX_CLASSES];
    memset(used, 0, sizeof(used));

    for (int r = 0; r < m; r++) {
        int best_k = -1;
        int32_t best_s = -2147483647;
        for (int k = 0; k < n_cls; k++) {
            if (!used[k] && (best_k < 0 || logits[k] > best_s)) {
                best_k = k;
                best_s = logits[k];
            }
        }
        if (best_k >= 0) {
            used[best_k] = 1U;
            out_indices[r] = best_k;
        }
    }
    return m;
}

void extract_dtw_envelope_int8(
    const dtw_model_config_t *cfg,
    const int16_t *win_q16,
    float inv_adc_scale,
    int8_t *out_env_q
) {
    const int n_ch = cfg->num_channels;
    const int n_env = cfg->env_len;
    const float block_inv = 1.0f / (float)DTW_SUBBLOCK_SAMPLES;
    const float inv_q_scale = (cfg->env_quant_scale > 0.0f) ? (1.0f / cfg->env_quant_scale) : 1.0f;

    for (int t = 0; t < n_env; t++) {
        const int s_base = t * DTW_SUBBLOCK_SAMPLES;
        for (int c = 0; c < n_ch; c++) {
            int32_t sum_abs = 0;
            for (int s = 0; s < DTW_SUBBLOCK_SAMPLES; s++) {
                int32_t v = (int32_t)win_q16[(s_base + s) * n_ch + c];
                sum_abs += (v >= 0) ? v : -v;
            }
            float mav = (float)sum_abs * inv_adc_scale * block_inv;
            float z = (mav - cfg->env_mean[c]) * cfg->env_inv_std[c];
            int32_t q = (int32_t)lrintf(z * inv_q_scale);
            out_env_q[t * n_ch + c] = clip_int8_dtw(q);
        }
    }
}

/*
 * M03 helper: Exact Classic O(N^2) DTW distance matrix computation without optimization
 */
int32_t dtw_classic_dist_one(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int template_idx
) {
    const int n = cfg->env_len;
    const int n_ch = cfg->num_channels;
    const int8_t *tmpl = &cfg->templates_q[template_idx * n * n_ch];

    for (int i = 0; i < n; i++) {
        const int8_t *qi = &env_q[i * n_ch];
        for (int j = 0; j < n; j++) {
            int32_t d = point_sq_dist_i8(qi, &tmpl[j * n_ch], n_ch);
            if (i == 0 && j == 0) {
                g_dtw_full_matrix[0][0] = d;
            } else if (i == 0) {
                g_dtw_full_matrix[0][j] = d + g_dtw_full_matrix[0][j - 1];
            } else if (j == 0) {
                g_dtw_full_matrix[i][0] = d + g_dtw_full_matrix[i - 1][0];
            } else {
                g_dtw_full_matrix[i][j] = d + min3_i32(
                    g_dtw_full_matrix[i - 1][j - 1],
                    g_dtw_full_matrix[i - 1][j],
                    g_dtw_full_matrix[i][j - 1]
                );
            }
        }
    }
    return g_dtw_full_matrix[n - 1][n - 1];
}

/*
 * M04 / M05 helper: Sakoe-Chiba Band Constrained DTW O(w*N) using only two rows (O(w) space)
 * Pass early_abandon_limit = DTW_INF to disable pruning (pure M04), or best_cost for M05.
 */
int32_t dtw_sakoe_chiba_dist_one(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int template_idx,
    int32_t early_abandon_limit
) {
    const int n = cfg->env_len;
    const int w = cfg->band_w;
    const int n_ch = cfg->num_channels;
    const int8_t *tmpl = &cfg->templates_q[template_idx * n * n_ch];

    int32_t row_a[DTW_ENV_LEN];
    int32_t row_b[DTW_ENV_LEN];
    int32_t *prev = row_a;
    int32_t *curr = row_b;

    for (int j = 0; j < n; j++) {
        prev[j] = DTW_INF;
        curr[j] = DTW_INF;
    }

    for (int i = 0; i < n; i++) {
        const int j_start = (i - w > 0) ? (i - w) : 0;
        const int j_end   = (i + w < n - 1) ? (i + w) : (n - 1);
        const int8_t *qi = &env_q[i * n_ch];

        for (int j = j_start; j <= j_end; j++) {
            curr[j] = DTW_INF;
        }
        /* Clear adjacent outside cells from previous row reuse */
        if (j_start > 0) {
            curr[j_start - 1] = DTW_INF;
        }
        if (j_end + 1 < n) {
            curr[j_end + 1] = DTW_INF;
        }

        int32_t row_min = DTW_INF;
        for (int j = j_start; j <= j_end; j++) {
            int32_t d = point_sq_dist_i8(qi, &tmpl[j * n_ch], n_ch);
            int32_t best_prev;
            if (i == 0 && j == 0) {
                best_prev = 0;
            } else if (i == 0) {
                best_prev = curr[j - 1];
            } else if (j == 0) {
                best_prev = prev[0];
            } else {
                best_prev = min3_i32(prev[j - 1], prev[j], curr[j - 1]);
            }
            int32_t val = (best_prev >= DTW_INF) ? DTW_INF : (d + best_prev);
            curr[j] = val;
            if (val < row_min) {
                row_min = val;
            }
        }

        if (row_min >= early_abandon_limit) {
            return DTW_INF;
        }

        int32_t *tmp = prev;
        prev = curr;
        curr = tmp;
    }

    return prev[n - 1];
}

int32_t dtw_lb_keogh_dist_one(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int template_idx
) {
    const int n = cfg->env_len;
    const int n_ch = cfg->num_channels;
    const size_t offset = (size_t)template_idx * (size_t)n * (size_t)n_ch;
    const int8_t *u_ptr = &cfg->env_upper[offset];
    const int8_t *l_ptr = &cfg->env_lower[offset];

    int32_t lb_acc = 0;
    const int total_elems = n * n_ch;
    for (int idx = 0; idx < total_elems; idx++) {
        int32_t c_val = (int32_t)env_q[idx];
        int32_t u_val = (int32_t)u_ptr[idx];
        int32_t l_val = (int32_t)l_ptr[idx];
        if (c_val > u_val) {
            int32_t d = c_val - u_val;
            lb_acc += d * d;
        } else if (c_val < l_val) {
            int32_t d = l_val - c_val;
            lb_acc += d * d;
        }
    }
    return lb_acc;
}

/* M03: Classic DTW O(N^2) Nearest-Template Classifier */
int32_t dtw_classic_on2_predict(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int32_t *out_distances,
    dtw_hybrid_telemetry_t *tel
) {
    uint32_t t0 = cnn1d_read_ccount();
    const int n_cls = cfg->num_classes;
    int32_t best_cls = 0;
    int32_t best_cost = DTW_INF;

    for (int k = 0; k < n_cls; k++) {
        int32_t cost = dtw_classic_dist_one(cfg, env_q, k);
        if (out_distances != NULL) {
            out_distances[k] = cost;
        }
        if (k == 0 || cost < best_cost) {
            best_cost = cost;
            best_cls = k;
        }
    }
    uint32_t t1 = cnn1d_read_ccount();

    if (tel != NULL) {
        tel->cycles_total = t1 - t0;
        tel->pruned_lb_count = 0U;
        tel->dp_calls_count = (uint32_t)n_cls;
        tel->early_exit = 0U;
        tel->best_cost = best_cost;
    }
    return best_cls;
}

/* M04: Constrained Sakoe-Chiba DTW O(w*N) Nearest-Template Classifier */
int32_t dtw_sakoe_chiba_predict(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int32_t *out_distances,
    dtw_hybrid_telemetry_t *tel
) {
    uint32_t t0 = cnn1d_read_ccount();
    const int n_cls = cfg->num_classes;
    int32_t best_cls = 0;
    int32_t best_cost = DTW_INF;

    for (int k = 0; k < n_cls; k++) {
        int32_t cost = dtw_sakoe_chiba_dist_one(cfg, env_q, k, DTW_INF);
        if (out_distances != NULL) {
            out_distances[k] = cost;
        }
        if (k == 0 || cost < best_cost) {
            best_cost = cost;
            best_cls = k;
        }
    }
    uint32_t t1 = cnn1d_read_ccount();

    if (tel != NULL) {
        tel->cycles_total = t1 - t0;
        tel->pruned_lb_count = 0U;
        tel->dp_calls_count = (uint32_t)n_cls;
        tel->early_exit = 0U;
        tel->best_cost = best_cost;
    }
    return best_cls;
}

/*
 * M05: Linear LB_Keogh Pruned DTW O(N) best-case with tight best-first seeding
 * Guarantees exact equivalence to unpruned M04_dtw_sakoe_chiba while pruning uncompetitive candidates.
 */
int32_t dtw_lb_keogh_predict(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    dtw_hybrid_telemetry_t *tel
) {
    uint32_t t0 = cnn1d_read_ccount();
    const int n_cls = cfg->num_classes;
    int32_t lb_vals[DTW_MAX_CLASSES];

    int k_min = 0;
    int32_t lb_min = DTW_INF;
    for (int k = 0; k < n_cls; k++) {
        int32_t lb = dtw_lb_keogh_dist_one(cfg, env_q, k);
        lb_vals[k] = lb;
        if (k == 0 || lb < lb_min) {
            lb_min = lb;
            k_min = k;
        }
    }

    /* Evaluate strongest lower-bound candidate first to tighten best_cost immediately */
    int32_t best_cls = k_min;
    int32_t best_cost = dtw_sakoe_chiba_dist_one(cfg, env_q, k_min, DTW_INF);
    uint32_t dp_calls = 1U;
    uint32_t pruned_lb = 0U;

    for (int k = 0; k < n_cls; k++) {
        if (k == k_min) {
            continue;
        }
        if (lb_vals[k] >= best_cost) {
            pruned_lb++;
            continue;
        }
        dp_calls++;
        int32_t cost = dtw_sakoe_chiba_dist_one(cfg, env_q, k, best_cost);
        if (cost < best_cost || (cost == best_cost && k < best_cls)) {
            best_cost = cost;
            best_cls = k;
        }
    }
    uint32_t t1 = cnn1d_read_ccount();

    if (tel != NULL) {
        tel->cycles_total = t1 - t0;
        tel->pruned_lb_count = pruned_lb;
        tel->dp_calls_count = dp_calls;
        tel->early_exit = 0U;
        tel->best_cost = best_cost;
    }
    return best_cls;
}

/*
 * M06: Hybrid 1 — CNN-1D INT8 Backbone (GAP 24D) + Hudgins TD (5*C) -> Canonical LDA Head Q7
 */
int32_t hybrid_lda_cnn_head_predict_q7(
    const cnn1d_model_config_t *cnn_cfg,
    const hybrid_lda_head_t *head,
    const int8_t *x_q,
    const float *td_features,
    float *out_scores,
    dtw_hybrid_telemetry_t *tel
) {
    uint32_t t0 = cnn1d_read_ccount();

    int8_t gap_q[DTW_GAP_DIM];
    cnn1d_cycle_profile_t cnn_prof;
    cnn1d_backbone_gap_int8(cnn_cfg, x_q, gap_q, &cnn_prof);

    const int n_feat = head->num_features;
    const int n_td = head->num_channels * 5;
    const int n_cls = head->num_classes;
    int16_t feat_q8[MAX_HYBRID_FEATURES];

    for (int f = 0; f < DTW_GAP_DIM; f++) {
        if (head->zero_var_mask[f]) {
            feat_q8[f] = 0;
        } else {
            float z = ((float)gap_q[f] - head->mean[f]) * head->inv_scale[f];
            int32_t q = (int32_t)lrintf(z * 256.0f);
            if (q > 32767) q = 32767;
            if (q < -32768) q = -32768;
            feat_q8[f] = (int16_t)q;
        }
    }
    for (int j = 0; j < n_td; j++) {
        const int idx = DTW_GAP_DIM + j;
        if (head->zero_var_mask[idx]) {
            feat_q8[idx] = 0;
        } else {
            float z = (td_features[j] - head->mean[idx]) * head->inv_scale[idx];
            int32_t q = (int32_t)lrintf(z * 256.0f);
            if (q > 32767) q = 32767;
            if (q < -32768) q = -32768;
            feat_q8[idx] = (int16_t)q;
        }
    }

    int best_idx = 0;
    float best_score = -1e30f;
    for (int k = 0; k < n_cls; k++) {
        int32_t acc_int = 0;
        const int8_t *w_row = &head->coef_q7[k * n_feat];
        for (int j = 0; j < n_feat; j++) {
            acc_int += (int32_t)w_row[j] * (int32_t)feat_q8[j];
        }
        float score = ((float)acc_int * head->weight_scale[k] * (1.0f / 256.0f)) + head->intercept[k];
        if (out_scores != NULL) {
            out_scores[k] = score;
        }
        if (k == 0 || score > best_score) {
            best_score = score;
            best_idx = k;
        }
    }
    uint32_t t1 = cnn1d_read_ccount();

    if (tel != NULL) {
        tel->cycles_total = t1 - t0;
        tel->pruned_lb_count = 0U;
        tel->dp_calls_count = 0U;
        tel->early_exit = 0U;
        tel->best_cost = 0;
    }
    return head->classes[best_idx];
}

/*
 * M07: Hybrid 2 — Early-Exit Gatekeeper Cascade (Tier 1 Micro-LDA Q7 -> Margin Gate -> Tier 2)
 */
int32_t hybrid_lda_cnn_cascade_predict(
    const micro_lda_model_t *lda_model,
    const cnn1d_model_config_t *cnn_cfg,
    const hybrid_lda_head_t *head,
    const hybrid_fusion_config_t *fcfg,
    const float *win_fp32,
    const int8_t *pre_x_q,
    const int8_t *pre_aux_q,
    const float *td_features,
    dtw_hybrid_telemetry_t *tel
) {
    uint32_t t0 = cnn1d_read_ccount();
    float lda_scores[MAX_CLASSES];

    int32_t tier1_pred = micro_lda_predict_q7(lda_model, td_features, lda_scores);
    int top1_idx = 0;
    float top1_val = 0.0f;
    float top2_val = 0.0f;
    compute_top2_f32(lda_scores, lda_model->num_classes, &top1_idx, &top1_val, &top2_val);
    float margin = top1_val - top2_val;

    if (margin >= fcfg->cascade_margin_threshold) {
        uint32_t t1 = cnn1d_read_ccount();
        if (tel != NULL) {
            tel->cycles_total = t1 - t0;
            tel->pruned_lb_count = 0U;
            tel->dp_calls_count = 0U;
            tel->early_exit = 1U;
            tel->best_cost = 0;
        }
        return tier1_pred;
    }

    const int8_t *x_ptr = pre_x_q;
    const int8_t *aux_ptr = pre_aux_q;
    if (x_ptr == NULL && win_fp32 != NULL) {
        cnn1d_quantize_input(cnn_cfg, win_fp32, td_features, g_hyb_x_q, g_hyb_aux_q);
        x_ptr = g_hyb_x_q;
        aux_ptr = g_hyb_aux_q;
    }

    int32_t final_pred;
    if (fcfg->cascade_use_cnn1d_tier2) {
        cnn1d_cycle_profile_t cnn_prof;
        int32_t pred_idx = cnn1d_forward_int8(cnn_cfg, x_ptr, aux_ptr, NULL, &cnn_prof);
        final_pred = (pred_idx >= 0 && pred_idx < lda_model->num_classes)
            ? lda_model->classes[pred_idx]
            : pred_idx;
    } else {
        final_pred = hybrid_lda_cnn_head_predict_q7(cnn_cfg, head, x_ptr, td_features, NULL, NULL);
    }

    uint32_t t1 = cnn1d_read_ccount();
    if (tel != NULL) {
        tel->cycles_total = t1 - t0;
        tel->pruned_lb_count = 0U;
        tel->dp_calls_count = 0U;
        tel->early_exit = 0U;
        tel->best_cost = 0;
    }
    return final_pred;
}

/*
 * M08: Hybrid DTW + LDA (Top-M Micro-LDA candidates fused with Sakoe-Chiba DTW alignment)
 */
int32_t hybrid_dtw_lda_predict(
    const micro_lda_model_t *lda_model,
    const dtw_model_config_t *dtw_cfg,
    const hybrid_fusion_config_t *fcfg,
    const float *td_features,
    const int8_t *env_q,
    dtw_hybrid_telemetry_t *tel
) {
    uint32_t t0 = cnn1d_read_ccount();
    float lda_scores[MAX_CLASSES];
    (void)micro_lda_predict_q7(lda_model, td_features, lda_scores);

    int cand_indices[DTW_TOPM_HYBRID] = {0};
    int m = select_topm_f32(lda_scores, lda_model->num_classes, DTW_TOPM_HYBRID, cand_indices);

    const float inv_norm = 1.0f / (float)(dtw_cfg->env_len * dtw_cfg->num_channels);
    int best_idx = cand_indices[0];
    float best_fused = -1e30f;

    for (int r = 0; r < m; r++) {
        int k = cand_indices[r];
        int32_t dtw_cost = dtw_sakoe_chiba_dist_one(dtw_cfg, env_q, k, DTW_INF);
        float fused = lda_scores[k] - fcfg->gamma_dtw_lda * ((float)dtw_cost * inv_norm);
        if (r == 0 || fused > best_fused) {
            best_fused = fused;
            best_idx = k;
        }
    }
    uint32_t t1 = cnn1d_read_ccount();

    if (tel != NULL) {
        tel->cycles_total = t1 - t0;
        tel->pruned_lb_count = (uint32_t)(lda_model->num_classes - m);
        tel->dp_calls_count = (uint32_t)m;
        tel->early_exit = 0U;
        tel->best_cost = 0;
    }
    return lda_model->classes[best_idx];
}

/*
 * M09: Hybrid DTW + CNN (Top-M CNN-1D candidates fused with Sakoe-Chiba DTW alignment)
 */
int32_t hybrid_dtw_cnn_predict(
    const cnn1d_model_config_t *cnn_cfg,
    const dtw_model_config_t *dtw_cfg,
    const hybrid_fusion_config_t *fcfg,
    const int8_t *x_q,
    const int8_t *aux_q,
    const int8_t *env_q,
    dtw_hybrid_telemetry_t *tel
) {
    uint32_t t0 = cnn1d_read_ccount();
    int32_t cnn_logits[CNN1D_MAX_CLASSES];
    cnn1d_cycle_profile_t cnn_prof;
    (void)cnn1d_forward_int8(cnn_cfg, x_q, aux_q, cnn_logits, &cnn_prof);

    int cand_indices[DTW_TOPM_HYBRID] = {0};
    int m = select_topm_i32(cnn_logits, cnn_cfg->num_classes, DTW_TOPM_HYBRID, cand_indices);

    const float inv_norm = 1.0f / (float)(dtw_cfg->env_len * dtw_cfg->num_channels);
    const float logit_scale = cnn_cfg->fc_eff_scale;
    int best_idx = cand_indices[0];
    float best_fused = -1e30f;

    for (int r = 0; r < m; r++) {
        int k = cand_indices[r];
        int32_t dtw_cost = dtw_sakoe_chiba_dist_one(dtw_cfg, env_q, k, DTW_INF);
        float cnn_s = (float)cnn_logits[k] * logit_scale;
        float fused = cnn_s - fcfg->gamma_dtw_cnn * ((float)dtw_cost * inv_norm);
        if (r == 0 || fused > best_fused) {
            best_fused = fused;
            best_idx = k;
        }
    }
    uint32_t t1 = cnn1d_read_ccount();

    if (tel != NULL) {
        tel->cycles_total = t1 - t0;
        tel->pruned_lb_count = (uint32_t)(cnn_cfg->num_classes - m);
        tel->dp_calls_count = (uint32_t)m;
        tel->early_exit = 0U;
        tel->best_cost = 0;
    }
    return best_idx;
}

/*
 * M10: Tri-Hybrid DTW + LDA + CNN
 *   Stage 1: Micro-LDA Q7 margin gate; Early-Exit if margin >= tri_margin_threshold
 *   Stage 2: Micro-LDA picks Top-3 candidates + Hybrid 1 scores + LB_Keogh pruned Sakoe-Chiba DTW
 */
int32_t hybrid_dtw_lda_cnn_predict(
    const micro_lda_model_t *lda_model,
    const cnn1d_model_config_t *cnn_cfg,
    const hybrid_lda_head_t *head,
    const dtw_model_config_t *dtw_cfg,
    const hybrid_fusion_config_t *fcfg,
    const float *win_fp32,
    const int8_t *pre_x_q,
    const int8_t *pre_env_q,
    const int16_t *win_q16,
    float inv_adc_scale,
    const float *td_features,
    dtw_hybrid_telemetry_t *tel
) {
    uint32_t t0 = cnn1d_read_ccount();
    float lda_scores[MAX_CLASSES];

    int32_t stage1_pred = micro_lda_predict_q7(lda_model, td_features, lda_scores);
    int top1_idx = 0;
    float top1_val = 0.0f;
    float top2_val = 0.0f;
    compute_top2_f32(lda_scores, lda_model->num_classes, &top1_idx, &top1_val, &top2_val);
    float margin = top1_val - top2_val;

    if (margin >= fcfg->tri_margin_threshold) {
        uint32_t t1 = cnn1d_read_ccount();
        if (tel != NULL) {
            tel->cycles_total = t1 - t0;
            tel->pruned_lb_count = (uint32_t)lda_model->num_classes;
            tel->dp_calls_count = 0U;
            tel->early_exit = 1U;
            tel->best_cost = 0;
        }
        return stage1_pred;
    }

    const int8_t *x_ptr = pre_x_q;
    if (x_ptr == NULL && win_fp32 != NULL) {
        cnn1d_quantize_input(cnn_cfg, win_fp32, NULL, g_hyb_x_q, NULL);
        x_ptr = g_hyb_x_q;
    }

    const int8_t *env_ptr = pre_env_q;
    if (env_ptr == NULL && win_q16 != NULL) {
        extract_dtw_envelope_int8(dtw_cfg, win_q16, inv_adc_scale, g_hyb_env_q);
        env_ptr = g_hyb_env_q;
    }

    float hyb1_scores[MAX_CLASSES];
    (void)hybrid_lda_cnn_head_predict_q7(cnn_cfg, head, x_ptr, td_features, hyb1_scores, NULL);

    int cand_indices[DTW_TRI_TOPM] = {0};
    int m = select_topm_f32(lda_scores, lda_model->num_classes, DTW_TRI_TOPM, cand_indices);

    /* Sort the 3 shortlisted candidates descending by Hybrid 1 score to maximize LB_Keogh pruning */
    for (int a = 0; a < m - 1; a++) {
        for (int b = a + 1; b < m; b++) {
            if (hyb1_scores[cand_indices[b]] > hyb1_scores[cand_indices[a]]) {
                int tmp = cand_indices[a];
                cand_indices[a] = cand_indices[b];
                cand_indices[b] = tmp;
            }
        }
    }

    const float inv_norm = 1.0f / (float)(dtw_cfg->env_len * dtw_cfg->num_channels);
    const float gamma = fcfg->gamma_tri_hybrid;
    int best_idx = cand_indices[0];
    float best_fused = -1e30f;
    uint32_t dp_calls = 0U;
    uint32_t pruned = (uint32_t)(lda_model->num_classes - m);

    for (int r = 0; r < m; r++) {
        int k = cand_indices[r];
        if (r > 0 && gamma > 0.0f) {
            int32_t lb = dtw_lb_keogh_dist_one(dtw_cfg, env_ptr, k);
            float optimistic_upper = hyb1_scores[k] - gamma * ((float)lb * inv_norm);
            if (optimistic_upper <= best_fused) {
                pruned++;
                continue;
            }
        }
        dp_calls++;
        int32_t dtw_cost = dtw_sakoe_chiba_dist_one(dtw_cfg, env_ptr, k, DTW_INF);
        float fused = hyb1_scores[k] - gamma * ((float)dtw_cost * inv_norm);
        if (r == 0 || fused > best_fused) {
            best_fused = fused;
            best_idx = k;
        }
    }
    uint32_t t1 = cnn1d_read_ccount();

    if (tel != NULL) {
        tel->cycles_total = t1 - t0;
        tel->pruned_lb_count = pruned;
        tel->dp_calls_count = dp_calls;
        tel->early_exit = 0U;
        tel->best_cost = 0;
    }
    return head->classes[best_idx];
}
