/*
 * dtw_hybrid_engine.h — Pure C99 DTW (Classic O(N^2), Sakoe-Chiba O(w*N), LB_Keogh O(N))
 *                       and Hybrid Architectures (DTW, Micro-LDA Q7, CNN-1D INT8)
 *
 * Implements 8 additional classifiers alongside Micro-LDA Q7 (M01) and CNN-1D INT8 (M02):
 *   M03: Classic Dynamic Time Warping O(N^2) time & space (no asymptotic optimization)
 *   M04: Constrained DTW Sakoe-Chiba Band O(w*N) time, O(w) / two-row space
 *   M05: Linear Lower-Bound Pruned DTW LB_Keogh O(N) best-case with Early Abandoning
 *   M06: Hybrid 1 — CNN-1D INT8 Backbone (GAP 24D) + Hudgins TD (5*C) + LDA Head Q7
 *   M07: Hybrid 2 — Early-Exit Cascade (Tier 1 Micro-LDA Q7 -> Margin Gate -> Tier 2 CNN/Hyb1)
 *   M08: Hybrid DTW + LDA (Top-M LDA candidates fused with elastic DTW distance)
 *   M09: Hybrid DTW + CNN (Top-M CNN candidates fused with elastic DTW distance)
 *   M10: Tri-Hybrid DTW + LDA + CNN (Margin Gate -> Top-3 LDA + LB_Keogh + Hybrid 1)
 */

#ifndef DTW_HYBRID_ENGINE_H
#define DTW_HYBRID_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include "semg_pipeline.h"
#include "cnn1d_inference.h"

#define DTW_ENV_LEN          40
#define DTW_SUBBLOCK_SAMPLES 10
#define DTW_SAKOE_W          4
#define DTW_MAX_CHANNELS     12
#define DTW_MAX_CLASSES      50
#define DTW_GAP_DIM          24
#define MAX_HYBRID_FEATURES  (DTW_GAP_DIM + MAX_FEATURES) /* 24 + 60 = 84 */
#define DTW_TOPM_HYBRID      5
#define DTW_TRI_TOPM         3

/* Unified 10-model identifier table shared across QEMU and ESP32-S3 HIL */
typedef enum {
    MODEL_M01_MICRO_LDA_Q7          = 0,
    MODEL_M02_CNN1D_INT8            = 1,
    MODEL_M03_DTW_CLASSIC_ON2       = 2,
    MODEL_M04_DTW_SAKOE_CHIBA       = 3,
    MODEL_M05_DTW_LB_KEOGH          = 4,
    MODEL_M06_HYBRID_LDA_CNN_HEAD   = 5,
    MODEL_M07_HYBRID_LDA_CNN_CASC   = 6,
    MODEL_M08_HYBRID_DTW_LDA        = 7,
    MODEL_M09_HYBRID_DTW_CNN        = 8,
    MODEL_M10_HYBRID_DTW_LDA_CNN    = 9,
    MODEL_COUNT                     = 10
} semg_model_id_t;

/* Multi-channel DTW template bank (int8 Q8 envelopes + precomputed Keogh bounds U, L) */
typedef struct {
    int            num_channels;
    int            num_classes;
    int            env_len;         /* DTW_ENV_LEN (40) */
    int            band_w;          /* DTW_SAKOE_W (4) */
    const float   *env_mean;        /* [num_channels] */
    const float   *env_inv_std;     /* [num_channels] */
    float          env_quant_scale; /* s_env: q = clip(round((mav - mean) * inv_std / s_env), -127, 127) */
    const int8_t  *templates_q;     /* [num_classes * DTW_ENV_LEN * num_channels] */
    const int8_t  *env_upper;       /* [num_classes * DTW_ENV_LEN * num_channels] */
    const int8_t  *env_lower;       /* [num_classes * DTW_ENV_LEN * num_channels] */
} dtw_model_config_t;

/* Hybrid 1 LDA Head operating on concatenated [GAP (24) || TD (5*C)] features */
typedef struct {
    int            num_channels;
    int            num_features;    /* 24 + 5 * num_channels */
    int            num_classes;
    const int32_t *classes;         /* [num_classes] */
    const float   *mean;            /* [num_features] */
    const float   *inv_scale;       /* [num_features] */
    const uint8_t *zero_var_mask;   /* [num_features] */
    const int8_t  *coef_q7;         /* [num_classes * num_features] */
    const float   *weight_scale;    /* [num_classes] */
    const float   *intercept;       /* [num_classes] */
} hybrid_lda_head_t;

/* Calibrated thresholds & fusion weights for Cascaded and Multi-Modal Hybrids */
typedef struct {
    float   cascade_margin_threshold;  /* M07 threshold on Top1-Top2 Micro-LDA score margin */
    uint8_t cascade_use_cnn1d_tier2;   /* 1 = use M02 CNN1D as Tier2, 0 = use M06 Hybrid1 as Tier2 */
    float   gamma_dtw_lda;             /* M08 fusion penalty scale on normalized DTW distance */
    float   gamma_dtw_cnn;             /* M09 fusion penalty scale on normalized DTW distance */
    float   tri_margin_threshold;      /* M10 Stage-1 Micro-LDA Early-Exit threshold */
    float   gamma_tri_hybrid;          /* M10 Stage-2 DTW penalty scale on Hybrid 1 scores */
} hybrid_fusion_config_t;

/* Detailed runtime telemetry for single-window DTW / Hybrid execution */
typedef struct {
    uint32_t cycles_total;
    uint32_t pruned_lb_count;
    uint32_t dp_calls_count;
    uint8_t  early_exit;
    int32_t  best_cost;
} dtw_hybrid_telemetry_t;

/* Extract deterministic 40-step x C int8 temporal activation envelope from [400, C] int16 window */
void extract_dtw_envelope_int8(
    const dtw_model_config_t *cfg,
    const int16_t *win_q16,
    float inv_adc_scale,
    int8_t *out_env_q
);

/* Compute exact Classic O(N^2) squared DTW cost between query env_q and template k (full NxN matrix) */
int32_t dtw_classic_dist_one(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int template_idx
);

/* Compute Sakoe-Chiba constrained O(w*N) squared DTW cost using 2 static rows (O(w) space) */
int32_t dtw_sakoe_chiba_dist_one(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int template_idx,
    int32_t early_abandon_limit
);

/* Compute linear O(N) LB_Keogh squared lower bound for template_idx */
int32_t dtw_lb_keogh_dist_one(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int template_idx
);

/* M03: Classic DTW O(N^2) Nearest-Template Classifier */
int32_t dtw_classic_on2_predict(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int32_t *out_distances,
    dtw_hybrid_telemetry_t *tel
);

/* M04: Sakoe-Chiba Band DTW O(w*N) Nearest-Template Classifier */
int32_t dtw_sakoe_chiba_predict(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    int32_t *out_distances,
    dtw_hybrid_telemetry_t *tel
);

/* M05: LB_Keogh Lower-Bound Pruned DTW O(N) Best-Case Nearest-Template Classifier */
int32_t dtw_lb_keogh_predict(
    const dtw_model_config_t *cfg,
    const int8_t *env_q,
    dtw_hybrid_telemetry_t *tel
);

/* M06: Hybrid 1 — CNN-1D INT8 Backbone (GAP 24D) + Hudgins TD (5*C) + Canonical LDA Head Q7 */
int32_t hybrid_lda_cnn_head_predict_q7(
    const cnn1d_model_config_t *cnn_cfg,
    const hybrid_lda_head_t *head,
    const int8_t *x_q,
    const float *td_features,
    float *out_scores,
    dtw_hybrid_telemetry_t *tel
);

/* M07: Hybrid 2 — Early-Exit Gatekeeper Cascade (Micro-LDA Q7 -> Margin Gate -> Tier 2) */
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
);

/* M08: Hybrid DTW + LDA (Top-M Micro-LDA candidates reranked with Sakoe-Chiba DTW) */
int32_t hybrid_dtw_lda_predict(
    const micro_lda_model_t *lda_model,
    const dtw_model_config_t *dtw_cfg,
    const hybrid_fusion_config_t *fcfg,
    const float *td_features,
    const int8_t *env_q,
    dtw_hybrid_telemetry_t *tel
);

/* M09: Hybrid DTW + CNN (Top-M CNN-1D candidates reranked with Sakoe-Chiba DTW) */
int32_t hybrid_dtw_cnn_predict(
    const cnn1d_model_config_t *cnn_cfg,
    const dtw_model_config_t *dtw_cfg,
    const hybrid_fusion_config_t *fcfg,
    const int8_t *x_q,
    const int8_t *aux_q,
    const int8_t *env_q,
    dtw_hybrid_telemetry_t *tel
);

/* M10: Tri-Hybrid DTW + LDA + CNN (Margin Gate -> Top-3 LDA + LB_Keogh + Hybrid 1) */
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
);

#endif /* DTW_HYBRID_ENGINE_H */
