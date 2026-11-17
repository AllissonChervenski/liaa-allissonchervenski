/*
 * semg_rtos_app.c — Native ESP-IDF Dual-Core FreeRTOS sEMG Processing Pipeline
 *
 * Implements the two-core architecture on ESP32-S3-WROOM (Xtensa LX7 @ 240 MHz):
 *   - Core 0 (PRO_CPU, Priority 10): vTaskAquisicaoCore0
 *       Waits on xBufferReadySem, applies Causal Butterworth 20-450 Hz SOS DF2T
 *       filtering (unless bypassed via HIL), feeds StatefulWindowBuffer (W=400, S=100),
 *       and dispatches lightweight window descriptors via xQueueJanelaPronta.
 *   - Core 1 (APP_CPU, Priority 5): vTaskInferenciaCore1
 *       Receives window descriptors, executes Micro-LDA (Q7, Zero-FPU) or
 *       CNN-1D (Symmetric INT8, 4.61 KB arena), applies 3-window majority vote,
 *       updates optional status GPIO, and emits structured JSON telemetry.
 */

#include "../include/semg_rtos_app.h"
#include "../include/semg_hal_acq.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include "driver/gpio.h"
#include "esp_log.h"

#include "semg_pipeline.h"
#include "cnn1d_inference.h"
#include "cnn1d_weights_int8.h"
#include "micro_lda_weights.h"
#include "dtw_hybrid_engine.h"
#include "dtw_hybrid_weights.h"

static const char *TAG = "semg_rtos";

/* Static Ping-Pong DMA double buffers */
static int16_t dma_ping_buffer[SEMG_DMA_BLOCK_SAMPLES * MAX_CHANNELS];
static int16_t dma_pong_buffer[SEMG_DMA_BLOCK_SAMPLES * MAX_CHANNELS];

typedef struct {
    int16_t          *active_dma_buffer;
    const int16_t    *processing_buffer;
    uint32_t          pending_samples;
    volatile uint8_t  buffer_id;
    SemaphoreHandle_t xBufferReadySem;
} dma_ping_pong_ctrl_t;

static dma_ping_pong_ctrl_t g_dma_ctrl;
static portMUX_TYPE g_dma_spinlock = portMUX_INITIALIZER_UNLOCKED;

/* Static double-buffered window slots between Core 0 and Core 1 */
static float   g_slot_win_fp32[2][WINDOW_LEN * MAX_CHANNELS];
static int16_t g_slot_win_q16[2][WINDOW_LEN * MAX_CHANNELS];

typedef struct {
    uint32_t       window_index;
    uint32_t       core0_cycles;
    const float   *win_fp32_ptr;
    const int16_t *win_q16_ptr;
} window_descriptor_t;

/* Static Core 0 working buffers (never placed on task stack) */
static float g_core0_in_chunk_fp32[SEMG_DMA_BLOCK_SAMPLES * MAX_CHANNELS];
static float g_core0_out_chunk_fp32[SEMG_DMA_BLOCK_SAMPLES * MAX_CHANNELS];
static float g_core0_win_fp32[WINDOW_LEN * MAX_CHANNELS];

/* Static Core 1 working buffers for CNN-1D & DTW quantization */
static int8_t  g_core1_cnn_x_q[MAX_CHANNELS * WINDOW_LEN];
static int8_t  g_core1_cnn_aux_q[MAX_FEATURES];
static int8_t  g_core1_dtw_env_q[DTW_ENV_LEN * MAX_CHANNELS];
static int32_t g_core1_dtw_dists[DTW_MAX_CLASSES];

/* DSP & Classifier state */
static causal_sos_filter_t         g_sos_filter;
static stateful_window_buffer_t    g_win_buf;
static td_feature_extractor_t      g_td_cfg;
static micro_lda_model_t           g_lda_model;
static const cnn1d_model_config_t     *g_cnn_cfg    = NULL;
static const dtw_model_config_t       *g_dtw_cfg    = NULL;
static const hybrid_lda_head_t        *g_hyb1_head  = NULL;
static const hybrid_fusion_config_t   *g_fusion_cfg = NULL;

const char *semg_rtos_engine_name(uint8_t engine_mode) {
    static const char *const k_names[SEMG_ENGINE_COUNT] = {
        "micro_lda_q7",
        "cnn1d_int8",
        "dtw_classic_on2",
        "dtw_sakoe_chiba",
        "dtw_lb_keogh",
        "hybrid_lda_cnn_head",
        "hybrid_lda_cnn_cascade",
        "hybrid_dtw_lda",
        "hybrid_dtw_cnn",
        "hybrid_dtw_lda_cnn"
    };
    if (engine_mode < (uint8_t)SEMG_ENGINE_COUNT) {
        return k_names[engine_mode];
    }
    return k_names[0];
}

static volatile uint8_t  g_engine_mode   = SEMG_ENGINE_MICRO_LDA_Q7;
static volatile uint8_t  g_bypass_filter = 0U;
static volatile uint32_t g_window_seq    = 0U;
static uint32_t          g_num_channels  = 4U;
static uint32_t          g_num_classes   = 10U;
static float             g_adc_scale     = OZDEMIR_LDA_ADC_SCALE;

/* 3-window temporal majority voting ring */
static int32_t  g_majority_ring[3] = {-1, -1, -1};
static uint32_t g_majority_pos = 0U;

/* FreeRTOS handles */
static QueueHandle_t xQueueJanelaPronta = NULL;
static TaskHandle_t  xTaskCore0Handle   = NULL;
static TaskHandle_t  xTaskCore1Handle   = NULL;

static int32_t apply_majority_vote_3(int32_t current_pred) {
    g_majority_ring[g_majority_pos % 3U] = current_pred;
    g_majority_pos++;
    if (g_majority_pos < 3U) {
        return current_pred;
    }
    int32_t a = g_majority_ring[0];
    int32_t b = g_majority_ring[1];
    int32_t c = g_majority_ring[2];
    if (a == b || a == c) {
        return a;
    }
    if (b == c) {
        return b;
    }
    return current_pred;
}

static inline int16_t clamp_int16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

static void init_status_led(void) {
#if CONFIG_SEMG_STATUS_LED_GPIO >= 0
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << CONFIG_SEMG_STATUS_LED_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);
    gpio_set_level((gpio_num_t)CONFIG_SEMG_STATUS_LED_GPIO, 0);
#endif
}

static void update_status_led(int32_t predicted_class) {
#if CONFIG_SEMG_STATUS_LED_GPIO >= 0
    gpio_set_level((gpio_num_t)CONFIG_SEMG_STATUS_LED_GPIO, (predicted_class != 0) ? 1 : 0);
#else
    (void)predicted_class;
#endif
}

/*
 * Core 0 (PRO_CPU, Priority 10): Deterministic Signal Conditioning & Windowing
 */
static void vTaskAquisicaoCore0(void *pvParameters) {
    (void)pvParameters;
    char tx_line[128];

    while (xSemaphoreTake(g_dma_ctrl.xBufferReadySem, portMAX_DELAY) == pdTRUE) {
        uint32_t c_start = read_ccount();

        taskENTER_CRITICAL(&g_dma_spinlock);
        const int16_t *dma_block = g_dma_ctrl.processing_buffer;
        uint32_t n_samples = g_dma_ctrl.pending_samples;
        taskEXIT_CRITICAL(&g_dma_spinlock);

        if (n_samples > SEMG_DMA_BLOCK_SAMPLES) {
            n_samples = SEMG_DMA_BLOCK_SAMPLES;
        }

        const uint32_t C = g_num_channels;
        const float inv_scale = g_td_cfg.inv_scale;
        const uint32_t total_elems = n_samples * C;

        for (uint32_t i = 0; i < total_elems; i++) {
            g_core0_in_chunk_fp32[i] = (float)dma_block[i] * inv_scale;
        }

        if (g_bypass_filter) {
            memcpy(g_core0_out_chunk_fp32, g_core0_in_chunk_fp32, total_elems * sizeof(float));
        } else {
            sos_filter_process_chunk(&g_sos_filter, g_core0_in_chunk_fp32, g_core0_out_chunk_fp32, (int)n_samples);
        }

        int windows_emitted = 0;
        for (uint32_t s = 0; s < n_samples; s++) {
            int emitted = window_buffer_push_sample(
                &g_win_buf,
                &g_core0_out_chunk_fp32[s * C],
                g_core0_win_fp32
            );
            if (emitted) {
                windows_emitted++;
                uint32_t w_idx = g_window_seq++;
                uint32_t slot = w_idx & 1U;
                uint32_t win_elems = (uint32_t)WINDOW_LEN * C;

                memcpy(g_slot_win_fp32[slot], g_core0_win_fp32, win_elems * sizeof(float));
                for (uint32_t k = 0; k < win_elems; k++) {
                    g_slot_win_q16[slot][k] = clamp_int16((int32_t)lrintf(g_core0_win_fp32[k] * g_adc_scale));
                }

                uint32_t c_end = read_ccount();
                window_descriptor_t desc = {
                    .window_index = w_idx,
                    .core0_cycles = c_end - c_start,
                    .win_fp32_ptr = g_slot_win_fp32[slot],
                    .win_q16_ptr  = g_slot_win_q16[slot]
                };
                if (xQueueSend(xQueueJanelaPronta, &desc, 0) != pdTRUE) {
                    ESP_LOGW(TAG, "xQueueJanelaPronta full at win=%lu", (unsigned long)w_idx);
                }
            }
        }

#ifdef CONFIG_SEMG_ACQ_MODE_HIL_SERIAL
        if (windows_emitted == 0) {
            snprintf(
                tx_line,
                sizeof(tx_line),
                "SEMG_BUFFERED: {\"buffered_samples\":%d,\"window_len\":%d}",
                g_win_buf.buffered_samples,
                g_win_buf.window_length
            );
            semg_hal_tx_line(tx_line);
        }
#else
        (void)windows_emitted;
        (void)tx_line;
#endif
    }
}

/*
 * Core 1 (APP_CPU, Priority 5): Feature Extraction & Quantized Inference
 */
static void vTaskInferenciaCore1(void *pvParameters) {
    (void)pvParameters;
    window_descriptor_t desc;
    float td_features[MAX_FEATURES];
    float lda_scores[MAX_CLASSES];
    int32_t cnn_logits[CNN1D_MAX_CLASSES];
    cnn1d_cycle_profile_t cnn_prof;
    char tx_line[256];

    while (xQueueReceive(xQueueJanelaPronta, &desc, portMAX_DELAY) == pdPASS) {
        uint32_t c_start = read_ccount();
        uint32_t clf_cycles = 0U;
        uint8_t  early_exit = 0U;
        int32_t raw_pred = 0;
        const uint8_t active_mode = g_engine_mode;
        dtw_hybrid_telemetry_t hyb_tel;
        memset(&hyb_tel, 0, sizeof(hyb_tel));

        extract_td_features_int16(
            desc.win_q16_ptr,
            WINDOW_LEN,
            (int)g_num_channels,
            g_td_cfg.epsilon_int,
            g_td_cfg.inv_scale,
            td_features
        );

        switch (active_mode) {
            case SEMG_ENGINE_MICRO_LDA_Q7: {
                uint32_t c0 = read_ccount();
                raw_pred = micro_lda_predict_q7(&g_lda_model, td_features, lda_scores);
                uint32_t c1 = read_ccount();
                clf_cycles = c1 - c0;
                break;
            }
            case SEMG_ENGINE_CNN1D_INT8: {
                cnn1d_quantize_input(
                    g_cnn_cfg, desc.win_fp32_ptr, td_features, g_core1_cnn_x_q, g_core1_cnn_aux_q
                );
                int32_t pred_idx = cnn1d_forward_int8(
                    g_cnn_cfg, g_core1_cnn_x_q, g_core1_cnn_aux_q, cnn_logits, &cnn_prof
                );
                clf_cycles = cnn_prof.cycles_total;
                raw_pred = (pred_idx >= 0 && pred_idx < g_lda_model.num_classes)
                    ? g_lda_model.classes[pred_idx]
                    : pred_idx;
                break;
            }
            case SEMG_ENGINE_DTW_CLASSIC_ON2:
            case SEMG_ENGINE_DTW_SAKOE_CHIBA:
            case SEMG_ENGINE_DTW_LB_KEOGH: {
                extract_dtw_envelope_int8(g_dtw_cfg, desc.win_q16_ptr, g_td_cfg.inv_scale, g_core1_dtw_env_q);
                if (active_mode == SEMG_ENGINE_DTW_CLASSIC_ON2) {
                    raw_pred = dtw_classic_on2_predict(g_dtw_cfg, g_core1_dtw_env_q, g_core1_dtw_dists, &hyb_tel);
                } else if (active_mode == SEMG_ENGINE_DTW_SAKOE_CHIBA) {
                    raw_pred = dtw_sakoe_chiba_predict(g_dtw_cfg, g_core1_dtw_env_q, g_core1_dtw_dists, &hyb_tel);
                } else {
                    raw_pred = dtw_lb_keogh_predict(g_dtw_cfg, g_core1_dtw_env_q, &hyb_tel);
                }
                clf_cycles = hyb_tel.cycles_total;
                break;
            }
            case SEMG_ENGINE_HYBRID_LDA_CNN_HEAD: {
                cnn1d_quantize_input(
                    g_cnn_cfg, desc.win_fp32_ptr, td_features, g_core1_cnn_x_q, g_core1_cnn_aux_q
                );
                raw_pred = hybrid_lda_cnn_head_predict_q7(
                    g_cnn_cfg, g_hyb1_head, g_core1_cnn_x_q, td_features, lda_scores, &hyb_tel
                );
                clf_cycles = hyb_tel.cycles_total;
                break;
            }
            case SEMG_ENGINE_HYBRID_LDA_CNN_CASC: {
                raw_pred = hybrid_lda_cnn_cascade_predict(
                    &g_lda_model,
                    g_cnn_cfg,
                    g_hyb1_head,
                    g_fusion_cfg,
                    desc.win_fp32_ptr,
                    NULL,
                    NULL,
                    td_features,
                    &hyb_tel
                );
                clf_cycles = hyb_tel.cycles_total;
                early_exit = hyb_tel.early_exit;
                break;
            }
            case SEMG_ENGINE_HYBRID_DTW_LDA: {
                extract_dtw_envelope_int8(g_dtw_cfg, desc.win_q16_ptr, g_td_cfg.inv_scale, g_core1_dtw_env_q);
                raw_pred = hybrid_dtw_lda_predict(
                    &g_lda_model, g_dtw_cfg, g_fusion_cfg, td_features, g_core1_dtw_env_q, &hyb_tel
                );
                clf_cycles = hyb_tel.cycles_total;
                break;
            }
            case SEMG_ENGINE_HYBRID_DTW_CNN: {
                cnn1d_quantize_input(
                    g_cnn_cfg, desc.win_fp32_ptr, td_features, g_core1_cnn_x_q, g_core1_cnn_aux_q
                );
                extract_dtw_envelope_int8(g_dtw_cfg, desc.win_q16_ptr, g_td_cfg.inv_scale, g_core1_dtw_env_q);
                raw_pred = hybrid_dtw_cnn_predict(
                    g_cnn_cfg, g_dtw_cfg, g_fusion_cfg, g_core1_cnn_x_q, g_core1_cnn_aux_q, g_core1_dtw_env_q, &hyb_tel
                );
                clf_cycles = hyb_tel.cycles_total;
                break;
            }
            case SEMG_ENGINE_HYBRID_DTW_LDA_CNN: {
                raw_pred = hybrid_dtw_lda_cnn_predict(
                    &g_lda_model,
                    g_cnn_cfg,
                    g_hyb1_head,
                    g_dtw_cfg,
                    g_fusion_cfg,
                    desc.win_fp32_ptr,
                    NULL,
                    NULL,
                    desc.win_q16_ptr,
                    g_td_cfg.inv_scale,
                    td_features,
                    &hyb_tel
                );
                clf_cycles = hyb_tel.cycles_total;
                early_exit = hyb_tel.early_exit;
                break;
            }
            default: {
                uint32_t c0 = read_ccount();
                raw_pred = micro_lda_predict_q7(&g_lda_model, td_features, lda_scores);
                uint32_t c1 = read_ccount();
                clf_cycles = c1 - c0;
                break;
            }
        }

        int32_t vote3_pred = apply_majority_vote_3(raw_pred);
        update_status_led(vote3_pred);

        uint32_t c_end = read_ccount();
        uint32_t core1_cycles = c_end - c_start;
        float core1_us = (float)core1_cycles / (float)SEMG_CPU_FREQ_MHZ;
        float clf_us = (float)clf_cycles / (float)SEMG_CPU_FREQ_MHZ;

        uint32_t stack_c0 = (xTaskCore0Handle != NULL)
            ? (uint32_t)(uxTaskGetStackHighWaterMark(xTaskCore0Handle) * sizeof(StackType_t))
            : 0U;
        uint32_t stack_c1 = (uint32_t)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t));

        snprintf(
            tx_line,
            sizeof(tx_line),
            "SEMG_DECISION: {\"win\":%lu,\"engine\":\"%s\",\"raw_pred\":%ld,\"vote3_pred\":%ld,"
            "\"early_exit\":%u,\"cycles_c0\":%lu,\"cycles_c1\":%lu,\"clf_cycles\":%lu,"
            "\"us_c1\":%.2f,\"us_clf\":%.2f,\"stack_c0_free\":%lu,\"stack_c1_free\":%lu}",
            (unsigned long)desc.window_index,
            semg_rtos_engine_name(active_mode),
            (long)raw_pred,
            (long)vote3_pred,
            (unsigned)early_exit,
            (unsigned long)desc.core0_cycles,
            (unsigned long)core1_cycles,
            (unsigned long)clf_cycles,
            (double)core1_us,
            (double)clf_us,
            (unsigned long)stack_c0,
            (unsigned long)stack_c1
        );
        semg_hal_tx_line(tx_line);
    }
}

void semg_rtos_reset_pipeline(uint8_t bypass_filter, uint8_t engine_mode) {
    taskENTER_CRITICAL(&g_dma_spinlock);
    sos_filter_reset(&g_sos_filter);
    window_buffer_reset(&g_win_buf);
    g_bypass_filter = bypass_filter ? 1U : 0U;
    g_engine_mode = (engine_mode < (uint8_t)SEMG_ENGINE_COUNT)
        ? engine_mode
        : (uint8_t)SEMG_ENGINE_MICRO_LDA_Q7;
    g_window_seq = 0U;
    g_majority_ring[0] = -1;
    g_majority_ring[1] = -1;
    g_majority_ring[2] = -1;
    g_majority_pos = 0U;
    taskEXIT_CRITICAL(&g_dma_spinlock);

    if (xQueueJanelaPronta != NULL) {
        xQueueReset(xQueueJanelaPronta);
    }
}

void semg_rtos_set_engine(uint8_t engine_mode) {
    g_engine_mode = (engine_mode < (uint8_t)SEMG_ENGINE_COUNT)
        ? engine_mode
        : (uint8_t)SEMG_ENGINE_MICRO_LDA_Q7;
}

int semg_rtos_submit_block(const int16_t *samples, uint32_t num_samples, uint32_t num_channels) {
    if (samples == NULL || num_samples == 0U || num_samples > SEMG_DMA_BLOCK_SAMPLES || num_channels != g_num_channels) {
        return -1;
    }

    memcpy(g_dma_ctrl.active_dma_buffer, samples, (size_t)num_samples * (size_t)num_channels * sizeof(int16_t));

    taskENTER_CRITICAL(&g_dma_spinlock);
    g_dma_ctrl.processing_buffer = g_dma_ctrl.active_dma_buffer;
    g_dma_ctrl.pending_samples = num_samples;
    if (g_dma_ctrl.buffer_id == 0U) {
        g_dma_ctrl.active_dma_buffer = dma_pong_buffer;
        g_dma_ctrl.buffer_id = 1U;
    } else {
        g_dma_ctrl.active_dma_buffer = dma_ping_buffer;
        g_dma_ctrl.buffer_id = 0U;
    }
    taskEXIT_CRITICAL(&g_dma_spinlock);

    xSemaphoreGive(g_dma_ctrl.xBufferReadySem);
    return 0;
}

semg_selftest_result_t semg_rtos_run_boot_selftest(void) {
    semg_selftest_result_t res;
    memset(&res, 0, sizeof(res));

    /* Static allocations in internal SRAM to avoid stack overflow in main task */
    static micro_lda_model_t lda_st;
    static td_feature_extractor_t td_st;
    static stateful_window_buffer_t wb_st;
    semg_load_ozdemir_micro_lda(&lda_st, &td_st);
    window_buffer_init(&wb_st, WINDOW_LEN, STRIDE_LEN, OZDEMIR_LDA_NUM_CHANNELS);

    static float sample_fp32[OZDEMIR_LDA_NUM_CHANNELS];
    static float win_fp32[WINDOW_LEN * OZDEMIR_LDA_NUM_CHANNELS];
    static int16_t win_q16[WINDOW_LEN * OZDEMIR_LDA_NUM_CHANNELS];
    static float td_feat[OZDEMIR_LDA_NUM_FEATURES];
    static float lda_scores[OZDEMIR_LDA_NUM_CLASSES];
    static int8_t x_q[OZDEMIR_LDA_NUM_CHANNELS * WINDOW_LEN];
    static int8_t aux_q[OZDEMIR_LDA_NUM_FEATURES];
    static int32_t cnn_logits[CNN1D_MAX_CLASSES];
    cnn1d_cycle_profile_t prof;

    uint64_t sum_lda_cyc = 0U;
    uint64_t sum_cnn_cyc = 0U;
    uint32_t w_idx = 0U;

    for (int s = 0; s < OZDEMIR_SELFTEST_SAMPLES; s++) {
        for (int c = 0; c < OZDEMIR_LDA_NUM_CHANNELS; c++) {
            sample_fp32[c] = (float)g_ozdemir_selftest_stream_q16[s * OZDEMIR_LDA_NUM_CHANNELS + c] * td_st.inv_scale;
        }
        if (window_buffer_push_sample(&wb_st, sample_fp32, win_fp32)) {
            if (w_idx < OZDEMIR_SELFTEST_WINDOWS) {
                for (int i = 0; i < WINDOW_LEN * OZDEMIR_LDA_NUM_CHANNELS; i++) {
                    win_q16[i] = clamp_int16((int32_t)lrintf(win_fp32[i] * OZDEMIR_LDA_ADC_SCALE));
                }
                extract_td_features_int16(win_q16, WINDOW_LEN, OZDEMIR_LDA_NUM_CHANNELS, td_st.epsilon_int, td_st.inv_scale, td_feat);

                uint32_t c0 = read_ccount();
                int32_t pred_lda = micro_lda_predict_q7(&lda_st, td_feat, lda_scores);
                uint32_t c1 = read_ccount();
                sum_lda_cyc += (uint64_t)(c1 - c0);

                cnn1d_quantize_input(&g_cnn1d_ozdemir_config, win_fp32, td_feat, x_q, aux_q);
                int32_t pred_cnn = cnn1d_forward_int8(&g_cnn1d_ozdemir_config, x_q, aux_q, cnn_logits, &prof);
                sum_cnn_cyc += (uint64_t)prof.cycles_total;

                if (pred_lda == g_ozdemir_selftest_lda_oracle[w_idx]) {
                    res.lda_matches++;
                }
                if (pred_cnn == g_ozdemir_selftest_cnn_oracle[w_idx]) {
                    res.cnn_matches++;
                }
                res.windows_total++;
                w_idx++;
            }
        }
    }

    if (res.windows_total > 0U) {
        res.lda_avg_cycles = (uint32_t)(sum_lda_cyc / res.windows_total);
        res.cnn_avg_cycles = (uint32_t)(sum_cnn_cyc / res.windows_total);
    }
    return res;
}

uint32_t semg_rtos_get_num_channels(void) { return g_num_channels; }
uint32_t semg_rtos_get_num_classes(void)  { return g_num_classes; }
float    semg_rtos_get_adc_scale(void)    { return g_adc_scale; }
uint8_t  semg_rtos_get_engine_mode(void)  { return g_engine_mode; }

void semg_rtos_app_init(void) {
    init_status_led();

#ifdef CONFIG_SEMG_DATASET_NINAPRO_12CH
    semg_load_ninapro_micro_lda(&g_lda_model, &g_td_cfg);
    g_cnn_cfg    = &g_cnn1d_ninapro_config;
    g_dtw_cfg    = &g_dtw_ninapro_config;
    g_hyb1_head  = &g_hyb1_ninapro_head;
    g_fusion_cfg = &g_fusion_ninapro_config;
    g_num_channels = NINAPRO_LDA_NUM_CHANNELS;
    g_num_classes  = NINAPRO_LDA_NUM_CLASSES;
    g_adc_scale    = NINAPRO_LDA_ADC_SCALE;
#else
    semg_load_ozdemir_micro_lda(&g_lda_model, &g_td_cfg);
    g_cnn_cfg    = &g_cnn1d_ozdemir_config;
    g_dtw_cfg    = &g_dtw_ozdemir_config;
    g_hyb1_head  = &g_hyb1_ozdemir_head;
    g_fusion_cfg = &g_fusion_ozdemir_config;
    g_num_channels = OZDEMIR_LDA_NUM_CHANNELS;
    g_num_classes  = OZDEMIR_LDA_NUM_CLASSES;
    g_adc_scale    = OZDEMIR_LDA_ADC_SCALE;
#endif

    sos_filter_init(&g_sos_filter, MAX_SOS_SECTIONS, (int)g_num_channels, g_semg_sos_20_450hz);
    window_buffer_init(&g_win_buf, WINDOW_LEN, STRIDE_LEN, (int)g_num_channels);

#ifdef CONFIG_SEMG_MODEL_CNN1D_INT8
    g_engine_mode = SEMG_ENGINE_CNN1D_INT8;
#else
    g_engine_mode = SEMG_ENGINE_MICRO_LDA_Q7;
#endif

#if CONFIG_SEMG_ENABLE_BANDPASS_FILTER
    g_bypass_filter = 0U;
#else
    g_bypass_filter = 1U;
#endif

    memset(&g_dma_ctrl, 0, sizeof(g_dma_ctrl));
    g_dma_ctrl.active_dma_buffer = dma_ping_buffer;
    g_dma_ctrl.processing_buffer = dma_pong_buffer;
    g_dma_ctrl.buffer_id = 0U;
    g_dma_ctrl.xBufferReadySem = xSemaphoreCreateBinary();
    configASSERT(g_dma_ctrl.xBufferReadySem != NULL);

    xQueueJanelaPronta = xQueueCreate(8, sizeof(window_descriptor_t));
    configASSERT(xQueueJanelaPronta != NULL);

    xTaskCreatePinnedToCore(
        vTaskAquisicaoCore0,
        "vTaskAquisicaoCore0",
        4096,
        NULL,
        10,
        &xTaskCore0Handle,
        0 /* PRO_CPU */
    );
    xTaskCreatePinnedToCore(
        vTaskInferenciaCore1,
        "vTaskInferenciaCore1",
        6144,
        NULL,
        5,
        &xTaskCore1Handle,
        1 /* APP_CPU */
    );
}
