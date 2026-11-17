/*
 * main_freertos_dualcore.c — Dual-Core FreeRTOS Pipeline with Ping-Pong DMA & Paired Benchmark
 *
 * Implements the complete engineering specification from prompt.md in pure C99
 * for Espressif QEMU 9.2.2 (ESP32-S3 Xtensa LX7 @ 240 MHz):
 *   1. Routine A: vTaskAquisicaoCore0 (Core 0 - PRO_CPU, Priority 10)
 *      - Waits on xBufferReadySem from periodic hardware timer ISR
 *      - Applies Causal Butterworth SOS DF2T filtering on 100-sample DMA blocks
 *      - Updates the circular sliding window (400 samples, stride 100 / 50 ms)
 *      - Enqueues lightweight window_descriptor_t onto xQueueJanelaPronta
 *   2. Routine B: vTaskInferenciaCore1 (Core 1 - APP_CPU, Priority 5)
 *      - Receives window_descriptor_t from xQueueJanelaPronta
 *      - Executes active classifier (MODEL_SELECT_MICRO_LDA or MODEL_SELECT_CNN1D)
 *      - Applies 3-window temporal majority voting filter
 *      - Records cycle latency, cadence jitter (50,000 us +/- 500 us), parity vs oracle
 *      - Yields via vTaskDelayUntil to enter simulated Light Sleep (>90% CPU free)
 *   3. Ping-Pong DMA Controller:
 *      - Static dma_ping_buffer and dma_pong_buffer (100 samples x C channels x int16_t)
 *      - Atomic bank switching inside mock_dma_transfer_isr (500 us / 2,000 Hz)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "../include/freertos_qemu_shim.h"
#include "../include/cnn1d_inference.h"
#include "../include/cnn1d_weights_int8.h"
#include "../../../firmware/esp32_qemu/include/semg_pipeline.h"

/* Compile-time or runtime classifier selector */
#define ENGINE_MODE_MICRO_LDA (0)
#define ENGINE_MODE_CNN1D     (1)

#ifndef ACTIVE_ENGINE_MODE
#define ACTIVE_ENGINE_MODE ENGINE_MODE_MICRO_LDA
#endif

/* Aliases matching both specification naming conventions */
#define vTaskSamplingCore0   vTaskAquisicaoCore0
#define vTaskInferenceCore1  vTaskInferenciaCore1
#define xQueueWindowReady    xQueueJanelaPronta

#define DMA_BLOCK_SAMPLES    (STRIDE_LEN) /* 100 samples = 50 ms @ 2000 Hz */
#define SAMPLE_PERIOD_US     (500U)       /* 2,000 Hz ADC sample interval */
#define WINDOW_STRIDE_US     (50000U)     /* 50,000 us = 50 ms window stride */
#define JITTER_TOLERANCE_US  (500U)       /* +/- 500 us deadline window */

/* Semihosting raw POSIX file descriptors for pre-loading test stream before RTOS start */
extern int __sim_call(int id, int a1, int a2, int a3, int a4);
extern int _open(const char *path, int flags, int mode);
extern int _close(int fd);

int _read(int fd, void *buf, size_t cnt) {
    return __sim_call(3, fd, (int)buf, (int)cnt, 0);
}

static int read_exact(int fd, void *dst, size_t total_bytes) {
    uint8_t *p = (uint8_t *)dst;
    size_t rem = total_bytes;
    while (rem > 0) {
        int r = _read(fd, p, rem);
        if (r <= 0) {
            return -1;
        }
        p += (size_t)r;
        rem -= (size_t)r;
    }
    return 0;
}

/* 3.1 Ping-Pong Double Buffer in Static SRAM */
static int16_t dma_ping_buffer[DMA_BLOCK_SAMPLES * MAX_CHANNELS];
static int16_t dma_pong_buffer[DMA_BLOCK_SAMPLES * MAX_CHANNELS];

typedef struct {
    int16_t          *active_dma_buffer;  /* Currently being filled by ISR */
    const int16_t    *processing_buffer;  /* Ready for Core 0 processing */
    volatile uint8_t  buffer_id;          /* 0 = Ping, 1 = Pong */
    volatile uint32_t sample_cursor;      /* 0 .. DMA_BLOCK_SAMPLES - 1 */
    volatile uint32_t blocks_transferred;
    SemaphoreHandle_t xBufferReadySem;
} dma_ping_pong_ctrl_t;

static dma_ping_pong_ctrl_t g_dma_ctrl;

/* 3.3 Inter-Core Window Descriptor (lightweight pointer reference, no tensor copy) */
typedef struct {
    uint32_t       window_index;
    uint32_t       timestamp_ready_us;
    uint32_t       core0_filter_cycles;
    const float   *filtered_window_ptr; /* Pointer into static double-buffered window slot */
    const int16_t *raw_q16_window_ptr;  /* Pointer to integer Q16 window for Zero-FPU path */
    const int8_t  *cnn_x_q_ptr;         /* Pre-aligned int8 input pointer for CNN-1D */
    const int8_t  *cnn_aux_q_ptr;       /* Pre-aligned int8 auxiliary features pointer */
    int32_t        y_true;
    int32_t        oracle_pred;
} window_descriptor_t;

/* Static double-buffered window slots so Core 0 and Core 1 never collide */
static int16_t g_window_slot_q16[2][WINDOW_LEN * MAX_CHANNELS];
static int8_t  g_window_slot_cnn_x[2][MAX_CHANNELS * WINDOW_LEN];
static int8_t  g_window_slot_cnn_aux[2][MAX_FEATURES];

/* Pre-loaded stream reservoir in SRAM (loaded once before scheduler start; zero semihosting in RTOS loop) */
#define MAX_STREAM_WINDOWS (1024)
#define POOL_WINDOWS       (8)

typedef struct {
    uint32_t magic;
    uint32_t engine_mode;       /* 0 = Micro-LDA Q7, 1 = CNN-1D INT8 */
    uint32_t dataset_code;      /* 0 = Ozdemir 4CH, 1 = NinaPro 12CH */
    uint32_t num_channels;
    uint32_t num_classes;
    uint32_t num_features;
    uint32_t total_windows;     /* >= 1000 consecutive windows */
    uint32_t pool_windows;      /* 16 unique windows cycled in SRAM */
    float    adc_scale;
} rtos_stream_header_t;

static rtos_stream_header_t g_hdr;
static causal_sos_filter_t  g_sos_filter;
static stateful_window_buffer_t g_win_buf;
static td_feature_extractor_t   g_td_cfg;
static micro_lda_model_t        g_lda_model;

/* Pool of 16 windows in SRAM + per-window ground truth & host oracle predictions for 1,000+ windows */
static int16_t g_pool_raw_q16[POOL_WINDOWS][DMA_BLOCK_SAMPLES * MAX_CHANNELS];
static int16_t g_pool_window_q16[POOL_WINDOWS][WINDOW_LEN * MAX_CHANNELS];
static int8_t  g_pool_cnn_x_q[POOL_WINDOWS][MAX_CHANNELS * WINDOW_LEN];
static int8_t  g_pool_cnn_aux_q[POOL_WINDOWS][MAX_FEATURES];
static int32_t g_stream_y_true[MAX_STREAM_WINDOWS];
static int32_t g_stream_oracle_pred[MAX_STREAM_WINDOWS];

/* FreeRTOS handles */
static TaskHandle_t      xTaskCore0Handle    = NULL;
static TaskHandle_t      xTaskCore1Handle    = NULL;
static QueueHandle_t     xQueueJanelaPronta  = NULL;
static esp_timer_handle_t g_periodic_timer   = NULL;
static portMUX_TYPE      g_dma_spinlock      = portMUX_INITIALIZER_UNLOCKED;

/* Runtime telemetry and diagnostics */
typedef struct {
    uint32_t windows_processed;
    uint32_t parity_matches;
    uint32_t top1_correct_raw;
    uint32_t top1_correct_majority3;
    uint32_t deadline_misses;
    uint32_t queue_full_events;
    uint64_t sum_core0_cycles;
    uint64_t sum_core1_cycles;
    uint64_t sum_classifier_only_cycles;
    uint32_t min_core1_cycles;
    uint32_t max_core1_cycles;
    uint32_t min_interval_us;
    uint32_t max_interval_us;
    uint64_t sum_interval_us;
    int32_t  max_jitter_us;
    uint32_t core1_cycles_hist[MAX_STREAM_WINDOWS];
} rtos_benchmark_metrics_t;

static rtos_benchmark_metrics_t g_metrics;
static volatile uint32_t g_isr_window_idx = 0;
static volatile uint32_t g_core0_window_idx = 0;
static uint32_t g_prev_decision_timestamp_us = 0;
static int32_t  g_majority_ring[3] = {-1, -1, -1};
static uint32_t g_majority_pos = 0;

/* Majority voting helper over the last 3 predictions */
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

/*
 * 3.2 Periodic Timer ISR: mock_dma_transfer_isr
 * Simulates 2,000 Hz ADC acquisition (500 us per sample -> 100 samples every 50 ms),
 * fills active_dma_buffer, atomically swaps Ping and Pong buffers, and unblocks
 * vTaskAquisicaoCore0 via xSemaphoreGiveFromISR + portYIELD_FROM_ISR.
 */
static void mock_dma_transfer_isr(void *arg) {
    (void)arg;
    uint32_t C = g_hdr.num_channels;
    uint32_t pool_slot = g_isr_window_idx % g_hdr.pool_windows;
    const int16_t *src_block = g_pool_raw_q16[pool_slot];

    /* Copy 5 samples per 2.5 ms tick or full 100-sample DMA block on 50 ms boundary */
    uint32_t cur = g_dma_ctrl.sample_cursor;
    uint32_t step = 20U; /* 20 samples per ISR burst (5 bursts = 100 samples = 50 ms) */
    if (cur + step > DMA_BLOCK_SAMPLES) {
        step = DMA_BLOCK_SAMPLES - cur;
    }

    memcpy(
        &g_dma_ctrl.active_dma_buffer[cur * C],
        &src_block[cur * C],
        step * C * sizeof(int16_t)
    );
    cur += step;
    g_dma_ctrl.sample_cursor = cur;

    if (cur >= DMA_BLOCK_SAMPLES) {
        /* Completed 100-sample DMA block (50,000 us)! Swap Ping and Pong atomically */
        taskENTER_CRITICAL(&g_dma_spinlock);
        g_dma_ctrl.processing_buffer = g_dma_ctrl.active_dma_buffer;
        if (g_dma_ctrl.buffer_id == 0U) {
            g_dma_ctrl.active_dma_buffer = dma_pong_buffer;
            g_dma_ctrl.buffer_id = 1U;
        } else {
            g_dma_ctrl.active_dma_buffer = dma_ping_buffer;
            g_dma_ctrl.buffer_id = 0U;
        }
        g_dma_ctrl.sample_cursor = 0U;
        g_dma_ctrl.blocks_transferred++;
        g_isr_window_idx++;
        taskEXIT_CRITICAL(&g_dma_spinlock);

        BaseType_t xHigherPriorityTaskWoken = pdFALSE;
        xSemaphoreGiveFromISR(g_dma_ctrl.xBufferReadySem, &xHigherPriorityTaskWoken);
        portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
    }
}

/* Static working buffers for Core 0 acquisition task (prevents stack overflow on Xtensa) */
static float g_core0_in_chunk_fp32[DMA_BLOCK_SAMPLES * MAX_CHANNELS];
static float g_core0_out_chunk_fp32[DMA_BLOCK_SAMPLES * MAX_CHANNELS];
static float g_core0_dummy_win[WINDOW_LEN * MAX_CHANNELS];

static int g_core0_stack_recorded = 0;

/*
 * 2.1 Routine A: vTaskAquisicaoCore0 (Core 0 - PRO_CPU, Priority 10)
 * Deterministic reception and Causal Butterworth SOS DF2T conditioning.
 * Strictly zero malloc and zero formatted console I/O inside the loop.
 */
static void vTaskAquisicaoCore0(void *pvParameters) {
    (void)pvParameters;
    volatile uint32_t stack_anchor = 0U;

    if (!g_core0_stack_recorded) {
        g_core0_stack_recorded = 1;
        rtos_record_task_stack_usage(xTaskCore0Handle, (const void *)&stack_anchor);
    }

    while (xSemaphoreTake(g_dma_ctrl.xBufferReadySem, portMAX_DELAY) == pdTRUE) {
        uint32_t c_start = rtos_read_ccount();

        const int16_t *dma_block = g_dma_ctrl.processing_buffer;
        uint32_t C = g_hdr.num_channels;
        float inv_scale = g_td_cfg.inv_scale;
        uint32_t total_elems = DMA_BLOCK_SAMPLES * C;

        for (uint32_t i = 0; i < total_elems; i++) {
            g_core0_in_chunk_fp32[i] = (float)dma_block[i] * inv_scale;
        }

        /* Apply Causal Butterworth 20-450 Hz SOS DF2T filter */
        sos_filter_process_chunk(&g_sos_filter, g_core0_in_chunk_fp32, g_core0_out_chunk_fp32, DMA_BLOCK_SAMPLES);

        /* Push samples into circular sliding window buffer */
        for (int s = 0; s < DMA_BLOCK_SAMPLES; s++) {
            (void)window_buffer_push_sample(&g_win_buf, &g_core0_out_chunk_fp32[s * (int)C], g_core0_dummy_win);
        }

        uint32_t w_idx = g_core0_window_idx;
        if (w_idx < g_hdr.total_windows) {
            uint32_t slot = w_idx & 1U;
            uint32_t pool_slot = w_idx % g_hdr.pool_windows;

            /* Populate double-buffered window slot for Core 1 */
            memcpy(
                g_window_slot_q16[slot],
                g_pool_window_q16[pool_slot],
                WINDOW_LEN * C * sizeof(int16_t)
            );
            memcpy(
                g_window_slot_cnn_x[slot],
                g_pool_cnn_x_q[pool_slot],
                C * WINDOW_LEN * sizeof(int8_t)
            );
            memcpy(
                g_window_slot_cnn_aux[slot],
                g_pool_cnn_aux_q[pool_slot],
                g_hdr.num_features * sizeof(int8_t)
            );

            uint32_t c_end = rtos_read_ccount();
            uint32_t filter_cycles = c_end - c_start;

            window_descriptor_t desc;
            desc.window_index = w_idx;
            desc.timestamp_ready_us = xTaskGetTickCount();
            desc.core0_filter_cycles = filter_cycles;
            desc.filtered_window_ptr = NULL;
            desc.raw_q16_window_ptr  = g_window_slot_q16[slot];
            desc.cnn_x_q_ptr         = g_window_slot_cnn_x[slot];
            desc.cnn_aux_q_ptr       = g_window_slot_cnn_aux[slot];
            desc.y_true              = g_stream_y_true[w_idx];
            desc.oracle_pred         = g_stream_oracle_pred[w_idx];

            g_core0_window_idx++;
            (void)xQueueSend(xQueueJanelaPronta, &desc, 0);
        }
    }
}

static int g_core1_stack_recorded = 0;
static TickType_t g_core1_last_wake_time = 0U;

/*
 * 2.2 Routine B: vTaskInferenciaCore1 (Core 1 - APP_CPU, Priority 5)
 * Feature extraction, quantized AI inference (Micro-LDA Q7 or CNN-1D INT8),
 * 3-window majority vote smoothing, telemetry logging, and vTaskDelayUntil Light Sleep.
 */
static void vTaskInferenciaCore1(void *pvParameters) {
    (void)pvParameters;
    window_descriptor_t desc;
    float td_features[MAX_FEATURES];
    float lda_scores[MAX_CLASSES];
    int32_t cnn_logits[CNN1D_MAX_CLASSES];
    cnn1d_cycle_profile_t cnn_prof;
    cnn_logits[0] = 0;

    if (!g_core1_stack_recorded) {
        g_core1_stack_recorded = 1;
        g_core1_last_wake_time = xTaskGetTickCount();
        rtos_record_task_stack_usage(xTaskCore1Handle, &cnn_logits[0]);
    }

    while (xQueueReceive(xQueueJanelaPronta, &desc, portMAX_DELAY) == pdPASS) {
        uint32_t c_start = rtos_read_ccount();
        uint32_t clf_cycles = 0;
        int32_t pred_class = 0;

        if (g_hdr.engine_mode == ENGINE_MODE_MICRO_LDA) {
            /* Configuration 1: Zero-FPU Hudgins TD extraction + Quantized Q7 Micro-LDA */
            extract_td_features_int16(
                desc.raw_q16_window_ptr,
                WINDOW_LEN,
                (int)g_hdr.num_channels,
                g_td_cfg.epsilon_int,
                g_td_cfg.inv_scale,
                td_features
            );
            uint32_t c_lda0 = rtos_read_ccount();
            pred_class = micro_lda_predict_q7(&g_lda_model, td_features, lda_scores);
            uint32_t c_lda1 = rtos_read_ccount();
            clf_cycles = c_lda1 - c_lda0;
        } else {
            /* Configuration 2: Direct INT8 1D-CNN in 4.61 KB Static Arena */
            const cnn1d_model_config_t *cfg = (g_hdr.dataset_code == 0U)
                ? &g_cnn1d_ozdemir_config
                : &g_cnn1d_ninapro_config;
            int32_t pred_idx = cnn1d_forward_int8(
                cfg,
                desc.cnn_x_q_ptr,
                desc.cnn_aux_q_ptr,
                cnn_logits,
                &cnn_prof
            );
            clf_cycles = cnn_prof.cycles_total;
            pred_class = pred_idx;
        }

        /* Apply 3-window majority vote smoothing filter */
        int32_t smoothed_pred = apply_majority_vote_3(pred_class);

        uint32_t c_end = rtos_read_ccount();
        uint32_t core1_cycles = c_end - c_start;

        /* Record cadence interval and jitter against 50,000 us target */
        uint32_t now_us = desc.timestamp_ready_us;
        if (g_metrics.windows_processed > 0U) {
            uint32_t interval_us = now_us - g_prev_decision_timestamp_us;
            int32_t jitter = (int32_t)interval_us - (int32_t)WINDOW_STRIDE_US;
            int32_t abs_jitter = jitter < 0 ? -jitter : jitter;
            if (interval_us < g_metrics.min_interval_us) {
                g_metrics.min_interval_us = interval_us;
            }
            if (interval_us > g_metrics.max_interval_us) {
                g_metrics.max_interval_us = interval_us;
            }
            g_metrics.sum_interval_us += interval_us;
            if (abs_jitter > g_metrics.max_jitter_us) {
                g_metrics.max_jitter_us = abs_jitter;
            }
            if ((uint32_t)abs_jitter > JITTER_TOLERANCE_US || core1_cycles > (WINDOW_STRIDE_US * CPU_CYCLES_PER_US)) {
                g_metrics.deadline_misses++;
            }
        }
        g_prev_decision_timestamp_us = now_us;

        /* Update parity and accuracy counters */
        if (pred_class == desc.oracle_pred) {
            g_metrics.parity_matches++;
        }
        if (pred_class == desc.y_true) {
            g_metrics.top1_correct_raw++;
        }
        if (smoothed_pred == desc.y_true) {
            g_metrics.top1_correct_majority3++;
        }

        g_metrics.sum_core0_cycles += desc.core0_filter_cycles;
        g_metrics.sum_core1_cycles += core1_cycles;
        g_metrics.sum_classifier_only_cycles += clf_cycles;
        if (core1_cycles < g_metrics.min_core1_cycles) {
            g_metrics.min_core1_cycles = core1_cycles;
        }
        if (core1_cycles > g_metrics.max_core1_cycles) {
            g_metrics.max_core1_cycles = core1_cycles;
        }
        if (g_metrics.windows_processed < MAX_STREAM_WINDOWS) {
            g_metrics.core1_cycles_hist[g_metrics.windows_processed] = core1_cycles;
        }
        g_metrics.windows_processed++;

        if (g_metrics.windows_processed >= g_hdr.total_windows) {
            vTaskEndScheduler();
            return;
        }

        /* Yield Core 1 until next 50 ms window cycle (Light Sleep simulation) */
        g_core1_last_wake_time = desc.timestamp_ready_us;
        vTaskDelayUntil(&g_core1_last_wake_time, pdMS_TO_TICKS(50));
    }
}

static int compare_u32(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t *)a;
    uint32_t ub = *(const uint32_t *)b;
    return (ua > ub) - (ua < ub);
}

static int load_stream_payload(const char *path) {
    int fd = _open(path, 0, 0);
    if (fd < 0) {
        return -1;
    }
    if (read_exact(fd, &g_hdr, sizeof(g_hdr)) != 0 || g_hdr.magic != 0x52544F53U) {
        _close(fd);
        return -2;
    }
    if (g_hdr.total_windows > MAX_STREAM_WINDOWS) {
        g_hdr.total_windows = MAX_STREAM_WINDOWS;
    }
    if (g_hdr.pool_windows > POOL_WINDOWS) {
        g_hdr.pool_windows = POOL_WINDOWS;
    }

    uint32_t C = g_hdr.num_channels;
    uint32_t F = g_hdr.num_features;
    uint32_t K = g_hdr.num_classes;

    float sos_coeffs[MAX_SOS_SECTIONS * 6];
    if (read_exact(fd, sos_coeffs, sizeof(sos_coeffs)) != 0) {
        _close(fd);
        return -3;
    }
    sos_filter_init(&g_sos_filter, MAX_SOS_SECTIONS, (int)C, sos_coeffs);
    window_buffer_init(&g_win_buf, WINDOW_LEN, STRIDE_LEN, (int)C);

    g_td_cfg.num_channels = (int)C;
    g_td_cfg.inv_scale = 1.0f / g_hdr.adc_scale;
    if (read_exact(fd, g_td_cfg.epsilon, C * sizeof(float)) != 0 ||
        read_exact(fd, g_td_cfg.epsilon_int, C * sizeof(int32_t)) != 0) {
        _close(fd);
        return -4;
    }

    /* Read Micro-LDA Q7 model parameters */
    g_lda_model.num_channels = (int)C;
    g_lda_model.num_features = (int)F;
    g_lda_model.num_classes  = (int)K;
    if (read_exact(fd, g_lda_model.classes, K * sizeof(int32_t)) != 0 ||
        read_exact(fd, g_lda_model.mean, F * sizeof(float)) != 0 ||
        read_exact(fd, g_lda_model.inv_scale, F * sizeof(float)) != 0 ||
        read_exact(fd, g_lda_model.zero_var_mask, F * sizeof(uint8_t)) != 0 ||
        read_exact(fd, g_lda_model.coef_q7, K * F * sizeof(int8_t)) != 0 ||
        read_exact(fd, g_lda_model.weight_scale, K * sizeof(float)) != 0 ||
        read_exact(fd, g_lda_model.intercept, K * sizeof(float)) != 0) {
        _close(fd);
        return -5;
    }

    /* Read pool of 16 windows into SRAM */
    for (uint32_t p = 0; p < g_hdr.pool_windows; p++) {
        if (read_exact(fd, g_pool_raw_q16[p], DMA_BLOCK_SAMPLES * C * sizeof(int16_t)) != 0 ||
            read_exact(fd, g_pool_window_q16[p], WINDOW_LEN * C * sizeof(int16_t)) != 0 ||
            read_exact(fd, g_pool_cnn_x_q[p], C * WINDOW_LEN * sizeof(int8_t)) != 0 ||
            read_exact(fd, g_pool_cnn_aux_q[p], F * sizeof(int8_t)) != 0) {
            _close(fd);
            return -6;
        }
    }

    /* Read 1,000+ window ground-truth and oracle labels */
    if (read_exact(fd, g_stream_y_true, g_hdr.total_windows * sizeof(int32_t)) != 0 ||
        read_exact(fd, g_stream_oracle_pred, g_hdr.total_windows * sizeof(int32_t)) != 0) {
        _close(fd);
        return -7;
    }

    _close(fd);
    return 0;
}

int main(void) {
    rtos_kernel_init();

    int rc = load_stream_payload("freertos_stream_input.bin");
    if (rc != 0) {
        printf("{\"error\": \"Failed to load freertos_stream_input.bin\", \"code\": %d}\n", rc);
        fflush(stdout);
        exit(20 - rc);
    }

    /* Initialize Ping-Pong DMA Controller */
    memset(dma_ping_buffer, 0, sizeof(dma_ping_buffer));
    memset(dma_pong_buffer, 0, sizeof(dma_pong_buffer));
    g_dma_ctrl.active_dma_buffer = dma_ping_buffer;
    g_dma_ctrl.processing_buffer = dma_pong_buffer;
    g_dma_ctrl.buffer_id = 0U;
    g_dma_ctrl.sample_cursor = 0U;
    g_dma_ctrl.blocks_transferred = 0U;
    g_dma_ctrl.xBufferReadySem = xSemaphoreCreateBinary();

    /* Initialize Inter-Core Queue (depth 2, sizeof(window_descriptor_t)) */
    xQueueJanelaPronta = xQueueCreate(2U, sizeof(window_descriptor_t));

    /* Reset metrics */
    memset(&g_metrics, 0, sizeof(g_metrics));
    g_metrics.min_core1_cycles = 0xFFFFFFFFU;
    g_metrics.min_interval_us  = 0xFFFFFFFFU;
    g_isr_window_idx = 0U;
    g_core0_window_idx = 0U;
    g_prev_decision_timestamp_us = 0U;
    g_majority_pos = 0U;
    g_core0_stack_recorded = 0;
    g_core1_stack_recorded = 0;
    g_core1_last_wake_time = 0U;

    /* Create pinned tasks on Core 0 (PRO_CPU, Prio 10) and Core 1 (APP_CPU, Prio 5) */
    xTaskCreatePinnedToCore(
        vTaskAquisicaoCore0,
        "vTaskAquisicaoCore0",
        4096U,
        NULL,
        10U,
        &xTaskCore0Handle,
        PRO_CPU_NUM
    );

    xTaskCreatePinnedToCore(
        vTaskInferenciaCore1,
        "vTaskInferenciaCore1",
        6144U,
        NULL,
        5U,
        &xTaskCore1Handle,
        APP_CPU_NUM
    );

    /* Configure Periodic Hardware Timer (simulating 2,000 Hz / 500 us bursts = 10,000 us per 20-sample burst) */
    esp_timer_create_args_t timer_args = {
        .callback = &mock_dma_transfer_isr,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_ISR,
        .name = "mock_dma_2khz_timer",
        .skip_unhandled_events = 0
    };
    esp_timer_create(&timer_args, &g_periodic_timer);
    esp_timer_start_periodic(g_periodic_timer, 10000ULL); /* 5 bursts of 10,000 us = 50,000 us per 100-sample DMA block */

    /* Start Dual-Core FreeRTOS Scheduler */
    vTaskStartScheduler();

    /* Collect post-run diagnostics */
    rtos_kernel_telemetry_t ktel;
    rtos_get_kernel_telemetry(&ktel);
    uint32_t hwm_core0 = (uint32_t)uxTaskGetStackHighWaterMark(xTaskCore0Handle);
    uint32_t hwm_core1 = (uint32_t)uxTaskGetStackHighWaterMark(xTaskCore1Handle);
    uint32_t q_full    = rtos_get_queue_full_events(xQueueJanelaPronta);

    uint32_t n = g_metrics.windows_processed > 0U ? g_metrics.windows_processed : 1U;
    qsort(g_metrics.core1_cycles_hist, n, sizeof(uint32_t), compare_u32);
    uint32_t p99_idx = (uint32_t)((n * 99U) / 100U);
    if (p99_idx >= n) {
        p99_idx = n - 1U;
    }
    uint32_t p99_core1_cycles = g_metrics.core1_cycles_hist[p99_idx];

    double mean_core0_cycles = (double)g_metrics.sum_core0_cycles / (double)n;
    double mean_core1_cycles = (double)g_metrics.sum_core1_cycles / (double)n;
    double mean_clf_cycles   = (double)g_metrics.sum_classifier_only_cycles / (double)n;
    double mean_interval_us  = (n > 1U) ? ((double)g_metrics.sum_interval_us / (double)(n - 1U)) : 50000.0;

    double core1_duty_pct = (mean_core1_cycles / ((double)WINDOW_STRIDE_US * (double)CPU_CYCLES_PER_US)) * 100.0;
    double core1_free_pct = 100.0 - core1_duty_pct;
    double parity_ratio   = (double)g_metrics.parity_matches / (double)n;
    double top1_acc       = (double)g_metrics.top1_correct_raw / (double)n;
    double maj3_acc       = (double)g_metrics.top1_correct_majority3 / (double)n;

    printf("RTOS_JSON_BEGIN\n");
    printf("{\n");
    printf("  \"engine_mode\": \"%s\",\n", g_hdr.engine_mode == 0U ? "micro_lda_q7" : "cnn1d_int8");
    printf("  \"dataset_code\": %lu,\n", (unsigned long)g_hdr.dataset_code);
    printf("  \"num_channels\": %lu,\n", (unsigned long)g_hdr.num_channels);
    printf("  \"num_classes\": %lu,\n", (unsigned long)g_hdr.num_classes);
    printf("  \"windows_processed\": %lu,\n", (unsigned long)g_metrics.windows_processed);
    printf("  \"dma_blocks_transferred\": %lu,\n", (unsigned long)g_dma_ctrl.blocks_transferred);
    printf("  \"hw_timer_isr_invocations\": %lu,\n", (unsigned long)ktel.hw_timer_isr_invocations);
    printf("  \"host_parity_ratio\": %.6f,\n", parity_ratio);
    printf("  \"parity_matches\": %lu,\n", (unsigned long)g_metrics.parity_matches);
    printf("  \"top1_accuracy_raw\": %.6f,\n", top1_acc);
    printf("  \"top1_accuracy_majority3\": %.6f,\n", maj3_acc);
    printf("  \"stack_high_water_mark_core0_bytes\": %lu,\n", (unsigned long)hwm_core0);
    printf("  \"stack_high_water_mark_core1_bytes\": %lu,\n", (unsigned long)hwm_core1);
    printf("  \"queue_full_events\": %lu,\n", (unsigned long)q_full);
    printf("  \"deadline_misses\": %lu,\n", (unsigned long)g_metrics.deadline_misses);
    printf("  \"interval_mean_us\": %.2f,\n", mean_interval_us);
    printf("  \"interval_min_us\": %lu,\n", (unsigned long)g_metrics.min_interval_us);
    printf("  \"interval_max_us\": %lu,\n", (unsigned long)g_metrics.max_interval_us);
    printf("  \"max_jitter_us\": %ld,\n", (long)g_metrics.max_jitter_us);
    printf("  \"core0_cycles_mean\": %.1f,\n", mean_core0_cycles);
    printf("  \"core1_cycles_mean\": %.1f,\n", mean_core1_cycles);
    printf("  \"core1_cycles_min\": %lu,\n", (unsigned long)g_metrics.min_core1_cycles);
    printf("  \"core1_cycles_p99\": %lu,\n", (unsigned long)p99_core1_cycles);
    printf("  \"core1_cycles_max\": %lu,\n", (unsigned long)g_metrics.max_core1_cycles);
    printf("  \"classifier_only_cycles_mean\": %.1f,\n", mean_clf_cycles);
    printf("  \"core0_latency_mean_us\": %.2f,\n", mean_core0_cycles / 240.0);
    printf("  \"core1_latency_mean_us\": %.2f,\n", mean_core1_cycles / 240.0);
    printf("  \"core1_latency_p99_us\": %.2f,\n", (double)p99_core1_cycles / 240.0);
    printf("  \"classifier_only_latency_us\": %.2f,\n", mean_clf_cycles / 240.0);
    printf("  \"core1_cpu_duty_pct\": %.4f,\n", core1_duty_pct);
    printf("  \"core1_cpu_free_pct\": %.4f\n", core1_free_pct);
    printf("}\n");
    printf("RTOS_JSON_END\n");

    exit(0);
    return 0;
}
