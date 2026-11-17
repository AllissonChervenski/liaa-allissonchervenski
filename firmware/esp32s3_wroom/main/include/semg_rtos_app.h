/*
 * semg_rtos_app.h — Native ESP-IDF Dual-Core FreeRTOS sEMG Processing Pipeline
 */

#ifndef SEMG_RTOS_APP_H
#define SEMG_RTOS_APP_H

#include <stdint.h>
#include "semg_app_config.h"

typedef struct {
    uint32_t windows_total;
    uint32_t lda_matches;
    uint32_t cnn_matches;
    uint32_t lda_avg_cycles;
    uint32_t cnn_avg_cycles;
} semg_selftest_result_t;

/* Initialize DSP, models (Micro-LDA Q7 + CNN-1D INT8), FreeRTOS queues/semaphores, and Dual-Core tasks */
void semg_rtos_app_init(void);

/* Submit a 100-sample int16 block into the Ping-Pong DMA buffer and signal Core 0 */
int semg_rtos_submit_block(const int16_t *samples, uint32_t num_samples, uint32_t num_channels);

/* Reset causal SOS filter, sliding window buffer, and majority voting state */
void semg_rtos_reset_pipeline(uint8_t bypass_filter, uint8_t engine_mode);

/* Dynamically switch active classifier (0..9 matching semg_engine_mode_t) */
void semg_rtos_set_engine(uint8_t engine_mode);

/* Human-readable telemetry name for engine_mode */
const char *semg_rtos_engine_name(uint8_t engine_mode);

/* Query active configuration */
uint32_t semg_rtos_get_num_channels(void);
uint32_t semg_rtos_get_num_classes(void);
float    semg_rtos_get_adc_scale(void);
uint8_t  semg_rtos_get_engine_mode(void);

/* Run deterministic 5-window built-in Flash self-test against host oracle predictions */
semg_selftest_result_t semg_rtos_run_boot_selftest(void);

#endif /* SEMG_RTOS_APP_H */
