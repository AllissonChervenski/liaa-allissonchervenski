/*
 * semg_hal_acq.c — ESP32-S3-WROOM Hardware Acquisition & HIL Serial Transport
 *
 * Implements three Kconfig-selectable acquisition backends:
 *   1. CONFIG_SEMG_ACQ_MODE_HIL_SERIAL:
 *        Receives CRC16-protected binary int16 frames over native USB-Serial/JTAG
 *        (/dev/ttyACM*) or UART0 (/dev/ttyUSB*) and feeds Ping-Pong DMA blocks.
 *   2. CONFIG_SEMG_ACQ_MODE_ADC_DMA:
 *        Acquires live multi-channel sEMG via ESP32-S3 continuous SAR ADC + GDMA
 *        with hardware oversampling decimation to exact 2,000 Hz/channel.
 *   3. CONFIG_SEMG_ACQ_MODE_SELFTEST_LOOP:
 *        Replays the embedded Flash Ozdemir S1 stream at 2,000 Hz via esp_timer.
 */

#include "../include/semg_hal_acq.h"
#include "../include/semg_hil_protocol.h"
#include "../include/semg_rtos_app.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "semg_pipeline.h"
#include "micro_lda_weights.h"

#if defined(CONFIG_SEMG_ACQ_MODE_HIL_SERIAL)
  #if defined(CONFIG_SEMG_HIL_TRANSPORT_USB_JTAG)
    #include "driver/usb_serial_jtag.h"
  #elif defined(CONFIG_SEMG_HIL_TRANSPORT_UART0)
    #include "driver/uart.h"
  #endif
#elif defined(CONFIG_SEMG_ACQ_MODE_ADC_DMA)
  #include "esp_adc/adc_continuous.h"
  #include "soc/soc_caps.h"
#endif

static const char *TAG = "semg_hal";
static SemaphoreHandle_t g_tx_mutex = NULL;

void semg_hal_transport_init(void) {
    if (g_tx_mutex == NULL) {
        g_tx_mutex = xSemaphoreCreateMutex();
    }

#if defined(CONFIG_SEMG_ACQ_MODE_HIL_SERIAL)
  #if defined(CONFIG_SEMG_HIL_TRANSPORT_USB_JTAG)
    usb_serial_jtag_driver_config_t usb_cfg = {
        .rx_buffer_size = 4096,
        .tx_buffer_size = 2048
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_cfg));
  #elif defined(CONFIG_SEMG_HIL_TRANSPORT_UART0)
    const uart_config_t uart_cfg = {
        .baud_rate  = 921600,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 4096, 2048, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_0, &uart_cfg));
  #endif
#endif
}

void semg_hal_tx_line(const char *line) {
    if (line == NULL) {
        return;
    }
    if (g_tx_mutex != NULL) {
        xSemaphoreTake(g_tx_mutex, portMAX_DELAY);
    }

#if defined(CONFIG_SEMG_ACQ_MODE_HIL_SERIAL) && defined(CONFIG_SEMG_HIL_TRANSPORT_USB_JTAG)
    size_t len = strlen(line);
    if (len > 0U) {
        usb_serial_jtag_write_bytes((const uint8_t *)line, len, pdMS_TO_TICKS(50));
    }
    const uint8_t nl = '\n';
    usb_serial_jtag_write_bytes(&nl, 1, pdMS_TO_TICKS(50));
#elif defined(CONFIG_SEMG_ACQ_MODE_HIL_SERIAL) && defined(CONFIG_SEMG_HIL_TRANSPORT_UART0)
    size_t len = strlen(line);
    if (len > 0U) {
        uart_write_bytes(UART_NUM_0, line, len);
    }
    uart_write_bytes(UART_NUM_0, "\n", 1);
#else
    printf("%s\n", line);
    fflush(stdout);
#endif

    if (g_tx_mutex != NULL) {
        xSemaphoreGive(g_tx_mutex);
    }
}

#if defined(CONFIG_SEMG_ACQ_MODE_HIL_SERIAL)

static semg_hil_parser_t g_hil_parser;
static semg_hil_frame_t  g_hil_frame;
static int16_t           g_hil_sample_buf[SEMG_DMA_BLOCK_SAMPLES * MAX_CHANNELS];

static void handle_hil_frame(const semg_hil_frame_t *frame) {
    char reply[224];

    switch (frame->cmd) {
        case SEMG_HIL_CMD_PING: {
            snprintf(
                reply,
                sizeof(reply),
                "SEMG_PONG: {\"target\":\"esp32s3-wroom\",\"channels\":%lu,\"classes\":%lu,"
                "\"engine\":\"%s\",\"adc_scale\":%.4f}",
                (unsigned long)semg_rtos_get_num_channels(),
                (unsigned long)semg_rtos_get_num_classes(),
                semg_rtos_engine_name(semg_rtos_get_engine_mode()),
                (double)semg_rtos_get_adc_scale()
            );
            semg_hal_tx_line(reply);
            break;
        }

        case SEMG_HIL_CMD_RESET: {
            uint8_t flags = (frame->payload_len >= 1U) ? frame->payload[0] : 0U;
            uint8_t bypass = (flags & SEMG_HIL_FLAG_BYPASS_FILTER) ? 1U : 0U;
            uint8_t engine = (frame->payload_len >= 2U)
                ? frame->payload[1]
                : ((flags & SEMG_HIL_FLAG_ENGINE_CNN1D)
                    ? (uint8_t)SEMG_ENGINE_CNN1D_INT8
                    : (uint8_t)SEMG_ENGINE_MICRO_LDA_Q7);
            if (engine >= (uint8_t)SEMG_ENGINE_COUNT) {
                engine = (uint8_t)SEMG_ENGINE_MICRO_LDA_Q7;
            }
            semg_rtos_reset_pipeline(bypass, engine);
            snprintf(
                reply,
                sizeof(reply),
                "SEMG_ACK: {\"cmd\":\"reset\",\"bypass_filter\":%u,\"engine\":\"%s\"}",
                (unsigned)bypass,
                semg_rtos_engine_name(semg_rtos_get_engine_mode())
            );
            semg_hal_tx_line(reply);
            break;
        }

        case SEMG_HIL_CMD_SET_ENGINE: {
            uint8_t engine = (frame->payload_len >= 1U) ? frame->payload[0] : 0U;
            semg_rtos_set_engine(engine);
            snprintf(
                reply,
                sizeof(reply),
                "SEMG_ACK: {\"cmd\":\"set_engine\",\"engine\":\"%s\"}",
                semg_rtos_engine_name(semg_rtos_get_engine_mode())
            );
            semg_hal_tx_line(reply);
            break;
        }

        case SEMG_HIL_CMD_RUN_SELFTEST: {
            semg_selftest_result_t st = semg_rtos_run_boot_selftest();
            snprintf(
                reply,
                sizeof(reply),
                "SEMG_SELFTEST: {\"windows\":%lu,\"lda_matches\":%lu,\"cnn_matches\":%lu,"
                "\"lda_avg_cyc\":%lu,\"cnn_avg_cyc\":%lu}",
                (unsigned long)st.windows_total,
                (unsigned long)st.lda_matches,
                (unsigned long)st.cnn_matches,
                (unsigned long)st.lda_avg_cycles,
                (unsigned long)st.cnn_avg_cycles
            );
            semg_hal_tx_line(reply);
            break;
        }

        case SEMG_HIL_CMD_PUSH_SAMPLES: {
            if (frame->payload_len < 4U) {
                semg_hal_tx_line("SEMG_ERR: {\"reason\":\"push_too_short\"}");
                break;
            }
            uint16_t n_samp = (uint16_t)frame->payload[0] | ((uint16_t)frame->payload[1] << 8);
            uint16_t n_ch   = (uint16_t)frame->payload[2] | ((uint16_t)frame->payload[3] << 8);
            uint32_t expected_bytes = 4U + (uint32_t)n_samp * (uint32_t)n_ch * sizeof(int16_t);

            if (n_samp == 0U || n_samp > SEMG_DMA_BLOCK_SAMPLES ||
                n_ch != semg_rtos_get_num_channels() || frame->payload_len != expected_bytes) {
                snprintf(
                    reply,
                    sizeof(reply),
                    "SEMG_ERR: {\"reason\":\"bad_shape\",\"n_samp\":%u,\"n_ch\":%u,\"len\":%u}",
                    (unsigned)n_samp,
                    (unsigned)n_ch,
                    (unsigned)frame->payload_len
                );
                semg_hal_tx_line(reply);
                break;
            }

            memcpy(g_hil_sample_buf, &frame->payload[4], (size_t)n_samp * (size_t)n_ch * sizeof(int16_t));
            if (semg_rtos_submit_block(g_hil_sample_buf, n_samp, n_ch) != 0) {
                semg_hal_tx_line("SEMG_ERR: {\"reason\":\"submit_failed\"}");
            }
            break;
        }

        default:
            snprintf(reply, sizeof(reply), "SEMG_ERR: {\"reason\":\"unknown_cmd\",\"cmd\":%u}", (unsigned)frame->cmd);
            semg_hal_tx_line(reply);
            break;
    }
}

static void vTaskHilSerialRx(void *pvParameters) {
    (void)pvParameters;
    uint8_t rx_chunk[256];
    semg_hil_parser_init(&g_hil_parser);

    while (1) {
        int n_read = 0;
#if defined(CONFIG_SEMG_HIL_TRANSPORT_USB_JTAG)
        n_read = usb_serial_jtag_read_bytes(rx_chunk, sizeof(rx_chunk), pdMS_TO_TICKS(20));
#elif defined(CONFIG_SEMG_HIL_TRANSPORT_UART0)
        n_read = uart_read_bytes(UART_NUM_0, rx_chunk, sizeof(rx_chunk), pdMS_TO_TICKS(20));
#endif
        if (n_read <= 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        for (int i = 0; i < n_read; i++) {
            semg_hil_parse_status_t st = semg_hil_parser_feed_byte(&g_hil_parser, rx_chunk[i], &g_hil_frame);
            if (st == SEMG_HIL_PARSE_FRAME_OK) {
                handle_hil_frame(&g_hil_frame);
            } else if (st == SEMG_HIL_PARSE_CRC_ERROR) {
                semg_hal_tx_line("SEMG_ERR: {\"reason\":\"crc_error\"}");
            } else if (st == SEMG_HIL_PARSE_LEN_ERROR) {
                semg_hal_tx_line("SEMG_ERR: {\"reason\":\"len_error\"}");
            }
        }
    }
}

#elif defined(CONFIG_SEMG_ACQ_MODE_ADC_DMA)

static adc_continuous_handle_t g_adc_handle = NULL;
static uint32_t g_adc_num_channels = 4U;
static float    g_adc_scale_factor = 1.0f;

static void vTaskAdcDmaReader(void *pvParameters) {
    (void)pvParameters;
    const uint32_t C = g_adc_num_channels;
    const uint32_t osr = CONFIG_SEMG_ADC_OVERSAMPLE_FACTOR;
    const float mv_per_lsb = (float)CONFIG_SEMG_ADC_VREF_MV / 2048.0f;

    uint8_t dma_rx_buf[1024];
    int32_t ch_accum[MAX_CHANNELS] = {0};
    uint32_t ch_count[MAX_CHANNELS] = {0};
    int16_t block_buf[SEMG_DMA_BLOCK_SAMPLES * MAX_CHANNELS];
    uint32_t sample_idx = 0U;

    while (1) {
        uint32_t ret_num = 0U;
        esp_err_t ret = adc_continuous_read(g_adc_handle, dma_rx_buf, sizeof(dma_rx_buf), &ret_num, portMAX_DELAY);
        if (ret != ESP_OK && ret != ESP_ERR_TIMEOUT) {
            continue;
        }

        for (uint32_t i = 0U; i + SOC_ADC_DIGI_RESULT_BYTES <= ret_num; i += SOC_ADC_DIGI_RESULT_BYTES) {
            const adc_digi_output_data_t *p = (const adc_digi_output_data_t *)&dma_rx_buf[i];
            uint32_t unit_offset = (p->type2.unit == 0U) ? 0U : 10U;
            uint32_t hw_ch = unit_offset + p->type2.channel;
            uint32_t raw   = p->type2.data;

            if (hw_ch < C) {
                ch_accum[hw_ch] += (int32_t)raw - 2048;
                ch_count[hw_ch]++;

                int all_ready = 1;
                for (uint32_t c = 0U; c < C; c++) {
                    if (ch_count[c] < osr) {
                        all_ready = 0;
                        break;
                    }
                }
                if (all_ready) {
                    for (uint32_t c = 0U; c < C; c++) {
                        float mean_lsb = (float)ch_accum[c] / (float)ch_count[c];
                        float mv = mean_lsb * mv_per_lsb;
                        int32_t q = (int32_t)lrintf(mv * g_adc_scale_factor);
                        if (q > 32767) q = 32767;
                        if (q < -32768) q = -32768;
                        block_buf[sample_idx * C + c] = (int16_t)q;
                        ch_accum[c] = 0;
                        ch_count[c] = 0U;
                    }
                    sample_idx++;
                    if (sample_idx >= SEMG_DMA_BLOCK_SAMPLES) {
                        (void)semg_rtos_submit_block(block_buf, SEMG_DMA_BLOCK_SAMPLES, C);
                        sample_idx = 0U;
                    }
                }
            }
        }
    }
}

static void init_adc_continuous_dma(uint32_t num_channels, float adc_scale) {
    /* ESP32-S3 ADC1 has 10 channels (GPIO1..10); channels 10..11 map to ADC2 (GPIO11..12) */
    uint32_t active_ch = (num_channels > MAX_CHANNELS) ? MAX_CHANNELS : num_channels;
    g_adc_num_channels = active_ch;
    g_adc_scale_factor = adc_scale;

    uint32_t hw_freq_hz = SEMG_SAMPLE_RATE_HZ * active_ch * CONFIG_SEMG_ADC_OVERSAMPLE_FACTOR;
    if (hw_freq_hz < 20000U) {
        hw_freq_hz = 20000U;
    }

    uint32_t frame_bytes = active_ch * CONFIG_SEMG_ADC_OVERSAMPLE_FACTOR * 16U * SOC_ADC_DIGI_RESULT_BYTES;
    adc_continuous_handle_cfg_t adc_cfg = {
        .max_store_buf_size = frame_bytes * 4U,
        .conv_frame_size = frame_bytes
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_cfg, &g_adc_handle));

    adc_digi_pattern_config_t pattern[MAX_CHANNELS];
    for (uint32_t c = 0U; c < active_ch; c++) {
#ifdef ADC_ATTEN_DB_12
        pattern[c].atten = ADC_ATTEN_DB_12;
#else
        pattern[c].atten = ADC_ATTEN_DB_11;
#endif
        if (c < 10U) {
            pattern[c].channel = (uint8_t)c;
            pattern[c].unit    = ADC_UNIT_1;
        } else {
            pattern[c].channel = (uint8_t)(c - 10U);
            pattern[c].unit    = ADC_UNIT_2;
        }
        pattern[c].bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
    }

    adc_continuous_config_t dig_cfg = {
        .sample_freq_hz = hw_freq_hz,
        .conv_mode      = (active_ch <= 10U) ? ADC_CONV_SINGLE_UNIT_1 : ADC_CONV_BOTH_UNIT,
        .format_mode    = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
        .pattern_num    = active_ch,
        .adc_pattern    = pattern
    };
    ESP_ERROR_CHECK(adc_continuous_config(g_adc_handle, &dig_cfg));
    ESP_ERROR_CHECK(adc_continuous_start(g_adc_handle));

    xTaskCreatePinnedToCore(vTaskAdcDmaReader, "vTaskAdcDmaReader", 4096, NULL, 11, NULL, 0);
}

#elif defined(CONFIG_SEMG_ACQ_MODE_SELFTEST_LOOP)

static SemaphoreHandle_t g_selftest_tick_sem = NULL;

static void selftest_timer_cb(void *arg) {
    (void)arg;
    if (g_selftest_tick_sem != NULL) {
        xSemaphoreGive(g_selftest_tick_sem);
    }
}

static void vTaskSelfTestLoop(void *pvParameters) {
    (void)pvParameters;
    uint32_t offset_samples = 0U;
    int16_t block[SEMG_DMA_BLOCK_SAMPLES * OZDEMIR_LDA_NUM_CHANNELS];

    while (xSemaphoreTake(g_selftest_tick_sem, portMAX_DELAY) == pdTRUE) {
        for (uint32_t s = 0U; s < SEMG_DMA_BLOCK_SAMPLES; s++) {
            uint32_t src_idx = (offset_samples + s) % OZDEMIR_SELFTEST_SAMPLES;
            for (uint32_t c = 0U; c < OZDEMIR_LDA_NUM_CHANNELS; c++) {
                block[s * OZDEMIR_LDA_NUM_CHANNELS + c] =
                    g_ozdemir_selftest_stream_q16[src_idx * OZDEMIR_LDA_NUM_CHANNELS + c];
            }
        }
        offset_samples = (offset_samples + SEMG_DMA_BLOCK_SAMPLES) % OZDEMIR_SELFTEST_SAMPLES;
        (void)semg_rtos_submit_block(block, SEMG_DMA_BLOCK_SAMPLES, OZDEMIR_LDA_NUM_CHANNELS);
    }
}

#endif

void semg_hal_acq_start(uint32_t num_channels, float adc_scale) {
#if defined(CONFIG_SEMG_ACQ_MODE_HIL_SERIAL)
    (void)num_channels;
    (void)adc_scale;
    ESP_LOGI(TAG, "Starting HIL Binary Serial RX task on Core 0");
    xTaskCreatePinnedToCore(vTaskHilSerialRx, "vTaskHilSerialRx", 4096, NULL, 8, NULL, 0);
#elif defined(CONFIG_SEMG_ACQ_MODE_ADC_DMA)
    ESP_LOGI(TAG, "Starting ESP32-S3 Continuous ADC DMA on Core 0 (ch=%lu)", (unsigned long)num_channels);
    init_adc_continuous_dma(num_channels, adc_scale);
#elif defined(CONFIG_SEMG_ACQ_MODE_SELFTEST_LOOP)
    (void)num_channels;
    (void)adc_scale;
    g_selftest_tick_sem = xSemaphoreCreateBinary();
    configASSERT(g_selftest_tick_sem != NULL);
    xTaskCreatePinnedToCore(vTaskSelfTestLoop, "vTaskSelfTestLoop", 3072, NULL, 8, NULL, 0);

    esp_timer_handle_t timer = NULL;
    const esp_timer_create_args_t targs = {
        .callback = selftest_timer_cb,
        .arg = NULL,
        .name = "semg_selftest_50ms"
    };
    ESP_ERROR_CHECK(esp_timer_create(&targs, &timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(timer, 50000ULL)); /* 50 ms = 100 samples @ 2000 Hz */
    ESP_LOGI(TAG, "Starting Built-in Flash Self-Test Loop (50 ms periodic timer)");
#endif
}
