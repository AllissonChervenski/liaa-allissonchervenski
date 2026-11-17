/*
 * semg_app_config.h — Unified ESP-IDF Kconfig / Standalone C99 Configuration
 */

#ifndef SEMG_APP_CONFIG_H
#define SEMG_APP_CONFIG_H

#ifdef ESP_PLATFORM
#include "sdkconfig.h"
#endif

/* Acquisition modes */
#if !defined(CONFIG_SEMG_ACQ_MODE_HIL_SERIAL) && \
    !defined(CONFIG_SEMG_ACQ_MODE_ADC_DMA) && \
    !defined(CONFIG_SEMG_ACQ_MODE_SELFTEST_LOOP)
#define CONFIG_SEMG_ACQ_MODE_HIL_SERIAL 1
#endif

/* HIL Transport */
#if defined(CONFIG_SEMG_ACQ_MODE_HIL_SERIAL) && \
    !defined(CONFIG_SEMG_HIL_TRANSPORT_USB_JTAG) && \
    !defined(CONFIG_SEMG_HIL_TRANSPORT_UART0)
#define CONFIG_SEMG_HIL_TRANSPORT_USB_JTAG 1
#endif

/* Default classifier */
#if !defined(CONFIG_SEMG_MODEL_MICRO_LDA_Q7) && \
    !defined(CONFIG_SEMG_MODEL_CNN1D_INT8)
#define CONFIG_SEMG_MODEL_MICRO_LDA_Q7 1
#endif

/* Dataset topology */
#if !defined(CONFIG_SEMG_DATASET_OZDEMIR_4CH) && \
    !defined(CONFIG_SEMG_DATASET_NINAPRO_12CH)
#define CONFIG_SEMG_DATASET_OZDEMIR_4CH 1
#endif

#ifndef CONFIG_SEMG_RUN_BOOT_SELFTEST
#define CONFIG_SEMG_RUN_BOOT_SELFTEST 1
#endif

#ifndef CONFIG_SEMG_ENABLE_BANDPASS_FILTER
#define CONFIG_SEMG_ENABLE_BANDPASS_FILTER 1
#endif

#ifndef CONFIG_SEMG_ADC_OVERSAMPLE_FACTOR
#define CONFIG_SEMG_ADC_OVERSAMPLE_FACTOR 4
#endif

#ifndef CONFIG_SEMG_ADC_VREF_MV
#define CONFIG_SEMG_ADC_VREF_MV 1650
#endif

#ifndef CONFIG_SEMG_STATUS_LED_GPIO
#define CONFIG_SEMG_STATUS_LED_GPIO (-1)
#endif

#define SEMG_CPU_FREQ_MHZ      (240U)
#define SEMG_SAMPLE_RATE_HZ    (2000U)
#define SEMG_DMA_BLOCK_SAMPLES (100U)  /* 50 ms @ 2000 Hz */

typedef enum {
    SEMG_ENGINE_MICRO_LDA_Q7        = 0,
    SEMG_ENGINE_CNN1D_INT8          = 1,
    SEMG_ENGINE_DTW_CLASSIC_ON2     = 2,
    SEMG_ENGINE_DTW_SAKOE_CHIBA     = 3,
    SEMG_ENGINE_DTW_LB_KEOGH        = 4,
    SEMG_ENGINE_HYBRID_LDA_CNN_HEAD = 5,
    SEMG_ENGINE_HYBRID_LDA_CNN_CASC = 6,
    SEMG_ENGINE_HYBRID_DTW_LDA      = 7,
    SEMG_ENGINE_HYBRID_DTW_CNN      = 8,
    SEMG_ENGINE_HYBRID_DTW_LDA_CNN  = 9,
    SEMG_ENGINE_COUNT               = 10
} semg_engine_mode_t;

typedef enum {
    SEMG_DATASET_OZDEMIR_4CH  = 0,
    SEMG_DATASET_NINAPRO_12CH = 1
} semg_dataset_topology_t;

#endif /* SEMG_APP_CONFIG_H */
