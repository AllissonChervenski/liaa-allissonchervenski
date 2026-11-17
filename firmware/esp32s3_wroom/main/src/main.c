/*
 * main.c — ESP32-S3-WROOM sEMG Real-Time Classification Entry Point (ESP-IDF v5.x)
 */

#include <stdio.h>
#include "../include/semg_app_config.h"
#include "../include/semg_hal_acq.h"
#include "../include/semg_rtos_app.h"

void app_main(void) {
    char banner[224];

    semg_hal_transport_init();
    semg_rtos_app_init();

#if CONFIG_SEMG_RUN_BOOT_SELFTEST
    semg_selftest_result_t st = semg_rtos_run_boot_selftest();
    snprintf(
        banner,
        sizeof(banner),
        "SEMG_BOOT_SELFTEST: {\"windows\":%lu,\"lda_matches\":%lu,\"cnn_matches\":%lu,"
        "\"lda_avg_cyc\":%lu,\"cnn_avg_cyc\":%lu}",
        (unsigned long)st.windows_total,
        (unsigned long)st.lda_matches,
        (unsigned long)st.cnn_matches,
        (unsigned long)st.lda_avg_cycles,
        (unsigned long)st.cnn_avg_cycles
    );
    semg_hal_tx_line(banner);
#endif

#if defined(CONFIG_SEMG_ACQ_MODE_ADC_DMA)
    const char *acq_name = "adc_continuous_dma";
#elif defined(CONFIG_SEMG_ACQ_MODE_SELFTEST_LOOP)
    const char *acq_name = "selftest_loop";
#else
    const char *acq_name = "hil_serial";
#endif

    snprintf(
        banner,
        sizeof(banner),
        "SEMG_READY: {\"target\":\"esp32s3-wroom\",\"acq_mode\":\"%s\","
        "\"default_engine\":\"%s\",\"channels\":%lu,\"classes\":%lu}",
        acq_name,
        semg_rtos_engine_name(semg_rtos_get_engine_mode()),
        (unsigned long)semg_rtos_get_num_channels(),
        (unsigned long)semg_rtos_get_num_classes()
    );
    semg_hal_tx_line(banner);

    semg_hal_acq_start(semg_rtos_get_num_channels(), semg_rtos_get_adc_scale());
}
