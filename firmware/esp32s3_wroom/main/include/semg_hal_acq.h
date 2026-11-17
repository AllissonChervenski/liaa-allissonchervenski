/*
 * semg_hal_acq.h — Hardware Abstraction Layer for ESP32-S3-WROOM Acquisition & HIL
 */

#ifndef SEMG_HAL_ACQ_H
#define SEMG_HAL_ACQ_H

#include <stdint.h>
#include <stddef.h>
#include "semg_app_config.h"

/* Initialize configured serial/USB-JTAG transport and TX mutex */
void semg_hal_transport_init(void);

/* Thread-safe telemetry line writer (adds '\n' if needed) */
void semg_hal_tx_line(const char *line);

/* Start configured acquisition backend (HIL Serial, ADC Continuous DMA, or Self-Test Loop) */
void semg_hal_acq_start(uint32_t num_channels, float adc_scale);

#endif /* SEMG_HAL_ACQ_H */
