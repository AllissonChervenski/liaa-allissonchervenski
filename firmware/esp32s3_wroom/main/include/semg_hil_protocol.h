/*
 * semg_hil_protocol.h — Deterministic Binary Framing Protocol for ESP32-S3 HIL
 *
 * Pure C99 state-machine parser with zero dynamic allocation and CRC-16/CCITT-FALSE
 * integrity verification. Designed for USB-Serial/JTAG and UART0 streaming.
 */

#ifndef SEMG_HIL_PROTOCOL_H
#define SEMG_HIL_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#define SEMG_HIL_MAGIC_0        (0xAAU)
#define SEMG_HIL_MAGIC_1        (0x55U)
#define SEMG_HIL_MAX_PAYLOAD    (4096U)

typedef enum {
    SEMG_HIL_CMD_PING         = 0x01U,
    SEMG_HIL_CMD_RESET        = 0x02U,
    SEMG_HIL_CMD_PUSH_SAMPLES = 0x03U,
    SEMG_HIL_CMD_SET_ENGINE   = 0x04U,
    SEMG_HIL_CMD_RUN_SELFTEST = 0x05U
} semg_hil_cmd_t;

#define SEMG_HIL_FLAG_BYPASS_FILTER (1U << 0)
#define SEMG_HIL_FLAG_ENGINE_CNN1D  (1U << 1)
#define SEMG_HIL_FLAG_HAS_EXT_ENG   (1U << 2) /* When payload[1] specifies exact engine_id 0..9 */

typedef enum {
    HIL_STATE_WAIT_MAGIC_0 = 0,
    HIL_STATE_WAIT_MAGIC_1,
    HIL_STATE_READ_CMD,
    HIL_STATE_READ_LEN_LO,
    HIL_STATE_READ_LEN_HI,
    HIL_STATE_READ_PAYLOAD,
    HIL_STATE_READ_CRC_LO,
    HIL_STATE_READ_CRC_HI
} semg_hil_rx_state_t;

typedef enum {
    SEMG_HIL_PARSE_INCOMPLETE = 0,
    SEMG_HIL_PARSE_FRAME_OK   = 1,
    SEMG_HIL_PARSE_CRC_ERROR  = -1,
    SEMG_HIL_PARSE_LEN_ERROR  = -2
} semg_hil_parse_status_t;

typedef struct {
    uint8_t  cmd;
    uint16_t payload_len;
    uint8_t  payload[SEMG_HIL_MAX_PAYLOAD];
    uint16_t crc_received;
    uint16_t crc_computed;
} semg_hil_frame_t;

typedef struct {
    semg_hil_rx_state_t state;
    uint8_t  cmd;
    uint16_t payload_len;
    uint16_t payload_pos;
    uint16_t crc_rx;
    uint16_t crc_calc;
    uint8_t  payload[SEMG_HIL_MAX_PAYLOAD];
    uint32_t frames_ok;
    uint32_t frames_crc_err;
    uint32_t frames_len_err;
} semg_hil_parser_t;

uint16_t semg_hil_crc16_ccitt(const uint8_t *data, size_t len, uint16_t init);
void semg_hil_parser_init(semg_hil_parser_t *p);
semg_hil_parse_status_t semg_hil_parser_feed_byte(
    semg_hil_parser_t *p,
    uint8_t byte,
    semg_hil_frame_t *out_frame
);

/* Encode a frame into out_buf; returns total encoded byte length, or 0 on buffer overflow */
size_t semg_hil_encode_frame(
    uint8_t cmd,
    const uint8_t *payload,
    uint16_t payload_len,
    uint8_t *out_buf,
    size_t out_buf_cap
);

#endif /* SEMG_HIL_PROTOCOL_H */
