/*
 * semg_hil_protocol.c — Deterministic Binary Framing Protocol Implementation
 */

#include "../include/semg_hil_protocol.h"
#include <string.h>

static uint16_t crc16_update_byte(uint16_t crc, uint8_t b) {
    crc ^= (uint16_t)b << 8;
    for (int i = 0; i < 8; i++) {
        if (crc & 0x8000U) {
            crc = (uint16_t)((crc << 1) ^ 0x1021U);
        } else {
            crc = (uint16_t)(crc << 1);
        }
    }
    return crc;
}

uint16_t semg_hil_crc16_ccitt(const uint8_t *data, size_t len, uint16_t init) {
    uint16_t crc = init;
    for (size_t i = 0; i < len; i++) {
        crc = crc16_update_byte(crc, data[i]);
    }
    return crc;
}

void semg_hil_parser_init(semg_hil_parser_t *p) {
    memset(p, 0, sizeof(*p));
    p->state = HIL_STATE_WAIT_MAGIC_0;
}

semg_hil_parse_status_t semg_hil_parser_feed_byte(
    semg_hil_parser_t *p,
    uint8_t byte,
    semg_hil_frame_t *out_frame
) {
    switch (p->state) {
        case HIL_STATE_WAIT_MAGIC_0:
            if (byte == SEMG_HIL_MAGIC_0) {
                p->state = HIL_STATE_WAIT_MAGIC_1;
            }
            return SEMG_HIL_PARSE_INCOMPLETE;

        case HIL_STATE_WAIT_MAGIC_1:
            if (byte == SEMG_HIL_MAGIC_1) {
                p->state = HIL_STATE_READ_CMD;
            } else if (byte == SEMG_HIL_MAGIC_0) {
                p->state = HIL_STATE_WAIT_MAGIC_1;
            } else {
                p->state = HIL_STATE_WAIT_MAGIC_0;
            }
            return SEMG_HIL_PARSE_INCOMPLETE;

        case HIL_STATE_READ_CMD:
            p->cmd = byte;
            p->crc_calc = crc16_update_byte(0xFFFFU, byte);
            p->state = HIL_STATE_READ_LEN_LO;
            return SEMG_HIL_PARSE_INCOMPLETE;

        case HIL_STATE_READ_LEN_LO:
            p->payload_len = (uint16_t)byte;
            p->crc_calc = crc16_update_byte(p->crc_calc, byte);
            p->state = HIL_STATE_READ_LEN_HI;
            return SEMG_HIL_PARSE_INCOMPLETE;

        case HIL_STATE_READ_LEN_HI:
            p->payload_len |= (uint16_t)((uint16_t)byte << 8);
            p->crc_calc = crc16_update_byte(p->crc_calc, byte);
            if (p->payload_len > SEMG_HIL_MAX_PAYLOAD) {
                p->frames_len_err++;
                p->state = HIL_STATE_WAIT_MAGIC_0;
                return SEMG_HIL_PARSE_LEN_ERROR;
            }
            p->payload_pos = 0U;
            if (p->payload_len == 0U) {
                p->state = HIL_STATE_READ_CRC_LO;
            } else {
                p->state = HIL_STATE_READ_PAYLOAD;
            }
            return SEMG_HIL_PARSE_INCOMPLETE;

        case HIL_STATE_READ_PAYLOAD:
            p->payload[p->payload_pos++] = byte;
            p->crc_calc = crc16_update_byte(p->crc_calc, byte);
            if (p->payload_pos >= p->payload_len) {
                p->state = HIL_STATE_READ_CRC_LO;
            }
            return SEMG_HIL_PARSE_INCOMPLETE;

        case HIL_STATE_READ_CRC_LO:
            p->crc_rx = (uint16_t)byte;
            p->state = HIL_STATE_READ_CRC_HI;
            return SEMG_HIL_PARSE_INCOMPLETE;

        case HIL_STATE_READ_CRC_HI:
            p->crc_rx |= (uint16_t)((uint16_t)byte << 8);
            p->state = HIL_STATE_WAIT_MAGIC_0;
            if (p->crc_rx == p->crc_calc) {
                p->frames_ok++;
                if (out_frame != NULL) {
                    out_frame->cmd = p->cmd;
                    out_frame->payload_len = p->payload_len;
                    if (p->payload_len > 0U) {
                        memcpy(out_frame->payload, p->payload, p->payload_len);
                    }
                    out_frame->crc_received = p->crc_rx;
                    out_frame->crc_computed = p->crc_calc;
                }
                return SEMG_HIL_PARSE_FRAME_OK;
            } else {
                p->frames_crc_err++;
                return SEMG_HIL_PARSE_CRC_ERROR;
            }

        default:
            p->state = HIL_STATE_WAIT_MAGIC_0;
            return SEMG_HIL_PARSE_INCOMPLETE;
    }
}

size_t semg_hil_encode_frame(
    uint8_t cmd,
    const uint8_t *payload,
    uint16_t payload_len,
    uint8_t *out_buf,
    size_t out_buf_cap
) {
    if (payload_len > SEMG_HIL_MAX_PAYLOAD || out_buf == NULL) {
        return 0U;
    }
    const size_t total = 7U + (size_t)payload_len;
    if (out_buf_cap < total) {
        return 0U;
    }

    out_buf[0] = SEMG_HIL_MAGIC_0;
    out_buf[1] = SEMG_HIL_MAGIC_1;
    out_buf[2] = cmd;
    out_buf[3] = (uint8_t)(payload_len & 0xFFU);
    out_buf[4] = (uint8_t)((payload_len >> 8) & 0xFFU);
    if (payload_len > 0U && payload != NULL) {
        memcpy(&out_buf[5], payload, payload_len);
    }

    uint16_t crc = semg_hil_crc16_ccitt(&out_buf[2], 3U + (size_t)payload_len, 0xFFFFU);
    out_buf[5U + payload_len] = (uint8_t)(crc & 0xFFU);
    out_buf[6U + payload_len] = (uint8_t)((crc >> 8) & 0xFFU);
    return total;
}
