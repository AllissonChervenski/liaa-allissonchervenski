/*
 * main_cnn.c — Bare-Metal ESP32 / ESP32-S3 QEMU Harness for Quantized INT8 1D-CNN
 *
 * Implements the test harness specified in Section 4 (Step 3):
 *   1. Loads test vectors (`cnn1d_test_vectors.bin`) via Xtensa Semihosting
 *   2. Reads initial hardware cycle counter (`CCOUNT`)
 *   3. Dispatches pure C99 INT8 1D-CNN inference (`cnn1d_forward_int8`)
 *   4. Reads final `CCOUNT` and records per-window cycles, mean, p99, and max latency
 *   5. Compares C99 firmware predictions and raw integer logits against the Python/PyTorch oracle
 *   6. Emits formatted JSON telemetry over virtual UART (`QEMU_CNN1D_REPORT_JSON: {...}`)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../include/cnn1d_inference.h"
#include "../include/cnn1d_weights_int8.h"

/* Semihosting raw syscall interface from libsys_qemu.a */
extern int __sim_call(int id, int a1, int a2, int a3, int a4);
extern int _open(const char *name, int flags, int mode);
extern int _close(int fd);

int _read(int fd, void *buf, size_t cnt) {
    return __sim_call(3, fd, (int)buf, (int)cnt, 0);
}

typedef struct {
    uint32_t magic;        /* 0x434E4E31 ("CNN1") */
    uint32_t dataset_code; /* 0 = Ozdemir 4CH, 1 = NinaPro DB2 12CH */
    uint32_t in_channels;
    uint32_t num_classes;
    uint32_t num_aux;
    uint32_t num_vectors;
} cnn1d_vector_header_t;

#define MAX_EVAL_WINDOWS 16000
static uint32_t g_window_cycles[MAX_EVAL_WINDOWS];
static int8_t   g_x_q[12 * CNN1D_WINDOW_LEN];
static int8_t   g_aux_q[60];
static int32_t  g_oracle_logits[CNN1D_MAX_CLASSES];
static int32_t  g_mcu_logits[CNN1D_MAX_CLASSES];

static uint32_t g_tp[CNN1D_MAX_CLASSES];
static uint32_t g_fp[CNN1D_MAX_CLASSES];
static uint32_t g_fn[CNN1D_MAX_CLASSES];

static void read_exact(int fd, void *dst, size_t nbytes) {
    uint8_t *p = (uint8_t *)dst;
    size_t rem = nbytes;
    while (rem > 0) {
        int r = _read(fd, p, rem);
        if (r <= 0) {
            printf("{\"error\": \"EOF while reading cnn1d_test_vectors.bin (%u bytes left)\"}\n", (unsigned)rem);
            exit(2);
        }
        p += (size_t)r;
        rem -= (size_t)r;
    }
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t va = *(const uint32_t *)a;
    uint32_t vb = *(const uint32_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

int main(void) {
    int fd = _open("cnn1d_test_vectors.bin", 0, 0);
    if (fd < 0) {
        printf("{\"error\": \"Could not open cnn1d_test_vectors.bin\"}\n");
        return 1;
    }

    cnn1d_vector_header_t hdr;
    read_exact(fd, &hdr, sizeof(cnn1d_vector_header_t));
    if (hdr.magic != 0x434E4E31U) {
        printf("{\"error\": \"Invalid magic 0x%08x\"}\n", (unsigned)hdr.magic);
        _close(fd);
        return 1;
    }

    const cnn1d_model_config_t *cfg = (hdr.dataset_code == 0)
        ? &g_cnn1d_ozdemir_config
        : &g_cnn1d_ninapro_config;

    const int n_ch  = (int)hdr.in_channels;
    const int n_cls = (int)hdr.num_classes;
    const int n_aux = (int)hdr.num_aux;
    const int n_vec = (int)((hdr.num_vectors <= MAX_EVAL_WINDOWS) ? hdr.num_vectors : MAX_EVAL_WINDOWS);

    memset(g_tp, 0, sizeof(g_tp));
    memset(g_fp, 0, sizeof(g_fp));
    memset(g_fn, 0, sizeof(g_fn));

    uint32_t oracle_matches = 0;
    uint32_t label_correct  = 0;
    int32_t  max_logit_diff = 0;

    uint64_t sum_cycles_total = 0;
    uint64_t sum_cycles_conv1 = 0;
    uint64_t sum_cycles_pool1 = 0;
    uint64_t sum_cycles_conv2 = 0;
    uint64_t sum_cycles_gap   = 0;
    uint64_t sum_cycles_fc    = 0;

    for (int i = 0; i < n_vec; i++) {
        int32_t y_true = 0;
        int32_t y_oracle = 0;

        read_exact(fd, &y_true, sizeof(int32_t));
        read_exact(fd, &y_oracle, sizeof(int32_t));
        read_exact(fd, g_oracle_logits, (size_t)n_cls * sizeof(int32_t));
        read_exact(fd, g_x_q, (size_t)(n_ch * CNN1D_WINDOW_LEN) * sizeof(int8_t));
        read_exact(fd, g_aux_q, (size_t)n_aux * sizeof(int8_t));

        cnn1d_cycle_profile_t prof;
        uint32_t c_start = cnn1d_read_ccount();
        int32_t y_mcu = cnn1d_forward_int8(cfg, g_x_q, g_aux_q, g_mcu_logits, &prof);
        uint32_t c_end = cnn1d_read_ccount();

        uint32_t win_cyc = c_end - c_start;
        g_window_cycles[i] = win_cyc;
        sum_cycles_total += (uint64_t)win_cyc;
        sum_cycles_conv1 += (uint64_t)prof.cycles_conv1;
        sum_cycles_pool1 += (uint64_t)prof.cycles_pool1;
        sum_cycles_conv2 += (uint64_t)prof.cycles_conv2;
        sum_cycles_gap   += (uint64_t)prof.cycles_gap;
        sum_cycles_fc    += (uint64_t)prof.cycles_fc;

        if (y_mcu == y_oracle) {
            oracle_matches++;
        }
        for (int k = 0; k < n_cls; k++) {
            int32_t diff = g_mcu_logits[k] - g_oracle_logits[k];
            if (diff < 0) diff = -diff;
            if (diff > max_logit_diff) max_logit_diff = diff;
        }

        if (y_mcu == y_true) {
            label_correct++;
            g_tp[y_true]++;
        } else {
            g_fn[y_true]++;
            g_fp[y_mcu]++;
        }
    }
    _close(fd);

    /* Compute Top-1, Balanced Accuracy, and Macro F1 */
    float top1_acc = (n_vec > 0) ? ((float)label_correct / (float)n_vec) : 0.0f;
    float sum_recall = 0.0f;
    float sum_f1 = 0.0f;
    int active_cls = 0;

    for (int c = 0; c < n_cls; c++) {
        uint32_t tp = g_tp[c];
        uint32_t fn = g_fn[c];
        uint32_t fp = g_fp[c];
        if (tp + fn > 0) {
            active_cls++;
            float rec = (float)tp / (float)(tp + fn);
            float prec = (tp + fp > 0) ? ((float)tp / (float)(tp + fp)) : 0.0f;
            float f1 = (prec + rec > 0.0f) ? (2.0f * prec * rec / (prec + rec)) : 0.0f;
            sum_recall += rec;
            sum_f1 += f1;
        }
    }
    float bal_acc = (active_cls > 0) ? (sum_recall / (float)active_cls) : 0.0f;
    float macro_f1 = (active_cls > 0) ? (sum_f1 / (float)active_cls) : 0.0f;
    float parity_ratio = (n_vec > 0) ? ((float)oracle_matches / (float)n_vec) : 0.0f;

    /* Sort cycle counts to extract p99 and max latency */
    qsort(g_window_cycles, (size_t)n_vec, sizeof(uint32_t), cmp_u32);
    int p99_idx = (int)((n_vec * 99) / 100);
    if (p99_idx >= n_vec) p99_idx = n_vec - 1;
    uint32_t cycles_min  = (n_vec > 0) ? g_window_cycles[0] : 0;
    uint32_t cycles_p99  = (n_vec > 0) ? g_window_cycles[p99_idx] : 0;
    uint32_t cycles_max  = (n_vec > 0) ? g_window_cycles[n_vec - 1] : 0;
    uint32_t cycles_mean = (n_vec > 0) ? (uint32_t)(sum_cycles_total / (uint64_t)n_vec) : 0;

    uint32_t cyc_c1   = (n_vec > 0) ? (uint32_t)(sum_cycles_conv1 / (uint64_t)n_vec) : 0;
    uint32_t cyc_p1   = (n_vec > 0) ? (uint32_t)(sum_cycles_pool1 / (uint64_t)n_vec) : 0;
    uint32_t cyc_c2   = (n_vec > 0) ? (uint32_t)(sum_cycles_conv2 / (uint64_t)n_vec) : 0;
    uint32_t cyc_gap  = (n_vec > 0) ? (uint32_t)(sum_cycles_gap   / (uint64_t)n_vec) : 0;
    uint32_t cyc_fc   = (n_vec > 0) ? (uint32_t)(sum_cycles_fc    / (uint64_t)n_vec) : 0;

    printf("QEMU_CNN1D_REPORT_JSON: {");
    printf("\"dataset_code\": %u, ", (unsigned)hdr.dataset_code);
    printf("\"in_channels\": %d, ", n_ch);
    printf("\"num_classes\": %d, ", n_cls);
    printf("\"num_vectors\": %d, ", n_vec);
    printf("\"host_parity_ratio\": %.6f, ", (double)parity_ratio);
    printf("\"max_logit_diff\": %d, ", (int)max_logit_diff);
    printf("\"top1_accuracy\": %.6f, ", (double)top1_acc);
    printf("\"balanced_accuracy\": %.6f, ", (double)bal_acc);
    printf("\"macro_f1\": %.6f, ", (double)macro_f1);
    printf("\"cycles_mean\": %u, ", (unsigned)cycles_mean);
    printf("\"cycles_min\": %u, ", (unsigned)cycles_min);
    printf("\"cycles_p99\": %u, ", (unsigned)cycles_p99);
    printf("\"cycles_max\": %u, ", (unsigned)cycles_max);
    printf("\"cycles_conv1\": %u, ", (unsigned)cyc_c1);
    printf("\"cycles_pool1\": %u, ", (unsigned)cyc_p1);
    printf("\"cycles_conv2\": %u, ", (unsigned)cyc_c2);
    printf("\"cycles_gap\": %u, ", (unsigned)cyc_gap);
    printf("\"cycles_fc\": %u, ", (unsigned)cyc_fc);
    printf("\"latency_mean_us_240mhz\": %.2f, ", (double)cycles_mean / 240.0);
    printf("\"latency_p99_us_240mhz\": %.2f, ", (double)cycles_p99 / 240.0);
    printf("\"latency_mean_ms_240mhz\": %.5f, ", (double)cycles_mean / 240000.0);
    printf("\"latency_p99_ms_240mhz\": %.5f", (double)cycles_p99 / 240000.0);
    printf("}\n");

    return (parity_ratio >= 0.995f) ? 0 : 3;
}
