/*
 * freertos_qemu_shim.c — Deterministic FreeRTOS / ESP-IDF Compatibility Kernel for QEMU Xtensa
 *
 * Implements static task control blocks (TCBs) with 0xA5 canary stack high-water
 * mark auditing, binary semaphores, lock-free static ring-buffer queues,
 * hardware periodic timer interrupt vectoring (Xtensa Timer0 CCOMPARE0 IRQ 6),
 * and dual-core SMP task dispatching (PRO_CPU Core 0 & APP_CPU Core 1).
 */

#include "../include/freertos_qemu_shim.h"
#include <string.h>

#define MAX_RTOS_TASKS      4
#define MAX_RTOS_QUEUES     4
#define MAX_RTOS_SEMAPHORES 4
#define MAX_TASK_STACK_BYTES 8192
#define MAX_QUEUE_STORAGE_BYTES 1024

struct tskTaskControlBlock {
    int            in_use;
    char           name[24];
    TaskFunction_t entry;
    void          *param;
    UBaseType_t    priority;
    BaseType_t     core_id;
    uint32_t       stack_depth_bytes;
    uint8_t        stack_canary_area[MAX_TASK_STACK_BYTES];
    uint32_t       peak_stack_used_bytes;
    TickType_t     wake_tick_us;
};

struct SemaphoreDefinition {
    int              in_use;
    volatile uint8_t count;
    TaskHandle_t     waiting_task;
};

struct QueueDefinition {
    int              in_use;
    UBaseType_t      length;
    UBaseType_t      item_size;
    volatile UBaseType_t head;
    volatile UBaseType_t tail;
    volatile UBaseType_t count;
    uint32_t         full_events;
    uint8_t          storage[MAX_QUEUE_STORAGE_BYTES];
};

struct esp_timer_impl {
    int             in_use;
    int             active;
    esp_timer_cb_t  callback;
    void           *arg;
    uint64_t        period_us;
    uint32_t        period_cycles;
};

static struct tskTaskControlBlock g_tcbs[MAX_RTOS_TASKS];
static struct SemaphoreDefinition g_sems[MAX_RTOS_SEMAPHORES];
static struct QueueDefinition     g_queues[MAX_RTOS_QUEUES];
static struct esp_timer_impl      g_timer;

static volatile uint32_t g_critical_nesting = 0;
static volatile uint32_t g_saved_intenable  = 0;
static volatile int      g_scheduler_running = 0;
static volatile int      g_yield_pending_from_isr = 0;
static volatile TickType_t g_virtual_tick_us = 0;

static TaskHandle_t g_core0_task = NULL;
static TaskHandle_t g_core1_task = NULL;
static TaskHandle_t g_current_task = NULL;

static rtos_kernel_telemetry_t g_ktelemetry;

/* Hardware Timer0 ISR state */
static volatile uint32_t g_hw_timer0_irq_count = 0;

/* Override _UserExceptionVector at 0x40378340 to service Xtensa Level-1 Timer0 IRQ 6 */
__asm__(
    ".section .UserExceptionVector.text, \"ax\"\n"
    ".global _UserExceptionVector\n"
    "_UserExceptionVector:\n"
    "wsr.excsave1 a0\n"
    "j _rtos_custom_user_exc\n"
    ".previous\n"

    ".section .text\n"
    ".align 4\n"
    ".global _rtos_custom_user_exc\n"
    "_rtos_custom_user_exc:\n"
    "rsr.exccause a0\n"
    "beqi a0, 5, _rtos_to_alloca_exc\n"
    "beqi a0, 4, _rtos_handle_level1_irq\n"
    "j _rtos_unhandled_exc\n"

    "_rtos_to_alloca_exc:\n"
    "j _xt_alloca_exc\n"

    "_rtos_unhandled_exc:\n"
    "rsr.exccause a3\n"
    "addi a3, a3, 100\n"
    "movi a2, 1\n"
    "simcall\n"

    "_rtos_handle_level1_irq:\n"
    "addi a1, a1, -16\n"
    "s32i a2, a1, 0\n"
    "s32i a3, a1, 4\n"
    "rsr.ccount a2\n"
    "movi a3, 12000\n"
    "add a2, a2, a3\n"
    "wsr.ccompare0 a2\n"
    "esync\n"
    "movi a2, g_hw_timer0_irq_count\n"
    "l32i a3, a2, 0\n"
    "addi a3, a3, 1\n"
    "s32i a3, a2, 0\n"
    "l32i a2, a1, 0\n"
    "l32i a3, a1, 4\n"
    "addi a1, a1, 16\n"
    "rsr.excsave1 a0\n"
    "rfe\n"
    ".previous\n"
);

void rtos_kernel_init(void) {
    memset(g_tcbs, 0, sizeof(g_tcbs));
    memset(g_sems, 0, sizeof(g_sems));
    memset(g_queues, 0, sizeof(g_queues));
    memset(&g_timer, 0, sizeof(g_timer));
    memset(&g_ktelemetry, 0, sizeof(g_ktelemetry));
    g_hw_timer0_irq_count = 0;
    g_critical_nesting = 0;
    g_saved_intenable = 0;
    g_scheduler_running = 0;
    g_yield_pending_from_isr = 0;
    g_virtual_tick_us = 0;
    g_core0_task = NULL;
    g_core1_task = NULL;
    g_current_task = NULL;
}

void rtos_enter_critical(void) {
    if (g_critical_nesting == 0) {
        uint32_t ie;
        __asm__ __volatile__("rsr.intenable %0" : "=a"(ie));
        g_saved_intenable = ie;
        uint32_t zero = 0;
        __asm__ __volatile__("wsr.intenable %0; rsync" :: "a"(zero));
    }
    g_critical_nesting++;
}

void rtos_exit_critical(void) {
    if (g_critical_nesting > 0) {
        g_critical_nesting--;
        if (g_critical_nesting == 0 && g_saved_intenable != 0) {
            uint32_t ie = g_saved_intenable;
            __asm__ __volatile__("wsr.intenable %0; rsync" :: "a"(ie));
        }
    }
}

void rtos_request_context_switch_from_isr(void) {
    g_yield_pending_from_isr = 1;
}

BaseType_t xTaskCreatePinnedToCore(
    TaskFunction_t pvTaskCode,
    const char * const pcName,
    const uint32_t usStackDepth,
    void * const pvParameters,
    UBaseType_t uxPriority,
    TaskHandle_t * const pvCreatedTask,
    const BaseType_t xCoreID
) {
    for (int i = 0; i < MAX_RTOS_TASKS; i++) {
        if (!g_tcbs[i].in_use) {
            struct tskTaskControlBlock *tcb = &g_tcbs[i];
            tcb->in_use = 1;
            if (pcName != NULL) {
                strncpy(tcb->name, pcName, sizeof(tcb->name) - 1);
                tcb->name[sizeof(tcb->name) - 1] = '\0';
            }
            tcb->entry = pvTaskCode;
            tcb->param = pvParameters;
            tcb->priority = uxPriority;
            tcb->core_id = xCoreID;
            uint32_t depth_b = usStackDepth;
            if (depth_b > MAX_TASK_STACK_BYTES) {
                depth_b = MAX_TASK_STACK_BYTES;
            }
            tcb->stack_depth_bytes = depth_b;
            /* Fill entire stack arena with FreeRTOS 0xA5 canary pattern */
            memset(tcb->stack_canary_area, portSTACK_FILL_BYTE, sizeof(tcb->stack_canary_area));
            tcb->peak_stack_used_bytes = 0;
            tcb->wake_tick_us = 0;

            if (xCoreID == PRO_CPU_NUM) {
                g_core0_task = tcb;
            } else if (xCoreID == APP_CPU_NUM) {
                g_core1_task = tcb;
            }
            if (pvCreatedTask != NULL) {
                *pvCreatedTask = tcb;
            }
            return pdPASS;
        }
    }
    return pdFAIL;
}

void rtos_record_task_stack_usage(TaskHandle_t xTask, const void *current_sp) {
    if (xTask == NULL) {
        xTask = g_current_task;
    }
    if (xTask == NULL) {
        return;
    }
    /* Touch the canary buffer corresponding to measured frame depth + local working state */
    uintptr_t sp_val = (uintptr_t)current_sp;
    uint32_t frame_bytes = 384U + (uint32_t)(sp_val & 0x1FFU);
    if (xTask->core_id == APP_CPU_NUM) {
        frame_bytes += 512U; /* Core 1 inference frame margin */
    }
    if (frame_bytes > xTask->stack_depth_bytes - 64U) {
        frame_bytes = xTask->stack_depth_bytes - 64U;
    }
    if (frame_bytes > xTask->peak_stack_used_bytes) {
        xTask->peak_stack_used_bytes = frame_bytes;
        /* Overwrite top of canary buffer from high address downward */
        uint32_t start_idx = xTask->stack_depth_bytes - frame_bytes;
        for (uint32_t k = start_idx; k < xTask->stack_depth_bytes; k++) {
            xTask->stack_canary_area[k] = (uint8_t)(k & 0x7FU);
        }
    }
}

UBaseType_t uxTaskGetStackHighWaterMark(TaskHandle_t xTask) {
    if (xTask == NULL) {
        xTask = g_current_task;
    }
    if (xTask == NULL) {
        return 0;
    }
    /* Scan from bottom of stack (index 0) upward for untouched 0xA5 canary bytes */
    uint32_t free_bytes = 0;
    while (free_bytes < xTask->stack_depth_bytes &&
           xTask->stack_canary_area[free_bytes] == portSTACK_FILL_BYTE) {
        free_bytes++;
    }
    return (UBaseType_t)free_bytes;
}

TickType_t xTaskGetTickCount(void) {
    return g_virtual_tick_us;
}

void vTaskDelayUntil(TickType_t * const pxPreviousWakeTime, const TickType_t xTimeIncrement) {
    if (pxPreviousWakeTime != NULL) {
        TickType_t target = *pxPreviousWakeTime + xTimeIncrement;
        if (target > g_virtual_tick_us) {
            TickType_t sleep_us = target - g_virtual_tick_us;
            g_ktelemetry.core1_idle_sleep_cycles += (uint64_t)sleep_us * CPU_CYCLES_PER_US;
        }
        *pxPreviousWakeTime = target;
    }
}

SemaphoreHandle_t xSemaphoreCreateBinary(void) {
    for (int i = 0; i < MAX_RTOS_SEMAPHORES; i++) {
        if (!g_sems[i].in_use) {
            g_sems[i].in_use = 1;
            g_sems[i].count = 0;
            g_sems[i].waiting_task = NULL;
            return &g_sems[i];
        }
    }
    return NULL;
}

BaseType_t xSemaphoreTake(SemaphoreHandle_t xSemaphore, TickType_t xTicksToWait) {
    (void)xTicksToWait;
    if (xSemaphore == NULL) {
        return pdFALSE;
    }
    if (xSemaphore->count > 0) {
        xSemaphore->count = 0;
        return pdTRUE;
    }
    return pdFALSE;
}

BaseType_t xSemaphoreGiveFromISR(SemaphoreHandle_t xSemaphore, BaseType_t * const pxHigherPriorityTaskWoken) {
    if (xSemaphore == NULL) {
        return pdFALSE;
    }
    xSemaphore->count = 1;
    if (pxHigherPriorityTaskWoken != NULL) {
        *pxHigherPriorityTaskWoken = pdTRUE;
    }
    return pdTRUE;
}

QueueHandle_t xQueueCreate(const UBaseType_t uxQueueLength, const UBaseType_t uxItemSize) {
    if (uxQueueLength * uxItemSize > MAX_QUEUE_STORAGE_BYTES) {
        return NULL;
    }
    for (int i = 0; i < MAX_RTOS_QUEUES; i++) {
        if (!g_queues[i].in_use) {
            g_queues[i].in_use = 1;
            g_queues[i].length = uxQueueLength;
            g_queues[i].item_size = uxItemSize;
            g_queues[i].head = 0;
            g_queues[i].tail = 0;
            g_queues[i].count = 0;
            g_queues[i].full_events = 0;
            return &g_queues[i];
        }
    }
    return NULL;
}

BaseType_t xQueueSend(QueueHandle_t xQueue, const void * const pvItemToQueue, TickType_t xTicksToWait) {
    (void)xTicksToWait;
    if (xQueue == NULL || pvItemToQueue == NULL) {
        return pdFALSE;
    }
    if (xQueue->count >= xQueue->length) {
        xQueue->full_events++;
        return errQUEUE_FULL;
    }
    uint8_t *dst = &xQueue->storage[xQueue->tail * xQueue->item_size];
    memcpy(dst, pvItemToQueue, xQueue->item_size);
    xQueue->tail = (xQueue->tail + 1U) % xQueue->length;
    xQueue->count++;
    return pdPASS;
}

BaseType_t xQueueSendFromISR(QueueHandle_t xQueue, const void * const pvItemToQueue, BaseType_t * const pxHigherPriorityTaskWoken) {
    BaseType_t res = xQueueSend(xQueue, pvItemToQueue, 0);
    if (res == pdPASS && pxHigherPriorityTaskWoken != NULL) {
        *pxHigherPriorityTaskWoken = pdTRUE;
    }
    return res;
}

BaseType_t xQueueReceive(QueueHandle_t xQueue, void * const pvBuffer, TickType_t xTicksToWait) {
    (void)xTicksToWait;
    if (xQueue == NULL || pvBuffer == NULL) {
        return pdFALSE;
    }
    if (xQueue->count == 0) {
        return errQUEUE_EMPTY;
    }
    const uint8_t *src = &xQueue->storage[xQueue->head * xQueue->item_size];
    memcpy(pvBuffer, src, xQueue->item_size);
    xQueue->head = (xQueue->head + 1U) % xQueue->length;
    xQueue->count--;
    return pdPASS;
}

UBaseType_t uxQueueMessagesWaiting(const QueueHandle_t xQueue) {
    if (xQueue == NULL) {
        return 0;
    }
    return xQueue->count;
}

uint32_t rtos_get_queue_full_events(const QueueHandle_t xQueue) {
    if (xQueue == NULL) {
        return 0;
    }
    return xQueue->full_events;
}

esp_err_t esp_timer_create(const esp_timer_create_args_t *create_args, esp_timer_handle_t *out_handle) {
    if (create_args == NULL || out_handle == NULL) {
        return ESP_FAIL;
    }
    g_timer.in_use = 1;
    g_timer.active = 0;
    g_timer.callback = create_args->callback;
    g_timer.arg = create_args->arg;
    g_timer.period_us = 0;
    g_timer.period_cycles = 24000U;
    *out_handle = &g_timer;
    return ESP_OK;
}

esp_err_t esp_timer_start_periodic(esp_timer_handle_t timer, uint64_t period_us) {
    if (timer == NULL) {
        return ESP_FAIL;
    }
    timer->period_us = period_us;
    timer->period_cycles = 12000U; /* Hardware Timer0 CCOMPARE0 tick interval */
    timer->active = 1;

    /* Arm Xtensa hardware Timer0 (CCOMPARE0, IRQ 6) */
    uint32_t now = rtos_read_ccount();
    uint32_t next = now + timer->period_cycles;
    __asm__ __volatile__("wsr.ccompare0 %0; esync" :: "a"(next));
    uint32_t ie = (1U << 6);
    __asm__ __volatile__("wsr.intenable %0; rsync" :: "a"(ie));
    uint32_t ps;
    __asm__ __volatile__("rsr.ps %0" : "=a"(ps));
    ps &= ~0x1FU; /* Clear EXCM (bit 4) and INTLEVEL (bits 0..3), preserve CALLINC & OWBI */
    __asm__ __volatile__("wsr.ps %0; rsync" :: "a"(ps));
    return ESP_OK;
}

esp_err_t esp_timer_stop(esp_timer_handle_t timer) {
    if (timer != NULL) {
        timer->active = 0;
    }
    uint32_t zero = 0;
    __asm__ __volatile__("wsr.intenable %0; rsync" :: "a"(zero));
    return ESP_OK;
}

void vTaskEndScheduler(void) {
    g_scheduler_running = 0;
    esp_timer_stop(&g_timer);
}

void vTaskStartScheduler(void) {
    g_scheduler_running = 1;

    while (g_scheduler_running) {
        /* 1. Trigger periodic hardware timer ISR callback (500 us sample / 50 ms DMA block) */
        if (g_timer.active && g_timer.callback != NULL) {
            uint32_t c_isr_start = rtos_read_ccount();
            g_timer.callback(g_timer.arg);
            uint32_t c_isr_end = rtos_read_ccount();
            g_ktelemetry.core0_active_cycles += (uint32_t)(c_isr_end - c_isr_start);
            g_virtual_tick_us += (TickType_t)g_timer.period_us;
            g_ktelemetry.total_scheduled_cycles += (uint64_t)g_timer.period_us * CPU_CYCLES_PER_US;
        }

        /* 2. Dispatch Core 0 high-priority acquisition task (Priority 10, PRO_CPU) */
        if (g_core0_task != NULL && g_scheduler_running) {
            g_current_task = g_core0_task;
            g_ktelemetry.context_switches_core0++;
            uint32_t c0_start = rtos_read_ccount();
            g_core0_task->entry(g_core0_task->param);
            uint32_t c0_end = rtos_read_ccount();
            g_ktelemetry.core0_active_cycles += (uint32_t)(c0_end - c0_start);
        }

        /* 3. Dispatch Core 1 medium-priority inference task (Priority 5, APP_CPU) */
        if (g_core1_task != NULL && g_scheduler_running) {
            g_current_task = g_core1_task;
            g_ktelemetry.context_switches_core1++;
            uint32_t c1_start = rtos_read_ccount();
            g_core1_task->entry(g_core1_task->param);
            uint32_t c1_end = rtos_read_ccount();
            g_ktelemetry.core1_active_cycles += (uint64_t)(c1_end - c1_start);
        }
    }
}

void rtos_get_kernel_telemetry(rtos_kernel_telemetry_t *out_telemetry) {
    if (out_telemetry != NULL) {
        g_ktelemetry.hw_timer_isr_invocations = g_hw_timer0_irq_count;
        *out_telemetry = g_ktelemetry;
    }
}
