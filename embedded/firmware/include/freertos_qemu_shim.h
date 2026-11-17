/*
 * freertos_qemu_shim.h — Deterministic FreeRTOS / ESP-IDF Compatibility Kernel for QEMU Xtensa
 *
 * Implements the ESP-IDF / FreeRTOS SMP API in pure C99 for cycle-accurate
 * emulation on Espressif QEMU 9.2.2 (ESP32-S3 Xtensa LX7 @ 240 MHz):
 *   - xTaskCreatePinnedToCore, vTaskDelete, vTaskDelayUntil, vTaskStartScheduler
 *   - uxTaskGetStackHighWaterMark (static stack canary watermark auditing)
 *   - xSemaphoreCreateBinary, xSemaphoreTake, xSemaphoreGiveFromISR
 *   - xQueueCreate, xQueueSend, xQueueReceive, uxQueueMessagesWaiting
 *   - portYIELD_FROM_ISR, taskENTER_CRITICAL, taskEXIT_CRITICAL
 *   - esp_timer_create, esp_timer_start_periodic, esp_timer_stop (backed by
 *     Xtensa hardware Timer0 CCOMPARE0 interrupt line 6)
 */

#ifndef FREERTOS_QEMU_SHIM_H
#define FREERTOS_QEMU_SHIM_H

#include <stdint.h>
#include <stddef.h>

/* FreeRTOS primitive types */
typedef int32_t  BaseType_t;
typedef uint32_t UBaseType_t;
typedef uint32_t TickType_t;
typedef uint32_t StackType_t;
typedef void (*TaskFunction_t)(void *);

typedef struct tskTaskControlBlock *TaskHandle_t;
typedef struct QueueDefinition     *QueueHandle_t;
typedef struct SemaphoreDefinition *SemaphoreHandle_t;

typedef uint32_t portMUX_TYPE;

#define pdFALSE         ((BaseType_t)0)
#define pdTRUE          ((BaseType_t)1)
#define pdPASS          (pdTRUE)
#define pdFAIL          (pdFALSE)
#define errQUEUE_FULL   ((BaseType_t)0)
#define errQUEUE_EMPTY  ((BaseType_t)0)

#define portMAX_DELAY   ((TickType_t)0xFFFFFFFFUL)
#define portSTACK_FILL_BYTE (0xA5U)

/* CPU clock and tick configuration (240 MHz Xtensa LX7, 1 tick = 1 us resolution) */
#define CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ (240U)
#define CPU_CYCLES_PER_US               (240U)
#define pdMS_TO_TICKS(ms)               ((TickType_t)((ms) * 1000U))

#define PRO_CPU_NUM (0)
#define APP_CPU_NUM (1)

#define portMUX_INITIALIZER_UNLOCKED (0U)

/* Critical section macros */
#define taskENTER_CRITICAL(mux) do { (void)(mux); rtos_enter_critical(); } while (0)
#define taskEXIT_CRITICAL(mux)  do { (void)(mux); rtos_exit_critical(); } while (0)

#define portYIELD_FROM_ISR(higher_prio_woken) \
    do { \
        if ((higher_prio_woken) != pdFALSE) { \
            rtos_request_context_switch_from_isr(); \
        } \
    } while (0)

/* ESP-IDF esp_timer compatibility API */
typedef int32_t esp_err_t;
#define ESP_OK   (0)
#define ESP_FAIL (-1)

typedef void (*esp_timer_cb_t)(void *arg);

typedef enum {
    ESP_TIMER_TASK = 0,
    ESP_TIMER_ISR  = 1
} esp_timer_dispatch_t;

typedef struct {
    esp_timer_cb_t       callback;
    void                *arg;
    esp_timer_dispatch_t dispatch_method;
    const char          *name;
    int                  skip_unhandled_events;
} esp_timer_create_args_t;

typedef struct esp_timer_impl *esp_timer_handle_t;

/* Read hardware cycle counter (CCOUNT) on Xtensa LX6 / LX7 */
static inline uint32_t rtos_read_ccount(void) {
    uint32_t ccount;
    __asm__ __volatile__("rsr.ccount %0" : "=a"(ccount));
    return ccount;
}

/* Kernel lifecycle and critical section functions */
void rtos_kernel_init(void);
void rtos_enter_critical(void);
void rtos_exit_critical(void);
void rtos_request_context_switch_from_isr(void);

/* Task Management API */
BaseType_t xTaskCreatePinnedToCore(
    TaskFunction_t pvTaskCode,
    const char * const pcName,
    const uint32_t usStackDepth,
    void * const pvParameters,
    UBaseType_t uxPriority,
    TaskHandle_t * const pvCreatedTask,
    const BaseType_t xCoreID
);

UBaseType_t uxTaskGetStackHighWaterMark(TaskHandle_t xTask);
TickType_t xTaskGetTickCount(void);
void vTaskDelayUntil(TickType_t * const pxPreviousWakeTime, const TickType_t xTimeIncrement);
void vTaskStartScheduler(void);
void vTaskEndScheduler(void);

/* Binary Semaphore API */
SemaphoreHandle_t xSemaphoreCreateBinary(void);
BaseType_t xSemaphoreTake(SemaphoreHandle_t xSemaphore, TickType_t xTicksToWait);
BaseType_t xSemaphoreGiveFromISR(SemaphoreHandle_t xSemaphore, BaseType_t * const pxHigherPriorityTaskWoken);

/* Queue API */
QueueHandle_t xQueueCreate(const UBaseType_t uxQueueLength, const UBaseType_t uxItemSize);
BaseType_t xQueueSend(QueueHandle_t xQueue, const void * const pvItemToQueue, TickType_t xTicksToWait);
BaseType_t xQueueSendFromISR(QueueHandle_t xQueue, const void * const pvItemToQueue, BaseType_t * const pxHigherPriorityTaskWoken);
BaseType_t xQueueReceive(QueueHandle_t xQueue, void * const pvBuffer, TickType_t xTicksToWait);
UBaseType_t uxQueueMessagesWaiting(const QueueHandle_t xQueue);
uint32_t rtos_get_queue_full_events(const QueueHandle_t xQueue);

/* ESP-IDF Hardware Periodic Timer API (backed by Xtensa CCOMPARE0 IRQ 6) */
esp_err_t esp_timer_create(const esp_timer_create_args_t *create_args, esp_timer_handle_t *out_handle);
esp_err_t esp_timer_start_periodic(esp_timer_handle_t timer, uint64_t period_us);
esp_err_t esp_timer_stop(esp_timer_handle_t timer);

/* Kernel runtime telemetry */
typedef struct {
    uint32_t hw_timer_isr_invocations;
    uint32_t context_switches_core0;
    uint32_t context_switches_core1;
    uint64_t core0_active_cycles;
    uint64_t core1_active_cycles;
    uint64_t core1_idle_sleep_cycles;
    uint64_t total_scheduled_cycles;
} rtos_kernel_telemetry_t;

void rtos_get_kernel_telemetry(rtos_kernel_telemetry_t *out_telemetry);
void rtos_record_task_stack_usage(TaskHandle_t xTask, const void *current_sp);

#endif /* FREERTOS_QEMU_SHIM_H */
