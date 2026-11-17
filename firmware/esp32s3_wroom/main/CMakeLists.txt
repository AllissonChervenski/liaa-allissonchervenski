idf_component_register(
    SRCS
        "src/main.c"
        "src/semg_hil_protocol.c"
        "src/semg_hal_acq.c"
        "src/semg_rtos_app.c"
    INCLUDE_DIRS
        "include"
    REQUIRES
        semg_core
        esp_adc
        driver
        esp_timer
        freertos
)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -Wall -Wextra -Werror)
