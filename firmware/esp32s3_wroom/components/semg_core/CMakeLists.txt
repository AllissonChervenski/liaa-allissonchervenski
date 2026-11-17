idf_component_register(
    SRCS
        "../../../esp32_qemu/src/semg_pipeline.c"
        "../../../../embedded/firmware/src/cnn1d_engine.c"
        "../../../../embedded/firmware/src/dtw_hybrid_engine.c"
    INCLUDE_DIRS
        "include"
        "../../../esp32_qemu/include"
        "../../../../embedded/firmware/include"
)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -Wall -Wextra -Werror)
