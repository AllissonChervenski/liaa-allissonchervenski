# Relatório Técnico Consolidado: Configuração do Emulador Espressif QEMU, Classificador Micro-LDA (Q7) e Rede Neural Convolucional CNN-1D (INT8)

> **Escopo do Documento:** Consolidação completa da arquitetura de emulação de hardware **Espressif QEMU 9.2.2 (`esp32` Xtensa LX6 e `esp32s3` Xtensa LX7 @ 240 MHz)**, da formulação e implementação do pipeline **DSP Causal + Extrator Zero-FPU + Micro-LDA Quantizado (Q7)** e do motor bare-metal/FreeRTOS da **CNN-1D Quantizada Simétrica (INT8)**, cobrindo os três regimes validados experimentalmente no projeto:
> 1. **Fase 1 — Emulação Bare-Metal Multi-Precisão e Multi-Canal (`110 cenários`, `1.201.280 inferências`)**: [`esp32_qemu_quantized_pipeline_report.md`](file:///home/allisson/Projects/agents_emg/reports/esp32_qemu/esp32_qemu_quantized_pipeline_report.md)
> 2. **Fase 2 — Benchmark Bare-Metal Pareado Micro-LDA (Q7) vs. CNN-1D (INT8) (`CCOUNT` isolado)**: [`micro_lda_vs_cnn1d_int8_comparison_report.md`](file:///home/allisson/Projects/agents_emg/reports/cnn1d_qemu/micro_lda_vs_cnn1d_int8_comparison_report.md)
> 3. **Fase 3 — Arquitetura Dual-Core sob FreeRTOS com DMA Ping-Pong e Timer ISR (`4.000 janelas` contínuas a `2.000 Hz`)**: [`qemu_freertos_dualcore_comparison_report.md`](file:///home/allisson/Projects/agents_emg/reports/freertos_qemu/qemu_freertos_dualcore_comparison_report.md)

---

## 1. Configuração do Emulador Espressif QEMU e Toolchain Xtensa

### 1.1. Especificações do Ambiente de Emulação e Compilação Cruzada

| Componente de Infraestrutura | Versão / Especificação Técnica | Caminho / Comando no Projeto |
| :--- | :--- | :--- |
| **Emulador Oficial Espressif** | `QEMU emulator version 9.2.2 (esp_develop_9.2.2_20261109)` | `~/.espressif/tools/qemu-xtensa/qemu/bin/qemu-system-xtensa` |
| **Máquina Alvo Primária (`ESP32-S3`)** | `-machine esp32s3` (Xtensa Dual-Core 32-bit **LX7** @ `240 MHz`, `512 KB` SRAM interna, Cache L1 de Dados `32 KB`) | `-nographic -machine esp32s3 -semihosting -kernel <firmware.elf>` |
| **Máquina Alvo Secundária (`ESP32`)** | `-machine esp32` (Xtensa Dual-Core 32-bit **LX6** @ `240 MHz`, `520 KB` SRAM interna) | `-nographic -machine esp32 -semihosting -kernel <firmware.elf>` |
| **Toolchain C99 Cross-Compiler** | `xtensa-esp-elf-gcc 14.4.0 (crosstool-NG esp-14.4.0_20261030)` | `xtensa-esp32s3-elf-gcc` / `xtensa-esp32-elf-gcc` |
| **Flags Estritas de Compilação** | `-std=c99 -O3 -Wall -Wextra -Werror` (**0 warnings** em todos os binários) | `-specs=board.elf.specs -specs=sys.qemu.specs -Wl,--defsym=entire_dram_seg=1 -lm` |
| **Interface Host ↔ MCU (Semihosting)** | Instrução privilegiada `simcall` do Xtensa (`SYS_open`, `SYS_read`, `SYS_write`, `SYS_exit`) via `libsys_qemu.a` | Leitura determinística de bundles binários (`.bin`) e emissão de telemetria JSON pela UART0 virtual |
| **Instrumentação Ciclo-Exata (`CCOUNT`)** | Registrador especial Xtensa `CCOUNT` (`rsr %0, ccount`) incrementado a cada ciclo de clock (`1 ciclo = 4,167 ns @ 240 MHz`) | [`read_ccount()`](file:///home/allisson/Projects/agents_emg/firmware/esp32_qemu/include/semg_pipeline.h#L94-L98) e [`rtos_read_ccount()`](file:///home/allisson/Projects/agents_emg/embedded/firmware/include/freertos_qemu_shim.h#L88-L92) |
| **Vetor de Interrupção de Timer (`IRQ 6`)** | Comparador de ciclo `CCOMPARE0` (`wsr.ccompare0`) acoplado à linha de interrupção de nível 1 (`IRQ 6`, `EXCCAUSE = 4`) em `0x40378340` (`_UserExceptionVector`) | [`freertos_qemu_shim.c`](file:///home/allisson/Projects/agents_emg/embedded/firmware/src/freertos_qemu_shim.c#L75-L124) (`-Wl,--allow-multiple-definition`) |

### 1.2. Diagrama da Arquitetura de Emulação (Bare-Metal e Dual-Core FreeRTOS)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ HOST LINUX (Python 3.13 + PyTorch + NumPy + SciPy)                                       │
│  • Datasets sEMG: Ozdemir S1 (4CH, 10 classes, 2 kHz) & NinaPro DB2 S4 (12CH, 50 classes)│
│  • Treinamento LOTO + Quantização estática (Micro-LDA Q7 e CNN-1D INT8 Simétrico)        │
│  • Exportação de headers C99 (.h) e streams binários de validação (.bin)                 │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │ Xtensa Semihosting (simcall)
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ ESPRESSIF QEMU 9.2.2 (qemu-system-xtensa -machine esp32s3 / esp32 @ 240 MHz)             │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ MODO 1: BARE-METAL ISOLADO (semg_esp32s3.elf / cnn1d_esp32s3.elf)                  │  │
│  │  • Medição pura de ciclos de clock via rsr.ccount sem interferência de SO          │  │
│  │  • Validação de 110 cenários de quantização + perfil por camada da CNN-1D          │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ MODO 2: DUAL-CORE FREERTOS + DMA PING-PONG (freertos_dualcore_esp32s3.elf)         │  │
│  │  • Timer0 ISR (CCOMPARE0 IRQ 6 @ 2.000 Hz): Preenche dma_ping_buffer / pong_buffer │  │
│  │  • Core 0 (PRO_CPU, Prio 10 — vTaskAquisicaoCore0): Filtro Causal Butterworth      │  │
│  │    SOS DF2T (20–450 Hz) + Janela Deslizante (W=400, S=100) -> xQueueJanelaPronta   │  │
│  │  • Core 1 (APP_CPU, Prio 5 — vTaskInferenciaCore1): Desempacota descritor, executa │  │
│  │    Micro-LDA (Q7) ou CNN-1D (INT8), aplica Voto Majoritário (K=3) e entra em       │  │
│  │    Light Sleep via vTaskDelayUntil (>99,5% de CPU livre no Core 1)                 │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3. Pegada de Memória dos Três Binários ELF Compilados (`xtensa-esp32s3-elf-size`)

| Binário ELF Compilado | Modo de Operação | `.text` (Flash Code + `.rodata`) | `.data` (DRAM Init) | `.bss` (SRAM Estática) | Flash Total | SRAM Total |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [`semg_esp32s3.elf`](file:///home/allisson/Projects/agents_emg/firmware/esp32_qemu/build/semg_esp32s3.elf) | Bare-Metal DSP + Micro-LDA | `59.935 B` (`58,53 KB`) | `288 B` | `211.824 B` (`206,86 KB`) | `60.223 B` | `212.112 B` |
| [`cnn1d_esp32s3.elf`](file:///home/allisson/Projects/agents_emg/embedded/firmware/build/cnn1d_esp32s3.elf) | Bare-Metal CNN-1D INT8 | `68.751 B` (`67,14 KB`) | `288 B` | `134.816 B` (`131,66 KB`) | `69.039 B` | `135.104 B` |
| [`freertos_dualcore_esp32s3.elf`](file:///home/allisson/Projects/agents_emg/embedded/firmware/build/freertos_dualcore_esp32s3.elf) | Dual-Core FreeRTOS Unificado (LDA + CNN) | `75.251 B` (`73,49 KB`) | `300 B` | `288.620 B` (`281,86 KB`) | `75.551 B` | `288.920 B` |

---

## 2. Configuração e Arquitetura do Pipeline Micro-LDA (Q7) e Extração Zero-FPU

### 2.1. Cadeia de Condicionamento Causal e Extração de Características (`semg_pipeline.c`)

Implementada em [`semg_pipeline.h`](file:///home/allisson/Projects/agents_emg/firmware/esp32_qemu/include/semg_pipeline.h) e [`semg_pipeline.c`](file:///home/allisson/Projects/agents_emg/firmware/esp32_qemu/src/semg_pipeline.c), a cadeia clássica é composta por quatro estágios determinísticos:

1. **Filtro Passa-Faixa Causal Butterworth (`CausalSosFilter`)**:
   - Ordem 4 ($20\text{–}450\text{ Hz}$ a $F_s = 2.000\text{ Hz}$), decomposto em $M = 4$ seções biquadráticas de segunda ordem na **Forma Direta II Transposta (DF2T)** ([`sos_filter_process_chunk`](file:///home/allisson/Projects/agents_emg/firmware/esp32_qemu/src/semg_pipeline.c#L26-L54)).
   - Estado persistente entre blocos de DMA (`float state[4][2][C]`), ocupando apenas **`128 bytes`** (4 canais) ou **`384 bytes`** (12 canais), com erro residual máximo frente ao `scipy.signal.sosfilt` de **$3,51 \times 10^{-7}\text{ mV}$**.
2. **Acumulador Deslizante Circular (`StatefulWindowBuffer`)**:
   - Janela de análise $W = 400$ amostras ($200\text{ ms}$) com avanço (stride) $S = 100$ amostras ($50\text{ ms}$, frequência de decisão de $20\text{ Hz}$) ([`window_buffer_push_sample`](file:///home/allisson/Projects/agents_emg/firmware/esp32_qemu/src/semg_pipeline.c#L66-L90)).
3. **Extrator Zero-FPU de Características de Hudgins (`extract_td_features_int16`)**:
   - Extrai 5 descritores por canal ($F = 5 \times C$: **MAV**, **RMS**, **WL**, **ZC**, **SSC**) diretamente sobre as amostras quantizadas `int16_t` / `int8_t` ([`extract_td_features_int16`](file:///home/allisson/Projects/agents_emg/firmware/esp32_qemu/src/semg_pipeline.c#L155-L216)).
   - Elimina completamente instruções de ponto flutuante dentro do laço de $N = 400$ amostras usando acumuladores inteiros de 32/64 bits (`int64_t sum_sq`, `int64_t sum_abs`, `int64_t wl_int`) e comparação inteira de zona morta (`epsilon_q`), aplicando a conversão de escala apenas uma vez ao final da janela.
   - **Ganho medido no QEMU (`ESP32-S3`):** Reduz o tempo de extração de `3.809 ciclos (15,87 µs)` para **`1.413 ciclos (5,89 µs)`** em 4 canais (**`2,70×` mais rápido**) e de `12.152 ciclos (50,63 µs)` para **`3.120 ciclos (13,00 µs)`** em 12 canais `int8` (**`3,89×` mais rápido**).
4. **Classificador Linear Discriminante Canônico Quantizado em Q7 (`micro_lda_predict_q7`)**:
   - Treinado em protocolo *Leave-One-Trial-Out (LOTO)* estritamente livre de vazamento (*train-only Z-score scaler* $\mu_f, \sigma_f^{-1}$ com máscara de variância zero).
   - Os pesos discriminantes $W_{k,f}$ são quantizados simetricamente por classe em `int8_t` no formato **Q7** ($W_{k,f}^{(Q7)} = \text{clip}(\lfloor W_{k,f} / s_k \rceil, -127, 127)$ com fator de escala $s_k = \max_f |W_{k,f}| / 127$) ([`micro_lda_predict_q7`](file:///home/allisson/Projects/agents_emg/firmware/esp32_qemu/src/semg_pipeline.c#L244-L276)):
     $$\hat{y} = \arg\max_{k \in \{1..K\}} \left( b_k + s_k \sum_{f=1}^{F} W_{k,f}^{(Q7)} \cdot z_f \right)$$
   - **Pegada Ultra-Compacta em Cache L1:** Apenas **`200 bytes`** ($10 \times 20 \times 1\text{ B}$) para Ozdemir 4CH e **`3.000 bytes`** ($50 \times 60 \times 1\text{ B}$) para NinaPro DB2 12CH — redução de **75%** frente ao `float32`, residindo 100% no Cache L1 de Dados de `32 KB` do `ESP32-S3`.

### 2.2. Desempenho do Micro-LDA (Q7) por Resolução de Quantização e Número de Canais

| Dataset (Sujeito / Protocolo LOTO) | Precisão ADC | Canais ($C$) | Features ($F$) | Pesos Q7 (ROM) | Acurácia Top-1 (`FP32` / `Q7`) | Macro-F1 (`Q7`) | Paridade Host ↔ QEMU | Ciclos `CCOUNT` Classificador Q7 | Latência Classificador Q7 (`240 MHz`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ozdemir S1** (`10 classes`, `2.340 janelas`) | `float32` | 4 | 20 | `200 B` | `82,31%` / **`82,22%`** | `82,64%` | `100,00%` | `305 cyc` (`224` no LX6) | **`1,27 µs`** (`0,93 µs` no LX6) |
| **Ozdemir S1** (`10 classes`, `2.340 janelas`) | `int16` | 4 | 20 | `200 B` | `81,97%` / **`82,05%`** | `82,31%` | `100,00%` | `305 cyc` (`220` no LX6) | **`1,27 µs`** (`0,92 µs` no LX6) |
| **Ozdemir S1** (`10 classes`, `2.340 janelas`) | `int16` | 3 | 15 | `150 B` | `80,04%` / **`80,09%`** | `79,66%` | `100,00%` | `242 cyc` | **`1,01 µs`** |
| **Ozdemir S1** (`10 classes`, `2.340 janelas`) | `int16` | 2 | 10 | `100 B` | `64,19%` / **`64,06%`** | `62,90%` | `100,00%` | `180 cyc` | **`0,75 µs`** |
| **Ozdemir S1** (`10 classes`, `2.340 janelas`) | `int8` | 4 | 20 | `200 B` | `76,50%` / **`76,07%`** | `76,49%` | `100,00%` | `309 cyc` | **`1,29 µs`** |
| **NinaPro DB2 S4** (`50 classes`, `15.824 janelas`) | `float32` | 12 | 60 | `3.000 B` | `48,20%` / **`48,24%`** | `48,80%` | `100,00%` | `1.275 cyc` | **`5,31 µs`** |
| **NinaPro DB2 S4** (`50 classes`, `15.824 janelas`) | `int16` | 12 | 60 | `3.000 B` | `48,21%` / **`48,17%`** | `48,72%` | `100,00%` | `1.273 cyc` | **`5,30 µs`** |
| **NinaPro DB2 S4** (`50 classes`, `15.824 janelas`) | `int16` | 8 | 40 | `2.000 B` | `42,08%` / **`42,10%`** | `41,79%` | `100,00%` | `882 cyc` | **`3,68 µs`** |
| **NinaPro DB2 S4** (`50 classes`, `15.824 janelas`) | `int8` | 12 | 60 | `3.000 B` | `46,35%` / **`46,36%`** | `46,98%` | `100,00%` | `1.274 cyc` | **`5,31 µs`** |

---

## 3. Configuração e Arquitetura da Rede Neural Convolucional CNN-1D (INT8)

### 3.1. Topologia da Rede e Quantização Simétrica em Ponto Fixo (`Q24`)

Implementada em C99 puro sem dependências externas ([`cnn1d_inference.h`](file:///home/allisson/Projects/agents_emg/embedded/firmware/include/cnn1d_inference.h), [`cnn1d_engine.c`](file:///home/allisson/Projects/agents_emg/embedded/firmware/src/cnn1d_engine.c) e [`cnn1d_weights_int8.h`](file:///home/allisson/Projects/agents_emg/embedded/firmware/include/cnn1d_weights_int8.h)) a partir do treinamento e exportação em [`train_cnn1d_quantized.py`](file:///home/allisson/Projects/agents_emg/src/models/train_cnn1d_quantized.py) e [`export_cnn1d_c_header.py`](file:///home/allisson/Projects/agents_emg/src/export/export_cnn1d_c_header.py):

- **Esquema de Quantização Simétrica (`INT8`, Zero-Point $= 0$):**
  - Pesos e ativações em `int8_t` ($\in [-127, 127]$), vieses (*biases*) em `int32_t`.
  - Produto escalar e convoluções executados com multiplicação `8x8 -> 32 bits` (`int32_t acc`) e requantização em ponto fixo **Q24** (`mult_q24 = round(eff_scale * 2^24)`), usando arredondamento simétrico *half-away-from-zero* (`(prod + (1LL << 23)) >> 24`) idêntico entre Python e C99.
- **Arena Estática Ping-Pong em SRAM (`4.720 bytes = 4,61 KB`):**
  - Duas regiões estáticas alinhadas em 16 bytes (`g_arena_buf_a[3152]` e `g_arena_buf_b[1568]`) recicladas camada a camada, com **zero chamadas a `malloc`/`free`**.

### 3.2. Detalhamento Camada a Camada e Decomposição Ciclo-Exata (`CCOUNT` @ 240 MHz)

| Estágio / Camada (`cnn1d_engine.c`) | Configuração do Operador | Tensor de Saída | Arena Usada | Ozdemir 4CH (`10 cls`) Ciclos / Tempo | NinaPro 12CH (`50 cls`) Ciclos / Tempo |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. `conv1d_relu_int8` (L1)** | `16` filtros, `kernel = 7`, `stride = 2`, `pad = 0`, `ReLU` + `Q24` | `[16, 197]` (`int8_t`) | `3.152 B` (`Buf A`) | `16.429 cyc` (`68,45 µs` — `55,5%`) | `36.878 cyc` (`153,66 µs` — `71,7%`) |
| **2. `maxpool1d_int8` (Pool1)** | `MaxPool1D`, `pool_size = 2`, `stride = 2` | `[16, 98]` (`int8_t`) | `1.568 B` (`Buf B`) | `309 cyc` (`1,29 µs` — `1,0%`) | `308 cyc` (`1,28 µs` — `0,6%`) |
| **3. `conv1d_relu_int8` (L2)** | `24` filtros, `kernel = 5`, `stride = 2`, `pad = 0`, `ReLU` + `Q24` | `[24, 47]` (`int8_t`) | `1.128 B` (`Buf A`) | `12.459 cyc` (`51,91 µs` — `42,1%`) | `12.593 cyc` (`52,47 µs` — `24,5%`) |
| **4. `gap1d_int8` (GAP)** | Global Average Pooling sobre $T = 47$ com arredondamento inteiro | `[24]` (`int8_t`) | `24 B` (`Buf B`) | `178 cyc` (`0,74 µs` — `0,6%`) | `180 cyc` (`0,75 µs` — `0,4%`) |
| **5. `dense_argmax_int8` (FC)** | Fusão `GAP [24]` + `Aux TD [5*C]` $\rightarrow$ Projeção `K` classes + `Argmax` | `[10]` ou `[50]` (`int32_t`) |Stack local | `144 cyc` (`0,60 µs` — `0,5%`) | `1.392 cyc` (`5,80 µs` — `2,7%`) |
| **TOTAL (`cnn1d_forward_int8`)** | **2.858 params (`3.066 B` ROM) / 7.554 params (`7.922 B` ROM)** | **Classe Predita** | **`4.720 B` (`4,61 KB`)** | **`29.588 cyc` (`123,28 µs`)** | **`51.424 cyc` (`214,27 µs`)** |

---

## 4. Síntese Comparativa: Micro-LDA (Q7) vs. CNN-1D (INT8) (Bare-Metal e Dual-Core FreeRTOS)

### 4.1. Tabela Comparativa Global no `ESP32-S3` (`Xtensa LX7 @ 240 MHz`)

| Dimensão Avaliada | Ozdemir 4CH (`10 Classes`): Micro-LDA (Q7) | Ozdemir 4CH (`10 Classes`): CNN-1D (INT8) | NinaPro DB2 12CH (`50 Classes`): Micro-LDA (Q7) | NinaPro DB2 12CH (`50 Classes`): CNN-1D (INT8) |
| :--- | :---: | :---: | :---: | :---: |
| **Acurácia Top-1 LOTO (%)** | **`82,05%`** | `80,90%` (`-1,15 pp`) | `48,17%` | **`59,28%` (`+11,11 pp`)** |
| **Macro-F1 LOTO (%)** | **`82,31%`** | `80,49%` (`-1,82 pp`) | `48,72%` | **`53,08%` (`+4,36 pp`)** |
| **Paridade Numérica Host ↔ QEMU** | **`100,00%`** (`2.340/2.340`) | **`100,00%`** (`2.340/2.340`, `max_diff=0`) | **`100,00%`** (`15.824/15.824`) | **`100,00%`** (`15.824/15.824`, `max_diff=0`) |
| **Parâmetros Totais** | **`210`** | `2.858` (`13,6×`) | **`3.050`** | `7.554` (`2,48×`) |
| **Flash ROM de Pesos (`.rodata`)** | **`200 B` (`0,20 KB`)** | `3.066 B` (`2,99 KB`) | **`3.000 B` (`2,93 KB`)** | `7.922 B` (`7,74 KB`) |
| **Arena de Ativações em SRAM** | **`0 B`** | `4.720 B` (`4,61 KB`) | **`0 B`** | `4.720 B` (`4,61 KB`) |
| **Latência Bare-Metal (Classificador Puro)** | **`1,27 µs`** (`305 cyc`) | `123,28 µs` (`29.588 cyc`) | **`5,30 µs`** (`1.273 cyc`) | `214,27 µs` (`51.424 cyc`) |
| **Latência Bare-Metal (Pipeline Completo)** | **`31,30 µs`** (`7.512 cyc`) | `153,31 µs` (`36.794 cyc`) | **`120,00 µs`** (`28.800 cyc`) | `328,97 µs` (`78.953 cyc`) |
| **Latência FreeRTOS Core 0 (Filtro + Janela)** | `23,75 µs` (`5.700 cyc`) | `30,84 µs` (`7.403 cyc`) | `72,37 µs` (`17.369 cyc`) | `76,24 µs` (`18.298 cyc`) |
| **Latência FreeRTOS Core 1 (Tarefa Completa)** | **`8,58 µs`** (`p99 = 18,50 µs`) | `157,34 µs` (`p99 = 277,72 µs`) | **`28,34 µs`** (`p99 = 47,07 µs`) | `239,89 µs` (`p99 = 442,74 µs`) |
| **Fração Livre do Core 1 (`Light Sleep`)** | **`99,9828%`** | **`99,6853%`** | **`99,9433%`** | **`99,5202%`** |
| **Stack High-Water Mark (`Core 0 / Core 1`)** | `3.580 B` / `5.040 B` | `3.580 B` / `5.040 B` | `3.580 B` / `5.040 B` | `3.580 B` / `5.040 B` |
| **Perdas de Deadline / Jitter (`1.000 janelas`)** | `0 misses` / `0 µs jitter` | `0 misses` / `0 µs jitter` | `0 misses` / `0 µs jitter` | `0 misses` / `0 µs jitter` |
| **Energia por Janela no Core 1 (`@ 148,5 mW`)** | **`1,274 µJ`** | `23,365 µJ` | **`4,208 µJ`** | `35,624 µJ` |
| **Índice de Responsividade ($\text{IRC}_{\text{Core1}}$)** | **`95.629,37`** | `5.141,73` (`18,60×` menor) | **`84.985,89`** | `12.355,66` (`6,88×` menor) |
| **Índice de Eficiência Embarcada ($\text{IEE}_{\text{Core1}}$)** | **`585.229,07`** | `28.018,28` (`20,89×` menor) | **`394.618,77`** | `54.883,10` (`7,19×` menor) |

---

## 5. Diretrizes de Seleção Arquitetural (Fronteira de Pareto)

1. **Cenário de Baixa Contagem de Canais e Vocabulário Compacto (`Ozdemir 4 canais, 10 gestos`):**
   - O **Micro-LDA (Q7)** com extração **Zero-FPU (`int16`)** é a escolha ótima: alcança **`82,05%` de acurácia Top-1 (`82,31%` Macro-F1)** superando a CNN-1D (`80,90%`), ocupa apenas **`200 bytes` de ROM** (`15,3×` menor que a CNN-1D), exige **`0 bytes` de arena extra em SRAM**, executa a tarefa completa de inferência no Core 1 em **`8,58 µs` (`1,27 µJ`)** e atinge um **$\text{IEE}$ `20,89×` superior**.
2. **Cenário de Alta Densidade de Canais e Grande Vocabulário (`NinaPro DB2 12 canais, 50 gestos`):**
   - A **CNN-1D (INT8)** justifica seu custo aritmético adicional ao capturar correlações espaço-temporais não-lineares entre os 12 eletrodos, elevando a acurácia Top-1 de `48,17%` para **`59,28%` (`+11,11 pp`)** e o Macro-F1 de `48,72%` para **`53,08%` (`+4,36 pp`)**.
   - Mesmo consumindo `57.573 ciclos` (`239,89 µs`) no Core 1 e `4,61 KB` de arena estática, a **CNN-1D (INT8)** mantém **`99,52%` do Core 1 livre** para `Light Sleep` a cada janela de `50 ms`, operando com folga de **mais de `60×`** abaixo do limiar de percepção neuromotora (`15 ms`).
3. **Como Reproduzir Todas as Emulações e Relatórios:**
   - **Suite Bare-Metal CNN-1D vs. Micro-LDA:** `make test_qemu_cnn` ([`run_qemu_cnn1d_benchmark.py`](file:///home/allisson/Projects/agents_emg/scripts/run_qemu_cnn1d_benchmark.py))
   - **Suite Dual-Core FreeRTOS (4.000 janelas em tempo real):** `make test_qemu_freertos` ([`run_qemu_freertos_dualcore_benchmark.py`](file:///home/allisson/Projects/agents_emg/scripts/run_qemu_freertos_dualcore_benchmark.py))
