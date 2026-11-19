# Relatório Consolidado de Latências nos Benchmarks: ESP32-S3 Físico & Emulador QEMU

> **Escopo:** Consolidação de todas as medições de latência obtidas nos benchmarks do projeto — cobrindo **silício real (`ESP32-S3-WROOM-1` @ 240 MHz)**, **emulador ciclo-exato (`Espressif QEMU 9.2.2` em `esp32` LX6 e `esp32s3` LX7)** e **latência algorítmica fim-a-fim** (janelamento + suavização por voto majoritário para execução de gesto em prótese mioelétrica).
>
> **Conclusão Principal:** **Nenhum modelo** ultrapassou a barreira clínica de **300 ms** (nem mesmo a de **200 ms**) em inferência pura. O maior tempo de inferência medido em todo o benchmark físico foi de **49,96 ms** (`49.957,65 µs` de média / `49.972,83 µs` de p99 no `M03_dtw_classic_on2` em 12 canais / 50 classes). Em operação contínua com **Vote-3**, a latência total para classificar e estabilizar um gesto fica em **~100,3 ms** (Micro-LDA Q7) a **~106,6 ms** (CNN-1D INT8) no Ozdemir 4CH.

---

## 1. Resumo Executivo (Hardware Físico vs. Emulador vs. Fim-a-Fim)

| Regime / Métrica | Micro-LDA Q7 (`M01`) | CNN-1D INT8 (`M02`) | Pior Caso Geral (`M03` DTW O(N²)) | Limite Clínico |
| :--- | :---: | :---: | :---: | :---: |
| **Silício Real ESP32-S3 (2CH Khushaba)** | **178,93 µs** (0,18 ms) | **5.901,15 µs** (5,90 ms) | 3.955,65 µs (3,96 ms) | ≤ 200–300 ms |
| **Silício Real ESP32-S3 (4CH Ozdemir, 40 suj.)** | **341,05 µs** (0,34 ms) | **7.549,96 µs** (7,55 ms) | 5.374,39 µs (5,37 ms) | ≤ 200–300 ms |
| **Silício Real ESP32-S3 (12CH NinaPro DB2, 40 suj.)** | **1.051,83 µs** (1,05 ms) | **14.148,59 µs** (14,15 ms) | **49.957,65 µs** (49,96 ms) | ≤ 200–300 ms |
| **QEMU Bare-Metal `esp32` LX6 (Pipeline `int16`)** | **32,4 µs** (4CH) / **92,0 µs** (12CH) | — | — | — |
| **QEMU Bare-Metal `esp32s3` LX7 (Pipeline `int16`)** | **31,3 µs** (4CH) / **120,0 µs** (12CH) | — | — | — |
| **QEMU Dual-Core FreeRTOS (Tarefa Core 1)** | **8,58 µs** (4CH) / **28,34 µs** (12CH) | **157,34 µs** (4CH) / **239,89 µs** (12CH) | — | — |
| **Fim-a-Fim com Vote-3 (+100 ms algorítmico, 4CH)** | **~100,3 ms** | **~106,6 ms** | ~105,4 ms (4CH) | ≤ 200–300 ms |

---

## 2. Silício Real (`ESP32-S3-WROOM-1` @ 240 MHz): 10 Modelos × 4 Datasets (101 Sujeitos)

Medições obtidas diretamente no **Core 1** do hardware físico via protocolo `SEW1` + `SEMG_HIL_CMD_PUSH_WINDOW` sobre `/dev/ttyACM0` (100% de paridade bit-exata com o Host; tempo total de execução da bateria: **6.967,3 s**).  
Fonte: `reports/benchmarks/esp32s3_full_population_10models_report.md`

| ID | Modelo | Khushaba (2CH, 10 cls, N=10) | Ozdemir (4CH, 10 cls, N=40) | NinaPro DB3 (12CH, 50 cls, N=11) | NinaPro DB2 (12CH, 50 cls, N=40) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `M01` | **M01_micro_lda_q7** | **178,93 µs** | **341,05 µs** | **1.049,62 µs** | **1.051,83 µs** |
| `M02` | **M02_cnn1d_int8** | **5.901,15 µs** | **7.549,96 µs** | **14.133,79 µs** | **14.148,59 µs** |
| `M03` | **M03_dtw_classic_on2** | **3.955,65 µs** | **5.374,39 µs** | **48.364,25 µs** | **49.957,65 µs** |
| `M04` | **M04_dtw_sakoe_chiba** | **1.322,73 µs** | **1.812,08 µs** | **12.684,94 µs** | **13.071,19 µs** |
| `M05` | **M05_dtw_lb_keogh** | **491,28 µs** | **788,72 µs** | **5.973,47 µs** | **4.758,16 µs** |
| `M06` | **M06_hybrid_lda_cnn_head** | **5.916,77 µs** | **7.562,99 µs** | **14.120,91 µs** | **14.137,42 µs** |
| `M07` | **M07_hybrid_lda_cnn_cascade** | **3.898,32 µs** | **3.313,55 µs** | **11.572,05 µs** | **12.282,12 µs** |
| `M08` | **M08_hybrid_dtw_lda** | **839,67 µs** | **1.197,41 µs** | **2.646,44 µs** | **2.651,23 µs** |
| `M09` | **M09_hybrid_dtw_cnn** | **6.562,05 µs** | **8.399,12 µs** | **15.720,46 µs** | **15.741,49 µs** |
| `M10` | **M10_hybrid_dtw_lda_cnn** | **3.505,72 µs** | **4.750,11 µs** | **9.785,67 µs** | **10.516,13 µs** |

### 2.1. Latência Média vs. p99 (Estabilidade Temporal / Jitter)
Nos ensaios físicos com registro de percentil 99 (`p99`), o jitter foi mínimo (diferença inferior a 1% entre média e p99 nos modelos principais):
- **Ozdemir População (40 sujeitos, 187.200 inferências reais em Flash):**
  - `micro_lda_q7`: Média **330,22 µs** | p99 **347,55 µs**
  - `cnn1d_int8`: Média **6.575,07 µs** | p99 **6.610,04 µs**
- **Sujeitos de Referência (500 janelas estratificadas por modelo):**
  - **NinaPro DB2 (`S4`, 12CH):** `M01` **1.059,13 µs** (p99 `1.075,45 µs`) | `M02` **14.201,21 µs** (p99 `14.254,17 µs`) | `M03` **49.960,18 µs** (p99 `49.972,45 µs`)
  - **NinaPro DB3 (`S11`, 12CH):** `M01` **1.060,52 µs** (p99 `1.073,52 µs`) | `M02` **14.129,97 µs** (p99 `14.249,19 µs`) | `M03` **49.960,16 µs** (p99 `49.972,83 µs`)
  - **Khushaba (`S1`, 2CH):** `M01` **180,65 µs** (p99 `189,01 µs`) | `M02` **5.915,67 µs** (p99 `5.934,20 µs`) | `M03` **3.963,14 µs** (p99 `3.971,76 µs`)

---

## 3. Emulador Espressif QEMU 9.2.2 (`esp32` LX6 vs. `esp32s3` LX7 @ 240 MHz)

Os tempos no QEMU isolado via registrador `CCOUNT` (`1 ciclo = 4,167 ns @ 240 MHz`) são menores que os do silício real porque medem o kernel computacional puro sem overhead do FreeRTOS SMP, cache misses de Flash SPI e pré-processamento do protocolo HIL.

### 3.1. Pipeline Bare-Metal Completo (`Filtro Butterworth + Features Zero-FPU + Micro-LDA Q7`)
Fonte: `reports/esp32_qemu/esp32_qemu_quantized_pipeline_report.md`

| Dataset | Máquina QEMU | Precisão | Canais | Latência Total TinyML @ 240 MHz | Duty Cycle (Stride 50 ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ozdemir (10 cls)** | `esp32` (LX6) | `float32` | 4 | **0,0345 ms** (`34,5 µs`) | 0,0691% |
| **Ozdemir (10 cls)** | `esp32s3` (LX7) | `float32` | 4 | **0,0314 ms** (`31,4 µs`) | 0,0629% |
| **Ozdemir (10 cls)** | `esp32` (LX6) | `int16` | 4 | **0,0324 ms** (`32,4 µs`) | 0,0647% |
| **Ozdemir (10 cls)** | `esp32s3` (LX7) | `int16` | 4 | **0,0313 ms** (`31,3 µs`) | 0,0626% |
| **Ozdemir (10 cls)** | `esp32` (LX6) | `int8` | 4 | **0,0342 ms** (`34,2 µs`) | 0,0685% |
| **Ozdemir (10 cls)** | `esp32s3` (LX7) | `int8` | 4 | **0,0612 ms** (`61,2 µs`) | 0,1223% |
| **NinaPro DB2 (50 cls)** | `esp32` (LX6) | `int16` | 12 | **0,0920 ms** (`92,0 µs`) | 0,1840% |
| **NinaPro DB2 (50 cls)** | `esp32s3` (LX7) | `int16` | 12 | **0,1200 ms** (`120,0 µs`) | 0,2400% |
| **NinaPro DB2 (50 cls)** | `esp32` (LX6) | `int8` | 12 | **0,0786 ms** (`78,6 µs`) | 0,1572% |
| **NinaPro DB2 (50 cls)** | `esp32s3` (LX7) | `int8` | 12 | **0,1384 ms** (`138,4 µs`) | 0,2768% |

### 3.2. Dual-Core FreeRTOS no QEMU (Latência da Tarefa Completa no Core 1)
Fonte: `reports/freertos_qemu/qemu_freertos_dualcore_comparison_report.md`
- **4 canais (Ozdemir):** **8,58 µs** (Micro-LDA Q7) vs. **157,34 µs** (CNN-1D INT8) — Micro-LDA **18,34×** mais rápido.
- **12 canais (NinaPro DB2):** **28,34 µs** (Micro-LDA Q7) vs. **239,89 µs** (CNN-1D INT8) — Micro-LDA **8,46×** mais rápido.

---

## 4. Latência Total para Classificar e Executar o Gesto (Pipeline + Suavização Temporal)

Para acionamento de uma prótese em tempo real (janela $W = 200\text{ ms}$, avanço/stride $S = 50\text{ ms}$ @ 2 kHz, decisão a cada 50 ms), o atraso total entre a mudança de intenção muscular e a estabilização do comando depende do tamanho $K$ do **voto majoritário temporal** (custo de CPU $< 0,5\ \mu\text{s}$):

| Política de Voto ($K$) | Atraso Algorítmico $(K-1) \times 50\text{ ms}$ | Latência Total (Ozdemir 4CH, Micro-LDA) | Latência Total (Ozdemir 4CH, CNN-1D) | Avaliação Clínica (Barreira 200–300 ms) |
| :---: | :---: | :---: | :---: | :--- |
| **$K=1$ (Sem voto)** | 0 ms | **~0,33 ms** | **~6,58 ms** | Instantâneo, porém mais sujeito a *jitter* transitório. |
| **$K=3$ (Padrão Firmware)** | +100 ms | **~100,3 ms** | **~106,6 ms** | Excelente para controle de alta velocidade; abaixo de 200 ms. |
| **$K=5$ (Produção)** | +200 ms | **~200,3 ms** | **~207,6 ms** | Ponto de equilíbrio ótimo entre estabilidade e resposta. |
| **$K=7$** | +300 ms | **~300,3 ms** | **~307,6 ms** | Atinge a barreira de 300 ms; ligeiro atraso em trocas rápidas. |
| **$K=9$** | +400 ms | **~400,3 ms** | **~407,6 ms** | Ultrapassa 300 ms (**não recomendado** para transições dinâmicas). |
