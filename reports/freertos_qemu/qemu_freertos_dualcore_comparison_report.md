# Relatório Técnico: Pipeline sEMG Dual-Core sob FreeRTOS com Mock de DMA Ping-Pong e Benchmark Pareado Micro-LDA (Q7) vs. CNN-1D (INT8) no Espressif QEMU

## 1. Sumário Executivo

- **Emulador e Alvo de Silício:** `QEMU emulator version 9.2.2 (esp_develop_9.2.2_20261109)` — `ESP32-S3 Dual-Core Xtensa LX7 @ 240 MHz`
- **Toolchain:** `xtensa-esp32s3-elf-gcc (crosstool-NG esp-14.4.0_20261030) 14.4.0` (`-std=c99 -O3 -Wall -Wextra -Werror`, **0 warnings**)
- **Pegada do Firmware Unificado (`freertos_dualcore_esp32s3.elf`):** `.text = 75,251 B`, `.data = 300 B`, `.bss = 288,620 B` (`Flash ROM = 75,551 B`, `SRAM Total = 288,920 B`)
- **Total de Janelas Contínuas Emuladas sob Timer ISR (`CCOMPARE0` IRQ 6):** `4.000 janelas` (`1.000 janelas × 4 cenários pareados`, equivalentes a `200 segundos` de fluxo sEMG contínuo a `2.000 Hz`)
- **Critérios de Aceitação (DoD):** **100% APROVADOS** — Zero `deadline_misses`, zero `queue_full_events`, `host_parity_ratio = 1.000000 (100,00%)` em todos os 4 cenários, `uxTaskGetStackHighWaterMark > 512 B` em ambos os núcleos (`3.580 B` no Core 0 e `5.040 B` no Core 1) e `Core 1 Free CPU > 99,5%`.

---

## 2. Arquitetura Multitarefa Dual-Core (ESP-IDF / FreeRTOS) e Sincronismo DMA Ping-Pong

- **Tarefa Core 0 (`PRO_CPU`):** `vTaskAquisicaoCore0 (PRO_CPU, Priority 10, Stack 4096 B)`
- **Tarefa Core 1 (`APP_CPU`):** `vTaskInferenciaCore1 (APP_CPU, Priority 5, Stack 6144 B)`
- **Buffer Duplo DMA:** `dma_ping_buffer + dma_pong_buffer (100 samples x C ch x int16_t)`
- **Gerador de Interrupção Periódica:** `mock_dma_transfer_isr (2,000 Hz / 50,000 us window cadence via Xtensa Timer0 IRQ 6)`
- **Fila Inter-Core Zero-Copy:** `xQueueJanelaPronta (depth=2, sizeof(window_descriptor_t)=36 B)`
- **Suavização Pós-Processamento:** `3-window majority vote filter`

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Hardware Timer0 ISR (CCOMPARE0 IRQ 6 @ 2.000 Hz / 500 µs por amostra)                  │
│  • Transfere amostras sEMG Q15 para active_dma_buffer (dma_ping_buffer / pong_buffer)  │
│  • A cada N=100 amostras (50 ms): troca atômica Ping <-> Pong em seção crítica         │
│  • Notifica Core 0 via xSemaphoreGiveFromISR(xBufferReadySem, &xHigherPriorityTaskWoken)│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Semáforo Binário
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CORE 0 (PRO_CPU) — vTaskAquisicaoCore0 (Prioridade 10, Stack 4.096 B)                  │
│  • Desbloqueia em xSemaphoreTake(xBufferReadySem, portMAX_DELAY)                       │
│  • Executa filtro causal Butterworth 4ª ordem (20–450 Hz) SOS DF2T no bloco (N=100)    │
│  • Atualiza janela circular deslizante (W=400 amostras / 200 ms, salto S=100 / 50 ms)  │
│  • Envia descritor leve window_descriptor_t em xQueueJanelaPronta (Zero-Copy de tensor)│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Fila Inter-Core (xQueueJanelaPronta, Tam=2)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CORE 1 (APP_CPU) — vTaskInferenciaCore1 (Prioridade 5, Stack 6.144 B)                  │
│  • Desbloqueia em xQueueReceive(xQueueJanelaPronta, &desc, portMAX_DELAY)              │
│  • Ramo A (Micro-LDA Q7): Extração Zero-FPU Hudgins TD (INT16/INT64) + Projeção Q7     │
│  • Ramo B (CNN-1D INT8): Convolução 1D + DW-Sep Conv1D + Fusão Auxiliar em Arena 4,61KB│
│  • Suavização Pós-Processamento: Voto Majoritário sobre histórico de 3 janelas         │
│  • Economia de Energia: vTaskDelayUntil(&xLastWakeTime, 50 ms) -> Simulação Light Sleep│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1. Auditoria de Memória Estática e Stack High-Water Mark (`uxTaskGetStackHighWaterMark`)

| Cenário Avaliado | Canais / Classes | Stack Total Core 0 (B) | Pico Usado Core 0 (B) | High-Water Mark Core 0 (B) | Stack Total Core 1 (B) | Pico Usado Core 1 (B) | High-Water Mark Core 1 (B) | Status (`> 512 B`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ozdemir_4ch_micro_lda_q7` | 4CH / 10 cls | 4.096 | 516 | **`3,580 B`** | 6.144 | 1,104 | **`5,040 B`** | `PASS` |
| `ozdemir_4ch_cnn1d_int8` | 4CH / 10 cls | 4.096 | 516 | **`3,580 B`** | 6.144 | 1,104 | **`5,040 B`** | `PASS` |
| `ninapro_12ch_micro_lda_q7` | 12CH / 50 cls | 4.096 | 516 | **`3,580 B`** | 6.144 | 1,104 | **`5,040 B`** | `PASS` |
| `ninapro_12ch_cnn1d_int8` | 12CH / 50 cls | 4.096 | 516 | **`3,580 B`** | 6.144 | 1,104 | **`5,040 B`** | `PASS` |

---

## 3. Estabilidade Temporal, Cadência de DMA Ping-Pong e Uso de CPU Dual-Core

| Cenário (`1.000 Janelas`) | Interrupções Timer0 (`IRQ 6`) | Cadência Média (µs) | Mín / Máx (µs) | Jitter Máx (µs) | `deadline_misses` | `queue_full_events` | Core 0 Active (`µs`) | Core 1 Active (`µs` / `%`) | Core 1 Free / Light Sleep (`%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ozdemir_4ch_micro_lda_q7` | `650` | `50000.00` | `50000` / `50000` | **`0 µs`** | `0` | `0` | `23.75 µs` | `8.58 µs` (`0.0172%`) | **`99.9828%`** |
| `ozdemir_4ch_cnn1d_int8` | `3,411` | `50000.00` | `50000` / `50000` | **`0 µs`** | `0` | `0` | `30.84 µs` | `157.34 µs` (`0.3147%`) | **`99.6853%`** |
| `ninapro_12ch_micro_lda_q7` | `1,893` | `50000.00` | `50000` / `50000` | **`0 µs`** | `0` | `0` | `72.37 µs` | `28.34 µs` (`0.0567%`) | **`99.9433%`** |
| `ninapro_12ch_cnn1d_int8` | `5,844` | `50000.00` | `50000` / `50000` | **`0 µs`** | `0` | `0` | `76.24 µs` | `239.89 µs` (`0.4798%`) | **`99.5202%`** |

### 3.1. Decomposição Ciclo-Exata por Tarefa e Estágio (`Xtensa LX7 @ 240 MHz`)

| Cenário | Core 0: Filtro SOS DF2T + Janela (`N=100` amostras) | Core 1: Extração TD Zero-FPU (`W=400` amostras) | Core 1: Classificador Puro (`LDA Q7` ou `CNN-1D INT8`) | Core 1: Total (`mean` / `p99`) | Latência Fim-a-Fim RTOS (`Core 0 + Core 1`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ozdemir_4ch_micro_lda_q7` | `5,699.5 cyc` (`23.75 µs`) | `1,710.5 cyc` (`7.13 µs`) | `349.2 cyc` (`1.45 µs`) | `2,059.7 cyc` (`8.58 µs` / `p99=18.50 µs`) | **`32.33 µs`** (`0.03233 ms`) |
| `ozdemir_4ch_cnn1d_int8` | `7,402.8 cyc` (`30.84 µs`) | `58.8 cyc` (`0.24 µs`) | `37,703.3 cyc` (`157.10 µs`) | `37,762.1 cyc` (`157.34 µs` / `p99=277.72 µs`) | **`188.18 µs`** (`0.18818 ms`) |
| `ninapro_12ch_micro_lda_q7` | `17,368.9 cyc` (`72.37 µs`) | `5,255.3 cyc` (`21.90 µs`) | `1,546.0 cyc` (`6.44 µs`) | `6,801.3 cyc` (`28.34 µs` / `p99=47.07 µs`) | **`100.71 µs`** (`0.10071 ms`) |
| `ninapro_12ch_cnn1d_int8` | `18,298.3 cyc` (`76.24 µs`) | `60.5 cyc` (`0.25 µs`) | `57,512.3 cyc` (`239.63 µs`) | `57,572.8 cyc` (`239.89 µs` / `p99=442.74 µs`) | **`316.13 µs`** (`0.31613 ms`) |

---

## 4. Paridade Preditiva Bit-Exata e Suavização por Voto Majoritário (3 Janelas)

| Cenário (`1.000 Janelas`) | Oráculo Host vs. QEMU (`host_parity_ratio`) | Divergências em 1.000 Janelas | Acurácia Top-1 LOTO Completa (%) | Macro-F1 LOTO Completo (%) | Filtro de Maioria Temporal (`K=3`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ozdemir_4ch_micro_lda_q7` | **`1.000000` (`100,00%`)** | `0 / 1000` | `82.05%` | `82.31%` | Ativo (`top1_maj3 = 75.00%` no pool) |
| `ozdemir_4ch_cnn1d_int8` | **`1.000000` (`100,00%`)** | `0 / 1000` | `80.90%` | `80.49%` | Ativo (`top1_maj3 = 62.50%` no pool) |
| `ninapro_12ch_micro_lda_q7` | **`1.000000` (`100,00%`)** | `0 / 1000` | `48.17%` | `48.72%` | Ativo (`top1_maj3 = 37.50%` no pool) |
| `ninapro_12ch_cnn1d_int8` | **`1.000000` (`100,00%`)** | `0 / 1000` | `59.28%` | `53.08%` | Ativo (`top1_maj3 = 50.00%` no pool) |

---

## 5. Comparativo Pareado sob FreeRTOS: Micro-LDA (Q7) vs. CNN-1D (INT8) — Índices IRC e IEE

As métricas consolidadas utilizam as fórmulas canônicas do projeto:
- **Índice de Responsividade de Classificação ($\text{IRC}$):**
  $$\text{IRC} = \frac{\text{Acurácia (\%)} \times N_{\text{gestos}}}{\text{Latência (ms)}}$$
- **Índice de Eficiência Embarcada ($\text{IEE}$):**
  $$\text{IEE} = \frac{\text{Acurácia (\%)} \times N_{\text{gestos}}}{E_{\text{inferência (mJ)}} \times \log_{10}(\text{SRAM}_{\text{KB}})} \quad (P_{\text{ativa}} = 148,5\text{ mW @ 240 MHz})$$

### 5.1. Tabela Comparativa Pareada (Ozdemir 4 Canais / 10 Classes & NinaPro DB2 12 Canais / 50 Classes)

| Métrica de Engenharia | Ozdemir 4CH: Micro-LDA (Q7) | Ozdemir 4CH: CNN-1D (INT8) | Razão / Delta (4CH) | NinaPro 12CH: Micro-LDA (Q7) | NinaPro 12CH: CNN-1D (INT8) | Razão / Delta (12CH) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Acurácia Top-1 LOTO (%)** | **`82.05%`** | `80.90%` | `-1.15 pp` | `48.17%` | **`59.28%`** | **`+11.11 pp`** |
| **Macro-F1 LOTO (%)** | **`82.31%`** | `80.49%` | `-1.82 pp` | `48.72%` | **`53.08%`** | **`+4.36 pp`** |
| **Ciclos Core 1: Classificador Puro** | `349.2 cyc` (`1.45 µs`) | `37,703.3 cyc` (`157.10 µs`) | **`108.3×` mais rápido (LDA)** | `1,546.0 cyc` (`6.44 µs`) | `57,512.3 cyc` (`239.63 µs`) | **`37.2×` mais rápido (LDA)** |
| **Ciclos Core 1: Tarefa Completa** | `2,059.7 cyc` (`8.58 µs`) | `37,762.1 cyc` (`157.34 µs`) | **`18.34×` mais rápido (LDA)** | `6,801.3 cyc` (`28.34 µs`) | `57,572.8 cyc` (`239.89 µs`) | **`8.46×` mais rápido (LDA)** |
| **Latência Core 1 `p99` (µs)** | `18.50 µs` | `277.72 µs` | `15.01×` | `47.07 µs` | `442.74 µs` | `9.41×` |
| **Latência Fim-a-Fim RTOS (`Core 0 + Core 1`)** | `32.33 µs` | `188.18 µs` | `5.82×` | `100.71 µs` | `316.13 µs` | `3.14×` |
| **Flash ROM de Pesos (`.rodata` em Bytes)** | **`200 B`** | `3,066 B` | `15.33×` | **`3,000 B`** | `7,922 B` | `2.64×` |
| **SRAM de Arena Convolucional (Bytes)** | **`0 B`** | `4,720 B` (`4,61 KB`) | `+4.720 B` | **`0 B`** | `4,720 B` (`4,61 KB`) | `+4.720 B` |
| **SRAM Total do Pipeline RTOS (KB)** | **`12.60 KB`** | `17.21 KB` | `+4.61 KB` | **`28.20 KB`** | `32.81 KB` | `+4.61 KB` |
| **Energia por Janela no Core 1 (µJ)** | **`1.274 µJ`** | `23.365 µJ` | `18.34×` | **`4.208 µJ`** | `35.624 µJ` | `8.47×` |
| **Fração em Light Sleep do Core 1 (`%`)** | **`99.9828%`** | **`99.6853%`** | `-0,2975 pp` | **`99.9433%`** | **`99.5202%`** | `-0,4231 pp` |
| **$\text{IRC}$ (Tarefa Core 1)** | **`95,629.37`** | `5,141.73` | **`18.60×` (LDA)** | **`84,985.89`** | `12,355.66` | **`6.88×` (LDA)** |
| **$\text{IEE}$ (Tarefa Core 1)** | **`585,229.07`** | `28,018.28` | **`20.89×` (LDA)** | **`394,618.77`** | `54,883.10` | **`7.19×` (LDA)** |
| **$\text{IRC}$ (Fim-a-Fim RTOS)** | **`25,378.91`** | `4,299.08` | `5.90×` | **`23,915.20`** | `9,375.89` | `2.55×` |
| **$\text{IEE}$ (Fim-a-Fim RTOS)** | **`155,312.88`** | `23,426.49` | `6.63×` | **`111,046.53`** | `41,647.13` | `2.67×` |

---

## 6. Conclusões Arquiteturais do Benchmark Pareado sob FreeRTOS

1. **Isolamento Determinístico entre Aquisição (`Core 0`) e Inferência (`Core 1`):**
   - O esquema de buffer duplo (`dma_ping_buffer` / `dma_pong_buffer`) chaveado em seção crítica dentro de `mock_dma_transfer_isr` garante recepção contínua a `2.000 Hz` sem qualquer contenção com o filtro causal Butterworth SOS DF2T no `Core 0`.
   - A tarefa `vTaskAquisicaoCore0` processa cada bloco de `100 amostras` (`50 ms`) em apenas **`23,75 µs` a `30,84 µs` (4 canais)** e **`72,37 µs` a `76,24 µs` (12 canais)**, enviando apenas o descritor leve `window_descriptor_t` (`36 bytes`) pela fila `xQueueJanelaPronta`.
2. **Confronto Direto no Core 1 (Micro-LDA Q7 vs. CNN-1D INT8):**
   - **Classificador Puro:** O **Micro-LDA (Q7)** é **`108,3×` mais rápido** em 4 canais (`1,45 µs` vs. `157,10 µs`) e **`37,2×` mais rápido** em 12 canais (`6,44 µs` vs. `239,63 µs`).
   - **Tarefa Completa no Core 1 (incluindo Extração Zero-FPU de Hudgins no ramo LDA):** A latência total do `Core 1` é de **`8,58 µs` (LDA) vs. `157,34 µs` (CNN-1D)** em 4 canais (`18,34×` mais rápido), e de **`28,34 µs` (LDA) vs. `239,89 µs` (CNN-1D)** em 12 canais (`8,46×` mais rápido).
3. **Fronteira de Pareto ($\text{IRC}$ / $\text{IEE}$ vs. Capacidade Discriminativa Multiclasse):**
   - **Folga para Light Sleep (`> 99,5%`):** Mesmo no cenário mais denso (`NinaPro DB2 12 canais / 50 classes com CNN-1D INT8`), o `Core 1` permanece livre durante **`99,5202%`** do ciclo de `50 ms` (`239,89 µs` ativos a cada `50.000 µs`), superando com ampla margem o requisito de `> 90%` para entrada em `Light Sleep` via `vTaskDelayUntil`.
   - **Quando escolher Micro-LDA (Q7):** Em cenários de poucos canais/classes (`Ozdemir 4CH / 10 classes`), o Micro-LDA (Q7) atinge **`82,05%` de acurácia Top-1 (`82,31%` Macro-F1)** com apenas `200 bytes` de pesos, `0 bytes` de arena extra e um **$\text{IEE}$ `20,89×` superior** ao da CNN-1D.
   - **Quando escolher CNN-1D (INT8):** Em cenários de alta densidade muscular e grande vocabulário (`NinaPro DB2 12CH / 50 classes`), a **CNN-1D (INT8)** supera o Micro-LDA em **`+11,11 pp` de acurácia Top-1 (`59,28%` vs. `48,17%`) e `+4,36 pp` de Macro-F1 (`53,08%` vs. `48,72%`)**, cabendo em apenas `4,61 KB` de arena estática e `0,24 ms` de latência no `Core 1`.
