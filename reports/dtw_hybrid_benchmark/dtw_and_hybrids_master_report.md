# Relatório Master Comparativo: DTW Assintoticamente Otimizado vs. Micro-LDA vs. CNN-1D vs. Híbridos (Host, QEMU & ESP32-S3)

> **Escopo Experimental:** Validação completa da proposta de mestrado (*Otimização Assintótica do DTW: Clássico $\mathcal{O}(N^2)$, Sakoe-Chiba $\mathcal{O}(w\cdot N)$, `LB_Keogh` $\mathcal{O}(N)$*) confrontada diretamente com os baselines congelados (`Micro-LDA Q7`, `CNN-1D INT8`) e 5 arquiteturas híbridas (`LDA+CNN H1/H2`, `DTW+LDA`, `DTW+CNN`, `Tri-Híbrido DTW+LDA+CNN`) nos datasets **Ozdemir et al. (4CH, 10 classes)** e **NinaPro DB2 (12CH, 50 classes)**.
> **Timestamp:** 2026-11-17 02:39:11 -03

---

## 1. Resumo Executivo e Veredito das 10 Arquiteturas

- **Compilação C99 Strict (`-std=c99 -O3 -Wall -Wextra -Werror`):** Zero warnings em QEMU (`esp32s3`, `esp32`) e no projeto nativo **ESP-IDF v5.3.2** (`semg_esp32s3.bin` = `240,512` bytes).
- **Paridade Numérica Host ↔ QEMU (`ESP32-S3`):** **100,00%** em todos os 10 modelos (`M01`–`M10`) com `env_max_diff = 0` (bit-exato).
- **Status Hardware Físico (`/dev/ttyACM0`):** `PHYSICAL_BOARD_DISCONNECTED`.

---

## 2. Benchmark Consolidado: Acurácia (Split LOTO Completo), Latência QEMU (`esp32s3` @ 240 MHz), Memória, Energia, IRC e IEE

### 2.1. Dataset Ozdemir et al. (Sujeito S1, 4 Canais, 10 Classes — 2.340 Janelas LOTO / 500 Janelas QEMU)

| ID | Arquitetura / Modelo | Acurácia Host LOTO | Macro-F1 Host | Paridade QEMU | Ciclos `CCOUNT` Médios | Latência QEMU (ms) | Flash ROM (Pesos) | SRAM Runtime | Energia (mJ) | **IRC** | **IEE** | Extra (`EE` / `Poda LB`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M01` | **M01_micro_lda_q7** | **81.20%** | 81.28% | 100.0% | `284` | **0.0012** | 200 B | 6.60 KB | 0.00018 | 686,361.8 | 5,639,679.2 | — |
| `M02` | **M02_cnn1d_int8** | **80.81%** | 80.24% | 100.0% | `41,793` | **0.1741** | 3,066 B | 11.21 KB | 0.02586 | 4,640.7 | 29,773.7 | — |
| `M03` | **M03_dtw_classic_on2** | **68.12%** | 67.43% | 100.0% | `38,305` | **0.1596** | 1,600 B | 12.85 KB | 0.02370 | 4,268.0 | 25,918.4 | — |
| `M04` | **M04_dtw_sakoe_chiba** | **67.73%** | 67.08% | 100.0% | `10,586` | **0.0441** | 1,600 B | 6.91 KB | 0.00655 | 15,356.6 | 123,185.6 | — |
| `M05` | **M05_dtw_lb_keogh** | **67.73%** | 67.08% | 100.0% | `2,452` | **0.0102** | 4,800 B | 7.11 KB | 0.00152 | 66,289.9 | 524,019.8 | Poda LB=84.7% |
| `M06` | **M06_hybrid_lda_cnn_head** | **81.79%** | 82.22% | 100.0% | `41,914` | **0.1746** | 3,306 B | 11.54 KB | 0.02593 | 4,683.6 | 29,692.2 | — |
| `M07` | **M07_hybrid_lda_cnn_cascade** | **81.79%** | 82.22% | 100.0% | `18,099` | **0.0754** | 3,266 B | 11.21 KB | 0.01120 | 10,846.1 | 69,586.0 | EE=57.6% |
| `M08` | **M08_hybrid_dtw_lda** | **81.32%** | 81.41% | 100.0% | `5,705` | **0.0238** | 1,800 B | 6.91 KB | 0.00353 | 34,210.3 | 274,423.6 | Top-M DP=5 |
| `M09` | **M09_hybrid_dtw_cnn** | **81.79%** | 81.37% | 100.0% | `47,103` | **0.1963** | 4,666 B | 11.52 KB | 0.02915 | 4,167.7 | 26,440.2 | Top-M DP=5 |
| `M10` | **M10_hybrid_dtw_lda_cnn** | **82.78%** | 83.15% | 100.0% | `22,321` | **0.0930** | 8,506 B | 11.72 KB | 0.01381 | 8,900.5 | 56,070.9 | EE=49.2% |

### 2.2. Dataset NinaPro DB2 (Sujeito S4, 12 Canais, 50 Classes — 15.824 Janelas LOTO / 500 Janelas QEMU)

| ID | Arquitetura / Modelo | Acurácia Host LOTO | Macro-F1 Host | Paridade QEMU | Ciclos `CCOUNT` Médios | Latência QEMU (ms) | Flash ROM (Pesos) | SRAM Runtime | Energia (mJ) | **IRC** | **IEE** | Extra (`EE` / `Poda LB`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M01` | **M01_micro_lda_q7** | **47.02%** | 48.34% | 100.0% | `1,202` | **0.0050** | 3,000 B | 22.20 KB | 0.00074 | 469,671.4 | 2,349,139.0 | — |
| `M02` | **M02_cnn1d_int8** | **59.21%** | 52.68% | 100.0% | `83,843` | **0.3493** | 7,922 B | 26.81 KB | 0.05188 | 8,474.1 | 39,952.8 | — |
| `M03` | **M03_dtw_classic_on2** | **43.63%** | 25.51% | 100.0% | `303,984` | **1.2666** | 24,000 B | 28.45 KB | 0.18809 | 1,722.3 | 7,976.3 | — |
| `M04` | **M04_dtw_sakoe_chiba** | **43.63%** | 25.51% | 100.0% | `77,617` | **0.3234** | 24,000 B | 22.51 KB | 0.04802 | 6,745.4 | 33,588.1 | — |
| `M05` | **M05_dtw_lb_keogh** | **43.63%** | 25.51% | 100.0% | `15,207` | **0.0634** | 72,000 B | 22.71 KB | 0.00941 | 34,428.5 | 170,947.5 | Poda LB=95.0% |
| `M06` | **M06_hybrid_lda_cnn_head** | **48.08%** | 48.67% | 100.0% | `84,899` | **0.3537** | 9,122 B | 27.14 KB | 0.05253 | 6,795.7 | 31,920.8 | — |
| `M07` | **M07_hybrid_lda_cnn_cascade** | **58.98%** | 53.22% | 100.0% | `64,878` | **0.2703** | 10,922 B | 26.81 KB | 0.04014 | 10,909.1 | 51,433.2 | EE=24.6% |
| `M08` | **M08_hybrid_dtw_lda** | **49.59%** | 45.92% | 100.0% | `9,237` | **0.0385** | 27,000 B | 22.51 KB | 0.00571 | 64,423.3 | 320,788.9 | Top-M DP=5 |
| `M09` | **M09_hybrid_dtw_cnn** | **59.73%** | 52.92% | 100.0% | `91,596` | **0.3816** | 31,922 B | 27.12 KB | 0.05668 | 7,825.5 | 36,766.6 | Top-M DP=5 |
| `M10` | **M10_hybrid_dtw_lda_cnn** | **49.26%** | 48.87% | 100.0% | `52,038` | **0.2168** | 87,122 B | 27.32 KB | 0.03220 | 11,359.4 | 53,251.3 | EE=42.0% |

---

## 3. Prova Empírica e Assintótica da Proposta de Mestrado (Modelo RAM e Otimizações DTW)

### 3.1. Escalonamento Assintótico de Tempo e Espaço vs. Limite Físico de 512 KB SRAM

| Comprimento ($N$) | Banda $w$ ($10\%N$) | DTW Clássico $\mathcal{O}(N^2)$ Células | SRAM Clássico (Int32/F32) | Cabe em 512 KB? | Sakoe-Chiba $\mathcal{O}(w\cdot N)$ Células | SRAM 2 Linhas ($\mathcal{O}(N)$) | Redução de Células | `LB_Keogh` $\mathcal{O}(N)$ Ops |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `20` | `2` | 400 | 1,600 B | Sim | 94 | 160 B | **4.26×** | 20 |
| `40` | `4` | 1,600 | 6,400 B | Sim | 340 | 320 B | **4.71×** | 40 |
| `100` | `10` | 10,000 | 40,000 B | Sim | 1,990 | 800 B | **5.03×** | 100 |
| `200` | `20` | 40,000 | 160,000 B | Sim | 7,780 | 1,600 B | **5.14×** | 200 |
| `400` | `40` | 160,000 | 640,000 B | **NÃO (ESTOURADO)** | 30,760 | 3,200 B | **5.20×** | 400 |

### 3.2. Speedup Medido em Silício Emulado (`ESP32-S3` QEMU `CCOUNT`) — Progressão DTW

- **Ozdemir 4CH (10 Classes):** Clássico $\mathcal{O}(N^2)$ = `0.160 ms` $\rightarrow$ Sakoe-Chiba $\mathcal{O}(w\cdot N)$ = `0.044 ms` (**3.62×** mais rápido) $\rightarrow$ `LB_Keogh` $\mathcal{O}(N)$ (**poda de 84.7% dos templates**) = **`0.010 ms`** (**15.62× speedup total vs. Clássico**, com 0,00% perda de acurácia frente ao Sakoe-Chiba).
- **NinaPro 12CH (50 Classes):** Clássico $\mathcal{O}(N^2)$ = `1.267 ms` $\rightarrow$ Sakoe-Chiba $\mathcal{O}(w\cdot N)$ = `0.323 ms` (**3.92×** mais rápido) $\rightarrow$ `LB_Keogh` $\mathcal{O}(N)$ (**poda de 95.0% dos templates**) = **`0.063 ms`** (**19.99× speedup total vs. Clássico**, com 0,00% perda de acurácia frente ao Sakoe-Chiba).

---

## 4. Testes de Robustez Fisiológica (Seção 3.3 da Proposta de Mestrado)

### 4.1. Variação de Velocidade do Gesto (*Time-Warping* até 200% — Subconjunto de 1.500 Janelas)

| Modelo | Ozdemir 100% | Ozdemir 125% | Ozdemir 150% | Ozdemir 200% | NinaPro 100% | NinaPro 125% | NinaPro 150% | NinaPro 200% |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **M01_micro_lda_q7** | 81.3% | 68.6% | 57.3% | 45.3% | 47.3% | 22.3% | 9.4% | 4.4% |
| **M02_cnn1d_int8** | 80.6% | 64.7% | 52.7% | 39.4% | 59.3% | 28.5% | 8.6% | 4.0% |
| **M03_dtw_classic_on2** | 68.2% | 66.4% | 65.8% | 68.5% | 42.9% | 43.3% | 43.5% | 42.9% |
| **M04_dtw_sakoe_chiba** | 68.2% | 66.4% | 65.8% | 68.5% | 42.9% | 43.3% | 43.5% | 42.9% |
| **M05_dtw_lb_keogh** | 68.2% | 66.4% | 65.8% | 68.5% | 42.9% | 43.3% | 43.5% | 42.9% |
| **M06_hybrid_lda_cnn_head** | 81.2% | 68.2% | 58.3% | 45.2% | 48.7% | 25.4% | 9.9% | 5.8% |
| **M07_hybrid_lda_cnn_cascade** | 81.2% | 68.2% | 58.3% | 46.2% | 58.6% | 28.7% | 9.2% | 4.2% |
| **M08_hybrid_dtw_lda** | 81.4% | 68.8% | 57.4% | 45.8% | 50.1% | 27.5% | 11.6% | 5.6% |
| **M09_hybrid_dtw_cnn** | 81.6% | 66.4% | 55.5% | 44.0% | 60.2% | 33.6% | 9.6% | 4.6% |
| **M10_hybrid_dtw_lda_cnn** | 82.5% | 71.1% | 61.5% | 48.9% | 50.3% | 24.9% | 9.9% | 4.7% |

### 4.2. Simulação de *Electrode Shift* (20% Acoplamento Cruzado + 15% Atenuação de Ganho)

| Modelo | Ozdemir Nominal | Ozdemir *Shift* ($\Delta$) | NinaPro Nominal | NinaPro *Shift* ($\Delta$) |
| :--- | :-: | :-: | :-: | :-: |
| **M01_micro_lda_q7** | 81.27% | 57.07% (-24.20 pp) | 47.27% | 41.07% (-6.20 pp) |
| **M02_cnn1d_int8** | 80.60% | 61.27% (-19.33 pp) | 59.33% | 43.60% (-15.73 pp) |
| **M03_dtw_classic_on2** | 68.20% | 36.00% (-32.20 pp) | 42.87% | 43.87% (+1.00 pp) |
| **M04_dtw_sakoe_chiba** | 68.20% | 36.00% (-32.20 pp) | 42.87% | 43.87% (+1.00 pp) |
| **M05_dtw_lb_keogh** | 68.20% | 36.00% (-32.20 pp) | 42.87% | 43.87% (+1.00 pp) |
| **M06_hybrid_lda_cnn_head** | 81.20% | 58.87% (-22.33 pp) | 48.73% | 40.47% (-8.27 pp) |
| **M07_hybrid_lda_cnn_cascade** | 81.20% | 59.13% (-22.07 pp) | 58.60% | 44.73% (-13.87 pp) |
| **M08_hybrid_dtw_lda** | 81.40% | 57.00% (-24.40 pp) | 50.07% | 47.60% (-2.47 pp) |
| **M09_hybrid_dtw_cnn** | 81.60% | 58.40% (-23.20 pp) | 60.20% | 45.80% (-14.40 pp) |
| **M10_hybrid_dtw_lda_cnn** | 82.47% | 57.80% (-24.67 pp) | 50.27% | 42.07% (-8.20 pp) |

---

## 5. Comparação Multi-Arquitetura ESP32 vs. ESP32-S3 & Build ESP-IDF v5.3.2

- **ELF Bare-Metal QEMU (`esp32s3`):** `.text=169,691 B`, `.data=288 B`, `.bss=90,360 B` (Total DRAM = `260,339 B`).
- **ELF Bare-Metal QEMU (`esp32` LX6):** `.text=169,747 B`, `.bss=90,360 B`.
- **Binário Nativo ESP-IDF v5.3.2 (`semg_esp32s3.bin`):** `240,512` bytes compilado com zero warnings.
- **Hardware Físico (`/dev/ttyACM0`):** `PHYSICAL_BOARD_DISCONNECTED`.
