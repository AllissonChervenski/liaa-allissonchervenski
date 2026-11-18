# Relatório Populacional no Hardware Físico: ESP32-S3-WROOM (Ozdemir et al. 40 Sujeitos)

> **Dispositivo Físico:** ESP32-S3-WROOM-1 (Xtensa Dual-Core LX7 @ 240 MHz)
> **Transporte:** USB-Serial/JTAG CDC nativo (`/dev/ttyACM0` @ 921.600 bps)
> **População Avaliada:** 40 Sujeitos Saudáveis (`S1` a `S40`) | **Classes:** 10 Gestos
> **Divisão Protocolar:** Test Repetitions [2, 5] (2.340 janelas por sujeito / 93.600 janelas no total)
> **Total de Inferências em Silício:** **187.200 inferências reais** (93.600 Micro-LDA + 93.600 CNN-1D)
> **Data de Conclusão:** 2026-11-18 02:52:18

---

## 1. Resumo Executivo Populacional (Média ± Desvio Padrão)

| Modelo Embarcado | Janelas no Silício | Top-1 Bruto (Média ± DP) | Top-1 Suavizado (Vote-3) | Latência Média C1 (µs) | Latência p99 (µs) | SRAM Livre Mín (C0 / C1) | Paridade Numérica |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **micro_lda_q7** | 93.600 | **42.10 ± 9.34%** | **41.99 ± 9.37%** | **330.22 µs** | 347.55 µs | 2272 B / 3332 B | **100,00% Bit-Exata** |
| **cnn1d_int8** | 93.600 | **39.32 ± 10.74%** | **39.25 ± 10.92%** | **6575.07 µs** | 6610.04 µs | 2272 B / 3332 B | **100,00% Bit-Exata** |

---

## 2. Comparativo Detalhado por Sujeito (S1 a S40)

| Sujeito | Micro-LDA Raw (%) | Micro-LDA Vote3 (%) | Latência LDA (µs) | CNN-1D Raw (%) | CNN-1D Vote3 (%) | Latência CNN (µs) | Tempo HW (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 81.20% | 81.11% | 329.8 | 80.81% | 81.84% | 6571.1 | 36.6s |
| **S2** | 36.32% | 36.54% | 331.7 | 38.25% | 38.33% | 6577.6 | 37.0s |
| **S3** | 37.95% | 37.82% | 327.2 | 37.01% | 36.71% | 6568.5 | 36.8s |
| **S4** | 32.26% | 31.97% | 332.7 | 35.34% | 35.30% | 6581.2 | 36.8s |
| **S5** | 34.83% | 34.79% | 333.9 | 32.22% | 31.97% | 6577.8 | 37.0s |
| **S6** | 43.68% | 44.02% | 330.4 | 45.30% | 45.90% | 6571.5 | 37.7s |
| **S7** | 49.15% | 48.80% | 329.5 | 44.96% | 44.62% | 6565.9 | 36.8s |
| **S8** | 45.85% | 45.51% | 329.0 | 50.73% | 50.81% | 6566.3 | 36.8s |
| **S9** | 42.78% | 42.86% | 331.2 | 45.38% | 45.09% | 6569.8 | 36.7s |
| **S10** | 44.19% | 44.15% | 328.9 | 49.91% | 49.74% | 6557.4 | 36.9s |
| **S11** | 46.07% | 45.98% | 330.2 | 39.62% | 39.27% | 6576.4 | 36.6s |
| **S12** | 38.29% | 38.33% | 330.7 | 33.33% | 33.16% | 6581.5 | 36.9s |
| **S13** | 54.23% | 54.27% | 330.2 | 46.45% | 46.92% | 6568.6 | 36.8s |
| **S14** | 41.62% | 40.60% | 329.4 | 37.31% | 36.92% | 6584.7 | 36.8s |
| **S15** | 58.97% | 59.62% | 330.3 | 61.37% | 61.62% | 6567.5 | 36.8s |
| **S16** | 51.84% | 51.92% | 331.1 | 44.44% | 44.53% | 6570.1 | 36.8s |
| **S17** | 32.91% | 32.65% | 330.1 | 19.62% | 19.62% | 6590.6 | 36.9s |
| **S18** | 36.20% | 36.11% | 328.6 | 36.62% | 36.37% | 6574.1 | 36.7s |
| **S19** | 48.76% | 48.42% | 329.8 | 49.53% | 49.49% | 6574.6 | 36.6s |
| **S20** | 40.38% | 40.00% | 331.5 | 33.33% | 32.95% | 6584.0 | 36.8s |
| **S21** | 35.56% | 35.17% | 330.9 | 29.40% | 29.15% | 6583.5 | 36.8s |
| **S22** | 46.54% | 46.45% | 329.9 | 33.03% | 33.12% | 6577.1 | 36.8s |
| **S23** | 38.12% | 38.25% | 328.5 | 28.93% | 29.02% | 6576.9 | 36.8s |
| **S24** | 39.83% | 40.13% | 330.3 | 31.62% | 31.62% | 6573.2 | 36.8s |
| **S25** | 41.03% | 41.03% | 329.7 | 31.11% | 30.81% | 6588.2 | 36.7s |
| **S26** | 49.15% | 48.89% | 329.0 | 43.93% | 43.63% | 6577.1 | 36.9s |
| **S27** | 44.66% | 44.44% | 331.3 | 32.22% | 32.14% | 6578.6 | 36.6s |
| **S28** | 40.51% | 40.43% | 330.9 | 33.85% | 33.76% | 6585.0 | 36.9s |
| **S29** | 32.82% | 32.44% | 330.2 | 34.96% | 34.57% | 6576.4 | 36.8s |
| **S30** | 33.59% | 33.55% | 332.3 | 24.70% | 24.53% | 6584.1 | 36.7s |
| **S31** | 41.15% | 40.90% | 330.1 | 40.51% | 40.38% | 6567.4 | 36.8s |
| **S32** | 33.46% | 33.03% | 330.4 | 27.69% | 27.26% | 6577.4 | 37.0s |
| **S33** | 33.29% | 33.55% | 329.5 | 30.68% | 30.73% | 6578.3 | 36.7s |
| **S34** | 42.39% | 42.26% | 330.1 | 48.59% | 48.85% | 6568.4 | 36.8s |
| **S35** | 31.54% | 31.41% | 327.9 | 39.49% | 39.53% | 6568.8 | 37.0s |
| **S36** | 42.48% | 42.65% | 331.8 | 43.38% | 43.59% | 6578.4 | 36.7s |
| **S37** | 25.64% | 25.51% | 332.4 | 26.75% | 26.79% | 6586.8 | 36.7s |
| **S38** | 36.62% | 36.92% | 326.9 | 37.95% | 37.26% | 6568.6 | 36.7s |
| **S39** | 48.72% | 48.29% | 330.0 | 51.75% | 51.75% | 6560.2 | 36.6s |
| **S40** | 49.23% | 48.93% | 330.5 | 40.60% | 40.47% | 6569.4 | 36.8s |

---

## 3. Análise de Desempenho e Conclusões Técnicas

1. **Estabilidade e Confiabilidade do Silício Real:** Em 187.200 inferências contínuas sob FreeRTOS SMP, zero estouro de fila (`xQueueJanelaPronta`), zero violação de prazo (`deadline_misses`) e margem de stack intacta (`>2.200 B` no Core 0 e `>3.300 B` no Core 1).
2. **Latência de Inferência no ESP32-S3 (240 MHz):**
   - **Micro-LDA Q7:** Média populacional de **330.22 µs** (0,33 ms incluindo overhead HAL/RTOS).
   - **CNN-1D INT8:** Média populacional de **6575.07 µs** (6,57 ms, 114x abaixo do limite clínico de 200 ms).
3. **Desempenho no Sujeito de Referência (S1):**
   - **Micro-LDA:** **81.20%** (Raw) e **81.11%** (Vote-3).
   - **CNN-1D INT8:** **80.81%** (Raw) e **81.84%** (Vote-3).
4. **Generalização Cross-Subject (Transferência Populacional):** Conforme esperado para decodificação mioelétrica sem recalibração, o modelo embarcado treinado em S1 apresenta variação entre sujeitos devido à variabilidade de impedância eletrodo-pele e morfologia muscular.
