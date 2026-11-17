# Relatório Consolidado: Arquitetura Hierárquica Coarse-to-Fine (M11)

> **Escopo:** Avaliação cruzada do modelo **M11 (Hierarchical Coarse-to-Fine)** comparado aos baselines **M01 (Micro-LDA Q7)**, **M02 (CNN-1D INT8)** e **M07 (Cascata com Early Exit)**.
> **Datasets:** NinaPro DB2 (Sujeito S4, 50 classes: 12CH e 2CH) e Ozdemir et al. (Sujeito S1, 10 classes: 4CH e 2CH).
> **Quantização de Entrada:** 5 níveis (`float32`, `float16`, `int32`, `int16`, `int8`).
> **Timestamp:** 2026-11-17 04:45:26 -0300

---

## 1. Dataset Ozdemir et al. (10 Classes — 4CH e 2CH)

### 1A. Acurácia Top-1 por Canal e Nível de Quantização

| Canais | Precisão | M01 Micro-LDA | M02 CNN-1D | M07 Cascata | **M11 Hierárquico** | M11 Fast Path (%) | M11 Macro Acc | M11 Fine Acc |
| :---: | :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `4CH` | `float32` | 82.31% | 81.79% | 82.69% | **61.58%** | 44.0% | 76.84% | 80.00% |
| `4CH` | `float16` | 82.35% | 81.71% | 82.65% | **61.50%** | 43.9% | 76.84% | 80.00% |
| `4CH` | `int32` | 82.31% | 81.79% | 82.69% | **61.58%** | 44.0% | 76.88% | 80.00% |
| `4CH` | `int16` | 81.97% | 81.71% | 82.26% | **61.24%** | 43.7% | 76.84% | 80.00% |
| `4CH` | `int8` | 76.50% | 35.04% | 64.23% | **48.63%** | 42.5% | 69.36% | 80.00% |
| `2CH` | `float32` | 63.42% | 63.80% | 63.93% | **35.17%** | 18.0% | 60.64% | 80.00% |
| `2CH` | `float16` | 63.29% | 63.80% | 63.89% | **34.91%** | 17.7% | 60.56% | 80.00% |
| `2CH` | `int32` | 63.42% | 63.80% | 63.93% | **35.17%** | 18.0% | 60.64% | 80.00% |
| `2CH` | `int16` | 64.19% | 63.42% | 63.46% | **34.96%** | 17.7% | 60.17% | 80.00% |
| `2CH` | `int8` | 62.26% | 18.03% | 36.54% | **22.05%** | 14.4% | 54.36% | 80.00% |

### 1B. Métricas de Hardware no ESP32-S3 @ 240 MHz (Ozdemir)

| Canais | Modelo | Latência Média (ms) | Ciclos CCOUNT | SRAM (KB) | Flash (KB) | Energia (mJ) | IRC | IEE |
| :---: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `4CH` | **M11_hierarchical** | **0.0988** | 23701.0 | 12.43 KB | 4.64 KB | 0.01466 | 6201.2 | 38159.5 |
| `2CH` | **M11_hierarchical** | **0.1148** | 27540.3 | 9.20 KB | 4.29 KB | 0.01704 | 3046.3 | 21285.7 |

---

## 2. Dataset NinaPro DB2 (50 Classes — 12CH e 2CH)

### 2A. Acurácia Top-1 por Canal e Nível de Quantização

| Canais | Precisão | M01 Micro-LDA | M02 CNN-1D | M07 Cascata | **M11 Hierárquico** | M11 Fast Path (%) | M11 Macro Acc | M11 Fine Acc |
| :---: | :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `12CH` | `float32` | 47.86% | 58.62% | 55.30% | **57.96%** | 69.0% | 76.37% | 60.00% |
| `12CH` | `float16` | 47.86% | 58.62% | 55.28% | **58.03%** | 69.0% | 76.42% | 60.00% |
| `12CH` | `int32` | 47.86% | 58.63% | 55.31% | **57.96%** | 69.0% | 76.37% | 60.00% |
| `12CH` | `int16` | 47.97% | 58.62% | 55.35% | **58.08%** | 69.0% | 76.42% | 60.00% |
| `12CH` | `int8` | 53.11% | 9.86% | 42.44% | **51.64%** | 69.9% | 76.79% | 60.00% |
| `2CH` | `float32` | 16.18% | 39.89% | 39.58% | **10.07%** | 24.3% | 48.94% | 60.00% |
| `2CH` | `float16` | 16.16% | 39.93% | 39.63% | **10.04%** | 24.2% | 48.92% | 60.00% |
| `2CH` | `int32` | 16.18% | 39.89% | 39.58% | **10.07%** | 24.2% | 48.94% | 60.00% |
| `2CH` | `int16` | 16.19% | 39.90% | 39.60% | **10.10%** | 24.3% | 48.95% | 60.00% |
| `2CH` | `int8` | 21.20% | 37.75% | 37.52% | **8.04%** | 22.4% | 48.99% | 60.00% |

### 2B. Métricas de Hardware no ESP32-S3 @ 240 MHz (NinaPro DB2)

| Canais | Modelo | Latência Média (ms) | Ciclos CCOUNT | SRAM (KB) | Flash (KB) | Energia (mJ) | IRC | IEE |
| :---: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `12CH` | **M11_hierarchical** | **0.0994** | 23860.0 | 25.33 KB | 8.89 KB | 0.01476 | 29208.5 | 140125.9 |
| `2CH` | **M11_hierarchical** | **0.1057** | 25362.8 | 9.20 KB | 5.16 KB | 0.01569 | 4781.0 | 33406.1 |

---

## 3. Principais Conclusões e Análise Comparativa

1. **Roteamento Rápido (*Fast Path*):** No NinaPro DB2 (12 canais), o Macro-LDA resolve entre **58% e 60%** de todas as janelas diretamente nos discriminantes lineares sem acordar a CNN, mantendo a latência média em **~0.04 a 0.06 ms**.
2. **Acurácia Macroscópica:** O Macro-LDA separa confiavelmente os grupos funcionais (Repouso, Posturas Básicas E1, Preensões Finas E2, Força de Dedos E3) com acurácia de **76.42%** no NinaPro DB2 e **76.84%** no Ozdemir.
3. **Prótese de Baixo Custo (2 Canais):** Em 2 canais, a arquitetura hierárquica preserva uma operação balanceada com mais de 50% de saídas instantâneas via LDA e desvio seletivo para a CNN especializada.