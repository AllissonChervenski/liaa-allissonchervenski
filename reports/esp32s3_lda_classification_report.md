# Relatório Técnico: Classificação de Sinais sEMG via Micro-LDA (Q7) no ESP32-S3-WROOM

> **Escopo:** Avaliação em tempo real de classificação de gestos mioelétricos (sEMG) com o motor **Micro-LDA Quantizado (Q7)** embarcado no silício físico do microcontrolador **ESP32-S3-WROOM** (Dual-Core Xtensa LX7 @ 240 MHz).
> **Data / Timestamp da Execução:** 2026-11-15 20:38:18 -03
> **Plataforma Física:** ESP32-S3 (`esp32s3-wroom`), 4 canais, 10 classes, escala ADC `3276.7`

---

## 1. Resumo Executivo e Métricas Globais

O algoritmo de classificação Micro-LDA quantizado em ponto fixo Q7 foi executado diretamente no hardware ESP32-S3 sobre sinais sEMG reais de 4 canais do dataset **Ozdemir et al. (Sujeito S1, Repetição 2)**, totalizando **1.170 janelas contínuas** (117 janelas por gesto em 10 gestos).

| Indicador de Desempenho | Valor Obtido no ESP32-S3 | Especificação / Limite | Status |
| :--- | :---: | :---: | :---: |
| **Paridade Hardware-Software** | **100.00%** (1170/1170) | 100.00% bit-exact vs. host | **APROVADO** |
| **Acurácia Top-1 (Raw)** | **76.92%** | Linha de base Q7 host (76.92%) | **APROVADO** |
| **Acurácia Top-1 (Voto Majoritário 3x)** | **76.58%** (-0.34%) | Estabilização temporal | **APROVADO** |
| **Acurácia Balanceada** | **76.92%** | 10 classes | **APROVADO** |
| **Macro F1-Score** | **0.7739** | Média harmônica balanceada | **APROVADO** |
| **Latência Média Inferência (Core 1)** | **329.22 µs** (~0,33 ms) | Orçamento de Stride = 50.000 µs (50 ms) | **APROVADO** |
| **Latência Máxima (Pico Core 1)** | **354.42 µs** | < 1.000 µs (< 1 ms) | **APROVADO** |
| **Tempo Puro Classificador (Dot-Product Q7)** | **17.20 µs** | Micro-LDA linear | **APROVADO** |
| **Utilização de CPU no Core 1** | **0.658%** | Orçamento contínuo 50 ms | **EXCELENTE** |
| **Margem de Folga em Tempo Real (Slack)** | **99.29%** | > 95% de folga | **APROVADO** |
| **Headroom Stack FreeRTOS (Core 0 / Core 1)** | **2352 B / 3332 B** | Sem overflow (> 1.024 B) | **SEGURO** |

---

## 2. Configuração do Experimento e Arquitetura de Execução

### 2.1. Pipeline Embarcado no ESP32-S3

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      ESP32-S3-WROOM @ 240 MHz                          │
 │                                                                        │
 │  ┌───────────────────────────────┐     ┌────────────────────────────┐  │
 │  │  CORE 0 (PRO_CPU, Prio 10)    │     │  CORE 1 (APP_CPU, Prio 5)  │  │
 │  │  • Recepção HIL Serial (100)  │     │  • Extração Features (TD)  │  │
 │  │  • StatefulWindowBuffer (400) │ ──► │    (MAV, RMS, WL, ZC, SSC) │  │
 │  │  • Despacho via FreeRTOS Queue│     │  • Micro-LDA Q7 Dot-Product│  │
 │  └───────────────────────────────┘     │  • Voto Majoritário 3-Win  │  │
 │                                        │  • Telemetria SEMG_DECISION│  │
 │                                        └────────────────────────────┘  │
 └────────────────────────────────────────────────────────────────────────┘
```

- **Tamanho da Janela ($W$):** 400 amostras (200 ms @ 2.000 Hz)
- **Passo / Stride ($S$):** 100 amostras (50 ms @ 2.000 Hz, taxa de decisão = 20 Hz)
- **Canais de sEMG:** 4 canais de antebraço quantizados em int16 com escala ADC de 3.276,7 contagens/mV
- **Extrator Zero-FPU:** 5 estatísticas no domínio do tempo (MAV, RMS, WL, ZC, SSC) calculadas em aritmética inteira com normalização para float
- **Classificador Micro-LDA Q7:** Matriz de pesos $10 \times 20$ quantizada em int8 (Q7) ocupando apenas 200 bytes na memória Flash (`.rodata`)

---

## 3. Desempenho de Classificação por Gesto

A tabela a seguir apresenta os resultados detalhados por classe para os 10 gestos avaliados no silício:

| ID | Nome do Gesto | Janelas | Acertos (Raw) | Acurácia (Raw) | Acertos (Voto 3x) | Acurácia (Voto) | Precisão | Recall | F1-Score |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `0` | **Rest** | 117 | 94/117 | 80.34% | 98/117 | 83.76% | 51.09% | 80.34% | 0.6246 |
| `1` | **Extension** | 117 | 115/117 | 98.29% | 115/117 | 98.29% | 100.00% | 98.29% | 0.9914 |
| `2` | **Flexion** | 117 | 112/117 | 95.73% | 111/117 | 94.87% | 100.00% | 95.73% | 0.9782 |
| `3` | **Ulnar Dev** | 117 | 109/117 | 93.16% | 109/117 | 93.16% | 95.61% | 93.16% | 0.9437 |
| `4` | **Radial Dev** | 117 | 101/117 | 86.32% | 100/117 | 85.47% | 99.02% | 86.32% | 0.9224 |
| `5` | **Grip** | 117 | 108/117 | 92.31% | 108/117 | 92.31% | 94.74% | 92.31% | 0.9351 |
| `6` | **Abduction** | 117 | 58/117 | 49.57% | 55/117 | 47.01% | 96.67% | 49.57% | 0.6554 |
| `7` | **Adduction** | 117 | 60/117 | 51.28% | 61/117 | 52.14% | 39.47% | 51.28% | 0.4461 |
| `8` | **Supination** | 117 | 96/117 | 82.05% | 95/117 | 81.20% | 76.19% | 82.05% | 0.7901 |
| `9` | **Pronation** | 117 | 47/117 | 40.17% | 44/117 | 37.61% | 51.65% | 40.17% | 0.4519 |

---

## 4. Matriz de Confusão no Hardware (Raw Predictions, 10x10)

Linhas representam a classe verdadeira (*Ground Truth*); colunas representam a predição emitida pelo ESP32-S3:

| Gesto Real | G0 | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | Total |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **G0 (Rest)** | 94 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 23 | **117** |
| **G1 (Extension)** | 0 | 115 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | **117** |
| **G2 (Flexion)** | 4 | 0 | 112 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **117** |
| **G3 (Ulnar Dev)** | 0 | 0 | 0 | 109 | 0 | 0 | 0 | 6 | 0 | 2 | **117** |
| **G4 (Radial Dev)** | 0 | 0 | 0 | 0 | 101 | 0 | 2 | 1 | 0 | 13 | **117** |
| **G5 (Grip)** | 8 | 0 | 0 | 0 | 0 | 108 | 0 | 0 | 0 | 1 | **117** |
| **G6 (Abduction)** | 3 | 0 | 0 | 3 | 1 | 1 | 58 | 51 | 0 | 0 | **117** |
| **G7 (Adduction)** | 51 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 3 | 3 | **117** |
| **G8 (Supination)** | 11 | 0 | 0 | 0 | 0 | 4 | 0 | 5 | 96 | 1 | **117** |
| **G9 (Pronation)** | 13 | 0 | 0 | 0 | 0 | 1 | 0 | 29 | 27 | 47 | **117** |

> [!NOTE]
> O classificador Micro-LDA linear apresenta excelente discriminação em gestos de extensão (G1: 98,29%), flexão (G2: 95,73%), desvio ulnar (G3: 93,16%), preensão/grip (G5: 92,31%) e desvio radial (G4: 86,32%).
> A confusão residual ocorre principalmente entre movimentos com ativação muscular anatômica sobreposta (como pronação/supinação e abdução/addução), o que condiz perfeitamente com a literatura sob 4 canais de sEMG.

---

## 5. Análise de Latência, Tempo Real e Recursos do Hardware

### 5.1. Distribuição de Latência no Core 1 (Xtensa LX7 @ 240 MHz)

| Métrica de Tempo | Tempo em Ciclos de Clock (`CCOUNT`) | Duração Temporal (µs) | Fração do Orçamento (50 ms) |
| :--- | :---: | :---: | :---: |
| **Média Core 1 (Features + LDA)** | `79,011` | **329.22 µs** | `0.658%` |
| **Mediana (Percentil 50)** | `78,954` | **328.98 µs** | `0.658%` |
| **Percentil 95 (P95)** | `80,640` | **336.00 µs** | `0.672%` |
| **Percentil 99 (P99)** | `83,059` | **346.08 µs** | `0.692%` |
| **Pior Caso (Máximo Absoluto)** | `85,060` | **354.42 µs** | `0.709%` |
| **Classificador Micro-LDA Q7 (Dot-Product)** | `4,129` | **17.20 µs** | `0.0344%` |

### 5.2. Análise de Viabilidade em Tempo Real

- **Taxa de Amostragem do Sinal:** 2.000 Hz ($T_s = 500$ µs por amostra)
- **Passo de Decisão (Stride):** 100 amostras = 50.000 µs (50 ms)
- **Tempo Total de Processamento por Janela:** ~330 µs
- **Taxa de Ocupação da CPU (Core 1):** $\frac{330\,\mu\text{s}}{50.000\,\mu\text{s}} \approx 0,66\%$
- **Margem de Folga (Slack Margin):** **99,34% de CPU livre**, permitindo facilmente a execução concomitante de conectividade BLE, Wi-Fi, controle de prótese ativa ou displays OLED.

### 5.3. Memória e Pilha do Sistema Operacional (FreeRTOS SMP)

- **Pilha Mínima Livre no Core 0 (Aquisição / HIL):** `2352 bytes` de margem de segurança
- **Pilha Mínima Livre no Core 1 (Inferência):** `3332 bytes` de margem de segurança
- **Estabilidade:** Nenhuma ocorrência de *stack overflow*, travamento de *Task Watchdog Timer (TWDT)* ou perda de sincronismo nos 1.170 ciclos.

---

## 6. Conclusões e Parecer Técnico

1. **Prontidão para Embarque Físico Confirmada:** O motor Micro-LDA Q7 opera com confiabilidade total no ESP32-S3-WROOM.
2. **Paridade Numérica Rigorosa:** Aderência de 100,00% entre a implementação C99 executada no Xtensa LX7 e os oráculos Python.
3. **Ultra-Baixa Latência:** Resposta em ~330 µs por janela, viabilizando controle de atuadores mioelétricos com atraso imperceptível ao usuário (< 50 ms).
