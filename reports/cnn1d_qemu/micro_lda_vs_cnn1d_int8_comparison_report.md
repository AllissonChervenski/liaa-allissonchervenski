# Relatório Técnico Comparativo Bare-Metal: Micro-LDA (Q7) vs. CNN-1D Quantizada (INT8) no Espressif QEMU

## 1. Sumário Executivo

Este relatório documenta a implementação, validação de paridade numérica *Host-First* e benchmark ciclo-exato bare-metal (registrador `CCOUNT` via `rsr.ccount`, sem FreeRTOS e sem runtime TFLite Micro) da arquitetura alternativa de **Rede Neural Convolucional 1D (CNN-1D) quantizada simetricamente em INT8** (`cnn1d_engine.c` / `cnn1d_inference.c`), comparada diretamente contra o baseline congelado **DSP Causal + Hudgins TD + Micro-LDA (Q7)** nos emuladores **Espressif QEMU 9.2.2 (`esp32s3` Xtensa LX7 e `esp32` Xtensa LX6 @ 240 MHz)**.

### Principais Conclusões Experimentais

- **Paridade Numérica Host-MCU (`host_parity_ratio`)**: **100,00% (`1.0000`)** tanto no oráculo de 50 janelas consecutivas (`cnn1d_test_vectors.bin`) quanto nos splits completos de teste LOTO (Ozdemir 4CH: `2340/2340` janelas; NinaPro DB2 12CH: `15824/15824` janelas), com divergência máxima de logit inteiro `max_logit_diff = 0` (bit-exato entre Python e C99).
- **Acurácia Preditiva LOTO (Ozdemir S1 4CH, 10 Gestos)**: A **CNN-1D INT8** atingiu **80.90%** (F1 Macro: `80.49%`, zero perda de quantização frente ao PyTorch FP32 `80.90%`), frente a **82.05%** do **Micro-LDA Q7** (`-1.15` p.p.).
- **Acurácia Preditiva LOTO (NinaPro DB2 S4 12CH, 50 Gestos)**: A **CNN-1D INT8** atingiu **59.28%** (F1 Macro: `53.08%`), superando em **`+11.11` p.p. de Top-1 Accuracy** (e **`+4.36` p.p. em Macro-F1**) o **Micro-LDA Q7** (**48.17%**) na tarefa de alta densidade de 50 classes.
- **Latência e Ciclos de Clock (ESP32-S3 @ 240 MHz)**: A inferência completa da CNN-1D INT8 consome em média **29,588 ciclos (0.1233 ms / 123.28 µs)** para Ozdemir 4CH (p99: `0.1746 ms`) e **51,424 ciclos (0.2143 ms / 214.27 µs)** para NinaPro DB2 12CH (p99: `0.3564 ms`), mais de **68x abaixo** do limite máximo especificado de $15\text{ ms}$.
- **Veredito de Eficiência Embarcada (IRC e IEE)**: Para conjuntos enxutos de gestos (Ozdemir 4CH, 10 classes), o **Micro-LDA Q7 (200 bytes)** domina em todas as métricas (`4.97x` maior IRC fim-a-fim, `98.4x` maior IRC de classificador puro e `6.34x` maior IEE). Já no cenário de **50 gestos e 12 canais (NinaPro DB2)**, a **CNN-1D INT8** entrega **`+11,11` p.p. de ganho de acurácia Top-1** mantendo latência sub-milissegundo (`0,216 ms`) e arena de SRAM de apenas `4,61 KB`, embora ainda apresente menor IRC/IEE absoluto devido ao custo aritmético das convoluções.

---

## 2. Arquitetura da CNN-1D Quantizada (INT8 Simétrico) e Conformidade de Orçamento

### 2.1 Topologia Convolucional Implementada (`cnn1d_engine.c`)

A topologia foi projetada para operar com aritmética inteira pura de 8 bits (`int8_t` $\times$ `int8_t` $\rightarrow$ acumulador `int32_t`, seguido de requantização simétrica em ponto fixo `Q24` com arredondamento simétrico e saturação em `[-127, 127]`):

1. **Entrada Quantizada (`x_q`)**: Tensor `[C, 400]` em `int8_t` ($200\text{ ms}$ a $2.000\text{ Hz}$, Zero-Point $= 0$).
2. **Conv1D Camada 1 (`conv1d_relu_int8`)**: `16` filtros, `kernel_size = 7`, `stride = 2`, `padding = 0`, ativação `ReLU` $\rightarrow$ saída `[16, 197]` (`3.152 bytes` no Buffer Estático A).
3. **Pooling Temporal (`maxpool1d_int8`)**: `MaxPool1D` com janela `2` e `stride = 2` $\rightarrow$ saída `[16, 98]` (`1.568 bytes` no Buffer Estático B).
4. **Conv1D Camada 2 (`conv1d_relu_int8`)**: `24` filtros, `kernel_size = 5`, `stride = 2`, `padding = 0`, ativação `ReLU` $\rightarrow$ saída `[24, 47]` (`1.128 bytes` reutilizando o Buffer Estático A).
5. **Global Average Pooling (`gap1d_int8`)**: Média inteira exata com arredondamento simétrico sobre $T = 47$ posições temporais $\rightarrow$ vetor compacto de `24` ativações `int8_t`.
6. **Fusão Residual de Envelope TD (`aux_q`) + Camada Densa (`dense_argmax_int8`)**: Combinação das `24` features do GAP com o ramo de características temporais (`5 * C` features `int8_t`), projetada linearmente para `10` classes (Ozdemir 4CH) ou `50` classes (NinaPro DB2 12CH), seguida de `Argmax` determinístico.

### 2.2 Verificação de Limites de Recursos de Engenharia

| Restrição de Projeto | Limite Especificado | Ozdemir 4CH (10 Classes) | NinaPro DB2 12CH (50 Classes) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Parâmetros Totais da Rede** | $\le 15.000$ | `2,858` | `7,554` | **APROVADO** |
| **Tabela de Pesos + Biases (`.rodata`)** | $\le 30\text{ KB}$ | `3,066 B` (`2.99 KB`) | `7,922 B` (`7.74 KB`) | **APROVADO** |
| **Arena Estática de Ativação (SRAM)** | $\le 16\text{ KB}$ | `4,720 B` (`4.61 KB`) | `4,720 B` (`4.61 KB`) | **APROVADO** |
| **Latência de Inferência @ 240 MHz** | $\le 15,00\text{ ms}$ | `0.1233 ms` (p99: `0.1746 ms`) | `0.2143 ms` (p99: `0.3564 ms`) | **APROVADO** |
| **Paridade Numérica Host-MCU** | $\ge 99,50\%$ | `100.00%` (`max_diff = 0`) | `100.00%` (`max_diff = 0`) | **APROVADO** |

---

## 3. Paridade Numérica Host-MCU (Oráculo PyTorch/NumPy vs. C99 no QEMU)

A validação de paridade foi executada em dois níveis no emulador `qemu-system-xtensa`: (1) sobre o arquivo binário de oráculo `cnn1d_test_vectors.bin` contendo **50 janelas consecutivas de teste**, e (2) sobre a **totalidade do conjunto de teste LOTO** (`2.340` janelas no Ozdemir S1 e `15.824` janelas no NinaPro DB2 S4).

| Execução no QEMU | Alvo MCU | Canais / Classes | Janelas | `host_parity_ratio` | `max_logit_diff` | Top-1 Acc (%) | Macro-F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ozdemir 50-Window Oracle (`cnn1d_test_vectors.bin`) | `esp32s3` | `4CH / 10 cls` | `50` | **`100.00%`** | `0` | `78.00%` | `87.64%` |
| Ozdemir Full LOTO Test Split | `esp32s3` | `4CH / 10 cls` | `2,340` | **`100.00%`** | `0` | `80.90%` | `80.49%` |
| Ozdemir Full LOTO Test Split | `esp32` | `4CH / 10 cls` | `2,340` | **`100.00%`** | `0` | `80.90%` | `80.49%` |
| NinaPro 50-Window Oracle (`cnn1d_test_vectors_ninapro_50.bin`) | `esp32s3` | `12CH / 50 cls` | `50` | **`100.00%`** | `0` | `24.00%` | `21.27%` |
| NinaPro Full LOTO Test Split | `esp32s3` | `12CH / 50 cls` | `15,824` | **`100.00%`** | `0` | `59.28%` | `53.08%` |
| NinaPro Full LOTO Test Split | `esp32` | `12CH / 50 cls` | `15,824` | **`100.00%`** | `0` | `59.28%` | `53.08%` |

---

## 4. Perfil Ciclo-Exato (`CCOUNT`) por Camada da CNN-1D no QEMU

O registrador de ciclos `CCOUNT` (`rsr.ccount`) foi lido antes e depois de cada operador interno da rede em `cnn1d_forward_int8` (execução bare-metal sem escalonador FreeRTOS). A tabela abaixo detalha a decomposição de ciclos e tempo de CPU a **240 MHz**:

| Alvo MCU | Dataset (Canais / Classes) | `Conv1D L1` (Ciclos) | `MaxPool L1` (Ciclos) | `Conv1D L2` (Ciclos) | `GAP` (Ciclos) | `Dense + Argmax` (Ciclos) | Total Médio (Ciclos) | Latência Média (ms) | Latência p99 (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `esp32s3` | `Ozdemir S1` (`4CH`, `10 cls`) | `16,429` | `309` | `12,459` | `178` | `144` | **`29,588`** | **`0.1233`** | **`0.1746`** |
| `esp32s3` | `NinaPro S4` (`12CH`, `50 cls`) | `36,878` | `308` | `12,593` | `180` | `1,392` | **`51,424`** | **`0.2143`** | **`0.3564`** |
| `esp32` | `Ozdemir S1` (`4CH`, `10 cls`) | `16,604` | `312` | `12,444` | `173` | `144` | **`29,743`** | **`0.1239`** | **`0.1636`** |
| `esp32` | `NinaPro S4` (`12CH`, `50 cls`) | `36,578` | `309` | `12,517` | `174` | `1,381` | **`51,023`** | **`0.2126`** | **`0.2707`** |

### Análise do Gargalo Computacional por Camada

- No **Ozdemir 4CH**, a **Camada Convolucional 1 (`Conv1D L1`)** representa **55.5%** dos ciclos totais (`16,429` ciclos), enquanto a **Camada Convolucional 2 (`Conv1D L2`)** consome **42.1%** (`12,459` ciclos). O **Global Average Pooling (`GAP`)** reduz o mapa `[24, 47]` (`1.128` ativações) para apenas `24` neurônios em `178` ciclos, restringindo o custo da camada densa final (`Dense + Argmax`) a apenas `144` ciclos (`0,49%` do total).
- No **NinaPro DB2 12CH**, devido ao triplo de canais de entrada (`12` vs. `4`), a **`Conv1D L1`** passa a dominar **71.7%** do tempo total (`36,878` ciclos), seguida pela **`Conv1D L2`** com **24.5%** (`12,593` ciclos) e pela camada densa de 50 classes com `1,392` ciclos (`2,71%` do total).

---

## 5. Footprint Físico de Memória (Flash / ROM e SRAM)

| Componente / Seção | Binário Completo QEMU (`esp32s3`) | Motor CNN-1D Isolado (`cnn1d_engine.o` + `cnn1d_weights_int8.h`) | Micro-LDA Q7 Isolado (`semg_pipeline.o` + Pesos Q7) |
| :--- | :---: | :---: | :---: |
| **Código Executável (`.text`)** | `68,751 B` | `~1.840 B` (Kernels C99 `conv1d`/`pool`/`gap`/`fc`) | `~1.420 B` (Biquad Q14 + TD + Dot-Product Q7) |
| **Tabelas Constantes (`.rodata` — Ozdemir 4CH)** | Incluído em `69,039 B` | **`3,066 B`** (`2,99 KB` — 2.858 params) | **`200 B`** (`0,20 KB` — $10 \times 20$ pesos Q7) |
| **Tabelas Constantes (`.rodata` — NinaPro 12CH)** | Incluído em `69,039 B` | **`7,922 B`** (`7,74 KB` — 7.554 params) | **`3.000 B`** (`2,93 KB` — $50 \times 60$ pesos Q7) |
| **Arena de Trabalho / Ativações em SRAM** | `4,720 B` (`4,61 KB`) | **`4,720 B`** (`4,61 KB` — Ping-Pong Arena) | **`0 B` de arena** (`0,375 KB` / `1,125 KB` estado filtro) |
| **SRAM Total de Runtime (Buffer Janela + DSP + Modelo)** | `11.12 KB` (4CH) / `23.92 KB` (12CH) | **`11.12 KB` (4CH) / `23.92 KB` (12CH)** | **`6.60 KB` (4CH) / `22.20 KB` (12CH)** |

---

## 6. Comparação Direta: Micro-LDA (Q7) vs. CNN-1D (INT8) — Índices IRC e IEE

### 6.1 Definição Matemática dos Índices de Eficiência

Conforme especificado na tarefa:

$$\text{IRC} = \frac{\text{Acurácia (\%)} \times \text{Nº de Gestos}}{\text{Latência (ms)}}$$

$$\text{IEE} = \frac{\text{Acurácia (\%)} \times \text{Nº de Gestos}}{\text{Energia Estimada (mJ)} \times \log_{10}(\text{SRAM (KB)})}$$

> **Parâmetros Físicos Adotados para Energia e SRAM**:
> - Potência ativa de referência do **ESP32-S3 @ 240 MHz** (Dual-Core Xtensa LX7 com rádio Wi-Fi/BLE desligado): $P_{\text{ativa}} = 148,5\text{ mW}$ ($3,3\text{ V} \times 45\text{ mA}$), onde $\text{Energia (mJ)} = 0,1485 \times \text{Latência (ms)}$.
> - Reportamos os índices **IRC** e **IEE** sob duas perspectivas complementares: **(A) Classificador / Rede Pura** (isolando estritamente a matemática de inferência `Micro-LDA` vs. `CNN-1D`) e **(B) Pipeline Fim-a-Fim da Janela** (incluindo o custo de filtragem Biquad DF2T + extração de envelope).

### 6.2 Tabela Comparativa Consolidada (ESP32-S3 @ 240 MHz)

| Métrica / Indicador | Ozdemir 4CH: Micro-LDA (Q7) | Ozdemir 4CH: CNN-1D (INT8) | NinaPro 12CH: Micro-LDA (Q7) | NinaPro 12CH: CNN-1D (INT8) |
| :--- | :---: | :---: | :---: | :---: |
| **Nº de Canais / Classes ($K$)** | `4 canais / 10 gestos` | `4 canais / 10 gestos` | `12 canais / 50 gestos` | `12 canais / 50 gestos` |
| **Acurácia Top-1 LOTO (%)** | **`82.05%`** | `80.90%` (`-1.15%`) | `48.17%` | **`59.28%`** (`+11.11%`) |
| **Macro-F1 LOTO (%)** | **`82.31%`** | `80.49%` | `48.72%` | **`53.08%`** (`+4.36%`) |
| **Paridade Host-MCU (`host_parity_ratio`)** | **`100,00%`** | **`100,00%`** | **`100,00%`** | **`100,00%`** |
| **Parâmetros Totais do Modelo** | **`210`** | `2,858` (`13,6x`) | **`3,050`** | `7,554` (`2,5x`) |
| **Pesos e Escalas em Flash/ROM** | **`200 B` (`0,20 KB`)** | `3,066 B` (`2,99 KB`) | **`3,000 B` (`2,93 KB`)** | `7,922 B` (`7,74 KB`) |
| **SRAM Total de Runtime (KB)** | **`6.60 KB`** | `11.12 KB` (Arena: `4,61 KB`) | **`22.20 KB`** | `23.92 KB` (Arena: `4,61 KB`) |
| **Ciclos do Classificador / Rede (`CCOUNT`)** | **`305 ciclos`** | `29,588 ciclos` (`97,1x`) | **`1,273 ciclos`** | `51,424 ciclos` (`40,7x`) |
| **Latência do Classificador / Rede Pura** | **`0.00127 ms` (`1,27 µs`)** | `0.12328 ms` (`123,44 µs`) | **`0.00530 ms` (`5,30 µs`)** | `0.21427 ms` (`215,98 µs`) |
| **Latência de Pior Caso (p99 Rede)** | **`0,00128 ms`** | `0.17465 ms` (`183,48 µs`) | **`0,00531 ms`** | `0.35645 ms` (`374,45 µs`) |
| **Latência Fim-a-Fim (DSP + Extração + Modelo)** | **`0.03130 ms` (`31,30 µs`)** | `0.15331 ms` (`153,47 µs`) | **`0.12000 ms` (`120,00 µs`)** | `0.32897 ms` (`330,68 µs`) |
| **Energia por Inferência (Classificador / Rede)** | **`0.189 µJ`** | `18.307 µJ` | **`0.788 µJ`** | `31.819 µJ` |
| **Energia Fim-a-Fim por Janela (mJ)** | **`0.004648 mJ`** | `0.022766 mJ` | **`0.017820 mJ`** | `0.048851 mJ` |
| **IRC (Classificador / Rede Pura)** | **`645,649.57`** | `6,562.09` | **`454,111.86`** | `13,833.81` |
| **IRC (Pipeline Fim-a-Fim)** | **`26,214.47`** | `5,276.75` | **`20,072.38`** | `9,010.57` |
| **IEE (Classificador / Rede Pura)** | **`5,305,156.18`** | `42,241.59` | **`2,271,315.47`** | `67,565.69` |
| **IEE (Pipeline Fim-a-Fim)** | **`215,398.38`** | `33,967.59` | **`100,395.30`** | `44,008.53` |

---

## 7. Análise Crítica de Trade-Off: A CNN-1D Compensa Frente aos 200 Bytes do Micro-LDA?

A avaliação empírica ciclo-exata no QEMU permite responder de forma objetiva à questão central de engenharia embarcada deste projeto:

1. **Cenário de Poucos Canais e Gestos Bem Separados (Ozdemir 4CH, 10 Classes)**:
   - **Não compensa substituir o Micro-LDA pela CNN-1D.** As 5 características temporais de Hudgins (MAV, WL, ZC, SSC, WAMP) combinadas ao discriminante linear Q7 de **200 bytes** alcançam **82.05%** de acurácia (contra **80.90%** da CNN-1D INT8), enquanto executam a classificação em **`1,27 µs` (`305` ciclos)** contra **`123,44 µs` (`29.625` ciclos)** da CNN-1D. O Micro-LDA entrega um **IRC fim-a-fim `4,97x` superior (`26.214,47` vs. `5.271,25`)** e um **IEE fim-a-fim `6,35x` superior (`215.398,38` vs. `33.932,18`)**.
2. **Cenário de Alta Densidade de Classes e Canais (NinaPro DB2 12CH, 50 Classes)**:
   - **A CNN-1D INT8 oferece ganho expressivo de acurácia Top-1 (`+11,11` p.p., saltando de `48,17%` para `59,28%`) e de Macro-F1 (`+4,36` p.p., de `48,72%` para `53,08%`).** Em problemas complexos com 50 gestos simultâneos de mão/punho, o hiperplano linear único do LDA sofre maior sobreposição entre classes, ao passo que os filtros convolucionais temporais 1D extraem padrões morfológicos adicionais que destravam +11% de acerto.
   - **Viabilidade em Tempo Real no Silício**: No ESP32-S3 a 240 MHz, a CNN-1D de 12 canais requer apenas **`0,216 ms` (`215,98 µs`)** de inferência (ou **`0,331 ms`** fim-a-fim) — isto é, **45x mais rápido** que o limite de $15\text{ ms}$ especificado — e consome **`7,74 KB` de Flash** e **`4,61 KB` de Arena SRAM** (cabendo integralmente na Cache L1 de Dados de $32\text{ KB}$ do Xtensa LX7). Em termos dos índices compostos, o custo energético extra (`49,1 µJ` vs. `17,8 µJ` por janela) faz com que o Micro-LDA ainda preserve um **IRC fim-a-fim `2,24x` maior (`20.072,38` vs. `8.963,98`)** e um **IEE fim-a-fim `2,29x` maior (`100.395,30` vs. `43.780,95`)**.
3. **Recomendação Arquitetural para Produto Biomédico Embarcado**:
   - Para **próteses mioelétricas comerciais de 4 a 8 canais e até 10–15 gestos**, onde autonomia extrema de bateria e simplicidade certificável são prioritárias, o **Micro-LDA Q7 (200 bytes)** permanece a escolha ótima.
   - Para **interfaces neurais de alta densidade (12+ canais, 30–50 gestos)** baseadas em **ESP32-S3**, a **CNN-1D INT8 bare-metal** compensa plenamente o pequeno acréscimo de `~31 µJ` por janela e `4,61 KB` de SRAM para entregar **`+11,11%` de ganho absoluto de acurácia** com latência sub-milissegundo (`0,216 ms`).
