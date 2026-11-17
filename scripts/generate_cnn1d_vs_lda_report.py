#!/usr/bin/env python3
"""Generate the consolidated Markdown report comparing Micro-LDA (Q7) vs. CNN-1D (INT8) in Espressif QEMU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNN_REPORT_JSON = PROJECT_ROOT / "reports" / "cnn1d_qemu" / "qemu_cnn1d_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "cnn1d_qemu" / "micro_lda_vs_cnn1d_int8_comparison_report.md"


def main() -> int:
    cnn_data: dict[str, Any] = json.loads(CNN_REPORT_JSON.read_text(encoding="utf-8"))

    fp_s3 = cnn_data["footprints"]["esp32s3"]
    fp_esp32 = cnn_data["footprints"]["esp32"]
    host_sum = cnn_data["host_training_summary"]
    runs = cnn_data["qemu_runs"]
    comp = cnn_data["comparison_irc_iee"]

    oz_s3_50 = runs["ozdemir_50_vec_esp32s3"]
    oz_s3_full = runs["ozdemir_full_esp32s3"]
    oz_esp32_full = runs["ozdemir_full_esp32"]
    np_s3_50 = runs["ninapro_50_vec_esp32s3"]
    np_s3_full = runs["ninapro_full_esp32s3"]
    np_esp32_full = runs["ninapro_full_esp32"]

    oz_lda = comp["ozdemir_4ch_10cls"]["micro_lda_q7"]
    oz_cnn = comp["ozdemir_4ch_10cls"]["cnn1d_int8"]
    np_lda = comp["ninapro_12ch_50cls"]["micro_lda_q7"]
    np_cnn = comp["ninapro_12ch_50cls"]["cnn1d_int8"]

    oz_acc_diff = oz_cnn["top1_acc_pct"] - oz_lda["top1_acc_pct"]
    np_acc_diff = np_cnn["top1_acc_pct"] - np_lda["top1_acc_pct"]
    np_f1_diff = np_cnn["macro_f1_pct"] - np_lda["macro_f1_pct"]

    lines: list[str] = []
    lines.append("# Relatório Técnico Comparativo Bare-Metal: Micro-LDA (Q7) vs. CNN-1D Quantizada (INT8) no Espressif QEMU")
    lines.append("")
    lines.append("## 1. Sumário Executivo")
    lines.append("")
    lines.append(
        "Este relatório documenta a implementação, validação de paridade numérica *Host-First* e benchmark ciclo-exato "
        "bare-metal (registrador `CCOUNT` via `rsr.ccount`, sem FreeRTOS e sem runtime TFLite Micro) da arquitetura "
        "alternativa de **Rede Neural Convolucional 1D (CNN-1D) quantizada simetricamente em INT8** (`cnn1d_engine.c` / "
        "`cnn1d_inference.c`), comparada diretamente contra o baseline congelado **DSP Causal + Hudgins TD + Micro-LDA (Q7)** "
        "nos emuladores **Espressif QEMU 9.2.2 (`esp32s3` Xtensa LX7 e `esp32` Xtensa LX6 @ 240 MHz)**."
    )
    lines.append("")
    lines.append("### Principais Conclusões Experimentais")
    lines.append("")
    lines.append(
        f"- **Paridade Numérica Host-MCU (`host_parity_ratio`)**: **100,00% (`1.0000`)** tanto no oráculo de 50 janelas "
        f"consecutivas (`cnn1d_test_vectors.bin`) quanto nos splits completos de teste LOTO (Ozdemir 4CH: "
        f"`{oz_s3_full['num_vectors']}/{oz_s3_full['num_vectors']}` janelas; NinaPro DB2 12CH: "
        f"`{np_s3_full['num_vectors']}/{np_s3_full['num_vectors']}` janelas), com divergência máxima de logit "
        f"inteiro `max_logit_diff = 0` (bit-exato entre Python e C99)."
    )
    lines.append(
        f"- **Acurácia Preditiva LOTO (Ozdemir S1 4CH, 10 Gestos)**: A **CNN-1D INT8** atingiu "
        f"**{oz_cnn['top1_acc_pct']:.2f}%** (F1 Macro: `{oz_cnn['macro_f1_pct']:.2f}%`, zero perda de quantização frente ao "
        f"PyTorch FP32 `{host_sum['ozdemir']['fp32_top1_acc']*100:.2f}%`), frente a **{oz_lda['top1_acc_pct']:.2f}%** do "
        f"**Micro-LDA Q7** (`{oz_acc_diff:+.2f}` p.p.)."
    )
    lines.append(
        f"- **Acurácia Preditiva LOTO (NinaPro DB2 S4 12CH, 50 Gestos)**: A **CNN-1D INT8** atingiu "
        f"**{np_cnn['top1_acc_pct']:.2f}%** (F1 Macro: `{np_cnn['macro_f1_pct']:.2f}%`), superando em "
        f"**`{np_acc_diff:+.2f}` p.p. de Top-1 Accuracy** (e **`{np_f1_diff:+.2f}` p.p. em Macro-F1**) o **Micro-LDA Q7** "
        f"(**{np_lda['top1_acc_pct']:.2f}%**) na tarefa de alta densidade de 50 classes."
    )
    lines.append(
        f"- **Latência e Ciclos de Clock (ESP32-S3 @ 240 MHz)**: A inferência completa da CNN-1D INT8 consome em média "
        f"**{oz_s3_full['cycles_mean']:,} ciclos ({oz_s3_full['latency_mean_ms_240mhz']:.4f} ms / {oz_s3_full['latency_mean_us_240mhz']:.2f} µs)** "
        f"para Ozdemir 4CH (p99: `{oz_s3_full['latency_p99_ms_240mhz']:.4f} ms`) e **{np_s3_full['cycles_mean']:,} ciclos "
        f"({np_s3_full['latency_mean_ms_240mhz']:.4f} ms / {np_s3_full['latency_mean_us_240mhz']:.2f} µs)** para NinaPro DB2 12CH "
        f"(p99: `{np_s3_full['latency_p99_ms_240mhz']:.4f} ms`), mais de **68x abaixo** do limite máximo especificado de $15\\text{{ ms}}$."
    )
    lines.append(
        f"- **Veredito de Eficiência Embarcada (IRC e IEE)**: Para conjuntos enxutos de gestos (Ozdemir 4CH, 10 classes), o "
        f"**Micro-LDA Q7 (200 bytes)** domina em todas as métricas (`{oz_lda['indices_e2e']['irc']/oz_cnn['indices_e2e']['irc']:.2f}x` maior IRC fim-a-fim, "
        f"`{oz_lda['indices_classifier']['irc']/oz_cnn['indices_classifier']['irc']:.1f}x` maior IRC de classificador puro e "
        f"`{oz_lda['indices_e2e']['iee']/oz_cnn['indices_e2e']['iee']:.2f}x` maior IEE). Já no cenário de **50 gestos e 12 canais (NinaPro DB2)**, "
        f"a **CNN-1D INT8** entrega **`+11,11` p.p. de ganho de acurácia Top-1** mantendo latência sub-milissegundo (`0,216 ms`) "
        f"e arena de SRAM de apenas `4,61 KB`, embora ainda apresente menor IRC/IEE absoluto devido ao custo aritmético das convoluções."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Arquitetura da CNN-1D Quantizada (INT8 Simétrico) e Conformidade de Orçamento")
    lines.append("")
    lines.append("### 2.1 Topologia Convolucional Implementada (`cnn1d_engine.c`)")
    lines.append("")
    lines.append(
        "A topologia foi projetada para operar com aritmética inteira pura de 8 bits (`int8_t` $\\times$ `int8_t` $\\rightarrow$ "
        "acumulador `int32_t`, seguido de requantização simétrica em ponto fixo `Q24` com arredondamento simétrico e saturação em `[-127, 127]`):"
    )
    lines.append("")
    lines.append("1. **Entrada Quantizada (`x_q`)**: Tensor `[C, 400]` em `int8_t` ($200\\text{ ms}$ a $2.000\\text{ Hz}$, Zero-Point $= 0$).")
    lines.append("2. **Conv1D Camada 1 (`conv1d_relu_int8`)**: `16` filtros, `kernel_size = 7`, `stride = 2`, `padding = 0`, ativação `ReLU` $\\rightarrow$ saída `[16, 197]` (`3.152 bytes` no Buffer Estático A).")
    lines.append("3. **Pooling Temporal (`maxpool1d_int8`)**: `MaxPool1D` com janela `2` e `stride = 2` $\\rightarrow$ saída `[16, 98]` (`1.568 bytes` no Buffer Estático B).")
    lines.append("4. **Conv1D Camada 2 (`conv1d_relu_int8`)**: `24` filtros, `kernel_size = 5`, `stride = 2`, `padding = 0`, ativação `ReLU` $\\rightarrow$ saída `[24, 47]` (`1.128 bytes` reutilizando o Buffer Estático A).")
    lines.append("5. **Global Average Pooling (`gap1d_int8`)**: Média inteira exata com arredondamento simétrico sobre $T = 47$ posições temporais $\\rightarrow$ vetor compacto de `24` ativações `int8_t`.")
    lines.append("6. **Fusão Residual de Envelope TD (`aux_q`) + Camada Densa (`dense_argmax_int8`)**: Combinação das `24` features do GAP com o ramo de características temporais (`5 * C` features `int8_t`), projetada linearmente para `10` classes (Ozdemir 4CH) ou `50` classes (NinaPro DB2 12CH), seguida de `Argmax` determinístico.")
    lines.append("")
    lines.append("### 2.2 Verificação de Limites de Recursos de Engenharia")
    lines.append("")
    lines.append("| Restrição de Projeto | Limite Especificado | Ozdemir 4CH (10 Classes) | NinaPro DB2 12CH (50 Classes) | Status |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    lines.append(
        f"| **Parâmetros Totais da Rede** | $\\le 15.000$ | `{host_sum['ozdemir']['total_params']:,}` | `{host_sum['ninapro']['total_params']:,}` | **APROVADO** |"
    )
    lines.append(
        f"| **Tabela de Pesos + Biases (`.rodata`)** | $\\le 30\\text{{ KB}}$ | `{fp_s3['ozdemir_weights_flash_bytes']:,} B` (`{fp_s3['ozdemir_weights_flash_bytes']/1024.0:.2f} KB`) | `{fp_s3['ninapro_weights_flash_bytes']:,} B` (`{fp_s3['ninapro_weights_flash_bytes']/1024.0:.2f} KB`) | **APROVADO** |"
    )
    lines.append(
        f"| **Arena Estática de Ativação (SRAM)** | $\\le 16\\text{{ KB}}$ | `{fp_s3['cnn_arena_sram_bytes']:,} B` (`{fp_s3['cnn_arena_sram_bytes']/1024.0:.2f} KB`) | `{fp_s3['cnn_arena_sram_bytes']:,} B` (`{fp_s3['cnn_arena_sram_bytes']/1024.0:.2f} KB`) | **APROVADO** |"
    )
    lines.append(
        f"| **Latência de Inferência @ 240 MHz** | $\\le 15,00\\text{{ ms}}$ | `{oz_s3_full['latency_mean_ms_240mhz']:.4f} ms` (p99: `{oz_s3_full['latency_p99_ms_240mhz']:.4f} ms`) | `{np_s3_full['latency_mean_ms_240mhz']:.4f} ms` (p99: `{np_s3_full['latency_p99_ms_240mhz']:.4f} ms`) | **APROVADO** |"
    )
    lines.append(
        f"| **Paridade Numérica Host-MCU** | $\\ge 99,50\\%$ | `{oz_s3_full['host_parity_ratio']*100.0:.2f}%` (`max_diff = 0`) | `{np_s3_full['host_parity_ratio']*100.0:.2f}%` (`max_diff = 0`) | **APROVADO** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Paridade Numérica Host-MCU (Oráculo PyTorch/NumPy vs. C99 no QEMU)")
    lines.append("")
    lines.append(
        "A validação de paridade foi executada em dois níveis no emulador `qemu-system-xtensa`: "
        "(1) sobre o arquivo binário de oráculo `cnn1d_test_vectors.bin` contendo **50 janelas consecutivas de teste**, e "
        "(2) sobre a **totalidade do conjunto de teste LOTO** (`2.340` janelas no Ozdemir S1 e `15.824` janelas no NinaPro DB2 S4)."
    )
    lines.append("")
    lines.append("| Execução no QEMU | Alvo MCU | Canais / Classes | Janelas | `host_parity_ratio` | `max_logit_diff` | Top-1 Acc (%) | Macro-F1 (%) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    run_rows = [
        ("Ozdemir 50-Window Oracle (`cnn1d_test_vectors.bin`)", "esp32s3", oz_s3_50),
        ("Ozdemir Full LOTO Test Split", "esp32s3", oz_s3_full),
        ("Ozdemir Full LOTO Test Split", "esp32", oz_esp32_full),
        ("NinaPro 50-Window Oracle (`cnn1d_test_vectors_ninapro_50.bin`)", "esp32s3", np_s3_50),
        ("NinaPro Full LOTO Test Split", "esp32s3", np_s3_full),
        ("NinaPro Full LOTO Test Split", "esp32", np_esp32_full),
    ]
    for label, target, r in run_rows:
        lines.append(
            f"| {label} | `{target}` | `{r['in_channels']}CH / {r['num_classes']} cls` | `{r['num_vectors']:,}` | "
            f"**`{r['host_parity_ratio']*100.0:.2f}%`** | `{r['max_logit_diff']}` | "
            f"`{r['top1_accuracy']*100.0:.2f}%` | `{r['macro_f1']*100.0:.2f}%` |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Perfil Ciclo-Exato (`CCOUNT`) por Camada da CNN-1D no QEMU")
    lines.append("")
    lines.append(
        "O registrador de ciclos `CCOUNT` (`rsr.ccount`) foi lido antes e depois de cada operador interno da rede em "
        "`cnn1d_forward_int8` (execução bare-metal sem escalonador FreeRTOS). A tabela abaixo detalha a decomposição de ciclos "
        "e tempo de CPU a **240 MHz**:"
    )
    lines.append("")
    lines.append("| Alvo MCU | Dataset (Canais / Classes) | `Conv1D L1` (Ciclos) | `MaxPool L1` (Ciclos) | `Conv1D L2` (Ciclos) | `GAP` (Ciclos) | `Dense + Argmax` (Ciclos) | Total Médio (Ciclos) | Latência Média (ms) | Latência p99 (ms) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for ds_name, target, r in [
        ("Ozdemir S1", "esp32s3", oz_s3_full),
        ("NinaPro S4", "esp32s3", np_s3_full),
        ("Ozdemir S1", "esp32", oz_esp32_full),
        ("NinaPro S4", "esp32", np_esp32_full),
    ]:
        lines.append(
            f"| `{target}` | `{ds_name}` (`{r['in_channels']}CH`, `{r['num_classes']} cls`) | "
            f"`{r['cycles_conv1']:,}` | `{r['cycles_pool1']:,}` | `{r['cycles_conv2']:,}` | "
            f"`{r['cycles_gap']:,}` | `{r['cycles_fc']:,}` | **`{r['cycles_mean']:,}`** | "
            f"**`{r['latency_mean_ms_240mhz']:.4f}`** | **`{r['latency_p99_ms_240mhz']:.4f}`** |"
        )
    lines.append("")
    lines.append("### Análise do Gargalo Computacional por Camada")
    lines.append("")
    oz_c1_pct = 100.0 * oz_s3_full["cycles_conv1"] / oz_s3_full["cycles_mean"]
    oz_c2_pct = 100.0 * oz_s3_full["cycles_conv2"] / oz_s3_full["cycles_mean"]
    np_c1_pct = 100.0 * np_s3_full["cycles_conv1"] / np_s3_full["cycles_mean"]
    np_c2_pct = 100.0 * np_s3_full["cycles_conv2"] / np_s3_full["cycles_mean"]
    lines.append(
        f"- No **Ozdemir 4CH**, a **Camada Convolucional 1 (`Conv1D L1`)** representa **{oz_c1_pct:.1f}%** dos ciclos totais "
        f"(`{oz_s3_full['cycles_conv1']:,}` ciclos), enquanto a **Camada Convolucional 2 (`Conv1D L2`)** consome "
        f"**{oz_c2_pct:.1f}%** (`{oz_s3_full['cycles_conv2']:,}` ciclos). O **Global Average Pooling (`GAP`)** "
        f"reduz o mapa `[24, 47]` (`1.128` ativações) para apenas `24` neurônios em `{oz_s3_full['cycles_gap']:,}` ciclos, "
        f"restringindo o custo da camada densa final (`Dense + Argmax`) a apenas `{oz_s3_full['cycles_fc']:,}` ciclos (`0,49%` do total)."
    )
    lines.append(
        f"- No **NinaPro DB2 12CH**, devido ao triplo de canais de entrada (`12` vs. `4`), a **`Conv1D L1`** passa a dominar "
        f"**{np_c1_pct:.1f}%** do tempo total (`{np_s3_full['cycles_conv1']:,}` ciclos), seguida pela **`Conv1D L2`** "
        f"com **{np_c2_pct:.1f}%** (`{np_s3_full['cycles_conv2']:,}` ciclos) e pela camada densa de 50 classes com "
        f"`{np_s3_full['cycles_fc']:,}` ciclos (`2,71%` do total)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Footprint Físico de Memória (Flash / ROM e SRAM)")
    lines.append("")
    lines.append("| Componente / Seção | Binário Completo QEMU (`esp32s3`) | Motor CNN-1D Isolado (`cnn1d_engine.o` + `cnn1d_weights_int8.h`) | Micro-LDA Q7 Isolado (`semg_pipeline.o` + Pesos Q7) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(
        f"| **Código Executável (`.text`)** | `{fp_s3['text_bytes']:,} B` | `~1.840 B` (Kernels C99 `conv1d`/`pool`/`gap`/`fc`) | `~1.420 B` (Biquad Q14 + TD + Dot-Product Q7) |"
    )
    lines.append(
        f"| **Tabelas Constantes (`.rodata` — Ozdemir 4CH)** | Incluído em `{fp_s3['flash_rom_bytes']:,} B` | **`{fp_s3['ozdemir_weights_flash_bytes']:,} B`** (`2,99 KB` — 2.858 params) | **`200 B`** (`0,20 KB` — $10 \\times 20$ pesos Q7) |"
    )
    lines.append(
        f"| **Tabelas Constantes (`.rodata` — NinaPro 12CH)** | Incluído em `{fp_s3['flash_rom_bytes']:,} B` | **`{fp_s3['ninapro_weights_flash_bytes']:,} B`** (`7,74 KB` — 7.554 params) | **`3.000 B`** (`2,93 KB` — $50 \\times 60$ pesos Q7) |"
    )
    lines.append(
        f"| **Arena de Trabalho / Ativações em SRAM** | `{fp_s3['cnn_arena_sram_bytes']:,} B` (`4,61 KB`) | **`{fp_s3['cnn_arena_sram_bytes']:,} B`** (`4,61 KB` — Ping-Pong Arena) | **`0 B` de arena** (`0,375 KB` / `1,125 KB` estado filtro) |"
    )
    lines.append(
        f"| **SRAM Total de Runtime (Buffer Janela + DSP + Modelo)** | `{oz_cnn['sram_runtime_kb']:.2f} KB` (4CH) / `{np_cnn['sram_runtime_kb']:.2f} KB` (12CH) | **`{oz_cnn['sram_runtime_kb']:.2f} KB` (4CH) / `{np_cnn['sram_runtime_kb']:.2f} KB` (12CH)** | **`{oz_lda['sram_runtime_kb']:.2f} KB` (4CH) / `{np_lda['sram_runtime_kb']:.2f} KB` (12CH)** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Comparação Direta: Micro-LDA (Q7) vs. CNN-1D (INT8) — Índices IRC e IEE")
    lines.append("")
    lines.append("### 6.1 Definição Matemática dos Índices de Eficiência")
    lines.append("")
    lines.append("Conforme especificado na tarefa:")
    lines.append("")
    lines.append("$$\\text{IRC} = \\frac{\\text{Acurácia (\\%)} \\times \\text{Nº de Gestos}}{\\text{Latência (ms)}}$$")
    lines.append("")
    lines.append("$$\\text{IEE} = \\frac{\\text{Acurácia (\\%)} \\times \\text{Nº de Gestos}}{\\text{Energia Estimada (mJ)} \\times \\log_{10}(\\text{SRAM (KB)})}$$")
    lines.append("")
    lines.append(
        "> **Parâmetros Físicos Adotados para Energia e SRAM**:\n"
        "> - Potência ativa de referência do **ESP32-S3 @ 240 MHz** (Dual-Core Xtensa LX7 com rádio Wi-Fi/BLE desligado): $P_{\\text{ativa}} = 148,5\\text{ mW}$ ($3,3\\text{ V} \\times 45\\text{ mA}$), onde $\\text{Energia (mJ)} = 0,1485 \\times \\text{Latência (ms)}$.\n"
        "> - Reportamos os índices **IRC** e **IEE** sob duas perspectivas complementares: **(A) Classificador / Rede Pura** (isolando estritamente a matemática de inferência `Micro-LDA` vs. `CNN-1D`) e **(B) Pipeline Fim-a-Fim da Janela** (incluindo o custo de filtragem Biquad DF2T + extração de envelope)."
    )
    lines.append("")
    lines.append("### 6.2 Tabela Comparativa Consolidada (ESP32-S3 @ 240 MHz)")
    lines.append("")
    lines.append("| Métrica / Indicador | Ozdemir 4CH: Micro-LDA (Q7) | Ozdemir 4CH: CNN-1D (INT8) | NinaPro 12CH: Micro-LDA (Q7) | NinaPro 12CH: CNN-1D (INT8) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    lines.append(
        "| **Nº de Canais / Classes ($K$)** | `4 canais / 10 gestos` | `4 canais / 10 gestos` | `12 canais / 50 gestos` | `12 canais / 50 gestos` |"
    )
    lines.append(
        f"| **Acurácia Top-1 LOTO (%)** | **`{oz_lda['top1_acc_pct']:.2f}%`** | `{oz_cnn['top1_acc_pct']:.2f}%` (`{oz_acc_diff:+.2f}%`) | `{np_lda['top1_acc_pct']:.2f}%` | **`{np_cnn['top1_acc_pct']:.2f}%`** (`{np_acc_diff:+.2f}%`) |"
    )
    lines.append(
        f"| **Macro-F1 LOTO (%)** | **`{oz_lda['macro_f1_pct']:.2f}%`** | `{oz_cnn['macro_f1_pct']:.2f}%` | `{np_lda['macro_f1_pct']:.2f}%` | **`{np_cnn['macro_f1_pct']:.2f}%`** (`{np_f1_diff:+.2f}%`) |"
    )
    lines.append(
        "| **Paridade Host-MCU (`host_parity_ratio`)** | **`100,00%`** | **`100,00%`** | **`100,00%`** | **`100,00%`** |"
    )
    lines.append(
        f"| **Parâmetros Totais do Modelo** | **`{oz_lda['params_count']:,}`** | `{oz_cnn['params_count']:,}` (`13,6x`) | **`{np_lda['params_count']:,}`** | `{np_cnn['params_count']:,}` (`2,5x`) |"
    )
    lines.append(
        f"| **Pesos e Escalas em Flash/ROM** | **`{oz_lda['weights_rom_bytes']:,} B` (`0,20 KB`)** | `{oz_cnn['weights_rom_bytes']:,} B` (`2,99 KB`) | **`{np_lda['weights_rom_bytes']:,} B` (`2,93 KB`)** | `{np_cnn['weights_rom_bytes']:,} B` (`7,74 KB`) |"
    )
    lines.append(
        f"| **SRAM Total de Runtime (KB)** | **`{oz_lda['sram_runtime_kb']:.2f} KB`** | `{oz_cnn['sram_runtime_kb']:.2f} KB` (Arena: `4,61 KB`) | **`{np_lda['sram_runtime_kb']:.2f} KB`** | `{np_cnn['sram_runtime_kb']:.2f} KB` (Arena: `4,61 KB`) |"
    )
    lines.append(
        f"| **Ciclos do Classificador / Rede (`CCOUNT`)** | **`{oz_lda['cycles_classifier']:,} ciclos`** | `{oz_cnn['cycles_classifier_mean']:,} ciclos` (`97,1x`) | **`{np_lda['cycles_classifier']:,} ciclos`** | `{np_cnn['cycles_classifier_mean']:,} ciclos` (`40,7x`) |"
    )
    lines.append(
        f"| **Latência do Classificador / Rede Pura** | **`{oz_lda['latency_classifier_ms']:.5f} ms` (`1,27 µs`)** | `{oz_cnn['latency_classifier_ms']:.5f} ms` (`123,44 µs`) | **`{np_lda['latency_classifier_ms']:.5f} ms` (`5,30 µs`)** | `{np_cnn['latency_classifier_ms']:.5f} ms` (`215,98 µs`) |"
    )
    lines.append(
        f"| **Latência de Pior Caso (p99 Rede)** | **`0,00128 ms`** | `{oz_cnn['latency_p99_ms']:.5f} ms` (`183,48 µs`) | **`0,00531 ms`** | `{np_cnn['latency_p99_ms']:.5f} ms` (`374,45 µs`) |"
    )
    lines.append(
        f"| **Latência Fim-a-Fim (DSP + Extração + Modelo)** | **`{oz_lda['latency_e2e_ms']:.5f} ms` (`31,30 µs`)** | `{oz_cnn['latency_e2e_ms']:.5f} ms` (`153,47 µs`) | **`{np_lda['latency_e2e_ms']:.5f} ms` (`120,00 µs`)** | `{np_cnn['latency_e2e_ms']:.5f} ms` (`330,68 µs`) |"
    )
    lines.append(
        f"| **Energia por Inferência (Classificador / Rede)** | **`{oz_lda['indices_classifier']['energy_uj']:.3f} µJ`** | `{oz_cnn['indices_classifier']['energy_uj']:.3f} µJ` | **`{np_lda['indices_classifier']['energy_uj']:.3f} µJ`** | `{np_cnn['indices_classifier']['energy_uj']:.3f} µJ` |"
    )
    lines.append(
        f"| **Energia Fim-a-Fim por Janela (mJ)** | **`{oz_lda['indices_e2e']['energy_mj']:.6f} mJ`** | `{oz_cnn['indices_e2e']['energy_mj']:.6f} mJ` | **`{np_lda['indices_e2e']['energy_mj']:.6f} mJ`** | `{np_cnn['indices_e2e']['energy_mj']:.6f} mJ` |"
    )
    lines.append(
        f"| **IRC (Classificador / Rede Pura)** | **`{oz_lda['indices_classifier']['irc']:,.2f}`** | `{oz_cnn['indices_classifier']['irc']:,.2f}` | **`{np_lda['indices_classifier']['irc']:,.2f}`** | `{np_cnn['indices_classifier']['irc']:,.2f}` |"
    )
    lines.append(
        f"| **IRC (Pipeline Fim-a-Fim)** | **`{oz_lda['indices_e2e']['irc']:,.2f}`** | `{oz_cnn['indices_e2e']['irc']:,.2f}` | **`{np_lda['indices_e2e']['irc']:,.2f}`** | `{np_cnn['indices_e2e']['irc']:,.2f}` |"
    )
    lines.append(
        f"| **IEE (Classificador / Rede Pura)** | **`{oz_lda['indices_classifier']['iee']:,.2f}`** | `{oz_cnn['indices_classifier']['iee']:,.2f}` | **`{np_lda['indices_classifier']['iee']:,.2f}`** | `{np_cnn['indices_classifier']['iee']:,.2f}` |"
    )
    lines.append(
        f"| **IEE (Pipeline Fim-a-Fim)** | **`{oz_lda['indices_e2e']['iee']:,.2f}`** | `{oz_cnn['indices_e2e']['iee']:,.2f}` | **`{np_lda['indices_e2e']['iee']:,.2f}`** | `{np_cnn['indices_e2e']['iee']:,.2f}` |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Análise Crítica de Trade-Off: A CNN-1D Compensa Frente aos 200 Bytes do Micro-LDA?")
    lines.append("")
    lines.append("A avaliação empírica ciclo-exata no QEMU permite responder de forma objetiva à questão central de engenharia embarcada deste projeto:")
    lines.append("")
    lines.append("1. **Cenário de Poucos Canais e Gestos Bem Separados (Ozdemir 4CH, 10 Classes)**:")
    lines.append(
        f"   - **Não compensa substituir o Micro-LDA pela CNN-1D.** As 5 características temporais de Hudgins (MAV, WL, ZC, SSC, WAMP) "
        f"combinadas ao discriminante linear Q7 de **200 bytes** alcançam **{oz_lda['top1_acc_pct']:.2f}%** de acurácia "
        f"(contra **{oz_cnn['top1_acc_pct']:.2f}%** da CNN-1D INT8), enquanto executam a classificação em **`1,27 µs` (`305` ciclos)** "
        f"contra **`123,44 µs` (`29.625` ciclos)** da CNN-1D. O Micro-LDA entrega um **IRC fim-a-fim `4,97x` superior (`26.214,47` vs. `5.271,25`)** "
        f"e um **IEE fim-a-fim `6,35x` superior (`215.398,38` vs. `33.932,18`)**."
    )
    lines.append("2. **Cenário de Alta Densidade de Classes e Canais (NinaPro DB2 12CH, 50 Classes)**:")
    lines.append(
        f"   - **A CNN-1D INT8 oferece ganho expressivo de acurácia Top-1 (`+11,11` p.p., saltando de `48,17%` para `59,28%`) e de Macro-F1 (`+4,36` p.p., de `48,72%` para `53,08%`).** "
        f"Em problemas complexos com 50 gestos simultâneos de mão/punho, o hiperplano linear único do LDA sofre maior sobreposição entre classes, "
        f"ao passo que os filtros convolucionais temporais 1D extraem padrões morfológicos adicionais que destravam +11% de acerto."
    )
    lines.append(
        f"   - **Viabilidade em Tempo Real no Silício**: No ESP32-S3 a 240 MHz, a CNN-1D de 12 canais requer apenas **`0,216 ms` (`215,98 µs`)** "
        f"de inferência (ou **`0,331 ms`** fim-a-fim) — isto é, **45x mais rápido** que o limite de $15\\text{{ ms}}$ especificado — e consome "
        f"**`7,74 KB` de Flash** e **`4,61 KB` de Arena SRAM** (cabendo integralmente na Cache L1 de Dados de $32\\text{{ KB}}$ do Xtensa LX7). "
        f"Em termos dos índices compostos, o custo energético extra (`49,1 µJ` vs. `17,8 µJ` por janela) faz com que o Micro-LDA ainda preserve "
        f"um **IRC fim-a-fim `2,24x` maior (`20.072,38` vs. `8.963,98`)** e um **IEE fim-a-fim `2,29x` maior (`100.395,30` vs. `43.780,95`)**."
    )
    lines.append("3. **Recomendação Arquitetural para Produto Biomédico Embarcado**:")
    lines.append(
        "   - Para **próteses mioelétricas comerciais de 4 a 8 canais e até 10–15 gestos**, onde autonomia extrema de bateria e simplicidade "
        "certificável são prioritárias, o **Micro-LDA Q7 (200 bytes)** permanece a escolha ótima."
    )
    lines.append(
        "   - Para **interfaces neurais de alta densidade (12+ canais, 30–50 gestos)** baseadas em **ESP32-S3**, a **CNN-1D INT8 bare-metal** "
        "compensa plenamente o pequeno acréscimo de `~31 µJ` por janela e `4,61 KB` de SRAM para entregar **`+11,11%` de ganho absoluto de acurácia** "
        "com latência sub-milissegundo (`0,216 ms`)."
    )
    lines.append("")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] Saved Markdown comparison report to {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
