# Guia Completo de Métricas, Fórmulas, Interpretação dos Resultados, Valor Científico e Ameaças à Validade

> **Objetivo deste Documento:** Traduzir, formalizar matematicamente e interpretar **todas as métricas** exibidas nas tabelas, nos gráficos e nos textos deste projeto (sEMG em tempo real no ESP32-S3), explicar o que o leitor está vendo em cada tabela/gráfico, destacar os modelos vencedores, analisar **por que certos modelos/datasets se sobressaem em múltiplas áreas** e apresentar com total transparência **por que o trabalho tem valor** e **quais pontos negativos e ameaças à validade podem comprometer ou anular seus resultados**.

---

## 1. Dicionário Matemático e Interpretativo de Todas as Métricas do Estudo

### 1.1. Acurácia Top-1 (`Top-1 Accuracy` / `Top-1 LOTO` / `Média ± DP`)
- **Significado por extenso e tradução:** *Top-1 Accuracy* = **Acurácia de Primeira Escolha** (%). *LOTO* = *Leave-One-Trial-Out* (**Validação Cruzada Deixando Repetições Inteiras Fora**). *DP* = **Desvio Padrão** (*Standard Deviation*).
- **Fórmula:**
  $$\text{Acurácia Top-1 (\%)} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i) \times 100 \qquad \text{DP} = \sqrt{\frac{1}{S-1}\sum_{s=1}^{S}(\text{Acc}_s - \overline{\text{Acc}})^2}$$
  onde $N$ é o número de janelas de teste, $\hat{y}_i$ é a classe de maior pontuação predita, $y_i$ é o gesto verdadeiro (*ground truth*) e $S$ é o número de sujeitos.
- **Como contribui e o que quer dizer para nós:** A acurácia Top-1 é o nível global de acerto do modelo ao tentar adivinhar, na primeira tentativa, qual gesto o usuário está fazendo em um dado dataset. O `± DP` mostra o quanto essa acurácia oscila de uma pessoa para outra.
- **Modelos em destaque:**
  - **Silício físico (`M01–M10`, 101 sujeitos):** `M02_cnn1d_int8` no Khushaba 2CH (**77,38 ± 4,40%**); `M07_hybrid_lda_cnn_cascade` no Ozdemir 4CH (**77,87 ± 4,84%**); `M09_hybrid_dtw_cnn` no NinaPro DB2 12CH (**67,66 ± 8,06%**) e no NinaPro DB3 Amputados 12CH (**49,47 ± 12,96%**).
  - **Simulação C99 INT8 (`M11–M14`):** `M14` atinge o recorde geral de **68,59%** (`K=5`) e **71,06%** (`K=9`) no NinaPro DB2 `S4`, e **76,36%** médios no Ozdemir 2CH (pico de **84,10%** em `S28`).
- **Interpretação para o projeto:** Em poucos gestos (10 classes), modelos lineares e cascatas já atingem controle funcional (>75%). Em 50 gestos e amputados, arquiteturas convolucionais/híbridas (`M02`, `M07`, `M09`, `M14`) tornam-se indispensáveis — mas a Top-1 sozinha pode enganar quando há desbalanceamento de classes (por isso deve sempre ser lida junto com **Macro-F1** e **Balanced Acc**).

---

### 1.2. Acurácia Balanceada (`Balanced Accuracy` / `Balanced Acc`)
- **Significado por extenso e tradução:** *Balanced Accuracy* = **Acurácia Balanceada** (%).
- **Fórmula:**
  $$\text{Balanced Acc (\%)} = \frac{1}{C}\sum_{c=1}^{C} \frac{\text{VP}_c}{\text{VP}_c + \text{FN}_c} \times 100 = \frac{1}{C}\sum_{c=1}^{C} \text{Recall}_c \times 100$$
  onde $C$ é o número total de gestos (10 ou 50), $\text{VP}_c$ são os verdadeiros positivos da classe $c$ e $\text{FN}_c$ são os falsos negativos.
- **Como contribui e o que quer dizer para nós:** Em datasets de sEMG, alguns gestos (e sobretudo o estado de **repouso**) aparecem com muito mais janelas que outros. A *Balanced Accuracy* dá exatamente o **mesmo peso** para cada gesto, impedindo que um modelo que "chuta repouso o tempo todo" pareça bom.
- **Modelos em destaque:** No NinaPro DB2 (`S4`), `M14` lidera com **64,47%** (`K=9`) / **61,96%** (`K=5`), seguido pelo `Micro-LDA Q7 int16` (**51,03%** no QEMU). No Ozdemir 4CH, `M01`/`M02`/`M07` mantêm *Balanced Acc* praticamente idêntica à Top-1 (~**77–82%**), provando que as 10 classes do Ozdemir são bem equilibradas.
- **Interpretação para o projeto:** Quando a *Balanced Acc* cai muito abaixo da *Top-1* (como ocorre no NinaPro 12CH sob quantização `int8`: `53,11%` Top-1 vs. `45,98%` Balanced Acc), é sinal claro de que o modelo está acertando a classe majoritária à custa de errar gestos finos de dedos.

---

### 1.3. Macro-F1 (`Macro-Averaged F1-Score`)
- **Significado por extenso e tradução:** *Macro-Averaged F1-Score* = **Pontuação F1 Média Macro** (%). Combina **Precisão** (*Precision*: quando o modelo diz que é o gesto $c$, quantas vezes acertou?) e **Revocação** (*Recall*: de todas as vezes que o gesto $c$ aconteceu, quantas o modelo detectou?).
- **Fórmula:**
  $$\text{Precision}_c = \frac{\text{VP}_c}{\text{VP}_c + \text{FP}_c}, \qquad \text{Recall}_c = \frac{\text{VP}_c}{\text{VP}_c + \text{FN}_c}$$
  $$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}, \qquad \text{Macro-F1 (\%)} = \frac{1}{C}\sum_{c=1}^{C} F1_c \times 100$$
- **Como contribui e o que quer dizer para nós:** É a métrica clínica mais honesta para próteses: penaliza severamente tanto **falsos acionamentos** (a prótese fechar sem querer e derrubar um copo = baixa *Precision*) quanto **falhas de acionamento** (o usuário fazer força e a prótese não responder = baixo *Recall*), tratando todos os gestos com igual importância.
- **Modelos em destaque (Silício físico):** `M02_cnn1d_int8` no Khushaba (**77,22 ± 4,51%**); `M07_hybrid_lda_cnn_cascade` no Ozdemir (**78,03 ± 4,85%**), no NinaPro DB2 (**59,11 ± 5,85%**) e no NinaPro DB3 Amputados (**35,73 ± 11,20%**). Na simulação (`M14`, DB2 `S4`): **65,81%** (`K=5`) e **69,34%** (`K=9`).
- **Interpretação para o projeto:** Repare que no NinaPro DB2 e DB3, `M09` vence em Top-1, mas **`M07` (Cascata LDA → CNN) vence em Macro-F1** — indicando decisões mais equilibradas entre as 50 classes. Já o fosso no DB3 (**49,47% Top-1** vs. **35,15% Macro-F1**) expõe a dificuldade real dos amputados em sustentar gestos raros e delicados de dedos.

---

### 1.4. Variação de Acurácia (`Δ vs. Float32` e `+Δ pp`)
- **Significado por extenso e tradução:** `Δ` (*Delta*) = diferença absoluta em **pontos percentuais (`pp`)**. `Δ vs. Float32` mede a perda ou ganho ao quantizar dados de 32 bits em ponto flutuante (`float32`) para formatos inteiros (`int16`, `int8`). `+Δ pp` mede o ganho trazido pela suavização temporal (`Vote-K` / `Soft Voting`).
- **Fórmula:**
  $$\Delta_{\text{quant}} = \text{Acc}_{\text{quant}} - \text{Acc}_{\text{float32}} \text{ (pp)} \qquad \Delta_{\text{smooth}}(K) = \text{Acc}(K) - \text{Acc}(K=1) \text{ (pp)}$$
- **Como contribui e o que quer dizer para nós:** Permite isolar o efeito puro de uma decisão de engenharia (quantizar o ADC ou adicionar voto temporal) sem confundir com a acurácia base do modelo.
- **Modelos em destaque:**
  - **Resiliência à quantização (`int8`):** `M13` (*SupCon + BatchNorm Folding*) perde apenas **`-0,08 pp`** (`59,27% → 59,19%`) e `M01` (`Micro-LDA`) mantém estabilidade quase total, enquanto `M02` (`CNN-1D` padrão) colapsa em **`-48,76 pp`** (`58,62% → 9,86%`).
  - **Ganho com suavização (`+Δ pp`):** `M14` com *Soft Voting* ganha **`+5,53 pp`** em `K=5` e **`+8,00 pp`** em `K=9`.
- **Interpretação para o projeto:** Fundamenta a **Lei da Invariância em 16 bits** (`Δ ≈ 0 pp` em `int16` vs. `float32`, cortando 50% da banda sem custo) e mostra que redes convolucionais ingênuas precisam de blindagem contrastiva (*SupCon*) para sobreviver a ADCs de 8 bits.

---

### 1.5. Latência no Core 1 (`Latência Core 1` / `Latência Média C1` / `CCOUNT`)
- **Significado por extenso e tradução:** Tempo decorrido para o processador de inferência (**Core 1 / APP_CPU** do ESP32-S3 a 240 MHz) receber uma janela sEMG pronta e concluir a classificação. Medido em **microssegundos (`µs`)** ou **milissegundos (`ms`)** via `esp_timer_get_time()` no silício real, ou via registrador de ciclos **`CCOUNT`** (*Cycle Count*) no emulador QEMU.
- **Fórmula:**
  $$t_{\text{silício}} = t_{\text{fim}} - t_{\text{início}} \quad (\mu\text{s}), \qquad t_{\text{QEMU}} = \frac{\text{CCOUNT}_{\text{fim}} - \text{CCOUNT}_{\text{início}}}{240 \times 10^6\text{ Hz}}$$
  $$\overline{t} = \frac{1}{N}\sum_{i=1}^{N} t_i$$
- **Como contribui e o que quer dizer para nós:** Mede a velocidade computacional pura do algoritmo no microcontrolador. Quanto menor, maior a folga de tempo real e menor o gasto de bateria.
- **Modelos em destaque (Silício físico):** **`M01_micro_lda_q7`** é líder absoluto em todos os 4 datasets: **178,93 µs** (Khushaba 2CH), **341,05 µs** (Ozdemir 4CH), **1.049,62 µs** (DB3 12CH) e **1.051,83 µs** (DB2 12CH). Em segundo lugar de velocidade geral vêm `M05_dtw_lb_keogh` (**491 µs – 5,97 ms**) e `M08_hybrid_dtw_lda` (**840 µs – 2,65 ms**). O mais lento é `M03_dtw_classic_on2` (**49,96 ms** no DB2 12CH).
- **Interpretação para o projeto:** Mesmo o modelo neural `M02_cnn1d_int8` (**5,90–14,15 ms**) e os híbridos (`M07`, `M09`: **3,31–15,74 ms**) rodam com folga em tempo real no ESP32-S3 físico. O `M03` (`49,96 ms`) bate no limite do *stride* de 50 ms em 12 canais, provando na prática por que otimizações como *Sakoe-Chiba* (`M04`, 13,07 ms) e *LB_Keogh* (`M05`, 4,76 ms) são vitais.

---

### 1.6. Percentil 99 de Latência (`p99` — *99th Percentile Latency*)
- **Significado por extenso e tradução:** *99th Percentile Latency* = **Percentil 99 da Latência** (`µs` ou `ms`). Valor abaixo do qual ocorrem 99% de todas as inferências medidas.
- **Fórmula:**
  $$p99 = t_{(\lceil 0{,}99 \cdot N \rceil)} \quad \text{com } t_{(1)} \le t_{(2)} \le \dots \le t_{(N)}$$
- **Como contribui e o que quer dizer para nós:** Em sistemas de tempo real (próteses, aviação, robótica), a média não basta — um único pico de atraso (*jitter*) pode causar travamento da fila do FreeRTOS. O `p99` mede o "pior caso típico" real.
- **Modelos em destaque (Silício físico):** No benchmark de 187.200 inferências do Ozdemir 4CH, `M01_micro_lda_q7` teve média **330,22 µs** e `p99` **347,55 µs**; `M02_cnn1d_int8` teve média **6.575,07 µs** e `p99` **6.610,04 µs** (variação < 1%). Em 12CH (`S4`/`S11`), `M01` teve `p99` de **1.075 µs** e `M02` de **14.254 µs**.
- **Interpretação para o projeto:** A proximidade extrema entre média e `p99` prova **determinismo temporal rígido** no firmware C99 sem `malloc/free` — ou seja, não há pausas imprevisíveis de sistema operacional nem degradação térmica.

---

### 1.7. Latência Algorítmica / Fim-a-Fim (`Atraso Algorítmico`, Janela $W$, Stride $S$, Voto $K$)
- **Significado por extenso e tradução:** Tempo total desde o início de uma nova contração muscular até o comando estabilizado ser liberado para o atuador da prótese. Combina o janelamento deslizante (*Window* $W = 200\text{ ms}$, passo/*Stride* $S = 50\text{ ms}$), a latência de inferência ($t_{\text{inf}}$) e o atraso causal do voto majoritário de $K$ janelas (*Majority Voting* / *Soft Voting*).
- **Fórmula:**
  $$t_{\text{algorítmico}}(K) = (K - 1) \times S, \qquad t_{\text{comando}} = t_{\text{algorítmico}}(K) + t_{\text{inf}} + t_{\text{voto}}$$
  *(com $t_{\text{voto}} < 0{,}5\ \mu\text{s}$ no ESP32-S3)*
- **Como contribui e o que quer dizer para nós:** Define se o usuário **percebe atraso** ao movimentar a prótese. A literatura clínica de reabilitação (critério de Hudson-Englehart / Farrell & Weir) fixa o limite de atraso aceitável entre **200 ms e 300 ms**.
- **Configurações em destaque:** **`K=3`** (`+100 ms` → total **~100,3 ms** LDA / **~106,6 ms** CNN no Ozdemir 4CH) é o padrão do firmware FreeRTOS; **`K=5`** (`+200 ms` → **~200,3–207,6 ms**) maximiza a acurácia dentro da barreira clínica. **`K=9`** (`+400 ms`) ultrapassa o limite clínico e é desaconselhado para gestos rápidos.
- **Interpretação para o projeto:** Mostra que o gargalo real de percepção humana numa prótese **não é a CPU do ESP32-S3** (que leva 0,3 a 14 ms), e sim o compromisso entre estabilidade temporal ($K$) e agilidade de troca de gesto.

---

### 1.8. Consumo do Orçamento de Tempo Real / Ciclo de Trabalho (`CPU Budget %` / `Duty Cycle`)
- **Significado por extenso e tradução:** *CPU Budget* = **Orçamento de Tempo Real da CPU** (%). *Duty Cycle* = **Ciclo de Trabalho** (% do período em que a CPU fica acordada processando).
- **Fórmula:**
  $$\text{CPU Budget (\%)} = \frac{t_{\text{Core1}}}{S} \times 100 = \frac{t_{\text{Core1}}}{50{,}0\text{ ms}} \times 100$$
  *(para Khushaba a 4 kHz com stride de 25 ms, ou normalizado na base padrão de 50 ms = 20 Hz)*
- **Como contribui e o que quer dizer para nós:** Como chega uma nova janela de dados a cada **50 ms**, se um modelo levar mais que 50 ms (**> 100%**), as janelas começam a acumular na fila (`xQueueJanelaPronta`) até estourar a memória (*deadline miss*). A fração complementar (`100% − CPU Budget`) é o tempo livre em que o ESP32-S3 pode dormir em `Light Sleep`.
- **Modelos em destaque (Silício físico):** **`M01_micro_lda_q7`** ocupa apenas **0,36%** (2CH), **0,68%** (4CH) e **2,10%** (12CH), deixando ~98–99% da CPU livre! **`M02_cnn1d_int8`** ocupa **11,80%** (2CH), **15,10%** (4CH) e **28,30%** (12CH) — ainda deixando >71% livre. Já **`M03_dtw_classic_on2`** atinge **99,92%** em 12CH (no limite absoluto de saturação).
- **Interpretação para o projeto:** Comprova que o ESP32-S3 suporta inferência contínua em tempo real com **zero *deadline misses*** em todos os modelos úteis (`M01`, `M02`, `M04`–`M10`), e expõe a inviabilidade prática do DTW quadrático clássico (`M03`) em 12 canais.

---

### 1.9. Energia por Inferência (`Energia / Inf.`) e Potência Ativa (`mW`)
- **Significado por extenso e tradução:** Quantidade de energia elétrica gasta pelo microcontrolador para classificar uma única janela sEMG, em **milijoules (`mJ`)** ou **microjoules (`µJ`)**, multiplicando a potência ativa nominal do ESP32-S3 (**148,5 mW** = miliwatts @ 240 MHz) pelo tempo de CPU.
- **Fórmula:**
  $$E_{\text{inf}}\ (\text{mJ}) = P_{\text{ativa}}\ (\text{mW}) \times \frac{t_{\text{Core1}}\ (\text{ms})}{1000} = 148{,}5 \times t_{\text{Core1}}\ (\text{s})$$
- **Como contribui e o que quer dizer para nós:** Próteses mioelétricas rodam em baterias LiPo pequenas (ex.: 500 mAh) embutidas no encaixe do antebraço. Menor energia por inferência significa mais horas de autonomia sem recarga e menor aquecimento dentro do soquete protético.
- **Modelos em destaque (Silício físico):** **`M01_micro_lda_q7`** gasta apenas **0,0266 mJ** (Khushaba), **0,0506 mJ** (Ozdemir) e **0,1562 mJ** (DB2 12CH) — cerca de **13× a 33× menos energia** que `M02_cnn1d_int8` (**0,876 mJ** a **2,101 mJ**) e **47× menos** que `M03` (**7,419 mJ**).
- **Interpretação para o projeto:** Mesmo os modelos mais pesados consomem poucos milijoules frente aos motores mecânicos da mão biônica, mas em sistemas de monitoramento contínuo (20 inferências/segundo = 72.000 inferências/hora), a economia do `Micro-LDA` e da poda `LB_Keogh` (`M05`, corte de ~90% frente a `M03`) maximiza o tempo de `Light Sleep`.

---

### 1.10. Recursos de Memória: ROM / Flash, SRAM e Stack High-Water Mark
- **Significado por extenso e tradução:**
  - **ROM / Flash** (*Read-Only Memory* / Memória Flash não-volátil): espaço ocupado pelo código executável (`.text`) e pelos pesos constantes do modelo (`.rodata`), em **Bytes (`B`)** ou **Kilobytes (`KB`)**.
  - **SRAM** (*Static Random-Access Memory* / Memória RAM Estática): memória volátil rápida de trabalho (limite físico total de **512 KB** no ESP32-S3), abrigando buffers DMA, variáveis globais (`.data`/`.bss`) e a **Arena de Ativação** (tensores intermediários das camadas convolucionais).
  - **Stack High-Water Mark** (`uxTaskGetStackHighWaterMark`): menor margem de pilha livre já registrada nas tarefas do Core 0 (`C0`) e Core 1 (`C1`) durante toda a execução.
- **Fórmula:**
  $$\text{ROM}_{\text{total}} = \text{bytes}(\texttt{.text}) + \text{bytes}(\texttt{.rodata}), \qquad \text{SRAM}_{\text{total}} = \text{bytes}(\texttt{.data}) + \text{bytes}(\texttt{.bss}) + \text{Arena}$$
- **Como contribui e o que quer dizer para nós:** Garante que o modelo cabe fisicamente dentro do microcontrolador de US$ 3 sem precisar de PSRAM externa (mais lenta e cara) e sem risco de *stack overflow* (travamento súbito da prótese).
- **Modelos em destaque:** **`M01_micro_lda_q7`** exige apenas **100 B** (2CH), **200 B** (4CH) e **3.000 B** (12CH) de pesos em ROM e **0 B de arena extra** (SRAM total da tarefa: **3,46 KB a 22,20 KB**), cabendo 100% no **Cache L1 de Dados de 32 KB**. A **`CNN-1D INT8`** usa **8,07 KB a 26,81 KB** de SRAM. Já `M14` usa **186 KB** de ROM mas apenas **7,62 KB** de SRAM (vs. **30,9 KB** na `CNN-2D` `M12`/`M13`). Durante 187.200+ inferências físicas, a margem livre de pilha permaneceu intacta (`≥ 2.272 B` no Core 0 e `≥ 3.268 B` no Core 1).
- **Interpretação para o projeto:** Prova viabilidade de memória estática determinística em C99 puro (zero alocação dinâmica `malloc`).

---

### 1.11. Largura de Banda de Aquisição (`kB/s`, `B/amostra`) e Figura de Mérito (`FoM`)
- **Significado por extenso e tradução:** *Bandwidth* = **Largura de Banda** (taxa de fluxo de dados gerados pelo ADC em kilobytes por segundo, `kB/s`). *FoM* = *Figure of Merit* (**Figura de Mérito** — acurácia obtida por byte de dado transmitido por amostra).
- **Fórmula:**
  $$\text{Taxa (kB/s)} = \frac{C_{\text{canais}} \times B_{\text{precisão}} \times F_s}{1000}, \qquad \text{FoM} = \frac{\text{Acurácia Top-1 (\%)}}{C_{\text{canais}} \times B_{\text{precisão}}}$$
  onde $B_{\text{precisão}} \in \{4\text{ B } (\texttt{float32}),\ 2\text{ B } (\texttt{int16}/\texttt{float16}),\ 1\text{ B } (\texttt{int8})\}$.
- **Como contribui e o que quer dizer para nós:** Em uma prótese sem fio (BLE) ou via SPI/DMA, reduzir a largura de banda corta o custo do hardware analógico (AFE), economiza RAM de buffer e reduz consumo de rádio.
- **Configuração em destaque:** **`int16` @ 4 canais no Ozdemir** (`16,0 kB/s`, corte de **-50%** vs. `float32`, **zero perda de acurácia**: `77,51%` = `77,51%`, `FoM = 9,69`) e **`int8` @ 4 canais** (`8,0 kB/s`, `70,31%` populacional), que supera **`float32` @ 2 canais** (`16,0 kB/s`, `59,87%`) gastando **metade da banda**.
- **Interpretação para o projeto:** Revela a regra de ouro do co-design sEMG: **invista em mais eletrodos (resolução espacial) com menor profundidade de bits, em vez de poucos eletrodos em alta precisão numérica**.

---

### 1.12. Paridade Numérica Bit-Exata (`Paridade HW` / `host_parity_ratio` / `max_logit_diff`)
- **Significado por extenso e tradução:** *Hardware Bit-Exact Parity* = **Paridade Numérica Bit-Exata entre o Computador Oráculo e o Microcontrolador** (%). `max_logit_diff` = **Diferença Máxima Absoluta nos Logits** (pontuações brutas antes da escolha da classe).
- **Fórmula:**
  $$\text{Paridade HW (\%)} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}\left(\hat{y}_i^{\text{ESP32}} = \hat{y}_i^{\text{Host}}\right) \times 100, \qquad \text{max\_logit\_diff} = \max_{i,c}\left|z_{i,c}^{\text{ESP32}} - z_{i,c}^{\text{Host}}\right|$$
- **Como contribui e o que quer dizer para nós:** Garante que a implementação embarcada em C99 quantizado no ESP32-S3 não introduziu bugs de compilador, *overflow* inteiro, erro de *endianness* ou erro de arredondamento frente ao modelo validado em Python.
- **Destaque obtido:** **100,00% de paridade bit-exata** em **100.630 inferências** do benchmark populacional de 10 modelos (101 sujeitos) e em **187.200 inferências** do benchmark físico contínuo do Ozdemir (e `99,8%` — 499/500 — apenas em um único empate de desempate numérico no `M03` em `S11` no teste prévio).
- **Interpretação para o projeto:** Valida a integridade absoluta da engenharia de firmware: o que você vê nas simulações C99 é exatamente o que roda no chip físico.

---

### 1.13. IRC — Índice de Responsividade Computacional (`Index of Responsive Classification`)
- **Significado por extenso e tradução:** *Index of Responsive Classification* = **Índice de Responsividade Computacional** (métrica composta criada no projeto).
- **Fórmula:**
  $$\text{IRC} = \frac{\text{Acurácia Top-1 (\%)} \times N_{\text{gestos}}}{\text{Latência}_{\text{Core1}}\ (\text{ms})}$$
- **Como contribui e o que quer dizer para nós:** Permite comparar de forma unificada um modelo que decodifica **50 gestos** contra um que decodifica **10 gestos**, medindo quantas "unidades de repertório acurado" o ESP32-S3 entrega por milissegundo de CPU. Quanto maior, mais responsivo.
- **Modelos em destaque (Silício físico):** **`M01_micro_lda_q7`** lidera com enorme folga em todos os 4 datasets: **3.940,65** (Khushaba), **2.592,62** (NinaPro DB2), **2.252,46** (Ozdemir) e **1.604,39** (NinaPro DB3) — contra **99,85 a 235,18** da `CNN-1D INT8`. (No QEMU bare-metal, `M14` atinge `8.075` em `K=5`, mas em latência simulada, não física).
- **Interpretação para o projeto:** Mostra a supremacia computacional do `Micro-LDA Q7` em velocidade pura, mas **exige leitura crítica**: como a latência está no denominador sem piso de acurácia, um modelo muito rápido com acurácia baixa (ex.: DB3 `33,68%`) ainda pontua alto em IRC!

---

### 1.14. IEE — Índice de Eficiência Embarcada (`Index of Embedded Efficiency`)
- **Significado por extenso e tradução:** *Index of Embedded Efficiency* = **Índice de Eficiência Embarcada** (métrica holística composta criada no projeto).
- **Fórmula:**
  $$\text{IEE} = \frac{\text{Acurácia Top-1 (\%)} \times N_{\text{gestos}}}{E_{\text{inf}}\ (\text{mJ}) \times \log_{10}(\text{SRAM}_{\text{KB}})}$$
- **Como contribui e o que quer dizer para nós:** Sintetiza em um único índice os 4 pilares do TinyML embarcado: precisão preditiva, tamanho do vocabulário de gestos, gasto elétrico da bateria e consumo de memória SRAM (em escala logarítmica para evitar que poucos bytes distorçam o índice).
- **Modelos em destaque (Silício físico):** **`M01_micro_lda_q7`** lidera por uma ordem de grandeza em todos os datasets: **49.225,60** (Khushaba), **18.507,92** (Ozdemir), **12.967,44** (NinaPro DB2) e **8.024,62** (NinaPro DB3) — frente a **640,64 a 1.108,82** da `CNN-1D INT8`.
- **Interpretação para o projeto:** Confirma que, sob restrições extremas de bateria e SRAM (próteses 2–4CH de baixo custo), o `Micro-LDA Q7` é imbatível em eficiência física; porém herda a mesma limitação estrutural do IRC quando aplicado a cenários onde sua acurácia absoluta fica abaixo do patamar clínico de 50%.

---

### 1.15. Taxa de Saída Antecipada (`EE` — *Early-Exit Rate*)
- **Significado por extenso e tradução:** *Early-Exit Rate* = **Taxa de Saída Antecipada** (%). Percentual de janelas classificadas imediatamente pelo estágio rápido (`Micro-LDA`), pulando a execução da rede neural pesada (`CNN-1D` ou especialista fino).
- **Fórmula:**
  $$\text{EE (\%)} = \frac{N_{\text{saídas rápidas (confiança} \ge \tau\text{)}}}{N_{\text{total}}} \times 100$$
- **Como contribui e o que quer dizer para nós:** Em arquiteturas em cascata (`M07`, `M10`, `M11`), mede quanta computação foi efetivamente economizada desviando gestos "fáceis" do caminho pesado.
- **Modelos em destaque:** No Ozdemir 4CH, a cascata `M07` reduz a latência média física de **7,55 ms** (`CNN-1D`) para **3,31 ms** (economia de **56%**) mantendo acurácia líder de **77,87%**. Na arquitetura hierárquica `M11` (QEMU), o *Fast Path* reduz a latência em ~72%. **Porém**, no NinaPro DB2/DB3 (`50 classes`, teste físico `S4`/`S11`), `EE` caiu para **0,0%** sob o limiar rígido `τ`, reduzindo drasticamente o ganho de velocidade (`11,57–12,28 ms`).
- **Interpretação para o projeto:** Cascatas *Early-Exit* brilham em problemas de 10 classes bem separadas, mas perdem eficiência se o limiar de confiança `τ` não for recalibrado para espaços de alta incerteza (50 classes / amputados).

---

### 1.16. Métricas de Condicionamento de Sinal (DSP) e Características de Hudgins
- **Taxa de Amostragem ($F_s$, em `Hz` = amostras/segundo):** `2.000 Hz` (Ozdemir, NinaPro DB2/DB3) e `4.000 Hz` (Khushaba). Atende com folga ao critério de Nyquist ($F_s > 2 \times 450\text{ Hz} = 900\text{ Hz}$) para a banda útil do sEMG (**20–450 Hz**).
- **Descritores de Hudgins (Domínio do Tempo — Zero-FPU):**
  - **MAV** (*Mean Absolute Value* — Valor Absoluto Médio): $\text{MAV} = \frac{1}{W}\sum_{n=1}^{W}|x_n|$ (força/intensidade média da contração).
  - **RMS** (*Root Mean Square* — Valor Quadrático Médio): $\text{RMS} = \sqrt{\frac{1}{W}\sum_{n=1}^{W}x_n^2}$ (potência elétrica muscular).
  - **WL** (*Waveform Length* — Comprimento da Forma de Onda): $\text{WL} = \sum_{n=2}^{W}|x_n - x_{n-1}|$ (complexidade combinada de amplitude e frequência).
  - **ZC** (*Zero Crossings* — Cruzamentos por Zero): $\text{ZC} = \sum_{n=1}^{W-1} \mathbb{I}(x_n x_{n+1} < 0 \land |x_n - x_{n+1}| \ge \varepsilon)$ (estimativa de frequência sem FFT).
  - **SSC** (*Slope Sign Changes* — Mudanças de Sinal da Inclinação): $\text{SSC} = \sum_{n=2}^{W-1} \mathbb{I}((x_n - x_{n-1})(x_n - x_{n+1}) \ge \varepsilon)$ (conteúdo espectral de alta frequência).

---

## 2. Análise de Modelos e Datasets Dominantes em Múltiplas Áreas & Ameaças que Podem "Anular" Esse Destaque

### 2.1. `M01_micro_lda_q7` — O Dominante Absoluto em Eficiência (7 Métricas em 4 Datasets)
- **Onde se sobressai:** Vence **Latência**, **p99**, **Orçamento de CPU**, **Energia**, **ROM**, **SRAM**, **IRC** e **IEE** em **100% dos 4 datasets físicos**, além de vencer a CNN-1D em acurácia no Ozdemir 4CH (`76,82%` vs. `75,39%`).
- **Por que se sobressai:** Não usa convoluções nem camadas ocultas; calcula apenas 5 estatísticas inteiras (Zero-FPU) e um produto escalar de `int8` (`100 B` a `3 KB`) que permanece 100% no Cache L1 de 32 KB do Xtensa LX7, sem arena de ativações.
- **Ameaças à validade que podem ANULAR esse destaque:**
  1. **Viés algébrico de `IRC` e `IEE` sem piso clínico de acurácia:** Como a latência e a energia entram no denominador de forma linear, o `Micro-LDA` ganha o 1º lugar em IRC (`1.604`) e IEE (`8.025`) no **NinaPro DB3 (amputados)** mesmo acertando apenas **33,68%** — um valor **clinicamente inutilizável** (abaixo da meta de 50%). Um índice de eficiência sem gate mínimo de acurácia premia "errar muito rápido".
  2. **Colapso sob alta densidade e não-linearidade:** Em 12 canais / 50 classes, fica **-12,01 pp** atrás da CNN-1D no DB2 e **-15,62 pp** atrás no DB3 (e **-16,5 pp** atrás do `M14`).
  3. **Fragilidade sem recalibração (`Cross-Subject`):** Com pesos fixos de `S1` na Flash, despenca de `81,20%` para `42,10%` na população (mínimo de `25,64%` em `S37`).
  4. **Desvantagem artificial da CNN-1D sem SIMD PIE:** A vantagem de `13×–33×` do Micro-LDA sobre a CNN-1D no ESP32-S3 físico foi medida com convoluções C99 escalares puras; ativar instruções vetoriais SIMD PIE (`esp-dsp`) reduziria drasticamente essa margem.

### 2.2. `M07` (Cascata Early-Exit) e `M09` (DTW + CNN-1D) — Os Dominantes em Acurácia no Silício Físico
- **Onde se sobressaem:** `M09` lidera Top-1 físico no NinaPro DB2 (**67,66%**) e DB3 (**49,47%**); `M07` lidera Top-1 no Ozdemir (**77,87%**) e lidera **Macro-F1 físico em 3 dos 4 datasets** (Ozdemir **78,03%**, DB2 **59,11%**, DB3 **35,73%**).
- **Por que se sobressaem:** Combinam a extração morfológica não-linear da CNN-1D com um segundo mecanismo complementar (alinhamento elástico temporal do DTW no `M09`, ou filtro de confiança rápido via LDA no `M07`).
- **Ameaças à validade que podem ANULAR esse destaque:**
  1. **Colapso da Saída Antecipada (`EE = 0,0%`) em 50 classes:** Sem sintonia fina do limiar `τ` por sujeito, a cascata `M07` acaba executando LDA + CNN em quase todas as janelas difíceis, pagando latência quase igual à CNN pura (`11,57–12,28 ms`).
  2. **Fosso Top-1 vs. Macro-F1 em amputados:** No DB3, os `49,47%` de Top-1 do `M09` escondem apenas `35,15%` de Macro-F1 (**-14,32 pp**), mostrando dependência das classes mais fáceis/frequentes.
  3. **Amostragem estratificada física (100 janelas/sujeito):** Embora o LOTO no host tenha avaliado todo o conjunto de teste, a validação física simultânea dos 10 modelos em 101 sujeitos usou 100 janelas/sujeito por limite prático de transmissão serial (~2h de execução contínua).

### 2.3. `M14` (`636 Features + SupCon Residual MLP INT8`) — O Dominante em Acurácia Geral (Simulada)
- **Onde se sobressai:** Recorde absoluto de acurácia no NinaPro DB2 (`71,06%` em `K=9`, `68,59%` em `K=5`), recuperação impressionante em 2 canais no Ozdemir (`80,77%` em `S1`, `76,36%` populacional nos 40 sujeitos, pico `84,10%` em `S28`) e excelente compromisso SRAM (`7,62 KB`).
- **Por que se sobressai:** Substitui a convolução pesada por um vetor hiper-rico de 636 características (tempo, frequência, coeficientes AR e 132 sinergias espaciais inter-canais) treinado com perda contrastiva supervisionada (*SupCon*) e suavização probabilística (*Soft Voting*).
- **Ameaças à validade que podem ANULAR esse destaque:**
  1. **Não validado no silício físico ESP32-S3:** Diferente de `M01`–`M10`, o `M14` (assim como `M11`–`M13`) rodou apenas em **simulação C99 INT8 + estimativa analítica de ciclos** (`0,425 ms`). O custo real de extrair 636 features (incluindo FFTs, coeficientes AR e 132 sinergias) com stalls de Flash SPI e FreeRTOS no chip físico ainda **não foi medido em hardware**.
  2. **O recorde de `71,06%` viola o limite clínico de tempo real:** Depende de `K=9` (**400 ms** de atraso algorítmico > limite de 300 ms). Sem suavização (`K=1`, INT8), a acurácia real instantânea é `63,06%`.
  3. **Validação em apenas 1 sujeito (`S4`) no NinaPro DB2:** Diferente dos 40 sujeitos físicos de `M01`–`M10`, o recorde de `M14` no NinaPro DB2 não foi replicado nos 40 sujeitos nem nos 11 amputados do DB3 (apenas no Ozdemir 2CH foi validado nos 40 sujeitos).

### 2.4. Datasets `Ozdemir 4CH` e `Khushaba 2CH` — Os Datasets com Melhores Pontuações
- **Por que se sobressaem:** Entregam as maiores acurácias (`76–78%`), menor variância inter-sujeitos (`DP = 4,4%–6,1%`), menor latência (`0,18–0,34 ms` LDA) e maiores `IRC`/`IEE`.
- **Ameaças que impedem generalizar esses números para próteses clínicas:**
  - Apenas **10 gestos bem separados** executados por **sujeitos jovens e saudáveis** sob contrações isométricas limpas de laboratório.
  - O **Ozdemir** já veio **pré-filtrado pelos autores** (20–450 Hz + Notch 50 Hz), eliminando artefatos severos de linha de base presentes em sensores reais de campo.
  - O **Khushaba** possui apenas **10 sujeitos** (`N = 10`, baixa potência estatística) e taxa de **4.000 Hz** (que dobra o custo de aquisição frente a 2.000 Hz).

---

## 3. Guia Claro de Leitura: O Que o Leitor Está Vendo em Cada Tabela e Gráfico do Site

1. **Seção 1 — Gráfico & Tabela Principal no ESP32-S3 Físico (`Micro-LDA Q7` vs. `CNN-1D INT8`):** Confronto direto no silício real entre os dois paradigmas fundamentais (linear vs. deep learning) usando a configuração ótima (`int16`, 100% dos canais) nos 4 datasets. Mostra o dilema central do estudo: `Micro-LDA` esmaga em velocidade/bateria (`IRC`/`IEE`), enquanto `CNN-1D` vence em acurácia nos cenários difíceis (Khushaba 2CH, DB2 12CH e DB3 Amputados).
2. **Seção 2.6 — Tabela de Condicionamento e Filtros DSP por Dataset:** Mostra por que não aplicamos o mesmo filtro em tudo — filtrar duas vezes um sinal já filtrado (Ozdemir) distorce a fase e atenua o sinal útil; já nos sinais brutos (NinaPro DB2/DB3), o filtro causal Butterworth DF2T + Notch 50 Hz no ESP32-S3 é obrigatório.
3. **Seção 5 — Os 13 Gráficos Interativos:**
   - *Gráfico Principal (10 Modelos Físicos):* Mapa completo `Acurácia × Latência Real` no ESP32-S3 dos 10 motores C99 (M01–M10) nos 101 sujeitos.
   - *Gráficos 1 e 2 (14 Arquiteturas & Fronteira de Pareto):* Incluem a evolução dos passos avançados (`M11–M14` em simulação C99/QEMU) mostrando quais modelos são ótimos de Pareto (`M01`, `M11`, `M14`) e quais são dominados (`M03`).
   - *Gráfico 3 (Energia) e Gráficos 5–6 (IRC / IEE em QEMU):* Avaliam autonomia de bateria e memória SRAM sob ciclos `CCOUNT` puros (atenção: não comparar diretamente o valor numérico de IRC/IEE do QEMU com o do silício físico!).
   - *Gráfico 4 (Quantização Float32 vs. INT8):* Demonstra visualmente o colapso da CNN-1D padrão em 8 bits (`9,86%`) e como o treinamento contrastivo (`M13`/`M14`) blinda a rede.
   - *Gráficos 7 e 8 (Suavização Temporal `K=1..9` e `+Δ pp`):* Quantificam quanto de acurácia se ganha gratuitamente no pós-processamento e onde o atraso algorítmico cruza o limite clínico de 200–300 ms.
   - *Gráficos 9, 10, 11 e 12 (QEMU vs. Silício, Orçamento 50 ms, Estabilidade S1–S40, Transferência S1):* Provam o determinismo temporal do hardware físico (`0 deadline misses`, `p99 ≈ média`, zero vazamento de RAM em 25 min contínuos) e justificam por que o protocolo `SEW1` de calibração em RAM foi criado.
4. **Seção 6 — Tabelas Hierárquicas (`Modelos > Canais > Quantização`):** Permitem auditar cada uma das 14 arquiteturas descendo do resumo físico populacional até a degradação controlada por número de canais (`12 → 1CH`) e precisão numérica (`float32 → int8`).
5. **Seção 6.1 — Tabelas do Relatório de Latências:** Consolidam todas as latências no ESP32-S3 físico e somam o atraso algorítmico do voto majoritário (`K=1..9`), provando que nenhum modelo ultrapassa 300 ms na inferência pura.

---

## 4. Por Que Nosso Trabalho Tem Valor? (Contribuições Científicas e de Engenharia)

Sim, este trabalho tem **alto valor científico e prático em engenharia biomédica embarcada**, por cinco razões concretas:
1. **Saída do "PyTorch em GPU" para Silício Real de US$ 3 com Prova Bit-Exata:** A vasta maioria dos artigos de sEMG para em simulações em PC/GPU. Este projeto rodou **287.830 inferências reais em hardware físico** (`100.630` nos 10 modelos × 101 sujeitos + `187.200` no stress test contínuo de 40 sujeitos) no `ESP32-S3-WROOM-1`, com **100,00% de paridade bit-exata** e **0 violações de tempo real** sob FreeRTOS SMP.
2. **Rigor Anti-Vazamento (LOTO + `PROVE-OR-QUARANTINE`):** Rejeita o *random split* de janelas sobrepostas (que infla falsamente artigos da literatura para 95–99%), utiliza divisão estritamente por repetições inteiras e auditou/corrigiu 18 gravações desalinhadas do NinaPro DB2 E3.
3. **Leis Práticas de Co-Design Hardware-Software:** Comprova empiricamente a *Lei da Invariância em 16 bits* (`int16 ≈ float32`, -50% banda), a superioridade de *mais canais em 8 bits sobre menos canais em 32 bits*, e o diagnóstico do colapso da CNN-1D em 8 bits (e sua mitigação via *SupCon*).
4. **Inovação de Engenharia no Firmware (`SEW1` em RAM):** Resolve o problema real de queda cross-subject (`42% → 77%`) permitindo personalizar pesos por paciente em `< 0,2 s` sem regravar Flash.
5. **Honestidade Clínica com Amputados (NinaPro DB3):** Não esconde a queda severa em coto residual (`-17 a -21 pp` frente a saudáveis), documentando com dados reais de silício o quanto falta para próteses acessíveis atingirem confiabilidade clínica plena.

---

## 5. Pontos Negativos, Limitações e Ameaças à Validade (O Que Pode Comprometer o Trabalho)

Para defender este projeto perante uma banca rigorosa, é essencial reconhecer abertamente seus pontos fracos, classificados pelos 4 pilares clássicos de validade científica:

### 5.1. Validade Externa (Generalização para o Mundo Real)
1. **Avaliação estritamente Offline / Hardware-in-the-Loop (HIL), sem teste Online com humano e atuador:** Os sinais foram injetados via USB-Serial (`/dev/ttyACM0`) a partir de arquivos pré-gravados. Não houve aquisição ao vivo via eletrodos + AFE (ex.: ADS1299), nem ruído eletromagnético dos motores da prótese, nem suor/fadiga muscular, nem feedback visual/proprioceptivo do paciente em malha fechada.
2. **Dependência de calibração intra-sujeito (LOTO) em sessão única:** Os resultados elevados (`67–78%`) pressupõem calibração prévia no próprio usuário na mesma sessão. Não medimos a degradação por **deslocamento de eletrodos (*electrode shift*)** ao vestir/retirar o encaixe da prótese em dias diferentes (*cross-day*).
3. **Amostra pequena e heterogênea de amputados (`N = 11` no DB3):** Desvio padrão muito alto (`±11,5%` a `±13,0%`) e número desigual de gestos executáveis por paciente (39 a 50 classes), limitando a generalização estatística para toda a população de amputados transradiais.

### 5.2. Validade de Construto (Se as Métricas Medem Realmente o Que Prometem)
4. **Energia estimada analiticamente, não medida eletricamente:** Os valores em `mJ` e o `IEE` usam potência fixa de datasheet (`148,5 mW × t`), sem medição física real com sensor de corrente/osciloscópio (ex.: INA219 / Nordic PPK2) que capture picos de Flash SPI, regulador de tensão, DMA e rádio BLE.
5. **Viés dos índices `IRC` e `IEE` sem restrição mínima de acurácia:** Como demonstrado no DB3 (`33,68%` liderando IRC/IEE), maximizar `IRC` ou `IEE` sem impor um piso mínimo (ex.: `Acc ≥ 60%` ou `Macro-F1 ≥ 50%`) pode apontar como "vencedor" um modelo inútil na prática.
6. **Top-1 inflada por desbalanceamento de classes no NinaPro (DB2/DB3):** A classe de repouso e movimentos longos elevam a Top-1 em até **+14 pp** acima da `Macro-F1` no DB3. Avaliar apenas Top-1 superestima a capacidade de controlar gestos finos individuais.

### 5.3. Validade Interna (Assimetrias Experimentais Dentro do Estudo)
7. **Assimetria entre modelos físicos (`M01–M10`) e modelos simulados (`M11–M14`):** `M11`–`M14` (inclusive o campeão `M14` de 636 features) ainda **não foram executados no silício real do ESP32-S3** nem avaliados nos 40 sujeitos do NinaPro DB2 / 11 amputados do DB3. Suas latências são estimativas de ciclos que não capturam stalls reais de cache/Flash nem o tempo embarcado completo de extração das 636 features.
8. **Falta de otimização SIMD PIE (`esp-dsp`) nas convoluções:** O firmware C99 da CNN-1D usa laços escalares portáveis. Isso penaliza artificialmente a latência física da CNN-1D (`5,9–14,1 ms`) frente ao Micro-LDA no ESP32-S3 (que possui acelerador vetorial PIE de 128 bits).
9. **Subamostragem estratificada (100 janelas/sujeito) no benchmark físico de 10 modelos:** Por restrição de tempo de transporte serial (`~2h`), o benchmark físico populacional de 10 modelos avaliou 100 janelas estratificadas por sujeito (embora o benchmark físico exclusivo de Ozdemir tenha validado 100% das 2.340 janelas/sujeito = 187.200 inferências).
10. **Falta de calibração automática do limiar `τ` nas cascatas (`M07`/`M10`):** Em 50 classes, `EE = 0,0%` nos sujeitos de referência mostrou que o limiar fixo anulou o benefício de saída antecipada.

### 5.4. Validade de Conclusão (Limites Clínicos e Estatísticos)
11. **Acurácia em amputados (`49,47%` Top-1 / `35,73%` Macro-F1 em 50 classes) ainda está abaixo do aceitável para uso comercial seguro:** Embora 50 classes sejam muito mais exigentes que próteses reais (que operam com 6–10 gestos), o próprio resultado confirma que ainda é necessário reduzir o repertório funcional (ex.: 8–10 AVDs críticas) e incorporar adaptação online para atingir >80% de segurança clínica em amputados.
12. **Uso de `K=9` (`400 ms`) em números de destaque do `M14`:** O pico de `71,06%` no NinaPro DB2 ultrapassa a barreira clínica de 300 ms; o número clinicamente defensável do `M14` é **`68,59%` em `K=5` (`200 ms`)** — e ainda pendente de validação física no silício e em toda a população.
