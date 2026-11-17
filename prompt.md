# ESPECIFICAÇÃO DE ENGENHARIA: Pipeline Multicore sob FreeRTOS no QEMU com Buffer Duplo e Avaliação Pareada Micro-LDA vs CNN-1D

## 1. OBJETIVO DO MÓDULO

Implementar e validar no ambiente de emulação Espressif QEMU (alvo ESP32-S3 Xtensa LX7 a 240 MHz) uma arquitetura multitarefa sob FreeRTOS para processamento determinístico de séries temporais contínuas em tempo real. O firmware deve desacoplar a recepção contínua de amostras da execução dos classificadores de aprendizado de máquina (Micro-LDA e CNN-1D quantizados em inteiros de 8 bits), aferindo ciclos de processamento, ocupação de memória SRAM e estabilidade temporal das tarefas.

---

## 2. ESTRUTURAÇÃO DAS TAREFAS NO FREERTOS (ESP-IDF)

A aplicação deve instanciar duas rotinas com afinidade estrita de núcleo via `xTaskCreatePinnedToCore`:

### 2.1. Rotina A: `vTaskAquisicaoCore0` (Core 0, Prioridade Alta: 10)

* **Finalidade:** Recepção determinística das amostras discretizadas e filtragem digital causal.
* **Funcionamento:** Permanece em espera aguardando notificação ou semáforo binário de bloco de dados pronto. Ao ser ativada, processa as amostras através do filtro biquadrático em seções de segunda ordem (SOS DF2T) e insere os dados no buffer circular deslizante.
* **Diretrizes de Determinismo:** Sem alocação dinâmica no heap (`malloc`) durante a execução em regime contínuo e sem operações de escrita formatada em console dentro do laço crítico.

### 2.2. Rotina B: `vTaskInferenciaCore1` (Core 1, Prioridade Média: 5)

* **Finalidade:** Extração de descritores matemáticos e execução dos modelos preditivos.
* **Funcionamento:** Aguarda itens na fila `xQueueJanelaPronta`. Ao receber o descritor de uma janela completa de 400 amostras (deslocamento a cada 100 amostras / 50 ms):
  1. Executa o algoritmo selecionado por diretiva de compilação:
     * **Configuração Micro-LDA:** calcula as 5 estatísticas no domínio do tempo (conjunto de Hudgins: MAV, RMS, WL, ZC e SSC em aritmética inteira) e projeta no discriminante linear quantizado em Q7;
     * **Configuração CNN-1D:** transfere o tensor de entrada diretamente para a área estática de ativações (4,61 KB) e executa as camadas convolucionais quantizadas em INT8;
  2. Registra o tempo de resposta e o identificador da classe prevista;
  3. Aplica filtro de consenso temporal sobre as últimas 3 predições para suavização de saída;
  4. Cede o processador temporariamente via `vTaskDelayUntil` para representar o período de economia de energia até o próximo ciclo.

---

## 3. MECANISMOS DE AQUISIÇÃO E SINCRONISMO

### 3.1. Buffer Duplo de Entrada (Esquema Ping-Pong)

* Definir estaticamente dois blocos contíguos de memória RAM (`dma_ping_buffer` e `dma_pong_buffer`) dimensionados para receber os blocos de aquisição (100 amostras x C canais x 2 bytes);
* Alternar ponteiros de escrita (`active_dma_buffer`) e leitura (`processing_buffer`) com identificador de banco (`buffer_id`) e semáforo binário (`xBufferReadySem`) para impedir concorrência de acesso entre a rotina de recepção e a rotina de processamento.

### 3.2. Temporizador Periódico de Amostragem (Timer ISR)

* Configurar um temporizador periódico de hardware da plataforma programado para operar na cadência equivalente à taxa de amostragem de 2.000 Hz (500 µs por amostra / blocos de 100 amostras a cada 50 ms);
* A rotina de serviço do temporizador (`mock_dma_transfer_isr`) transfere as amostras da sequência de teste para o buffer de entrada ativo, alterna os bancos Ping/Pong e sinaliza o semáforo de sincronismo para ativar a tarefa do Core 0 via `xSemaphoreGiveFromISR` com `portYIELD_FROM_ISR`.

### 3.3. Fila de Comunicação Inter-Core

* Instanciar a fila `xQueueJanelaPronta` (`xQueueWindowReady`) para enviar descritores leves de janela ou vetores compactos de atributos, evitando cópias desnecessárias de tensores entre os núcleos.

---

## 4. PARÂMETROS DOS CLASSIFICADORES AVALIADOS

O sistema deve permitir seleção por diretiva de compilação ou configuração entre os dois classificadores validados:

* **Modelo 1: Micro-LDA (Q7)**
  * Coeficientes estáticos em Flash (`.rodata`): 200 bytes para topologia de 4 canais / 3.000 bytes para 12 canais;
  * Alocação em memória volátil de ativação: 0 bytes adicionais;
  * Utiliza vetor pré-calculado de estatísticas temporais;
  * Latência esperada do classificador puro: cerca de 1 µs.
* **Modelo 2: CNN-1D (INT8)**
  * Coeficientes convolucionais em Flash (`.rodata`): aproximadamente 3.000 bytes para 4 canais / 7.740 bytes para 12 canais;
  * Área estática para tensores intermediários em SRAM: 4.720 bytes (4,61 KB);
  * Entrada temporal direta da janela sem cálculo prévio de atributos de Hudgins;
  * Latência esperada do classificador puro: cerca de 123 a 214 µs.

---

## 5. DIAGNÓSTICO E CRITÉRIOS DE AVALIAÇÃO NO EMULADOR

A rotina de validação executada via Espressif QEMU 9.2.2 deve registrar:

1. **Margem de Pilha das Tarefas:** consultar `uxTaskGetStackHighWaterMark` para verificar a folga estática de memória de cada tarefa (`stack_high_water_mark > 512 bytes`);
2. **Regularidade Temporal e Jitter:** verificar se o intervalo entre janelas consecutivas permanece estável em 50.000 µs ± 500 µs sem atrasos de ciclo (`deadline_misses == 0` ao longo de pelo menos 1.000 janelas);
3. **Paridade Preditiva sob Escalonamento:** confirmar que a concordância com a referência numérica do host seja de 100,00% (`host_parity_ratio == 1.000000`) tanto para o Micro-LDA quanto para a CNN-1D;
4. **Taxa de Ocupação e Índices Unificados:** aferir o percentual de tempo livre de CPU (`> 90%` no Core 1) e computar os índices de Responsividade (IRC) e Eficiência Embarcada (IEE) com as medições obtidas sob o FreeRTOS.
