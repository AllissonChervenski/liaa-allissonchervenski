# Firmware Nativo ESP-IDF v5.x — ESP32-S3-WROOM (`semg_esp32s3`)

Firmware de produção dual-core FreeRTOS SMP para o módulo **ESP32-S3-WROOM** (Xtensa LX7 @ 240 MHz), trazendo o pipeline DSP causal, extrator Zero-FPU e classificadores quantizados (**Micro-LDA Q7** e **CNN-1D INT8**) validados no emulador QEMU para silício real sem semihosting.

---

## 1. Arquitetura Dual-Core SMP

| Núcleo | Tarefa FreeRTOS | Prioridade | Responsabilidade |
| :--- | :--- | :---: | :--- |
| **Core 0 (`PRO_CPU`)** | `vTaskAquisicaoCore0` | `10` | Recebe blocos DMA Ping-Pong (100 amostras / 50 ms), aplica filtro causal Butterworth 4ª ordem ($20\text{–}450\text{ Hz}$ DF2T), alimenta o buffer circular ($W=400, S=100$) e envia descritores via `xQueueJanelaPronta`. |
| **Core 1 (`APP_CPU`)** | `vTaskInferenciaCore1` | `5` | Extrai atributos Hudgins em inteiros (`extract_td_features_int16`), executa **Micro-LDA (Q7)** ou **CNN-1D (INT8)** com pesos estáticos em Flash (`.rodata`), aplica voto majoritário (3 janelas) e emite telemetria JSON. |

---

## 2. Modos de Aquisição (`idf.py menuconfig`)

Selecione em `sEMG ESP32-S3-WROOM Configuration -> Signal Acquisition Mode`:

1. **`HIL_SERIAL` (Padrão):** Hardware-in-the-Loop via USB-Serial/JTAG (`/dev/ttyACM*`) ou UART0 (`/dev/ttyUSB*`). Recebe quadros binários protegidos por CRC16-CCITT do script [`scripts/validate_esp32s3_hil.py`](../../scripts/validate_esp32s3_hil.py) e valida paridade bit-exata na placa real.
2. **`ADC_DMA`:** Leitura contínua de eletrodos sEMG reais via SAR ADC1 + GDMA (`esp_adc/adc_continuous.h`) com sobreamostragem 4x ($32\text{ kHz}$ total $\rightarrow$ decimação exata para $2.000\text{ Hz}$/canal):
   * **Canal 0:** `GPIO1` (`ADC1_CH0`)
   * **Canal 1:** `GPIO2` (`ADC1_CH1`)
   * **Canal 2:** `GPIO3` (`ADC1_CH2`)
   * **Canal 3:** `GPIO4` (`ADC1_CH3`)
   * *(Para 12 canais, canais 0–9 usam `GPIO1..GPIO10` no ADC1 e 10–11 usam `GPIO11..GPIO12` no ADC2).*
3. **`SELFTEST_LOOP`:** Reprodução contínua a $2.000\text{ Hz}$ via `esp_timer` usando os vetores Ozdemir S1 embarcados na Flash.

---

## 3. Como Compilar e Gravar no ESP32-S3-WROOM

### Pré-requisitos
Instalar o **ESP-IDF v5.x** (recomendado v5.2+):
```bash
mkdir -p ~/esp && cd ~/esp
git clone -b v5.3.2 --recursive https://github.com/espressif/esp-idf.git
cd ~/esp/esp-idf && ./install.sh esp32s3
```

### Atualização de Pesos Estáticos em Flash (Opcional)
Para re-exportar os pesos estáticos de Micro-LDA Q7 e o stream de auto-teste:
```bash
.venv/bin/python src/export/export_micro_lda_c_header.py
.venv/bin/python src/export/export_cnn1d_c_header.py
```

### Build, Flash e Monitor Serial
```bash
source ~/esp/esp-idf/export.sh
cd firmware/esp32s3_wroom

idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults.esp32s3" set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

### Validação Hardware-in-the-Loop (HIL) na Placa Física
Após gravar o firmware no modo `HIL_SERIAL`, execute do diretório raiz do repositório:
```bash
.venv/bin/python scripts/validate_esp32s3_hil.py --port /dev/ttyACM0 --dataset ozdemir --windows 50
```
