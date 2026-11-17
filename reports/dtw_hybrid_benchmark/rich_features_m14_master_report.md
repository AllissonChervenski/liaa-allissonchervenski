# Relatório de Benchmark: M14 Extração Rica (636 Features) + SupCon MLP INT8

**Data:** 2026-11-17 08:38:26 -0300 | **Tempo Total:** 307.74 s

## 1. Engenharia de Features Multi-Domínio (60 -> 636 Features em 12CH)

- **TD Não-Linear & Log-Amplitude (8/ch):** MAV, LOG_RMS, WL, MFL, DASDV, WAMP, Form Factor, Crest Factor.
- **Morfologia, Hjorth & HOS (6/ch):** ZC, SSC, Mobilidade, Complexidade, Assimetria (Skew), Log-Curtose.
- **Polos Auto-Regressivos AR-4 (4/ch):** Yule-Walker regularizado em ordem 4.
- **Espectro Multi-Banda & Cepstro EMGCC (11/ch):** Potência Log em 5 bandas (20–500 Hz), MNF, Entropia Espectral e DCT Cepstral (EMGCC1..4).
- **Wavelet Haar 4 Níveis (5/ch):** Log-Energias `cD1, cD2, cD3, cD4, cA4`.
- **Envelope Temporal em Sub-janelas (8/ch):** 4x Log-MAV + 4x Log-WL (sub-janelas de 50 ms).
- **Matriz de Sinergia Muscular Inter-Canais (132 em 12CH):** 66 correlações cruzadas de Pearson + 66 razões Log-RMS par-a-par invariantes à força.

## 2. Resultados Consolidados (Enhanced-LDA vs. M14 SupCon MLP INT8 + Passo 1)

| Dataset | Canais | Feats | Enh-LDA Raw | M14 FP32 | M14 INT8 ($K=1$) | M14 + MV ($K=5$) | M14 + Soft ($K=5$, 200ms) | M14 + Soft ($K=9$, 400ms) | Latência ESP32-S3 | Flash / SRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NinaPro DB2 (50 cls) | 12CH | 636 | 54.59% | 64.35% | 63.06% | 67.65% | **68.59%** | **71.06%** | 0.425 ms | 182.0 KB / 7.6 KB |
| NinaPro DB2 (50 cls) | 2CH | 86 | 25.46% | 45.46% | 45.10% | 47.58% | **48.62%** | **50.94%** | 0.062 ms | 22.2 KB / 3.0 KB |
| Ozdemir (10 cls) | 4CH | 180 | 84.27% | 80.90% | 79.53% | 81.03% | **82.01%** | **82.14%** | 0.118 ms | 33.7 KB / 3.9 KB |
| Ozdemir (10 cls) | 2CH | 86 | 76.84% | 74.79% | 74.19% | 77.95% | **79.19%** | **80.77%** | 0.057 ms | 14.5 KB / 3.0 KB |
