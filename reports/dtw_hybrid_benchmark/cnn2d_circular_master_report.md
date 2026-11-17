# Relatório de Benchmark: Spatial-Temporal Circular CNN-2D (M12) — Passo 2

**Data:** 2026-11-17 06:44:16 -0300 | **Tempo Total:** 854.45 s

## 1. Resumo Arquitetural (M12 Circular CNN-2D)

- **Ramo Circumferencial em Anel (Circular Conv2D):** Modela os 8 eletrodos do anel do antebraço (NinaPro Ch 0..7) ou 4 eletrodos (Ozdemir) com *padding espacial circular* (`ch[C-1] <-> ch[0]`), capturando propagação transversal de MUAPs.
- **Preservação de Quadrantes Azimutais:** Ao invés de colapsar todos os eletrodos via GAP global único, retém **4 quadrantes anatômicos espaciais** (flexores vs. extensores vs. radial vs. ulnar).
- **Ramo Anatômico Dedicado (NinaPro 12CH):** Processa isoladamente os eletrodos Ch 8..11 (FDS, EDC, Bíceps, Tríceps).
- **Quantização INT8 com Folding de BatchNorm:** Todo `Conv + BatchNorm` é fundido antes da quantização simétrica INT8/Q24 (custo zero em runtime no ESP32-S3).

## 2. Resultados: Precisão (FP32 vs. INT8) + Suavização Temporal (Passo 1)

| Dataset | Canais | FP32 Raw | INT8 Raw ($K=1$) | INT8 + MV ($K=5$, 200ms) | INT8 + Soft ($K=5$, 200ms) | INT8 + Soft ($K=9$, 400ms) | Parâmetros | Flash (KB) | Latência ESP32-S3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NinaPro DB2 (50 cls) | 12CH | 58.94% | 58.36% | **60.50%** | **61.61%** | **63.42%** | 14,750 | 14.74 KB | 1.249 ms |
| NinaPro DB2 (50 cls) | 2CH | 40.00% | 39.74% | **40.42%** | **40.79%** | **41.42%** | 3,442 | 3.61 KB | 0.074 ms |
| Ozdemir (10 cls) | 4CH | 79.06% | 78.76% | **80.68%** | **81.07%** | **82.52%** | 5,850 | 5.85 KB | 0.590 ms |
| Ozdemir (10 cls) | 2CH | 63.42% | 62.52% | **64.66%** | **66.15%** | **67.48%** | 2,042 | 2.13 KB | 0.073 ms |
