# Relatório de Benchmark: M13 SupCon Circular CNN-2D (Passo 3)

**Data:** 2026-11-17 07:57:35 -0300 | **Temperatura SupCon:** $\tau=0.1$ | **Tempo Total:** 1433.3 s

## 1. Resumo da Abordagem (Passo 3)

- **Stage 1 (Class-Balanced SupCon):** Treina o backbone `CompactCircularSemgCnn2D` + *Projection Head* MLP (`112 -> 128 -> 64`) aproximando repetições do mesmo gesto e repelindo as demais classes. Âncoras ponderadas inversamente pela frequência no lote evitam que o repouso (36,3% do NinaPro DB2) domine os gradientes.
- **Stage 2 (Prototype Init & Fine-Tuning):** Descarta a *Projection Head*, inicializa `fc_conv` com os centróides L2-normalizados dos clusters contrastivos e refina ponta-a-ponta.
- **Custo Zero em TinyML:** O modelo exportado em INT8 possui exatamente os mesmos **14.750 parâmetros**, **14,74 KB de Flash** e **1,249 ms** de latência do M12 no ESP32-S3.

## 2. Resultados Consolidados (INT8 + Passo 1: Votação Majoritária & Soft Voting)

| Dataset | Canais | FP32 Raw | INT8 Raw ($K=1$) | INT8 + MV ($K=5$) | INT8 + Soft ($K=5$, 200ms) | INT8 + Soft ($K=9$, 400ms) | Macro F1 ($K=5$) | Latência ESP32-S3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NinaPro DB2 (50 cls) | 12CH | 59.27% | 59.19% | 61.19% | **62.08%** | **63.57%** | **57.65%** | 1.249 ms |
| NinaPro DB2 (50 cls) | 2CH | 39.91% | 39.90% | 40.71% | **41.13%** | **42.00%** | **18.74%** | 0.074 ms |
| Ozdemir (10 cls) | 4CH | 78.59% | 78.68% | 81.15% | **81.11%** | **81.79%** | **80.97%** | 0.590 ms |
| Ozdemir (10 cls) | 2CH | 62.65% | 62.22% | 63.89% | **65.34%** | **66.92%** | **64.34%** | 0.073 ms |
