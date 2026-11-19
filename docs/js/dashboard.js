/**
 * dashboard.js — LIAA / UNIPAMPA sEMG & TinyML Benchmarks Dashboard
 * Controla:
 *  1. Gráfico comparativo principal de abertura (ESP32-S3 físico: LDA vs CNN-1D nos 4 datasets, IRC/IEE/Acc/Lat/Energia/SRAM)
 *  2. Gráfico comparativo dos 10 modelos (M01–M10) no silício físico por dataset
 *  3. Os 12 gráficos técnicos existentes com detalhamento expandido no card e no modal lightbox
 *  4. Renderização hierárquica completa: Modelos (M01–M14) > Número de Canais > Quantização
 *  5. Menu sanduíche lateral direito e acessibilidade (Escape / ARIA)
 */

const chartRegistry = {};
let modalChartInstance = null;
let heroChartInstance = null;
let phys10ChartInstance = null;

const heroState = {
  metric: "irc_iee",     // "irc_iee" | "acc_lat" | "energy_sram"
  scope: "population",   // "population" (padrão: média dos sujeitos) | "best_subject"
};

let phys10Dataset = "ninapro_db2";
let activeHierarchyModel = "M01"; // "M01".."M14" | "ALL"

document.addEventListener("DOMContentLoaded", function () {
  initHamburgerMenu();
  if (typeof Chart !== "undefined") {
    initPhysicalHero();
    initPhysical10Models();
    initCharts();
  }
  initBenchmarkHierarchy();
  initChartModal();
});

function fmtNum(val, decimals = 2) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
  return Number(val).toLocaleString("pt-BR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtBytes(bytes) {
  if (bytes >= 1024) {
    return `${fmtNum(bytes / 1024, 2)} KB`;
  }
  return `${Math.round(bytes)} B`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ============================================================================
   1. GRÁFICO COMPARATIVO DE ABERTURA: ESP32-S3 FÍSICO (LDA vs CNN-1D, 4 DATASETS)
   ============================================================================ */

function initPhysicalHero() {
  const data = window.SEMG_BENCHMARK_DATA;
  if (!data || !data.physicalHero) return;

  const metricBtns = document.querySelectorAll("[data-hero-metric]");
  const scopeBtns = document.querySelectorAll("[data-hero-scope]");

  metricBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      heroState.metric = btn.getAttribute("data-hero-metric");
      metricBtns.forEach((b) => b.classList.toggle("active", b === btn));
      renderPhysicalHero();
    });
  });

  scopeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      heroState.scope = btn.getAttribute("data-hero-scope");
      scopeBtns.forEach((b) => b.classList.toggle("active", b === btn));
      renderPhysicalHero();
    });
  });

  renderPhysicalHero();
}

function buildHeroTooltipLines(dsMeta, row, scope) {
  const accStr =
    scope === "population"
      ? `${fmtNum(row.top1_mean, 2)}% ± ${fmtNum(row.top1_std, 2)}% (média de ${dsMeta.num_subjects} suj.)`
      : `${fmtNum(row.top1_mean, 2)}% (melhor sujeito: ${row.subject})`;
  return [
    `Modelo: ${row.model_label}`,
    `Dataset: ${dsMeta.label}`,
    `Acurácia Top-1: ${accStr}`,
    `Latência Core 1 (Físico): ${fmtNum(row.lat_mean_ms, 3)} ms (${fmtNum(row.lat_mean_us, 1)} µs — ${fmtNum(row.cpu_budget_pct, 2)}% de 50 ms)`,
    `Energia / Inferência: ${fmtNum(row.energy_mj, 5)} mJ (${fmtNum(row.energy_uj, 2)} µJ @ 148,5 mW)`,
    `Recursos MCU: ROM ${fmtBytes(row.rom_bytes)} | SRAM ${fmtNum(row.sram_kb, 2)} KB | Stack livre C1: ${row.min_c1_bytes} B`,
    `IRC (Responsividade): ${fmtNum(row.irc, 1)} gestos·%/ms`,
    `IEE (Eficiência Embarcada): ${fmtNum(row.iee, 1)}`,
    `Paridade Bit-Exata HW: ${row.parity_matches}/${row.parity_total} (${fmtNum(row.parity_pct, 2)}%)`,
  ];
}

function renderPhysicalHero() {
  const data = window.SEMG_BENCHMARK_DATA;
  const hero = data.physicalHero;
  const datasets = hero.datasets;
  const bucket = hero[heroState.scope];
  const labels = datasets.map((d) => d.short_label);

  const ldaRows = datasets.map((d) => bucket[d.key]["M01_micro_lda_q7"]);
  const cnnRows = datasets.map((d) => bucket[d.key]["M02_cnn1d_int8"]);

  const captionEl = document.getElementById("heroChartCaption");
  const scopeLabel =
    heroState.scope === "population"
      ? "Média populacional completa (N = 101 sujeitos reais, 100,00% de paridade bit-exata)"
      : "Melhor sujeito individual de cada dataset no ESP32-S3-WROOM-1 físico";

  let config;
  if (heroState.metric === "irc_iee") {
    if (captionEl) {
      captionEl.textContent =
        `Índices IRC e IEE no ESP32-S3 físico (escala logarítmica) — ${scopeLabel}. Passe o cursor sobre qualquer barra para ver acurácia, latência, ROM/SRAM e energia.`;
    }
    config = {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Micro-LDA Q7 — IRC",
            data: ldaRows.map((r) => r.irc),
            backgroundColor: "rgba(37, 99, 235, 0.85)",
            borderColor: "rgb(37, 99, 235)",
            borderWidth: 1,
            borderRadius: 4,
            _metaRows: ldaRows,
          },
          {
            label: "Micro-LDA Q7 — IEE",
            data: ldaRows.map((r) => r.iee),
            backgroundColor: "rgba(124, 58, 237, 0.82)",
            borderColor: "rgb(124, 58, 237)",
            borderWidth: 1,
            borderRadius: 4,
            _metaRows: ldaRows,
          },
          {
            label: "CNN-1D INT8 — IRC",
            data: cnnRows.map((r) => r.irc),
            backgroundColor: "rgba(13, 148, 136, 0.85)",
            borderColor: "rgb(13, 148, 136)",
            borderWidth: 1,
            borderRadius: 4,
            _metaRows: cnnRows,
          },
          {
            label: "CNN-1D INT8 — IEE",
            data: cnnRows.map((r) => r.iee),
            backgroundColor: "rgba(245, 158, 11, 0.85)",
            borderColor: "rgb(217, 119, 6)",
            borderWidth: 1,
            borderRadius: 4,
            _metaRows: cnnRows,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              title: (items) => {
                const idx = items[0].dataIndex;
                return datasets[idx].label;
              },
              label: () => "",
              afterBody: (items) => {
                const it = items[0];
                const dsMeta = datasets[it.dataIndex];
                const row = it.dataset._metaRows[it.dataIndex];
                return buildHeroTooltipLines(dsMeta, row, heroState.scope);
              },
            },
          },
        },
        scales: {
          y: {
            type: "logarithmic",
            min: 10,
            title: {
              display: true,
              text: "Pontuação IRC / IEE no ESP32-S3 Físico [Escala Log]",
            },
          },
        },
      },
    };
  } else if (heroState.metric === "acc_lat") {
    if (captionEl) {
      captionEl.textContent =
        `Acurácia Top-1 LOTO (%) vs. Latência Real no Core 1 (ms, escala log) — ${scopeLabel}.`;
    }
    config = {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Micro-LDA Q7 — Acurácia (%)",
            data: ldaRows.map((r) => r.top1_mean),
            backgroundColor: "rgba(37, 99, 235, 0.82)",
            borderRadius: 4,
            yAxisID: "yAcc",
            _metaRows: ldaRows,
          },
          {
            label: "CNN-1D INT8 — Acurácia (%)",
            data: cnnRows.map((r) => r.top1_mean),
            backgroundColor: "rgba(13, 148, 136, 0.82)",
            borderRadius: 4,
            yAxisID: "yAcc",
            _metaRows: cnnRows,
          },
          {
            type: "line",
            label: "Micro-LDA Q7 — Latência Física (ms)",
            data: ldaRows.map((r) => r.lat_mean_ms),
            borderColor: "rgb(29, 78, 216)",
            backgroundColor: "rgb(29, 78, 216)",
            borderWidth: 2.5,
            borderDash: [5, 4],
            pointRadius: 6,
            tension: 0.15,
            yAxisID: "yLat",
            _metaRows: ldaRows,
          },
          {
            type: "line",
            label: "CNN-1D INT8 — Latência Física (ms)",
            data: cnnRows.map((r) => r.lat_mean_ms),
            borderColor: "rgb(220, 38, 38)",
            backgroundColor: "rgb(220, 38, 38)",
            borderWidth: 2.5,
            pointRadius: 6,
            tension: 0.15,
            yAxisID: "yLat",
            _metaRows: cnnRows,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              title: (items) => datasets[items[0].dataIndex].label,
              label: () => "",
              afterBody: (items) => {
                const it = items[0];
                return buildHeroTooltipLines(
                  datasets[it.dataIndex],
                  it.dataset._metaRows[it.dataIndex],
                  heroState.scope
                );
              },
            },
          },
        },
        scales: {
          yAcc: {
            type: "linear",
            position: "left",
            min: 0,
            max: 100,
            title: { display: true, text: "Acurácia Top-1 LOTO (%)" },
          },
          yLat: {
            type: "logarithmic",
            position: "right",
            min: 0.1,
            max: 30,
            grid: { drawOnChartArea: false },
            title: { display: true, text: "Latência Core 1 (ms) [Log]" },
          },
        },
      },
    };
  } else {
    if (captionEl) {
      captionEl.textContent =
        `Consumo Energético por Inferência (mJ @ 148,5 mW, escala log) e Pegada de SRAM Runtime (KB) — ${scopeLabel}.`;
    }
    config = {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Micro-LDA Q7 — Energia (mJ)",
            data: ldaRows.map((r) => r.energy_mj),
            backgroundColor: "rgba(16, 185, 129, 0.85)",
            borderRadius: 4,
            yAxisID: "yEnergy",
            _metaRows: ldaRows,
          },
          {
            label: "CNN-1D INT8 — Energia (mJ)",
            data: cnnRows.map((r) => r.energy_mj),
            backgroundColor: "rgba(239, 68, 68, 0.82)",
            borderRadius: 4,
            yAxisID: "yEnergy",
            _metaRows: cnnRows,
          },
          {
            type: "line",
            label: "Micro-LDA Q7 — SRAM (KB)",
            data: ldaRows.map((r) => r.sram_kb),
            borderColor: "rgb(37, 99, 235)",
            backgroundColor: "rgb(37, 99, 235)",
            borderWidth: 2.5,
            borderDash: [5, 4],
            pointRadius: 6,
            yAxisID: "ySram",
            _metaRows: ldaRows,
          },
          {
            type: "line",
            label: "CNN-1D INT8 — SRAM (KB)",
            data: cnnRows.map((r) => r.sram_kb),
            borderColor: "rgb(124, 58, 237)",
            backgroundColor: "rgb(124, 58, 237)",
            borderWidth: 2.5,
            pointRadius: 6,
            yAxisID: "ySram",
            _metaRows: cnnRows,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              title: (items) => datasets[items[0].dataIndex].label,
              label: () => "",
              afterBody: (items) => {
                const it = items[0];
                return buildHeroTooltipLines(
                  datasets[it.dataIndex],
                  it.dataset._metaRows[it.dataIndex],
                  heroState.scope
                );
              },
            },
          },
        },
        scales: {
          yEnergy: {
            type: "logarithmic",
            position: "left",
            min: 0.01,
            max: 5,
            title: { display: true, text: "Energia / Inferência (mJ) [Log]" },
          },
          ySram: {
            type: "linear",
            position: "right",
            min: 0,
            max: 35,
            grid: { drawOnChartArea: false },
            title: { display: true, text: "SRAM Runtime (KB)" },
          },
        },
      },
    };
  }

  const canvas = document.getElementById("chartPhysicalHero");
  if (canvas) {
    if (heroChartInstance) {
      heroChartInstance.destroy();
    }
    heroChartInstance = new Chart(canvas, config);
  }

  // Atualiza a tabela executiva acompanhante logo abaixo do gráfico
  const tbody = document.getElementById("heroPhysicalTableBody");
  if (tbody) {
    const rowsHtml = [];
    datasets.forEach((ds) => {
      ["M01_micro_lda_q7", "M02_cnn1d_int8"].forEach((mkey) => {
        const r = bucket[ds.key][mkey];
        const isLda = mkey === "M01_micro_lda_q7";
        const accCell =
          heroState.scope === "population"
            ? `<strong>${fmtNum(r.top1_mean, 2)}%</strong> <small>&plusmn; ${fmtNum(r.top1_std, 2)}%</small>`
            : `<strong>${fmtNum(r.top1_mean, 2)}%</strong> <small>(${escapeHtml(r.subject)})</small>`;
        rowsHtml.push(`
          <tr class="${isLda ? "row-physical" : ""}">
            <td><strong>${escapeHtml(ds.label)}</strong><br><small style="color:var(--color-text-muted);">${escapeHtml(ds.population_desc)}</small></td>
            <td><span class="badge ${isLda ? "badge-linear" : "badge-dl"}">${escapeHtml(r.model_label)}</span></td>
            <td>${accCell}</td>
            <td><strong>${fmtNum(r.lat_mean_ms, 3)} ms</strong><br><small style="color:var(--color-text-muted);">${fmtNum(r.lat_mean_us, 1)} &mu;s (${fmtNum(r.cpu_budget_pct, 2)}% de 50 ms)</small></td>
            <td>${fmtBytes(r.rom_bytes)} ROM<br><small style="color:var(--color-text-muted);">${fmtNum(r.sram_kb, 2)} KB SRAM</small></td>
            <td>${fmtNum(r.energy_mj, 5)} mJ<br><small style="color:var(--color-text-muted);">${fmtNum(r.energy_uj, 1)} &mu;J</small></td>
            <td><strong>${fmtNum(r.irc, 1)}</strong></td>
            <td><strong>${fmtNum(r.iee, 1)}</strong></td>
            <td><span class="badge badge-success">${fmtNum(r.parity_pct, 0)}%</span></td>
          </tr>
        `);
      });
    });
    tbody.innerHTML = rowsHtml.join("");
  }
}

/* ============================================================================
   2. GRÁFICO FÍSICO DOS 10 MODELOS (M01–M10) NO ESP32-S3-WROOM-1
   ============================================================================ */

function initPhysical10Models() {
  const data = window.SEMG_BENCHMARK_DATA;
  if (!data || !data.physicalAll10) return;

  const btns = document.querySelectorAll("[data-phys10-ds]");
  btns.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      phys10Dataset = btn.getAttribute("data-phys10-ds");
      btns.forEach((b) => b.classList.toggle("active", b === btn));
      renderPhysical10Models();
    });
  });

  renderPhysical10Models();
}

function renderPhysical10Models() {
  const data = window.SEMG_BENCHMARK_DATA;
  const rows = data.physicalAll10[phys10Dataset];
  const dsMeta = data.physicalHero.datasets.find((d) => d.key === phys10Dataset);
  const canvas = document.getElementById("chartPhysical10Models");
  if (!canvas || !rows) return;

  const config = {
    type: "bar",
    data: {
      labels: rows.map((r) => r.model_label),
      datasets: [
        {
          label: `Acurácia Top-1 Populacional (${dsMeta.short_label}) [%]`,
          data: rows.map((r) => r.top1_mean),
          backgroundColor: "rgba(37, 99, 235, 0.8)",
          borderRadius: 4,
          yAxisID: "yAcc",
        },
        {
          type: "line",
          label: "Latência Média Core 1 ESP32-S3 Físico [ms]",
          data: rows.map((r) => r.lat_mean_ms),
          borderColor: "rgb(220, 38, 38)",
          backgroundColor: "rgb(220, 38, 38)",
          borderWidth: 2.5,
          pointRadius: 5,
          tension: 0.2,
          yAxisID: "yLat",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const r = rows[items[0].dataIndex];
              return [
                `Top-1: ${fmtNum(r.top1_mean, 2)}% ± ${fmtNum(r.top1_std, 2)}% | Macro-F1: ${fmtNum(r.f1_mean, 2)}%`,
                `Latência Física: ${fmtNum(r.lat_mean_ms, 3)} ms (${fmtNum(r.lat_mean_us, 1)} µs)`,
                `Energia: ${fmtNum(r.energy_mj, 5)} mJ | ROM: ${fmtBytes(r.rom_bytes)} | SRAM: ${fmtNum(r.sram_kb, 2)} KB`,
                `IRC: ${fmtNum(r.irc, 1)} | IEE: ${fmtNum(r.iee, 1)} | Paridade: 100,00%`,
              ];
            },
          },
        },
      },
      scales: {
        yAcc: {
          position: "left",
          min: 0,
          max: 100,
          title: { display: true, text: "Acurácia Top-1 LOTO (%)" },
        },
        yLat: {
          type: "logarithmic",
          position: "right",
          min: 0.1,
          max: 60,
          grid: { drawOnChartArea: false },
          title: { display: true, text: "Latência Core 1 (ms) [Log]" },
        },
      },
    },
  };

  if (phys10ChartInstance) {
    phys10ChartInstance.destroy();
  }
  phys10ChartInstance = new Chart(canvas, config);

  chartRegistry["chartPhysical10Models"] = {
    title: `10 Modelos no ESP32-S3 Físico — ${dsMeta.label}`,
    desc: `Média populacional (${dsMeta.population_desc}) com carregamento dinâmico SEW1 em RAM e 100% de paridade bit-exata.`,
    techDetails: `<strong>O que foi realizado:</strong> Execução real dos 10 modelos C99 (M01–M10) no silício físico ESP32-S3-WROOM-1 sobre todos os ${dsMeta.num_subjects} sujeitos do dataset ${dsMeta.label} (${dsMeta.hw_windows_per_model * 10} inferências físicas). <strong>Por que essas escolhas:</strong> Carregamento dinâmico SEW1 em RAM evita re-flash por sujeito e valida simultaneamente acurácia, latência Core 1 via esp_timer_get_time() e paridade 100% contra o oráculo Host.`,
    config,
  };
}

/* ============================================================================
   3. OS 12 GRÁFICOS TÉCNICOS COMPLETOS (COM DETALHAMENTO EXPANDIDO)
   ============================================================================ */

function initCharts() {
  const models = [
    "M01 (LDA)", "M02 (CNN1D)", "M03 (DTW-O2)", "M04 (DTW-SC)", "M05 (DTW-LB)",
    "M06 (H-Head)", "M07 (H-Casc)", "M08 (H-DTW+L)", "M09 (H-DTW+C)", "M10 (Tri-H)",
    "M11 (Hier)", "M12 (CNN2D-Circ)", "M13 (SupCon-CNN2D)", "M14 (636f+SupCon)",
  ];

  // 1. Acurácia Top-1 Comparativa por Modelo
  registerAndRenderChart("chartAccuracy", {
    title: "Acurácia Top-1 Comparativa das 14 Arquiteturas",
    desc: "NinaPro DB2 (S4: 50 classes, 12 canais) vs. Ozdemir et al. (S1: 10 classes, 4 canais) sob protocolo LOTO.",
    techDetails:
      "<strong>O que foi realizado:</strong> Avaliação padronizada das 14 arquiteturas (M01–M14) sob split LOTO estritamente livre de vazamento (treino reps [1,3,4,6] vs teste [2,5] no NinaPro; [1,3,4] vs [2,5] no Ozdemir). <strong>Decisões técnicas:</strong> Uso de Z-score calibrado apenas no treino, quantização de entrada int16 e comparação direta entre bases de alta densidade (50 classes) e baixa densidade (10 classes) para revelar onde Deep Learning e 636 features superam o LDA clássico.",
    config: {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "NinaPro DB2 (12CH, 50 Classes) [%]",
            data: [47.02, 59.21, 43.63, 43.63, 43.63, 48.08, 58.98, 49.59, 59.73, 49.26, 58.15, 61.61, 62.08, 68.59],
            backgroundColor: "rgba(37, 99, 235, 0.78)",
            borderColor: "rgb(37, 99, 235)",
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: "Ozdemir et al. (4CH, 10 Classes) [%]",
            data: [81.20, 80.81, 68.12, 67.73, 67.73, 81.79, 81.79, 81.32, 81.79, 82.78, 64.87, 81.07, 81.15, 82.01],
            backgroundColor: "rgba(13, 148, 136, 0.78)",
            borderColor: "rgb(13, 148, 136)",
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)}%`,
            },
          },
        },
        scales: {
          y: { min: 0, max: 100, title: { display: true, text: "Acurácia Top-1 (%)" } },
        },
      },
    },
  });

  // 2. Fronteira de Pareto: Latência vs Acurácia (NinaPro DB2)
  const paretoData = [
    { x: 0.0050, y: 47.02, label: "M01 (LDA)" },
    { x: 0.3493, y: 59.21, label: "M02 (CNN-1D)" },
    { x: 1.2666, y: 43.63, label: "M03 (DTW O(N²))" },
    { x: 0.3234, y: 43.63, label: "M04 (DTW SC)" },
    { x: 0.0634, y: 43.63, label: "M05 (DTW LB-Keogh)" },
    { x: 0.3537, y: 48.08, label: "M06 (H-Head)" },
    { x: 0.2703, y: 58.98, label: "M07 (Cascata)" },
    { x: 0.0385, y: 49.59, label: "M08 (DTW+LDA)" },
    { x: 0.3816, y: 59.73, label: "M09 (DTW+CNN)" },
    { x: 0.2168, y: 49.26, label: "M10 (Tri-Híbrido)" },
    { x: 0.0970, y: 58.15, label: "M11 (Hierárquico)" },
    { x: 1.2490, y: 61.61, label: "M12 (CNN2D-Circ K=5)" },
    { x: 1.2490, y: 62.08, label: "M13 (SupCon-CNN2D K=5)" },
    { x: 0.4247, y: 68.59, label: "M14 (636f+SupCon K=5)" },
  ];

  registerAndRenderChart("chartPareto", {
    title: "Fronteira de Pareto: Latência vs. Acurácia no NinaPro DB2",
    desc: "Tempo de inferência no ESP32-S3 @ 240 MHz (escala log, QEMU CCOUNT) vs. Acurácia Top-1.",
    techDetails:
      "<strong>O que significa:</strong> Modelos mais próximos do canto superior esquerdo (menor latência e maior acurácia) dominam a fronteira ótima de Pareto. <strong>Destaques:</strong> M01 domina em velocidade extrema (0,005 ms), M11 cria um degrau intermediário eficiente (0,097 ms, 58,15%) e M14 domina o topo absoluto (68,59% em 0,425 ms — 3× mais rápido que M12/M13).",
    config: {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Arquiteturas (NinaPro 12CH)",
            data: paretoData,
            backgroundColor: function (ctx) {
              const item = ctx.raw;
              if (item && item.label && item.label.includes("M14")) return "rgb(219, 39, 119)";
              if (item && item.label && item.label.includes("M11")) return "rgb(220, 38, 38)";
              if (item && item.label && item.label.includes("M01")) return "rgb(37, 99, 235)";
              if (item && item.label && item.label.includes("M07")) return "rgb(13, 148, 136)";
              return "rgba(100, 116, 139, 0.78)";
            },
            pointRadius: 7,
            pointHoverRadius: 10,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const item = ctx.raw;
                return `${item.label} → Latência: ${fmtNum(item.x, 4)} ms | Acurácia: ${fmtNum(item.y, 2)}%`;
              },
            },
          },
        },
        scales: {
          x: { type: "logarithmic", title: { display: true, text: "Latência ESP32-S3 @ 240 MHz (ms) [Escala Log]" } },
          y: { min: 35, max: 75, title: { display: true, text: "Acurácia Top-1 NinaPro DB2 (%)" } },
        },
      },
    },
  });

  // 3. Consumo Energético Médio por Inferência (mJ)
  registerAndRenderChart("chartEnergy", {
    title: "Consumo Energético Médio por Inferência (mJ — QEMU CCOUNT)",
    desc: "Energia computada com potência ativa nominal do ESP32-S3 (148,5 mW @ 240 MHz): Verde < 0,015 mJ | Amarelo < 0,050 mJ | Vermelho ≥ 0,050 mJ.",
    techDetails:
      "<strong>Cálculo:</strong> E(mJ) = 148,5 mW × (Latência_ms / 1000). <strong>Por que importa:</strong> Em uma prótese a 20 inferências/segundo (stride 50 ms), reduzir energia por janela de 0,188 mJ (DTW clássico) para 0,00074 mJ (Micro-LDA) ou 0,0144 mJ (M11) multiplica drasticamente a autonomia diária da bateria permitindo permanência > 99% em Light Sleep.",
    config: {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "Energia no ESP32-S3 (mJ / inferência)",
            data: [0.00074, 0.05188, 0.18809, 0.04802, 0.00941, 0.05253, 0.04014, 0.00571, 0.05668, 0.03220, 0.01440, 0.18548, 0.18548, 0.06306],
            backgroundColor: function (ctx) {
              const val = ctx.raw;
              if (val < 0.015) return "rgba(16, 185, 129, 0.82)";
              if (val < 0.050) return "rgba(245, 158, 11, 0.82)";
              return "rgba(239, 68, 68, 0.82)";
            },
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Energia: ${fmtNum(ctx.parsed.y, 5)} mJ (${fmtNum(ctx.parsed.y * 1000, 2)} µJ)`,
            },
          },
        },
        scales: {
          y: { min: 0, title: { display: true, text: "Consumo de Energia Médio (mJ)" } },
        },
      },
    },
  });

  // 4. Resiliência à Quantização: Float32 vs INT8
  registerAndRenderChart("chartQuant", {
    title: "Resiliência à Quantização: Float32 vs. INT8 (NinaPro 12CH)",
    desc: "Colapso da CNN-1D pura (9,86%) sob entrada 8-bit vs. estabilidade do Micro-LDA, M11 e blindagem contrastiva do M13/M14.",
    techDetails:
      "<strong>O que foi realizado:</strong> Simulação de ADC de 8 bits (±5 mV → LSB = 39,37 µV) e quantização INT8 sobre 15.824 janelas LOTO do NinaPro DB2. <strong>Por que ocorreu o colapso na M02:</strong> O degrau de 39,4 µV destrói micro-gradientes da camada Conv1D de entrada, enquanto o M13 (SupCon) compacta clusters L2 reduzindo a perda de quantização para apenas 0,08 pp.",
    config: {
      type: "bar",
      data: {
        labels: ["M01 (Micro-LDA)", "M02 (CNN-1D)", "M07 (Cascata)", "M11 (Hierárquico)", "M12 (CNN2D-Circ)", "M13 (SupCon-CNN2D)", "M14 (636f+SupCon)"],
        datasets: [
          {
            label: "Precisão Float32 [%]",
            data: [47.86, 58.62, 55.30, 58.15, 58.94, 59.27, 64.35],
            backgroundColor: "rgba(59, 130, 246, 0.82)",
            borderRadius: 4,
          },
          {
            label: "Quantização INT8 (8 bits) [%]",
            data: [53.11, 9.86, 42.44, 51.74, 58.36, 59.19, 63.06],
            backgroundColor: "rgba(244, 63, 94, 0.82)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)}%`,
            },
          },
        },
        scales: {
          y: { min: 0, max: 70, title: { display: true, text: "Acurácia Top-1 (%)" } },
        },
      },
    },
  });

  // 5. Índice IRC (QEMU CCOUNT — todas as 14 arquiteturas)
  registerAndRenderChart("chartIrc", {
    title: "Índice IRC (QEMU Bare-Metal): Responsividade de Decodificação",
    desc: "IRC = (Acurácia % × Nº Gestos) / Latência (ms) [Escala Logarítmica] para as 14 arquiteturas.",
    techDetails:
      "<strong>Significado:</strong> Quantifica quantos gestos corretamente decodificados a CPU entrega por milissegundo de computação pura. <strong>Decisão técnica:</strong> O uso de escala logarítmica permite comparar desde o M01 (469 mil a 686 mil no CCOUNT bare-metal) até redes 2D profundas (1,3 mil a 2,5 mil) no mesmo plano visual.",
    config: {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "IRC NinaPro DB2 (50 classes, 12CH)",
            data: [469671.4, 8474.1, 1722.3, 6745.4, 34428.5, 6795.7, 10909.1, 64423.3, 7825.5, 11359.4, 30047.2, 2336.23, 2369.37, 8075.76],
            backgroundColor: "rgba(99, 102, 241, 0.82)",
            borderRadius: 4,
          },
          {
            label: "IRC Ozdemir et al. (10 classes, 4CH)",
            data: [686361.8, 4640.7, 4268.0, 15356.6, 66289.9, 4683.6, 10846.1, 34210.3, 4167.7, 8900.5, 7208.9, 1335.66, 1334.21, 6954.68],
            backgroundColor: "rgba(20, 184, 166, 0.82)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 1)}`,
            },
          },
        },
        scales: {
          y: { type: "logarithmic", title: { display: true, text: "IRC = (Acc × Gestos) / Latência (ms)" } },
        },
      },
    },
  });

  // 6. Índice IEE (QEMU CCOUNT — todas as 14 arquiteturas)
  registerAndRenderChart("chartIee", {
    title: "Índice IEE (QEMU Bare-Metal): Eficiência Embarcada em Silício",
    desc: "IEE = (Acurácia % × Nº Gestos) / (Energia_mJ × log10(SRAM_KB)) [Escala Logarítmica].",
    techDetails:
      "<strong>Significado:</strong> Penaliza simultaneamente consumo energético e pegada de memória SRAM (em escala log10 para equilibrar o teto de 512 KB do ESP32-S3). <strong>Destaque:</strong> M01 lidera por não exigir arena convolucional (0 B extra), seguido por M08 (DTW+LDA), M05 (LB-Keogh) e M11 (Hierárquico, 3,5× acima da CNN-1D).",
    config: {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "IEE NinaPro DB2 (50 classes, 12CH)",
            data: [2349139.0, 39952.8, 7976.3, 33588.1, 170947.5, 31920.8, 51433.2, 320788.9, 36766.6, 53251.3, 144149.2, 10558.17, 10707.94, 61656.63],
            backgroundColor: "rgba(168, 85, 247, 0.82)",
            borderRadius: 4,
          },
          {
            label: "IEE Ozdemir et al. (10 classes, 4CH)",
            data: [5639679.2, 29773.7, 25918.4, 123185.6, 524019.8, 29692.2, 69586.0, 274423.6, 26440.2, 56070.9, 44360.3, 7615.64, 7607.37, 78852.96],
            backgroundColor: "rgba(249, 115, 22, 0.82)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 1)}`,
            },
          },
        },
        scales: {
          y: { type: "logarithmic", title: { display: true, text: "IEE = (Acc × Gestos) / (Energia × log10(SRAM))" } },
        },
      },
    },
  });

  // 7. Suavização Temporal Causal (Passo 1)
  registerAndRenderChart("chartSmoothing", {
    title: "Suavização Temporal Causal: Acurácia Top-1 vs. Janela K",
    desc: "Buffer circular FIFO causal de predições passadas (K = 1 a K = 9, atraso 0 a 400 ms) sem retreino.",
    techDetails:
      "<strong>O que foi realizado:</strong> Aplicação de votação majoritária e Soft Voting causal (acumulação inteira de logits int32) com reinicialização estrita nas fronteiras de repetição (15.824 janelas NinaPro e 2.340 Ozdemir). <strong>Por que K=5 (200 ms) é o padrão clínico:</strong> Estabiliza glitches transitórios de contração respeitando o limiar perceptivo de próteses (≤ 200–300 ms) com apenas 5 a 200 bytes de SRAM e < 0,2 µs de CPU.",
    config: {
      type: "line",
      data: {
        labels: ["Raw (K=1, 0ms)", "K=3 (100ms)", "K=5 (200ms)", "K=7 (300ms)", "K=9 (400ms)"],
        datasets: [
          {
            label: "NinaPro M14 (636f + SupCon MLP, Soft) [50cls]",
            data: [63.06, 66.55, 68.59, 69.99, 71.06],
            borderColor: "rgb(219, 39, 119)",
            backgroundColor: "rgba(219, 39, 119, 0.15)",
            borderWidth: 3,
            tension: 0.25,
            pointRadius: 6,
          },
          {
            label: "NinaPro M13 (SupCon CNN-2D, Soft) [50cls]",
            data: [59.19, 60.59, 62.08, 62.82, 63.57],
            borderColor: "rgb(5, 150, 105)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5,
          },
          {
            label: "NinaPro M12 (CNN-2D Circ, Soft) [50cls]",
            data: [58.36, 59.95, 61.61, 62.49, 63.42],
            borderColor: "rgb(2, 132, 199)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5,
          },
          {
            label: "NinaPro M02 (CNN-1D) [50cls]",
            data: [58.62, 59.88, 60.55, 61.43, 61.99],
            borderColor: "rgb(37, 99, 235)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5,
          },
          {
            label: "NinaPro M11 (Hierárquico) [50cls]",
            data: [58.08, 59.23, 59.79, 60.83, 61.62],
            borderColor: "rgb(220, 38, 38)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5,
          },
          {
            label: "NinaPro M01 (Micro-LDA) [50cls]",
            data: [47.97, 48.69, 49.42, 50.18, 50.80],
            borderColor: "rgb(147, 51, 234)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5,
          },
          {
            label: "Ozdemir M07 (Cascata) [10cls, 4CH]",
            data: [82.26, 82.95, 84.15, 84.10, 85.00],
            borderColor: "rgb(13, 148, 136)",
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.25,
            pointRadius: 5,
          },
          {
            label: "Ozdemir M02 (CNN-1D) [10cls, 4CH]",
            data: [81.71, 82.48, 83.33, 83.29, 83.97],
            borderColor: "rgb(16, 185, 129)",
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.25,
            pointRadius: 5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)}%`,
            },
          },
        },
        scales: {
          y: { min: 45, max: 90, title: { display: true, text: "Acurácia Top-1 (%)" } },
          x: { title: { display: true, text: "Janela K e Atraso Causal (ms)" } },
        },
      },
    },
  });

  // 8. Ganho Absoluto de Acurácia (+Δ%) com Suavização Temporal
  registerAndRenderChart("chartSmoothingDelta", {
    title: "Ganho Absoluto de Acurácia (+Δ%) com Suavização Causal",
    desc: "Incremento obtido sem retreino em K = 5 (ótimo clínico, 200 ms) e K = 9 (máximo, 400 ms).",
    techDetails:
      "<strong>Insight técnico:</strong> Arquiteturas treinadas com SupCon + Soft Voting (M12, M13, M14) produzem logits melhor calibrados; por isso acumular evidência temporal no M14 gera +5,53 pp em K=5 e +8,00 pp em K=9 — mais que o dobro do ganho da votação discreta pura no LDA (+1,46 pp).",
    config: {
      type: "bar",
      data: {
        labels: ["M01 (LDA)", "M02 (CNN-1D)", "M07 (Cascata)", "M11 (Hier.)", "M12 (CNN2D Soft)", "M13 (SupCon Soft)", "M14 (636f Soft)"],
        datasets: [
          {
            label: "NinaPro 12CH: Ganho em K=5 (200ms) [+%]",
            data: [1.46, 1.93, 1.93, 1.71, 3.25, 2.89, 5.53],
            backgroundColor: "rgba(37, 99, 235, 0.82)",
            borderRadius: 4,
          },
          {
            label: "NinaPro 12CH: Ganho Máx em K=9 (400ms) [+%]",
            data: [2.83, 3.37, 2.73, 3.55, 5.06, 4.38, 8.00],
            backgroundColor: "rgba(220, 38, 38, 0.82)",
            borderRadius: 4,
          },
          {
            label: "Ozdemir 4CH: Ganho em K=5 (200ms) [+%]",
            data: [0.56, 1.62, 1.88, -0.04, 2.31, 2.43, 2.48],
            backgroundColor: "rgba(13, 148, 136, 0.82)",
            borderRadius: 4,
          },
          {
            label: "Ozdemir 4CH: Ganho Máx em K=9 (400ms) [+%]",
            data: [1.07, 2.26, 2.74, 0.51, 3.76, 3.11, 2.61],
            backgroundColor: "rgba(16, 185, 129, 0.82)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y >= 0 ? "+" : ""}${fmtNum(ctx.parsed.y, 2)} pp`,
            },
          },
        },
        scales: {
          y: { title: { display: true, text: "Ganho de Acurácia Top-1 (+Δ pp)" } },
        },
      },
    },
  });

  // 9. Latência: QEMU vs Silício Físico Real (µs, escala log)
  registerAndRenderChart("chartHwLatencyLog", {
    title: "Latência de Execução: Emulador QEMU vs. Silício Físico (ESP32-S3-WROOM-1)",
    desc: "Escala logarítmica (µs) — Ozdemir 4CH: QEMU Bare-Metal vs. QEMU FreeRTOS Dual-Core vs. ESP32-S3 Físico Real (93.600 janelas/modelo).",
    techDetails:
      "<strong>Por que existe diferença entre QEMU e silício real:</strong> O QEMU emula execução ideal de 1 instrução/ciclo sem latência de cache L1/flash SPI, preempção de interrupções USB-JTAG nem stalls de pipeline LX7. No silício real com FreeRTOS SMP, o Micro-LDA leva 330,22 µs e a CNN-1D 6.575,07 µs (p99 quase idêntico à média, provando altíssimo determinismo).",
    config: {
      type: "bar",
      data: {
        labels: ["Micro-LDA Q7 (Média)", "Micro-LDA Q7 (p99)", "CNN-1D INT8 (Média)", "CNN-1D INT8 (p99)"],
        datasets: [
          {
            label: "QEMU Bare-Metal (CCOUNT Ideal) [µs]",
            data: [1.27, null, 123.28, 174.65],
            backgroundColor: "rgba(37, 99, 235, 0.82)",
            borderRadius: 4,
          },
          {
            label: "QEMU FreeRTOS Dual-Core (Core 1) [µs]",
            data: [8.58, 18.50, 157.34, 277.72],
            backgroundColor: "rgba(13, 148, 136, 0.82)",
            borderRadius: 4,
          },
          {
            label: "ESP32-S3-WROOM Físico Real (40 Suj.) [µs]",
            data: [330.22, 347.55, 6575.07, 6610.04],
            backgroundColor: "rgba(220, 38, 38, 0.85)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                ctx.parsed.y === null ? `${ctx.dataset.label}: N/A` : `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)} µs`,
            },
          },
        },
        scales: {
          y: { type: "logarithmic", min: 1, max: 15000, title: { display: true, text: "Latência (µs) — Escala Log" } },
        },
      },
    },
  });

  // 10. Uso do Orçamento de Tempo Real (% do Stride de 50 ms)
  registerAndRenderChart("chartHwCpuBudget", {
    title: "Uso do Orçamento de Tempo Real (% do Stride de 50.000 µs)",
    desc: "Percentual da janela de 50 ms consumida por inferência. Mesmo no silício físico sem SIMD PIE, a CNN-1D deixa 86,85% livres.",
    techDetails:
      "<strong>Significado clínico:</strong> Como chega 1 nova janela a cada 50 ms (20 Hz), qualquer modelo abaixo de 100% opera em tempo real estrito sem perda de deadline. O Micro-LDA ocupa apenas 0,66% do Core 1 e a CNN-1D ocupa 13,15% (Ozdemir 4CH) ou 28,30% (NinaPro 12CH), garantindo folga ampla para BLE, controle PID de motores e Light Sleep.",
    config: {
      type: "bar",
      data: {
        labels: ["QEMU Bare-Metal", "QEMU FreeRTOS Dual-Core", "ESP32-S3 Físico (Ozdemir 4CH)", "ESP32-S3 Físico (NinaPro 12CH)"],
        datasets: [
          {
            label: "Micro-LDA Q7 (% de 50 ms)",
            data: [0.0626, 0.0172, 0.6604, 2.1037],
            backgroundColor: "rgba(37, 99, 235, 0.82)",
            borderRadius: 4,
          },
          {
            label: "CNN-1D INT8 (% de 50 ms)",
            data: [0.2466, 0.3147, 13.1501, 28.2972],
            backgroundColor: "rgba(217, 119, 6, 0.85)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)}% (${fmtNum(ctx.parsed.y * 500, 1)} µs)`,
            },
          },
        },
        scales: {
          y: { min: 0, max: 35, title: { display: true, text: "Ocupação do Stride de 50 ms (%)" } },
        },
      },
    },
  });

  // 11. Estabilidade da Latência por Sujeito (S1–S40) no Silício Físico vs. QEMU
  const subjects40 = Array.from({ length: 40 }, (_, i) => `S${i + 1}`);
  const physLdaLat = [329.82, 331.72, 327.22, 332.74, 333.88, 330.38, 329.48, 329.0, 331.23, 328.94, 330.2, 330.67, 330.18, 329.43, 330.27, 331.06, 330.12, 328.65, 329.79, 331.48, 330.86, 329.85, 328.46, 330.3, 329.73, 329.01, 331.31, 330.94, 330.17, 332.3, 330.11, 330.4, 329.48, 330.06, 327.95, 331.78, 332.39, 326.89, 329.96, 330.53];
  const physCnnLat = [6571.08, 6577.58, 6568.52, 6581.22, 6577.83, 6571.46, 6565.94, 6566.26, 6569.83, 6557.44, 6576.37, 6581.5, 6568.59, 6584.67, 6567.48, 6570.09, 6590.61, 6574.09, 6574.57, 6583.96, 6583.54, 6577.05, 6576.89, 6573.22, 6588.25, 6577.06, 6578.55, 6585.0, 6576.39, 6584.08, 6567.36, 6577.41, 6578.31, 6568.37, 6568.84, 6578.41, 6586.82, 6568.64, 6560.2, 6569.41];

  registerAndRenderChart("chartHwPopLatency", {
    title: "Estabilidade da Latência no Silício Físico Real (S1–S40) vs. QEMU FreeRTOS",
    desc: "Variância ínfima (< 0,5%) entre todos os 40 sujeitos do Ozdemir (2.340 janelas/sujeito, 187.200 inferências) no ESP32-S3.",
    techDetails:
      "<strong>O que prova:</strong> A curva perfeitamente plana ao longo de 40 sujeitos confirma determinismo temporal rígido (zero vazamento de memória, zero fragmentação de heap pois o firmware é 100% estático e zero deadline misses sob FreeRTOS SMP).",
    config: {
      type: "line",
      data: {
        labels: subjects40,
        datasets: [
          {
            label: "ESP32-S3 Físico — CNN-1D INT8 [µs]",
            data: physCnnLat,
            borderColor: "rgb(220, 38, 38)",
            borderWidth: 2,
            pointRadius: 2.5,
            tension: 0.15,
          },
          {
            label: "ESP32-S3 Físico — Micro-LDA Q7 [µs]",
            data: physLdaLat,
            borderColor: "rgb(37, 99, 235)",
            borderWidth: 2,
            pointRadius: 2.5,
            tension: 0.15,
          },
          {
            label: "QEMU FreeRTOS — CNN-1D INT8 (157,34 µs)",
            data: Array(40).fill(157.34),
            borderColor: "rgb(217, 119, 6)",
            borderDash: [6, 4],
            borderWidth: 2,
            pointRadius: 0,
          },
          {
            label: "QEMU FreeRTOS — Micro-LDA Q7 (8,58 µs)",
            data: Array(40).fill(8.58),
            borderColor: "rgb(13, 148, 136)",
            borderDash: [6, 4],
            borderWidth: 2,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)} µs` } },
        },
        scales: {
          y: { type: "logarithmic", min: 5, max: 10000, title: { display: true, text: "Latência (µs) — Escala Log" } },
          x: { ticks: { maxRotation: 45, autoSkip: true, maxTicksLimit: 20 } },
        },
      },
    },
  });

  // 12. Generalização Cross-Subject (S1-fixed em S1–S40) no Silício Real
  const physLdaAcc = [81.2, 36.32, 37.95, 32.26, 34.83, 43.68, 49.15, 45.85, 42.78, 44.19, 46.07, 38.29, 54.23, 41.62, 58.97, 51.84, 32.91, 36.2, 48.76, 40.38, 35.56, 46.54, 38.12, 39.83, 41.03, 49.15, 44.66, 40.51, 32.82, 33.59, 41.15, 33.46, 33.29, 42.39, 31.54, 42.48, 25.64, 36.62, 48.72, 49.23];
  const physCnnAcc = [80.81, 38.25, 37.01, 35.34, 32.22, 45.3, 44.96, 50.73, 45.38, 49.91, 39.62, 33.33, 46.45, 37.31, 61.37, 44.44, 19.62, 36.62, 49.53, 33.33, 29.4, 33.03, 28.93, 31.62, 31.11, 43.93, 32.22, 33.85, 34.96, 24.7, 40.51, 27.69, 30.68, 48.59, 39.49, 43.38, 26.75, 37.95, 51.75, 40.6];

  registerAndRenderChart("chartHwPopAccuracy", {
    title: "Transferência Cross-Subject sem Recalibração (Pesos de S1 em S1–S40 no Silício)",
    desc: "100% bit-exata vs. Host: pesos fixos de S1 testados em 40 sujeitos (média 42,10% LDA vs. 39,32% CNN) — justifica a calibração por sujeito SEW1 (76,82% média!).",
    techDetails:
      "<strong>Insight clínico crítico:</strong> Quando pesos estáticos gravados em Flash para o sujeito S1 (81,2%) são aplicados sem recalibração a outros 39 usuários, a acurácia cai para ~42% devido à variabilidade anatômica e impedância pele-eletrodo. Foi exatamente esse achado que motivou o protocolo de carregamento dinâmico SEW1 em RAM (que elevou a média real no Ozdemir para 76,82%!).",
    config: {
      type: "bar",
      data: {
        labels: subjects40,
        datasets: [
          {
            label: "Micro-LDA Q7 (Pesos S1 fixos) [%]",
            data: physLdaAcc,
            backgroundColor: "rgba(37, 99, 235, 0.78)",
            borderRadius: 2,
          },
          {
            label: "CNN-1D INT8 (Pesos S1 fixos) [%]",
            data: physCnnAcc,
            backgroundColor: "rgba(13, 148, 136, 0.78)",
            borderRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)}%` } },
        },
        scales: {
          y: { min: 0, max: 100, title: { display: true, text: "Acurácia Top-1 (%)" } },
          x: { ticks: { maxRotation: 45, autoSkip: true, maxTicksLimit: 20 } },
        },
      },
    },
  });
}

function registerAndRenderChart(canvasId, meta) {
  chartRegistry[canvasId] = meta;
  const canvas = document.getElementById(canvasId);
  if (canvas) {
    new Chart(canvas, meta.config);
  }
}

/* ============================================================================
   4. HIERARQUIA COMPLETA: MODELOS > NÚMERO DE CANAIS > QUANTIZAÇÃO
   ============================================================================ */

function initBenchmarkHierarchy() {
  const data = window.SEMG_BENCHMARK_DATA;
  const navRoot = document.getElementById("hierarchyModelNav");
  const contentRoot = document.getElementById("benchmarkHierarchyRoot");
  if (!data || !data.modelsHierarchy || !navRoot || !contentRoot) return;

  const models = data.modelsHierarchy;

  // Monta botões de navegação de nível 1 (Modelos)
  const navHtml = [
    ...models.map(
      (m) =>
        `<button type="button" class="hierarchy-model-btn ${m.id === activeHierarchyModel ? "active" : ""}" data-hier-model="${m.id}">${escapeHtml(m.short_name)}</button>`
    ),
    `<button type="button" class="hierarchy-model-btn ${activeHierarchyModel === "ALL" ? "active" : ""}" data-hier-model="ALL">Ver Todos (M01–M14)</button>`,
  ];
  navRoot.innerHTML = navHtml.join("");

  navRoot.querySelectorAll("[data-hier-model]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeHierarchyModel = btn.getAttribute("data-hier-model");
      navRoot.querySelectorAll("[data-hier-model]").forEach((b) => b.classList.toggle("active", b === btn));
      renderHierarchyModels();
    });
  });

  renderHierarchyModels();
}

function renderHierarchyModels() {
  const data = window.SEMG_BENCHMARK_DATA;
  const contentRoot = document.getElementById("benchmarkHierarchyRoot");
  if (!data || !contentRoot) return;

  const list =
    activeHierarchyModel === "ALL"
      ? data.modelsHierarchy
      : data.modelsHierarchy.filter((m) => m.id === activeHierarchyModel);

  contentRoot.innerHTML = list.map((m) => renderSingleModelHierarchy(m)).join("");

  // Bind botões "Expandir/Recolher todos os canais" dentro de cada modelo
  contentRoot.querySelectorAll("[data-toggle-channels]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mid = btn.getAttribute("data-toggle-channels");
      const sec = contentRoot.querySelector(`[data-model-section="${mid}"]`);
      if (!sec) return;
      const details = sec.querySelectorAll("details.hierarchy-channel-group");
      const anyClosed = Array.from(details).some((d) => !d.open);
      details.forEach((d) => {
        d.open = anyClosed;
      });
      btn.textContent = anyClosed ? "Recolher canais" : "Expandir todos os canais";
    });
  });
}

function renderSingleModelHierarchy(m) {
  // Tabela de validação física (se disponível em M01..M10)
  let physHtml = "";
  if (m.physical_summary && m.physical_summary.length > 0) {
    physHtml = `
      <div class="hierarchy-level-heading">
        <span>Validação Populacional no ESP32-S3-WROOM-1 Físico (4 Datasets, 101 Sujeitos, 100% Paridade)</span>
      </div>
      <div class="table-wrapper" style="margin-top:0.5rem; margin-bottom:1.25rem;">
        <table>
          <thead>
            <tr>
              <th>Dataset (Canais)</th>
              <th>Sujeitos</th>
              <th>Top-1 LOTO (Média &plusmn; DP)</th>
              <th>Macro-F1</th>
              <th>Latência Core 1</th>
              <th>Energia (mJ)</th>
              <th>ROM / SRAM</th>
              <th>IRC Físico</th>
              <th>IEE Físico</th>
            </tr>
          </thead>
          <tbody>
            ${m.physical_summary
              .map(
                (r) => `
              <tr class="row-physical">
                <td><strong>${escapeHtml(r.dataset_label)}</strong></td>
                <td>${r.num_subjects}</td>
                <td><strong>${fmtNum(r.top1_mean, 2)}%</strong> &plusmn; ${fmtNum(r.top1_std, 2)}%</td>
                <td>${fmtNum(r.f1_mean, 2)}%</td>
                <td><strong>${fmtNum(r.lat_mean_ms, 3)} ms</strong> (${fmtNum(r.lat_mean_us, 1)} &mu;s)</td>
                <td>${fmtNum(r.energy_mj, 5)}</td>
                <td>${fmtBytes(r.rom_bytes)} / ${fmtNum(r.sram_kb, 2)} KB</td>
                <td><strong>${fmtNum(r.irc, 1)}</strong></td>
                <td><strong>${fmtNum(r.iee, 1)}</strong></td>
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  // Nível 2: Subsessões por Número de Canais -> Nível 3: Subsubsessão por Quantização
  const channelsHtml = m.channels
    .map((ch, idx) => {
      const openByDefault = idx === 0 || (ch.dataset_key === "ozdemir" && ch.num_channels === 4);
      const qRowsHtml = ch.quantizations
        .map((q) => {
          const rowClass =
            q.highlight === "pareto"
              ? "row-pareto"
              : q.highlight === "collapse"
              ? "row-collapse"
              : q.highlight === "physical"
              ? "row-physical"
              : "";
          const badge =
            q.highlight === "pareto"
              ? ' <span class="badge badge-success">Ótimo Pareto</span>'
              : q.highlight === "collapse"
              ? ' <span class="badge badge-danger">Colapso INT8</span>'
              : q.highlight === "physical"
              ? ' <span class="badge badge-info">Silício Real</span>'
              : "";
          const deltaStr =
            q.delta_vs_fp32_pp === 0
              ? "REF"
              : `${q.delta_vs_fp32_pp > 0 ? "+" : ""}${fmtNum(q.delta_vs_fp32_pp, 2)} pp`;
          return `
            <tr class="${rowClass}">
              <td><code>${escapeHtml(q.precision)}</code>${badge}</td>
              <td>${q.bytes_per_sample} B (${fmtNum(q.rate_kbps, 1)} kB/s)</td>
              <td><strong>${fmtNum(q.top1_pct, 2)}%</strong></td>
              <td>${fmtNum(q.bal_acc_pct, 2)}%</td>
              <td>${fmtNum(q.macro_f1_pct, 2)}%</td>
              <td>${deltaStr}</td>
              <td>${fmtNum(q.energy_mj, 5)} mJ</td>
              <td><strong>${fmtNum(q.irc, 1)}</strong></td>
              <td><strong>${fmtNum(q.iee, 1)}</strong></td>
            </tr>
          `;
        })
        .join("");

      return `
        <details class="hierarchy-channel-group" ${openByDefault ? "open" : ""}>
          <summary>
            <span>
              <span class="badge badge-neutral">Subsessão: ${ch.num_channels} ${ch.num_channels === 1 ? "Canal" : "Canais"}</span>
              &nbsp; ${escapeHtml(ch.dataset_label)}
            </span>
            <span class="hierarchy-channel-meta">
              <span>${escapeHtml(ch.env_label)}</span>
              <span>&bull; Lat: ${fmtNum(ch.latency_ms, 4)} ms</span>
              <span>&bull; ROM: ${fmtBytes(ch.rom_bytes)}</span>
              <span>&bull; SRAM: ${fmtNum(ch.sram_kb, 2)} KB</span>
              ${ch.extra_note ? `<span>&bull; ${escapeHtml(ch.extra_note)}</span>` : ""}
            </span>
          </summary>
          <div class="hierarchy-quant-subpanel">
            <div class="hierarchy-quant-label">
              Subsubsessão: Quantização dos Dados (${ch.quantizations.length} ${ch.quantizations.length === 1 ? "nível" : "níveis"})
            </div>
            <div class="table-wrapper" style="margin-top:0.35rem;">
              <table>
                <thead>
                  <tr>
                    <th>Quantização</th>
                    <th>Largura de Banda</th>
                    <th>Acurácia Top-1</th>
                    <th>Balanced Acc</th>
                    <th>Macro-F1</th>
                    <th>&Delta; vs. Float32</th>
                    <th>Energia</th>
                    <th>IRC</th>
                    <th>IEE</th>
                  </tr>
                </thead>
                <tbody>${qRowsHtml}</tbody>
              </table>
            </div>
          </div>
        </details>
      `;
    })
    .join("");

  return `
    <article class="hierarchy-model-section" data-model-section="${m.id}">
      <div class="hierarchy-model-header">
        <div>
          <span class="badge ${escapeHtml(m.badge_class)}">${escapeHtml(m.category)}</span>
          <h3 class="hierarchy-model-title" style="margin-top:0.35rem;">${escapeHtml(m.name)}</h3>
        </div>
        <span class="status-badge ${m.physical_summary.length ? "badge-real-active" : "badge-emulated"}">${escapeHtml(m.physical_status)}</span>
      </div>

      <!-- Detalhes da Arquitetura -->
      <div class="hierarchy-arch-grid">
        <div class="hierarchy-arch-box">
          <strong>Topologia &amp; Operação</strong>
          <span>${escapeHtml(m.architecture.topology)}</span>
        </div>
        <div class="hierarchy-arch-box">
          <strong>Quantização &amp; Aritmética</strong>
          <span>${escapeHtml(m.architecture.quantization)}</span>
        </div>
        <div class="hierarchy-arch-box">
          <strong>Complexidade &amp; Latência</strong>
          <span>${escapeHtml(m.architecture.complexity)}</span>
        </div>
        <div class="hierarchy-arch-box">
          <strong>Consumo de Memória</strong>
          <span>${escapeHtml(m.architecture.memory)}</span>
        </div>
      </div>

      <!-- Por que os resultados são esses -->
      <div class="hierarchy-why-box">
        <strong>Por que os resultados do ${escapeHtml(m.id)} são esses:</strong> ${escapeHtml(m.why_results)}
      </div>

      ${physHtml}

      <!-- Subsessões: Número de Canais > Quantização -->
      <div class="hierarchy-level-heading" style="justify-content:space-between;">
        <span>Subsessões por Número de Canais &rarr; Subsubsessão por Quantização (${m.channels.length} configurações)</span>
        <button type="button" class="toggle-pill" data-toggle-channels="${m.id}">Expandir todos os canais</button>
      </div>
      ${channelsHtml}
    </article>
  `;
}

/* ============================================================================
   5. MODAL LIGHTBOX DE GRÁFICOS (COM DETALHAMENTO TÉCNICO EXPANDIDO)
   ============================================================================ */

function initChartModal() {
  const modal = document.getElementById("chartModal");
  const modalCanvas = document.getElementById("modalChartCanvas");
  const modalTitle = document.getElementById("modalChartTitle");
  const modalDesc = document.getElementById("modalChartDesc");
  const modalTech = document.getElementById("modalChartTechDetails");
  const closeBtn = document.getElementById("closeChartModal");

  if (!modal || !modalCanvas || !closeBtn) return;

  // Evita que cliques dentro de <details class="chart-tech-details"> ou botões internos abram o modal
  document.querySelectorAll(".chart-tech-details, .chart-card button").forEach((el) => {
    el.addEventListener("click", (e) => e.stopPropagation());
  });

  const expandableCards = document.querySelectorAll(".chart-card.expandable");
  expandableCards.forEach((card) => {
    card.addEventListener("click", () => {
      const canvas = card.querySelector("canvas");
      if (!canvas) return;
      const meta = chartRegistry[canvas.id];
      if (!meta) return;

      modalTitle.textContent = meta.title;
      modalDesc.textContent = meta.desc;
      if (modalTech) {
        if (meta.techDetails) {
          modalTech.innerHTML = meta.techDetails;
          modalTech.style.display = "block";
        } else {
          modalTech.style.display = "none";
        }
      }

      if (modalChartInstance) {
        modalChartInstance.destroy();
      }

      const expandedConfig = JSON.parse(JSON.stringify(meta.config));
      if (expandedConfig.options) {
        expandedConfig.options.responsive = true;
        expandedConfig.options.maintainAspectRatio = false;
      }

      // Re-injeta callbacks de funções perdidas no clone JSON
      if (meta.config.options && meta.config.options.plugins && meta.config.options.plugins.tooltip) {
        expandedConfig.options.plugins.tooltip.callbacks = meta.config.options.plugins.tooltip.callbacks;
      }
      if (meta.config.data && meta.config.data.datasets) {
        meta.config.data.datasets.forEach((origDs, idx) => {
          if (origDs.backgroundColor instanceof Function) {
            expandedConfig.data.datasets[idx].backgroundColor = origDs.backgroundColor;
          }
        });
      }

      modalChartInstance = new Chart(modalCanvas, expandedConfig);
      modal.classList.add("open");
      document.body.style.overflow = "hidden";
    });
  });

  function closeModal() {
    modal.classList.remove("open");
    document.body.style.overflow = "";
    if (modalChartInstance) {
      modalChartInstance.destroy();
      modalChartInstance = null;
    }
  }

  closeBtn.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("open")) closeModal();
  });
}

/* ============================================================================
   6. MENU SANDUÍCHE LATERAL DIREITO ESCURO
   ============================================================================ */

function initHamburgerMenu() {
  const navToggle = document.getElementById("navToggle");
  const navClose = document.getElementById("navClose");
  const siteNav = document.getElementById("siteNav");
  const navBackdrop = document.getElementById("navBackdrop");

  if (!navToggle || !siteNav || navToggle.dataset.navBound === "1") return;
  navToggle.dataset.navBound = "1";

  function setMenuOpen(isOpen) {
    siteNav.classList.toggle("open", isOpen);
    siteNav.setAttribute("aria-hidden", String(!isOpen));
    if (navBackdrop) navBackdrop.classList.toggle("open", isOpen);
    navToggle.classList.toggle("is-open", isOpen);
    navToggle.setAttribute("aria-expanded", String(isOpen));
  }

  navToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    setMenuOpen(!siteNav.classList.contains("open"));
  });

  if (navClose) navClose.addEventListener("click", () => setMenuOpen(false));
  if (navBackdrop) navBackdrop.addEventListener("click", () => setMenuOpen(false));
  siteNav.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => setMenuOpen(false));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && siteNav.classList.contains("open")) setMenuOpen(false);
  });
}
