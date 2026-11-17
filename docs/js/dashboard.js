/**
 * dashboard.js — LIAA sEMG TinyML Benchmarks Dashboard
 * Inicializa os gráficos interativos com Chart.js (Acurácia, Pareto, Energia, Quantização, IRC e IEE),
 * suporta expansão em modal lightbox ao clicar no card, e controla as abas por modelo.
 */

const chartRegistry = {};
let modalChartInstance = null;

document.addEventListener("DOMContentLoaded", function () {
  initHamburgerMenu();
  if (typeof Chart !== "undefined") {
    initCharts();
  }
  initModelTabs();
  initChartModal();
});

function initCharts() {
  const models = [
    "M01 (LDA)", "M02 (CNN1D)", "M03 (DTW-O2)", "M04 (DTW-SC)", "M05 (DTW-LB)",
    "M06 (H-Head)", "M07 (H-Casc)", "M08 (H-DTW+L)", "M09 (H-DTW+C)", "M10 (Tri-H)", "M11 (Hier)", "M12 (CNN2D-Circ)", "M13 (SupCon-CNN2D)", "M14 (636f+SupCon)"
  ];

  // 1. Gráfico de Acurácia Top-1 Comparativa
  registerAndRenderChart("chartAccuracy", {
    title: "Acurácia Top-1 Comparativa por Modelo",
    desc: "NinaPro DB2 (50 classes, 12 canais) vs. Ozdemir et al. (10 classes, 4 canais)",
    config: {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "NinaPro DB2 (12CH, 50 Classes) [%]",
            data: [47.02, 59.21, 43.63, 43.63, 43.63, 48.08, 58.98, 49.59, 59.73, 49.26, 58.15, 61.61, 62.08, 68.59],
            backgroundColor: "rgba(37, 99, 235, 0.75)",
            borderColor: "rgb(37, 99, 235)",
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: "Ozdemir et al. (4CH, 10 Classes) [%]",
            data: [81.20, 80.81, 68.12, 67.73, 67.73, 81.79, 81.79, 81.32, 81.79, 82.78, 64.87, 81.07, 81.15, 82.01],
            backgroundColor: "rgba(13, 148, 136, 0.75)",
            borderColor: "rgb(13, 148, 136)",
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`;
              }
            }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: "Acurácia Top-1 (%)" }
          }
        }
      }
    }
  });

  // 2. Gráfico da Fronteira de Pareto: Latência vs Acurácia
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
    { x: 0.4247, y: 68.59, label: "M14 (636f+SupCon K=5)" }
  ];

  registerAndRenderChart("chartPareto", {
    title: "Fronteira de Pareto: Latência vs. Acurácia no NinaPro DB2",
    desc: "Tempo de inferência no ESP32-S3 a 240 MHz (escala log) vs. Acurácia Top-1",
    config: {
      type: "scatter",
      data: {
        datasets: [{
          label: "Arquiteturas (NinaPro 12CH)",
          data: paretoData,
          backgroundColor: function(ctx) {
            const item = ctx.raw;
            if (item && item.label && item.label.includes("M11")) return "rgb(220, 38, 38)";
            if (item && item.label && item.label.includes("M01")) return "rgb(37, 99, 235)";
            if (item && item.label && item.label.includes("M07")) return "rgb(13, 148, 136)";
            return "rgba(100, 116, 139, 0.7)";
          },
          pointRadius: 7,
          pointHoverRadius: 10
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: function (ctx) {
                const item = ctx.raw;
                return `${item.label} → Latência: ${item.x.toFixed(4)} ms | Acurácia: ${item.y.toFixed(2)}%`;
              }
            }
          }
        },
        scales: {
          x: {
            type: "logarithmic",
            title: { display: true, text: "Latência ESP32-S3 @ 240 MHz (ms) [Escala Log]" }
          },
          y: {
            min: 35,
            max: 75,
            title: { display: true, text: "Acurácia Top-1 NinaPro DB2 (%)" }
          }
        }
      }
    }
  });

  // 3. Gráfico de Consumo Energético Médio
  registerAndRenderChart("chartEnergy", {
    title: "Consumo Energético Médio por Inferência (mJ)",
    desc: "Impacto no dreno da bateria da prótese no microcontrolador ESP32-S3",
    config: {
      type: "bar",
      data: {
        labels: models,
        datasets: [{
          label: "Energia no ESP32-S3 (mJ / inferência)",
          data: [0.00074, 0.05188, 0.18809, 0.04802, 0.00941, 0.05253, 0.04014, 0.00571, 0.05668, 0.03220, 0.01440, 0.18548, 0.18548, 0.06306],
          backgroundColor: function(ctx) {
            const val = ctx.raw;
            if (val < 0.015) return "rgba(16, 185, 129, 0.8)";
            if (val < 0.050) return "rgba(245, 158, 11, 0.8)";
            return "rgba(239, 68, 68, 0.8)";
          },
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return `Energia: ${ctx.parsed.y.toFixed(5)} mJ (${(ctx.parsed.y * 1000).toFixed(2)} µJ)`;
              }
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: "Consumo de Energia Médio (mJ)" },
            min: 0
          }
        }
      }
    }
  });

  // 4. Gráfico de Resiliência à Quantização (Float32 vs INT8)
  registerAndRenderChart("chartQuant", {
    title: "Resiliência à Quantização: Float32 vs. INT8",
    desc: "Demonstração de colapso vs estabilidade sob quantização em 8 bits (NinaPro 12CH)",
    config: {
      type: "bar",
      data: {
        labels: ["M01 (Micro-LDA)", "M02 (CNN-1D)", "M07 (Cascata)", "M11 (Hierárquico)", "M12 (CNN2D-Circ)", "M13 (SupCon-CNN2D)", "M14 (636f+SupCon)"],
        datasets: [
          {
            label: "Precisão Float32 [%]",
            data: [47.86, 58.62, 55.30, 58.15, 58.94, 59.27, 64.35],
            backgroundColor: "rgba(59, 130, 246, 0.8)",
            borderRadius: 4
          },
          {
            label: "Quantização INT8 (8 bits) [%]",
            data: [53.11, 9.86, 42.44, 51.74, 58.36, 59.19, 63.06],
            backgroundColor: "rgba(244, 63, 94, 0.8)",
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`;
              }
            }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 70,
            title: { display: true, text: "Acurácia Top-1 (%)" }
          }
        }
      }
    }
  });

  // 5. Gráfico de Índice IRC (Responsividade: Gestos/ms)
  registerAndRenderChart("chartIrc", {
    title: "Índice IRC: Responsividade de Decodificação (Gestos/ms)",
    desc: "Mede o repertório gestual acurado computado por milissegundo de CPU [Escala Log]",
    config: {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "IRC NinaPro DB2 (50 classes, 12CH)",
            data: [469671.4, 8474.1, 1722.3, 6745.4, 34428.5, 6795.7, 10909.1, 64423.3, 7825.5, 11359.4, 30047.2, 2336.23, 2369.37, 8075.76],
            backgroundColor: "rgba(99, 102, 241, 0.8)",
            borderRadius: 4
          },
          {
            label: "IRC Ozdemir et al. (10 classes, 4CH)",
            data: [686361.8, 4640.7, 4268.0, 15356.6, 66289.9, 4683.6, 10846.1, 34210.3, 4167.7, 8900.5, 7208.9, 1335.66, 1334.21, 6954.68],
            backgroundColor: "rgba(20, 184, 166, 0.8)",
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                return `${ctx.dataset.label}: ${ctx.parsed.y.toLocaleString('pt-BR', { maximumFractionDigits: 1 })}`;
              }
            }
          }
        },
        scales: {
          y: {
            type: "logarithmic",
            title: { display: true, text: "IRC = (Acc * Gestos) / Latência (ms)" }
          }
        }
      }
    }
  });

  // 6. Gráfico de Índice IEE (Eficiência Embarcada)
  registerAndRenderChart("chartIee", {
    title: "Índice IEE: Eficiência Embarcada em Silício (Bateria & SRAM)",
    desc: "Relação de repertório de gestos por milijoule consumido e pegada de memória [Escala Log]",
    config: {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "IEE NinaPro DB2 (50 classes, 12CH)",
            data: [2349139.0, 39952.8, 7976.3, 33588.1, 170947.5, 31920.8, 51433.2, 320788.9, 36766.6, 53251.3, 144149.2, 10558.17, 10707.94, 61656.63],
            backgroundColor: "rgba(168, 85, 247, 0.8)",
            borderRadius: 4
          },
          {
            label: "IEE Ozdemir et al. (10 classes, 4CH)",
            data: [5639679.2, 29773.7, 25918.4, 123185.6, 524019.8, 29692.2, 69586.0, 274423.6, 26440.2, 56070.9, 44360.3, 7615.64, 7607.37, 78852.96],
            backgroundColor: "rgba(249, 115, 22, 0.8)",
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                return `${ctx.dataset.label}: ${ctx.parsed.y.toLocaleString('pt-BR', { maximumFractionDigits: 1 })}`;
              }
            }
          }
        },
        scales: {
          y: {
            type: "logarithmic",
            title: { display: true, text: "IEE = (Acc * Gestos) / (Energia * log10(SRAM))" }
          }
        }
      }
    }
  });

  // 7. Gráfico de Suavização Temporal (Passo 1): Acurácia vs Janela K
  registerAndRenderChart("chartSmoothing", {
    title: "Suavização Temporal Causal: Acurácia Top-1 vs. Janela K (Atraso Algorítmico)",
    desc: "Impacto do buffer circular FIFO causal de predições passadas (K=1 a K=9) nas repetições de teste.",
    config: {
      type: "line",
      data: {
        labels: ["Raw (K=1, 0ms)", "K=3 (100ms)", "K=5 (200ms)", "K=7 (300ms)", "K=9 (400ms)"],
        datasets: [
          {
            label: "NinaPro M14 (636f + SupCon MLP, Soft Voting) [50 Classes, 12CH]",
            data: [63.06, 66.55, 68.59, 69.99, 71.06],
            borderColor: "rgb(219, 39, 119)",
            backgroundColor: "rgba(219, 39, 119, 0.15)",
            borderWidth: 3,
            tension: 0.25,
            pointRadius: 6
          },
          {
            label: "NinaPro M13 (SupCon CNN-2D, Soft Voting) [50 Classes, 12CH]",
            data: [59.19, 60.59, 62.08, 62.82, 63.57],
            borderColor: "rgb(5, 150, 105)",
            backgroundColor: "rgba(5, 150, 105, 0.12)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5
          },
          {
            label: "NinaPro M12 (CNN-2D Circ, Soft Voting) [50 Classes, 12CH]",
            data: [58.36, 59.95, 61.61, 62.49, 63.42],
            borderColor: "rgb(2, 132, 199)",
            backgroundColor: "rgba(2, 132, 199, 0.12)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5
          },
          {
            label: "NinaPro M02 (CNN-1D) [50 Classes, 12CH]",
            data: [58.62, 59.88, 60.55, 61.43, 61.99],
            borderColor: "rgb(37, 99, 235)",
            backgroundColor: "rgba(37, 99, 235, 0.1)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5
          },
          {
            label: "NinaPro M11 (Hierárquico) [50 Classes, 12CH]",
            data: [58.08, 59.23, 59.79, 60.83, 61.62],
            borderColor: "rgb(220, 38, 38)",
            backgroundColor: "rgba(220, 38, 38, 0.1)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5
          },
          {
            label: "NinaPro M07 (Cascata) [50 Classes, 12CH]",
            data: [55.35, 56.46, 57.28, 58.08, 58.08],
            borderColor: "rgb(234, 88, 12)",
            backgroundColor: "rgba(234, 88, 12, 0.1)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5
          },
          {
            label: "NinaPro M01 (Micro-LDA) [50 Classes, 12CH]",
            data: [47.97, 48.69, 49.42, 50.18, 50.80],
            borderColor: "rgb(147, 51, 234)",
            backgroundColor: "rgba(147, 51, 234, 0.1)",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 5
          },
          {
            label: "Ozdemir M07 (Cascata) [10 Classes, 4CH]",
            data: [82.26, 82.95, 84.15, 84.10, 85.00],
            borderColor: "rgb(13, 148, 136)",
            backgroundColor: "rgba(13, 148, 136, 0.1)",
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.25,
            pointRadius: 5
          },
          {
            label: "Ozdemir M02 (CNN-1D) [10 Classes, 4CH]",
            data: [81.71, 82.48, 83.33, 83.29, 83.97],
            borderColor: "rgb(16, 185, 129)",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.25,
            pointRadius: 5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`;
              }
            }
          }
        },
        scales: {
          y: {
            min: 45,
            max: 90,
            title: { display: true, text: "Acurácia Top-1 (%)" }
          },
          x: {
            title: { display: true, text: "Tamanho da Janela K e Atraso Causal (ms)" }
          }
        }
      }
    }
  });

  // 8. Gráfico de Ganho de Acurácia (Delta %) com Suavização Temporal (K=5 e K=9)
  registerAndRenderChart("chartSmoothingDelta", {
    title: "Ganho Absoluto de Acurácia (+Δ%) com Votação Majoritária Causal",
    desc: "Aumento de acurácia obtido sem retreino em K=5 (ótimo clínico, 200ms) e K=9 (máximo, 400ms).",
    config: {
      type: "bar",
      data: {
        labels: ["M01 (Micro-LDA)", "M02 (CNN-1D INT8)", "M07 (Cascata)", "M11 (Hierárquico)", "M12 (CNN-2D Circ Soft)", "M13 (SupCon CNN-2D Soft)", "M14 (636f+SupCon Soft)"],
        datasets: [
          {
            label: "NinaPro 12CH: Ganho em K=5 (200ms) [+%]",
            data: [1.46, 1.93, 1.93, 1.71, 3.25, 2.89, 5.53],
            backgroundColor: "rgba(37, 99, 235, 0.8)",
            borderRadius: 4
          },
          {
            label: "NinaPro 12CH: Ganho Máximo em K=9 (400ms) [+%]",
            data: [2.83, 3.37, 2.73, 3.55, 5.06, 4.38, 8.00],
            backgroundColor: "rgba(220, 38, 38, 0.8)",
            borderRadius: 4
          },
          {
            label: "Ozdemir 4CH: Ganho em K=5 (200ms) [+%]",
            data: [0.56, 1.62, 1.88, -0.04, 2.31, 2.43, 2.48],
            backgroundColor: "rgba(13, 148, 136, 0.8)",
            borderRadius: 4
          },
          {
            label: "Ozdemir 4CH: Ganho Máximo em K=9 (400ms) [+%]",
            data: [1.07, 2.26, 2.74, 0.51, 3.76, 3.11, 2.61],
            backgroundColor: "rgba(16, 185, 129, 0.8)",
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                const val = ctx.parsed.y;
                return `${ctx.dataset.label}: ${val >= 0 ? '+' : ''}${val.toFixed(2)}%`;
              }
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: "Ganho de Acurácia Top-1 (+Δ%)" }
          }
        }
      }
    }
  });
}

function registerAndRenderChart(canvasId, meta) {
  chartRegistry[canvasId] = meta;
  const canvas = document.getElementById(canvasId);
  if (canvas) {
    new Chart(canvas, meta.config);
  }
}

// Controle do Modal Lightbox para Expansão dos Gráficos
function initChartModal() {
  const modal = document.getElementById("chartModal");
  const modalCanvas = document.getElementById("modalChartCanvas");
  const modalTitle = document.getElementById("modalChartTitle");
  const modalDesc = document.getElementById("modalChartDesc");
  const closeBtn = document.getElementById("closeChartModal");

  if (!modal || !modalCanvas || !closeBtn) return;

  const expandableCards = document.querySelectorAll(".chart-card.expandable");
  expandableCards.forEach(card => {
    card.addEventListener("click", () => {
      const canvas = card.querySelector("canvas");
      if (!canvas) return;
      const canvasId = canvas.id;
      const meta = chartRegistry[canvasId];
      if (!meta) return;

      modalTitle.textContent = meta.title;
      modalDesc.textContent = meta.desc;

      // Destrói instância anterior do modal se existir
      if (modalChartInstance) {
        modalChartInstance.destroy();
      }

      // Clona a configuração para o modal com tamanhos e legendas aumentadas
      const expandedConfig = JSON.parse(JSON.stringify(meta.config));
      if (expandedConfig.options) {
        expandedConfig.options.responsive = true;
        expandedConfig.options.maintainAspectRatio = false;
      }

      // Re-injeta callbacks de funções que JSON.stringify perde
      if (meta.config.options && meta.config.options.plugins && meta.config.options.plugins.tooltip) {
        expandedConfig.options.plugins.tooltip.callbacks = meta.config.options.plugins.tooltip.callbacks;
      }
      if (meta.config.data && meta.config.data.datasets && meta.config.data.datasets[0].backgroundColor instanceof Function) {
        expandedConfig.data.datasets[0].backgroundColor = meta.config.data.datasets[0].backgroundColor;
      }

      modalChartInstance = new Chart(modalCanvas, expandedConfig);
      modal.classList.add("open");
      document.body.style.overflow = "hidden"; // Evita scroll do fundo
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
    if (e.target === modal) {
      closeModal();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("open")) {
      closeModal();
    }
  });
}

// Filtro de Abas da Tabela Separada por Modelo
function initModelTabs() {
  const tabButtons = document.querySelectorAll(".model-tab-btn");
  const tabPanels = document.querySelectorAll(".model-tab-panel");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetModel = btn.getAttribute("data-target");

      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(`panel-${targetModel}`);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });
}

// Menu Sanduíche (Gaveta Lateral Direita Escura com Escurecimento)
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

  if (navClose) {
    navClose.addEventListener("click", () => setMenuOpen(false));
  }
  if (navBackdrop) {
    navBackdrop.addEventListener("click", () => setMenuOpen(false));
  }

  // Fecha o sumário ao clicar em qualquer link
  const navLinks = siteNav.querySelectorAll(".nav-link");
  navLinks.forEach(link => {
    link.addEventListener("click", () => setMenuOpen(false));
  });

  // Fecha o sumário com Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && siteNav.classList.contains("open")) {
      setMenuOpen(false);
    }
  });
}
