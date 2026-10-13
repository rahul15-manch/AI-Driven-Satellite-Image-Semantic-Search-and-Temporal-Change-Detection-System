/**
 * Research Results Module for Official Milestone M11 Scientific Benchmarks
 * Renders verified summary metrics, bootstrap confidence intervals,
 * paired statistical tests, ablation tables, and publication figures.
 */

import { getBenchmarkData } from "./api.js";

export async function initResults() {
  const benchmarkTableBody = document.getElementById("benchmark-table-body");
  const bootstrapTableBody = document.getElementById("bootstrap-table-body");
  const wilcoxonTableBody = document.getElementById("wilcoxon-table-body");
  const ablationTableBody = document.getElementById("ablation-table-body");
  const figuresGalleryGrid = document.getElementById("figures-gallery-grid");

  try {
    const data = await getBenchmarkData();
    renderSummaryTable(data.models || []);
    renderBootstrapTable(data.bootstrap_ci || []);
    renderWilcoxonTable(data.wilcoxon_tests || []);
    renderAblationTable(data.ablation || []);
    renderFiguresGallery(data.figures || []);
  } catch (err) {
    console.error("Failed to load benchmark data:", err);
    benchmarkTableBody.innerHTML = `<tr><td colspan="11" class="error-cell">Failed to load M11 benchmark records: ${err.message}</td></tr>`;
    bootstrapTableBody.innerHTML = `<tr><td colspan="5" class="error-cell">Failed to load bootstrap data.</td></tr>`;
    wilcoxonTableBody.innerHTML = `<tr><td colspan="5" class="error-cell">Failed to load hypothesis tests.</td></tr>`;
    ablationTableBody.innerHTML = `<tr><td colspan="9" class="error-cell">Failed to load ablation records.</td></tr>`;
  }

  function renderSummaryTable(models) {
    if (!models.length) {
      benchmarkTableBody.innerHTML = `<tr><td colspan="11">No benchmark records found.</td></tr>`;
      return;
    }

    let rowsHtml = "";
    models.forEach(m => {
      const name = m.display_name || m.model_name || m.model_id;
      const isTop = (m.model_id === "M9_BIT_CD") || (m.model_id && m.model_id.startsWith("M10_QAT_CD"));
      const highlightClass = isTop ? 'class="row-highlight"' : "";
      const isLearned = (m.architecture_type && m.architecture_type.includes("learned")) || m.type === "Learned";
      const typeBadge = isLearned 
        ? '<span class="badge badge-accent">Learned</span>' 
        : '<span class="badge">Classical</span>';

      const thresh = m.threshold_value !== undefined ? m.threshold_value : (m.threshold || 0.0);
      const latency = m.mean_latency_ms !== undefined ? m.mean_latency_ms : (m.latency_mean_ms || 0.0);
      const fpCount = m.fp !== undefined ? m.fp : (m.false_positives || 0);

      rowsHtml += `
        <tr ${highlightClass}>
          <td><strong>${name}</strong></td>
          <td>${typeBadge}</td>
          <td><code>&tau;* = ${thresh.toFixed(4)}</code></td>
          <td>${(m.precision * 100).toFixed(2)}%</td>
          <td>${(m.recall * 100).toFixed(2)}%</td>
          <td><strong>${m.f1.toFixed(4)}</strong></td>
          <td>${(m.iou * 100).toFixed(2)}%</td>
          <td>${(m.accuracy * 100).toFixed(2)}%</td>
          <td>${fpCount.toLocaleString()}</td>
          <td>${latency.toFixed(1)} ms</td>
          <td>${(m.peak_rss_mb || 0).toFixed(1)} MB</td>
        </tr>
      `;
    });
    benchmarkTableBody.innerHTML = rowsHtml;
  }

  function renderBootstrapTable(ciList) {
    if (!ciList.length) {
      bootstrapTableBody.innerHTML = `<tr><td colspan="5">No bootstrap records found.</td></tr>`;
      return;
    }

    let rowsHtml = "";
    ciList.forEach(item => {
      const p = item.precision;
      const r = item.recall;
      const f = item.f1;
      const i = item.iou;

      rowsHtml += `
        <tr>
          <td><strong>${item.model}</strong></td>
          <td>${(p.mean * 100).toFixed(1)}% <span class="ci-span">[${(p.ci_lower * 100).toFixed(1)}, ${(p.ci_upper * 100).toFixed(1)}]</span></td>
          <td>${(r.mean * 100).toFixed(1)}% <span class="ci-span">[${(r.ci_lower * 100).toFixed(1)}, ${(r.ci_upper * 100).toFixed(1)}]</span></td>
          <td><strong>${f.mean.toFixed(4)}</strong> <span class="ci-span">[${f.ci_lower.toFixed(4)}, ${f.ci_upper.toFixed(4)}]</span></td>
          <td>${(i.mean * 100).toFixed(1)}% <span class="ci-span">[${(i.ci_lower * 100).toFixed(1)}, ${(i.ci_upper * 100).toFixed(1)}]</span></td>
        </tr>
      `;
    });
    bootstrapTableBody.innerHTML = rowsHtml;
  }

  function renderWilcoxonTable(tests) {
    if (!tests.length) {
      wilcoxonTableBody.innerHTML = `<tr><td colspan="5">No paired tests found.</td></tr>`;
      return;
    }

    let rowsHtml = "";
    tests.forEach(t => {
      const sigBadge = t.significant_f1
        ? '<span class="badge badge-success">Significant (p &lt; 0.05)</span>'
        : '<span class="badge">Not Significant</span>';

      rowsHtml += `
        <tr>
          <td><strong>${t.comparison}</strong></td>
          <td>${t.delta_f1_mean >= 0 ? "+" : ""}${t.delta_f1_mean.toFixed(4)}</td>
          <td>${t.statistic_f1.toFixed(1)}</td>
          <td><code>${t.p_value_f1 < 0.0001 ? "< 1e-4" : t.p_value_f1.toFixed(4)}</code></td>
          <td>${sigBadge}</td>
        </tr>
      `;
    });
    wilcoxonTableBody.innerHTML = rowsHtml;
  }

  function renderAblationTable(ablation) {
    if (!ablation.length) {
      ablationTableBody.innerHTML = `<tr><td colspan="9">No ablation records found.</td></tr>`;
      return;
    }

    let rowsHtml = "";
    ablation.forEach(a => {
      rowsHtml += `
        <tr>
          <td><strong>${a.condition}</strong></td>
          <td>${(a.precision * 100).toFixed(2)}%</td>
          <td>${(a.recall * 100).toFixed(2)}%</td>
          <td><strong>${a.f1.toFixed(4)}</strong></td>
          <td>${a.false_positives.toLocaleString()}</td>
          <td><span class="badge badge-success">${a.fp_reduction_pct.toFixed(2)}%</span></td>
          <td>${a.precision_delta >= 0 ? "+" : ""}${(a.precision_delta * 100).toFixed(2)} pp</td>
          <td>${a.recall_delta >= 0 ? "+" : ""}${(a.recall_delta * 100).toFixed(2)} pp</td>
          <td><code>${a.exchange_ratio.toFixed(2)}</code></td>
        </tr>
      `;
    });
    ablationTableBody.innerHTML = rowsHtml;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function renderFiguresGallery(figures) {
    if (!figuresGalleryGrid) return;

    if (!figures.length) {
      figuresGalleryGrid.innerHTML = `
        <div class="chart-unavailable-state" style="grid-column: 1 / -1;">
          <p>No publication research figures found.</p>
          <code>Verify figures exist in experiments/figures/</code>
        </div>
      `;
      return;
    }

    const figuresFilterGroup = document.getElementById("figures-filter-group");

    // Update filter buttons count
    if (figuresFilterGroup) {
      const counts = {
        all: figures.length,
        m13: figures.filter(f => f.category_key === "m13").length,
        m11: figures.filter(f => f.category_key === "m11").length,
        m10: figures.filter(f => f.category_key === "m10").length,
        "m6-m5": figures.filter(f => f.category_key === "m6-m5" || f.category_key === "m6" || f.category_key === "m5").length,
      };

      figuresFilterGroup.querySelectorAll(".fig-filter-btn").forEach(btn => {
        const filterKey = btn.getAttribute("data-filter");
        if (filterKey && counts[filterKey] !== undefined) {
          const originalText = btn.textContent.split(" (")[0];
          btn.textContent = `${originalText} (${counts[filterKey]})`;
        }
      });
    }

    let gridHtml = "";
    figures.forEach(fig => {
      const catKey = fig.category_key || "all";
      const catLabel = fig.category || "Publication Figure";
      const title = fig.title || fig.name;
      const desc = fig.description || "";

      gridHtml += `
        <div class="research-chart-card figure-thumb-card figure-card" data-category="${escapeHtml(catKey)}">
          <div class="research-chart-card-header">
            <span class="research-chart-tag" data-cat="${escapeHtml(catKey)}">${escapeHtml(catLabel)}</span>
            <h4 class="research-chart-title">${escapeHtml(title)}</h4>
          </div>
          <div class="research-chart-media figure-thumb-box figure-thumb-wrapper" 
               data-src="${escapeHtml(fig.url)}" 
               data-title="${escapeHtml(title)}"
               role="button" 
               tabindex="0" 
               aria-label="Enlarge chart: ${escapeHtml(title)}"
               title="Click to view full-resolution visualization">
            <img src="${escapeHtml(fig.url)}" 
                 alt="${escapeHtml(title)}" 
                 loading="lazy" 
                 onerror="this.onerror=null; this.parentElement.innerHTML='<div class=\\'chart-unavailable-state\\'><p>Figure unavailable</p><code>${escapeHtml(fig.url)}</code></div>';" />
            <div class="research-chart-overlay">
              <span>🔍 Click to expand</span>
            </div>
          </div>
          ${desc ? `
            <div class="research-chart-caption">
              <p>${escapeHtml(desc)}</p>
            </div>
          ` : ""}
        </div>
      `;
    });

    figuresGalleryGrid.innerHTML = gridHtml;

    // Attach click and keyboard modal handlers
    const attachMediaHandler = (wrap) => {
      const triggerModal = () => {
        const src = wrap.getAttribute("data-src");
        const title = wrap.getAttribute("data-title");
        if (src) {
          window.app?.openModal(src, title);
        }
      };

      wrap.addEventListener("click", triggerModal);
      wrap.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          triggerModal();
        }
      });
    };

    figuresGalleryGrid.querySelectorAll(".research-chart-media").forEach(attachMediaHandler);

    // Setup Category Filtering
    if (figuresFilterGroup) {
      const filterButtons = figuresFilterGroup.querySelectorAll(".fig-filter-btn");
      filterButtons.forEach(btn => {
        btn.addEventListener("click", () => {
          const filter = btn.getAttribute("data-filter") || "all";
          filterButtons.forEach(b => b.classList.remove("active"));
          btn.classList.add("active");

          const cards = figuresGalleryGrid.querySelectorAll(".research-chart-card");
          cards.forEach(card => {
            const cardCat = card.getAttribute("data-category");
            if (filter === "all") {
              card.classList.remove("hidden");
            } else if (filter === "m6-m5") {
              if (cardCat === "m6-m5" || cardCat === "m6" || cardCat === "m5") {
                card.classList.remove("hidden");
              } else {
                card.classList.add("hidden");
              }
            } else if (cardCat === filter) {
              card.classList.remove("hidden");
            } else {
              card.classList.add("hidden");
            }
          });
        });
      });
    }
  }
}

