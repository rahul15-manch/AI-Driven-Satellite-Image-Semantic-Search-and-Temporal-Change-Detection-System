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

  function renderFiguresGallery(figures) {
    if (!figures.length) {
      figuresGalleryGrid.innerHTML = `<p class="empty-state-text">No publication figures found.</p>`;
      return;
    }

    let gridHtml = "";
    figures.forEach(fig => {
      gridHtml += `
        <div class="figure-card">
          <div class="figure-thumb-wrapper" data-src="${fig.url}" data-title="${fig.title}">
            <img src="${fig.url}" alt="${fig.title}" loading="lazy">
          </div>
          <div class="figure-caption">
            <h4>${fig.title}</h4>
          </div>
        </div>
      `;
    });
    figuresGalleryGrid.innerHTML = gridHtml;

    figuresGalleryGrid.querySelectorAll(".figure-thumb-wrapper").forEach(wrap => {
      wrap.addEventListener("click", () => {
        const src = wrap.getAttribute("data-src");
        const title = wrap.getAttribute("data-title");
        window.app?.openModal(src, title);
      });
    });
  }
}
