import { getTestPairs, detectChange, explainChangeDetection } from "./api.js";

export async function initChangeDetection() {
  const form = document.getElementById("change-form");
  const pairSelect = document.getElementById("change-pair-select");
  const modelSelect = document.getElementById("change-model-select");
  const qatOptionsGroup = document.getElementById("qat-options-group");
  const qatGammaSelect = document.getElementById("qat-gamma-select");
  const threshOverride = document.getElementById("change-threshold-override");
  
  const submitBtn = document.getElementById("change-submit-btn");
  const btnText = submitBtn.querySelector(".btn-text");
  const btnSpinner = submitBtn.querySelector(".btn-spinner");
  
  const overlayToggle = document.getElementById("show-error-overlay-toggle");
  const resultsStage = document.getElementById("change-results-stage");
  const ribbonModel = document.getElementById("ribbon-model");
  const ribbonThreshold = document.getElementById("ribbon-threshold");
  const ribbonLatency = document.getElementById("ribbon-latency");

  const viewT1 = document.getElementById("view-t1");
  const viewT2 = document.getElementById("view-t2");
  const viewHeatmap = document.getElementById("view-heatmap");
  const viewMask = document.getElementById("view-mask");
  
  const diagnosticSection = document.getElementById("diagnostic-overlay-section");
  const viewOverlay = document.getElementById("view-overlay");
  
  const valPrecision = document.getElementById("val-precision");
  const valRecall = document.getElementById("val-recall");
  const valF1 = document.getElementById("val-f1");
  const valIou = document.getElementById("val-iou");
  const valAccuracy = document.getElementById("val-accuracy");
  const valFp = document.getElementById("val-fp");

  const btnExplainChange = document.getElementById("btn-explain-change");
  const changeExplanationSlot = document.getElementById("change-explanation-slot");

  let lastDetectionResult = null;

  // Load available test scene pairs from backend
  try {
    const pairs = await getTestPairs();
    if (pairs && pairs.length > 0) {
      pairSelect.innerHTML = "";
      pairs.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.pair_id || p.id || "";
        const rawIdentifier = p.pair_id || p.name || p.id || "";
        const sceneName = (rawIdentifier && rawIdentifier !== "undefined" && rawIdentifier !== "null")
          ? String(rawIdentifier).trim()
          : "Scene identifier unavailable";
        const gtLabel = p.has_ground_truth ? "(Ground Truth Available)" : "(Unlabeled)";
        opt.textContent = `${sceneName} ${gtLabel}`;
        pairSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load dynamic pairs list, using fallback defaults:", err);
  }

  // Model selection listener for QAT-CD options
  modelSelect.addEventListener("change", () => {
    const optionsGrid = form.querySelector(".change-options-grid, .change-detection-config");
    if (modelSelect.value === "M10_QAT_CD") {
      qatOptionsGroup.style.display = "block";
      optionsGrid?.classList.add("has-qat");
    } else {
      qatOptionsGroup.style.display = "none";
      optionsGrid?.classList.remove("has-qat");
    }
  });

  // Diagnostic overlay visibility toggle
  overlayToggle.addEventListener("change", () => {
    if (viewOverlay.getAttribute("src")) {
      diagnosticSection.style.display = overlayToggle.checked ? "block" : "none";
    }
  });

  // Attach modal trigger helper
  const attachModalPreview = (element, title) => {
    element.style.cursor = "pointer";
    element.onclick = () => {
      const src = element.getAttribute("src");
      if (src) {
        window.app?.openModal(src, title);
      }
    };
  };

  attachModalPreview(viewT1, "Optical Scene T1 (Pre-Change)");
  attachModalPreview(viewT2, "Optical Scene T2 (Post-Change)");
  attachModalPreview(viewHeatmap, "Continuous Difference Heatmap [0, 1]");
  attachModalPreview(viewMask, "Predicted Binary Change Mask (>= τ*)");
  attachModalPreview(viewOverlay, "Diagnostic Confusion Overlay (TP: Green, FP: Red, FN: Blue)");

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function renderExplanationCard(exp) {
    const evidenceItems = (exp.supporting_evidence || [])
      .map(e => `<li>${escapeHtml(e)}</li>`)
      .join("");
    const limitationItems = (exp.limitations || [])
      .map(l => `<li>${escapeHtml(l)}</li>`)
      .join("");
    const unknownItems = (exp.unknowns || [])
      .map(u => `<li>${escapeHtml(u)}</li>`)
      .join("");

    return `
      <div class="explanation-card">
        <div class="explanation-card-header">
          <div class="explanation-title">
            <span>💡</span> AI Model Explanation
          </div>
          <div class="explanation-badges">
            <span class="badge-source-model">Source: ${escapeHtml(exp.source_model)}</span>
            <span class="badge-qwen-model">Explainer: Qwen 27B (Groq)</span>
          </div>
        </div>
        <div class="explanation-card-body">
          <div class="explanation-summary-box">
            <p>${escapeHtml(exp.prediction_summary)}</p>
          </div>
          <div class="explanation-details-grid">
            <div class="explanation-detail-col col-evidence">
              <h5><span>✓</span> Supporting Evidence</h5>
              <ul>${evidenceItems || "<li>Evidence derived from detection telemetry.</li>"}</ul>
            </div>
            <div class="explanation-detail-col col-limitations">
              <h5><span>⚠</span> Documented Limitations</h5>
              <ul>${limitationItems || "<li>Subject to resolution and environmental calibration limits.</li>"}</ul>
            </div>
            <div class="explanation-detail-col col-unknowns">
              <h5><span>?</span> Unknowns / Missing</h5>
              <ul>${unknownItems || "<li>Semantic object classification unavailable.</li>"}</ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // Explain Button Click Listener
  btnExplainChange?.addEventListener("click", async () => {
    if (!lastDetectionResult) return;
    if (!changeExplanationSlot) return;

    // Toggle if already shown
    if (changeExplanationSlot.innerHTML.trim() !== "") {
      changeExplanationSlot.innerHTML = "";
      btnExplainChange.innerHTML = `<span class="explain-icon">💡</span> Explain This Prediction (Qwen 27B)`;
      return;
    }

    const originalBtnHtml = btnExplainChange.innerHTML;
    btnExplainChange.disabled = true;
    btnExplainChange.innerHTML = `<span>⏳ Analyzing Prediction with Qwen 27B...</span>`;

    try {
      const payload = {
        pair_id: lastDetectionResult.pair_id || pairSelect.value,
        model_id: lastDetectionResult.model || lastDetectionResult.model_id,
        threshold: lastDetectionResult.threshold,
        gamma: lastDetectionResult.diagnostics?.gamma || (modelSelect.value === "M10_QAT_CD" ? parseFloat(qatGammaSelect.value) : 0.0),
        metrics: lastDetectionResult.metrics || null
      };

      const explanation = await explainChangeDetection(payload);
      changeExplanationSlot.innerHTML = renderExplanationCard(explanation);
      btnExplainChange.innerHTML = `<span class="explain-icon">✕</span> Hide Prediction Explanation`;
    } catch (err) {
      changeExplanationSlot.innerHTML = `
        <div class="explanation-error-state">
          <span>Could not generate explanation: ${escapeHtml(err.message)}</span>
          <button type="button" class="btn-explain-retry" id="btn-explain-cd-retry">Retry</button>
        </div>
      `;
      document.getElementById("btn-explain-cd-retry")?.addEventListener("click", () => {
        changeExplanationSlot.innerHTML = "";
        btnExplainChange.click();
      });
      btnExplainChange.innerHTML = originalBtnHtml;
    } finally {
      btnExplainChange.disabled = false;
    }
  });

  // Form submission handler
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const pairId = pairSelect.value;
    const model = modelSelect.value;
    const gamma = model === "M10_QAT_CD" ? parseFloat(qatGammaSelect.value) : 0.0;
    const threshold = threshOverride.value ? parseFloat(threshOverride.value) : null;

    // Clear previous explanation
    if (changeExplanationSlot) {
      changeExplanationSlot.innerHTML = "";
      if (btnExplainChange) {
        btnExplainChange.innerHTML = `<span class="explain-icon">💡</span> Explain This Prediction (Qwen 27B)`;
      }
    }

    // Loading UI state
    submitBtn.disabled = true;
    btnText.textContent = "Running Inference...";
    btnSpinner.classList.remove("hidden");

    try {
      const result = await detectChange(pairId, model, threshold, gamma);
      lastDetectionResult = result;
      
      // Update Ribbon
      ribbonModel.textContent = result.model;
      ribbonThreshold.textContent = `τ* = ${result.threshold.toFixed(4)}`;
      ribbonLatency.textContent = `${result.latency_ms.toFixed(1)} ms`;

      // Set Visual Outputs
      viewT1.src = result.t1_url;
      viewT2.src = result.t2_url;
      viewHeatmap.src = result.heatmap_data_uri;
      viewMask.src = result.mask_data_uri;

      // Handle Diagnostic Overlay & Metrics
      if (result.overlay_data_uri && result.metrics) {
        viewOverlay.src = result.overlay_data_uri;
        diagnosticSection.style.display = overlayToggle.checked ? "block" : "none";

        const m = result.metrics;
        valPrecision.textContent = `${(m.precision * 100).toFixed(2)}%`;
        valRecall.textContent = `${(m.recall * 100).toFixed(2)}%`;
        valF1.textContent = m.f1.toFixed(4);
        valIou.textContent = `${(m.iou * 100).toFixed(2)}%`;
        valAccuracy.textContent = `${(m.accuracy * 100).toFixed(2)}%`;
        valFp.textContent = m.fp.toLocaleString();
      } else {
        viewOverlay.src = "";
        diagnosticSection.style.display = "none";
      }

      resultsStage.style.display = "block";
      resultsStage.scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (err) {
      window.app?.showAlert(`Change detection failed: ${err.message}`, "error");
    } finally {
      submitBtn.disabled = false;
      btnText.textContent = "Execute Change Detection";
      btnSpinner.classList.add("hidden");
    }
  });
}

