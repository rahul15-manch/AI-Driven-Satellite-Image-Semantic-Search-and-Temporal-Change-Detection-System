/**
 * Bi-Temporal Change Detection Module
 * Connects to /api/change-detection and visualizes continuous heatmaps,
 * binary masks, diagnostic overlays, and ground-truth metrics.
 */

import { getTestPairs, detectChange } from "./api.js";

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

  // Load available test scene pairs from backend
  try {
    const pairs = await getTestPairs();
    if (pairs && pairs.length > 0) {
      pairSelect.innerHTML = "";
      pairs.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.pair_id;
        opt.textContent = `${p.name} ${p.has_ground_truth ? "(Ground Truth Available)" : "(Unlabeled)"}`;
        pairSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load dynamic pairs list, using fallback defaults:", err);
  }

  // Model selection listener for QAT-CD options
  modelSelect.addEventListener("change", () => {
    if (modelSelect.value === "M10_QAT_CD") {
      qatOptionsGroup.style.display = "block";
    } else {
      qatOptionsGroup.style.display = "none";
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

  // Form submission handler
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const pairId = pairSelect.value;
    const model = modelSelect.value;
    const gamma = model === "M10_QAT_CD" ? parseFloat(qatGammaSelect.value) : 0.0;
    const threshold = threshOverride.value ? parseFloat(threshOverride.value) : null;

    // Loading UI state
    submitBtn.disabled = true;
    btnText.textContent = "Running Inference...";
    btnSpinner.classList.remove("hidden");

    try {
      const result = await detectChange(pairId, model, threshold, gamma);
      
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
