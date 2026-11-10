import { getTestPairs, detectChange, explainChangeDetection, analyzeTemporalPair } from "./api.js";

export async function initChangeDetection() {
  // Mode switcher elements
  const modeBtnBenchmark = document.getElementById("mode-btn-benchmark");
  const modeBtnTemporal = document.getElementById("mode-btn-temporal");
  const containerBenchmark = document.getElementById("cd-benchmark-container");
  const containerTemporal = document.getElementById("cd-temporal-container");

  function switchCDMode(mode) {
    if (mode === "temporal") {
      modeBtnBenchmark?.classList.remove("active");
      modeBtnTemporal?.classList.add("active");
      modeBtnBenchmark?.setAttribute("aria-selected", "false");
      modeBtnTemporal?.setAttribute("aria-selected", "true");
      if (containerBenchmark) containerBenchmark.style.display = "none";
      if (containerTemporal) containerTemporal.style.display = "block";
    } else {
      modeBtnTemporal?.classList.remove("active");
      modeBtnBenchmark?.classList.add("active");
      modeBtnTemporal?.setAttribute("aria-selected", "false");
      modeBtnBenchmark?.setAttribute("aria-selected", "true");
      if (containerTemporal) containerTemporal.style.display = "none";
      if (containerBenchmark) containerBenchmark.style.display = "block";
    }
  }

  modeBtnBenchmark?.addEventListener("click", () => switchCDMode("benchmark"));
  modeBtnTemporal?.addEventListener("click", () => switchCDMode("temporal"));

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

  // =========================================================================
  // 2. TEMPORAL UNDERSTANDING MODE CONTROLLER (Upload T1 & T2)
  // =========================================================================
  const temporalForm = document.getElementById("temporal-form");
  const fileInputT1 = document.getElementById("tu-file-t1");
  const fileInputT2 = document.getElementById("tu-file-t2");
  const dropzoneT1 = document.getElementById("dropzone-t1");
  const dropzoneT2 = document.getElementById("dropzone-t2");
  const previewContainerT1 = document.getElementById("preview-container-t1");
  const previewContainerT2 = document.getElementById("preview-container-t2");
  const previewImgT1 = document.getElementById("tu-preview-img-t1");
  const previewImgT2 = document.getElementById("tu-preview-img-t2");
  const metaBadgeT1 = document.getElementById("tu-meta-t1");
  const metaBadgeT2 = document.getElementById("tu-meta-t2");
  const dateInputT1 = document.getElementById("tu-date-t1");
  const dateInputT2 = document.getElementById("tu-date-t2");
  const gsdInput = document.getElementById("tu-gsd-input");
  const modelSelectTU = document.getElementById("tu-model-select");
  const maxRegionsInput = document.getElementById("tu-max-regions-input");
  const descInput = document.getElementById("tu-desc-input");
  const btnSubmitTU = document.getElementById("tu-submit-btn");
  const btnResetTU = document.getElementById("tu-reset-btn");
  const valMsgTU = document.getElementById("tu-validation-msg");

  const tuResultsStage = document.getElementById("tu-results-stage");
  const tuRejectionBanner = document.getElementById("tu-rejection-banner");
  const tuRejectionText = document.getElementById("tu-rejection-text");

  // Overview elements
  const tuMetaStatus = document.getElementById("tu-meta-status");
  const tuMetaModel = document.getElementById("tu-meta-model");
  const tuMetaChangedPct = document.getElementById("tu-meta-changed-pct");
  const tuMetaChangedPx = document.getElementById("tu-meta-changed-px");
  const tuMetaLatency = document.getElementById("tu-meta-latency");
  const tuMetaPhysicalSlot = document.getElementById("tu-meta-physical-slot");
  const tuMetaPhysicalArea = document.getElementById("tu-meta-physical-area");

  // Diagnostics elements
  const tuRegDiagnosticsCard = document.getElementById("tu-reg-diagnostics-card");
  const tuRegStatusBadge = document.getElementById("tu-reg-status-badge");
  const tuRegMetricsGrid = document.getElementById("tu-reg-metrics-grid");
  const tuRegWarningsBox = document.getElementById("tu-reg-warnings-box");

  // Quad viewer elements
  const tuQuadGrid = document.getElementById("tu-quad-grid");
  const tuViewT1 = document.getElementById("tu-view-t1");
  const tuViewT2 = document.getElementById("tu-view-t2");
  const tuViewMask = document.getElementById("tu-view-mask");
  const tuViewOverlay = document.getElementById("tu-view-overlay");

  attachModalPreview(tuViewT1, "Observation T1 (Earlier Optical Scene)");
  attachModalPreview(tuViewT2, "Observation T2 (Later Registered Optical Scene)");
  attachModalPreview(tuViewMask, "Binary Change Mask (τ* = 0.4500)");
  attachModalPreview(tuViewOverlay, "Spatial Change Overlay & Ranked Regions");

  // Regions & Semantic elements
  const tuRegionsSection = document.getElementById("tu-regions-section");
  const tuRegionsCardsContainer = document.getElementById("tu-regions-cards-container");
  const tuVariSection = document.getElementById("tu-vari-section");
  const tuVariTableBody = document.getElementById("tu-vari-table-body");
  const tuSemanticSection = document.getElementById("tu-semantic-section");
  const tuSemanticFindingsContainer = document.getElementById("tu-semantic-findings-container");

  // Explanation elements
  const tuExplanationSection = document.getElementById("tu-explanation-section");
  const tuExpSummary = document.getElementById("tu-exp-summary");
  const tuExpEvidence = document.getElementById("tu-exp-evidence");
  const tuExpLimitations = document.getElementById("tu-exp-limitations");
  const tuExpUnknowns = document.getElementById("tu-exp-unknowns");
  const tuExpRecommendations = document.getElementById("tu-exp-recommendations");

  let t1FileObject = null;
  let t2FileObject = null;
  let t1DataUrl = null;
  let t2DataUrl = null;

  function showValidationMessage(text, isError = true) {
    if (!valMsgTU) return;
    valMsgTU.textContent = text;
    valMsgTU.className = isError ? "validation-msg" : "validation-msg validation-info";
    valMsgTU.classList.remove("hidden");
  }

  function clearValidationMessage() {
    if (!valMsgTU) return;
    valMsgTU.textContent = "";
    valMsgTU.classList.add("hidden");
  }

  function handleFileSelection(file, target) {
    clearValidationMessage();
    const maxBytes = 25 * 1024 * 1024;
    const allowed = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();

    if (!allowed.includes(ext)) {
      showValidationMessage(`Unsupported format for ${target.toUpperCase()}: ${ext}. Allowed: PNG, JPG, JPEG, TIFF.`);
      return;
    }

    if (file.size > maxBytes) {
      showValidationMessage(`File ${file.name} exceeds 25 MB limit (${(file.size / (1024*1024)).toFixed(1)} MB).`);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      const img = new Image();
      img.onload = () => {
        const dims = `${img.naturalWidth} × ${img.naturalHeight} px`;
        const sizeStr = file.size > 1024 * 1024
          ? `${(file.size / (1024*1024)).toFixed(2)} MB`
          : `${(file.size / 1024).toFixed(1)} KB`;

        if (target === "t1") {
          t1FileObject = file;
          t1DataUrl = dataUrl;
          if (previewImgT1) previewImgT1.src = dataUrl;
          if (metaBadgeT1) metaBadgeT1.innerHTML = `<strong>${escapeHtml(file.name)}</strong> · ${dims} · ${sizeStr}`;
          if (previewContainerT1) previewContainerT1.style.display = "block";
        } else {
          t2FileObject = file;
          t2DataUrl = dataUrl;
          if (previewImgT2) previewImgT2.src = dataUrl;
          if (metaBadgeT2) metaBadgeT2.innerHTML = `<strong>${escapeHtml(file.name)}</strong> · ${dims} · ${sizeStr}`;
          if (previewContainerT2) previewContainerT2.style.display = "block";
        }
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  }

  fileInputT1?.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0], "t1");
    }
  });

  fileInputT2?.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0], "t2");
    }
  });

  const setupDragDrop = (dropzone, target) => {
    if (!dropzone) return;
    ["dragenter", "dragover"].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelection(e.dataTransfer.files[0], target);
      }
    });
  };

  setupDragDrop(dropzoneT1, "t1");
  setupDragDrop(dropzoneT2, "t2");

  btnResetTU?.addEventListener("click", () => {
    t1FileObject = null;
    t2FileObject = null;
    t1DataUrl = null;
    t2DataUrl = null;
    if (fileInputT1) fileInputT1.value = "";
    if (fileInputT2) fileInputT2.value = "";
    if (previewContainerT1) previewContainerT1.style.display = "none";
    if (previewContainerT2) previewContainerT2.style.display = "none";
    if (tuResultsStage) tuResultsStage.style.display = "none";
    clearValidationMessage();
  });

  async function generateClientOverlay(t1Src, maskUri, regions) {
    return new Promise((resolve) => {
      const img1 = new Image();
      img1.crossOrigin = "anonymous";
      img1.onload = () => {
        const maskImg = new Image();
        maskImg.crossOrigin = "anonymous";
        maskImg.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = img1.naturalWidth || img1.width;
          canvas.height = img1.naturalHeight || img1.height;
          const ctx = canvas.getContext("2d");

          ctx.drawImage(img1, 0, 0, canvas.width, canvas.height);

          ctx.globalAlpha = 0.38;
          ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
          ctx.globalAlpha = 1.0;

          if (regions && regions.length > 0) {
            regions.forEach(r => {
              const bbox = r.bbox || [];
              if (bbox.length === 4) {
                const [xmin, ymin, xmax, ymax] = bbox;
                const rw = xmax - xmin;
                const rh = ymax - ymin;

                ctx.strokeStyle = "#E1A95F";
                ctx.lineWidth = 2;
                ctx.strokeRect(xmin, ymin, rw, rh);

                const label = `#${r.region_id}`;
                ctx.font = "bold 12px Inter, sans-serif";
                const textWidth = ctx.measureText(label).width;

                ctx.fillStyle = "#4A2E2B";
                ctx.fillRect(xmin, Math.max(0, ymin - 18), textWidth + 8, 18);

                ctx.fillStyle = "#F1EBE4";
                ctx.fillText(label, xmin + 4, Math.max(14, ymin - 4));
              }
            });
          }

          resolve(canvas.toDataURL("image/png"));
        };
        maskImg.src = maskUri;
      };
      img1.src = t1Src;
    });
  }

  temporalForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearValidationMessage();

    if (!t1FileObject || !t2FileObject) {
      showValidationMessage("Please provide both Observation T1 and Observation T2 image files.");
      return;
    }

    if (t1FileObject.name === t2FileObject.name && t1FileObject.size === t2FileObject.size) {
      showValidationMessage("Observation T1 and T2 appear identical. Change detection between identical scenes will detect minimal or zero change.", false);
    }

    const btnTextTU = btnSubmitTU?.querySelector(".btn-text");
    const btnSpinnerTU = btnSubmitTU?.querySelector(".btn-spinner");

    if (btnSubmitTU) btnSubmitTU.disabled = true;
    if (btnTextTU) btnTextTU.textContent = "Analyzing Temporal Pair...";
    if (btnSpinnerTU) btnSpinnerTU.classList.remove("hidden");

    try {
      const formData = new FormData();
      formData.append("file_t1", t1FileObject);
      formData.append("file_t2", t2FileObject);

      if (dateInputT1?.value.trim()) {
        formData.append("observation_date_t1", dateInputT1.value.trim());
      }
      if (dateInputT2?.value.trim()) {
        formData.append("observation_date_t2", dateInputT2.value.trim());
      }
      if (gsdInput?.value && parseFloat(gsdInput.value) > 0) {
        formData.append("gsd_meters_per_pixel", parseFloat(gsdInput.value));
      }
      if (descInput?.value.trim()) {
        formData.append("description", descInput.value.trim());
      }
      if (modelSelectTU?.value) {
        formData.append("detector_model", modelSelectTU.value);
      }
      if (maxRegionsInput?.value) {
        formData.append("max_regions", parseInt(maxRegionsInput.value, 10));
      }

      const res = await analyzeTemporalPair(formData);

      if (res.status === "registration_rejected") {
        if (tuRejectionBanner) {
          tuRejectionBanner.classList.remove("hidden");
          if (tuRejectionText) {
            tuRejectionText.textContent = (res.errors && res.errors.length > 0)
              ? res.errors.join(" ")
              : "Observation images could not be reliably co-registered to meet geometric confidence standards. Processing halted safely.";
          }
        }
        if (tuRegDiagnosticsCard) tuRegDiagnosticsCard.style.display = "block";
        if (tuRegStatusBadge) {
          tuRegStatusBadge.textContent = "REGISTRATION REJECTED";
          tuRegStatusBadge.className = "badge badge-accent";
        }
        const diag = res.registration_diagnostics || {};
        if (tuRegMetricsGrid) {
          tuRegMetricsGrid.innerHTML = `
            <div class="reg-metric-item"><span>Match Count:</span><strong>${diag.match_count || 0}</strong></div>
            <div class="reg-metric-item"><span>Inliers:</span><strong>${diag.inlier_count || 0} (${((diag.inlier_ratio || 0)*100).toFixed(1)}%)</strong></div>
            <div class="reg-metric-item"><span>Reproj. Error:</span><strong>${(diag.reprojection_error || 0).toFixed(2)} px</strong></div>
            <div class="reg-metric-item"><span>Confidence:</span><strong>${((diag.confidence_score || 0)*100).toFixed(1)}%</strong></div>
          `;
        }
        if (tuQuadGrid) tuQuadGrid.style.display = "none";
        if (tuRegionsSection) tuRegionsSection.style.display = "none";
        if (tuVariSection) tuVariSection.style.display = "none";
        if (tuSemanticSection) tuSemanticSection.style.display = "none";
        if (tuExplanationSection) tuExplanationSection.style.display = "none";

        if (tuResultsStage) {
          tuResultsStage.style.display = "block";
          tuResultsStage.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        return;
      }

      if (tuRejectionBanner) tuRejectionBanner.classList.add("hidden");
      if (tuQuadGrid) tuQuadGrid.style.display = "grid";
      if (tuRegionsSection) tuRegionsSection.style.display = "block";
      if (tuVariSection) tuVariSection.style.display = "block";
      if (tuSemanticSection) tuSemanticSection.style.display = "block";
      if (tuExplanationSection) tuExplanationSection.style.display = "block";

      const meta = res.detector_metadata || {};
      const diag = res.registration_diagnostics || {};
      const maskUri = res.change_mask?.mask_data_uri || "";
      const regions = res.regions || [];
      const gsdProvided = res.image_metadata?.gsd_meters_per_pixel != null;

      if (tuMetaStatus) tuMetaStatus.textContent = "Aligned (Success)";
      if (tuMetaModel) tuMetaModel.textContent = `${meta.model_id || "M9_BIT_CD"} (τ*=${(meta.threshold_used || 0.45).toFixed(4)})`;
      if (tuMetaChangedPct) tuMetaChangedPct.textContent = `${(meta.total_changed_area_pct || 0).toFixed(2)}%`;
      if (tuMetaChangedPx) tuMetaChangedPx.textContent = `${(meta.total_changed_pixels || 0).toLocaleString()} px`;
      if (tuMetaLatency) tuMetaLatency.textContent = `${(meta.latency_ms || 0).toFixed(1)} ms`;

      if (gsdProvided && tuMetaPhysicalSlot && tuMetaPhysicalArea) {
        const totalAreaSqm = regions.reduce((acc, r) => acc + (r.physical_area_sqm || 0), 0);
        tuMetaPhysicalArea.textContent = `${totalAreaSqm.toLocaleString(undefined, { maximumFractionDigits: 1 })} m²`;
        tuMetaPhysicalSlot.style.display = "block";
      } else if (tuMetaPhysicalSlot) {
        tuMetaPhysicalSlot.style.display = "none";
      }

      if (tuRegDiagnosticsCard) tuRegDiagnosticsCard.style.display = "block";
      if (tuRegStatusBadge) {
        tuRegStatusBadge.textContent = `ALIGNED (${(diag.transformation_type || "HOMOGRAPHY").toUpperCase()})`;
        tuRegStatusBadge.className = "badge badge-success";
      }
      if (tuRegMetricsGrid) {
        tuRegMetricsGrid.innerHTML = `
          <div class="reg-metric-item"><span>Match Count:</span><strong>${diag.match_count || 0}</strong></div>
          <div class="reg-metric-item"><span>RANSAC Inliers:</span><strong>${diag.inlier_count || 0} (${((diag.inlier_ratio || 0)*100).toFixed(1)}%)</strong></div>
          <div class="reg-metric-item"><span>Reproj. Error:</span><strong>${(diag.reprojection_error || 0).toFixed(2)} px</strong></div>
          <div class="reg-metric-item"><span>ECC Correlation:</span><strong>${diag.ecc_correlation != null ? diag.ecc_correlation.toFixed(4) : "N/A"}</strong></div>
          <div class="reg-metric-item"><span>Confidence:</span><strong>${((diag.confidence_score || 0)*100).toFixed(1)}%</strong></div>
        `;
      }
      if (tuRegWarningsBox) {
        if (diag.warnings && diag.warnings.length > 0) {
          tuRegWarningsBox.innerHTML = `⚠️ <strong>Registration Warnings:</strong> ${diag.warnings.map(escapeHtml).join("; ")}`;
          tuRegWarningsBox.style.display = "block";
        } else {
          tuRegWarningsBox.style.display = "none";
        }
      }

      if (tuViewT1) tuViewT1.src = t1DataUrl || "";
      if (tuViewT2) tuViewT2.src = t2DataUrl || "";
      if (tuViewMask) tuViewMask.src = maskUri;

      if (tuViewOverlay && t1DataUrl && maskUri) {
        try {
          const overlayDataUrl = await generateClientOverlay(t1DataUrl, maskUri, regions);
          tuViewOverlay.src = overlayDataUrl;
        } catch (_) {
          tuViewOverlay.src = maskUri;
        }
      }

      if (tuRegionsCardsContainer) {
        if (regions.length > 0) {
          tuRegionsCardsContainer.innerHTML = regions.map(r => {
            const bbox = r.bbox || [0, 0, 0, 0];
            const areaPct = ((r.area_ratio || 0) * 100).toFixed(3);
            const centroid = r.centroid || [0, 0];
            const physHtml = (gsdProvided && r.physical_area_sqm != null)
              ? ` · <strong>Physical Area:</strong> ${r.physical_area_sqm.toLocaleString(undefined, { maximumFractionDigits: 1 })} m²`
              : "";
            return `
              <div class="region-card">
                <div class="region-card-header">
                  <span class="region-card-title">Region #${r.region_id}</span>
                  <span class="badge">[${bbox[0]}, ${bbox[1]}, ${bbox[2]}, ${bbox[3]}]</span>
                </div>
                <div class="region-card-body">
                  <strong>Pixel Area:</strong> ${r.pixel_area.toLocaleString()} px (${areaPct}% of scene)<br>
                  <strong>Dimensions:</strong> ${r.bbox_width} &times; ${r.bbox_height} px · <strong>Centroid:</strong> (${centroid[0]}, ${centroid[1]})${physHtml}
                </div>
              </div>
            `;
          }).join("");
        } else {
          tuRegionsCardsContainer.innerHTML = `<p class="section-caveat-text">No significant change regions exceeded the minimum morphological threshold.</p>`;
        }
      }

      if (tuVariTableBody) {
        const variObs = res.vari_observations || [];
        if (variObs.length > 0) {
          tuVariTableBody.innerHTML = variObs.map((v, i) => `
            <tr>
              <td><strong>#${v.region_id || (i + 1)}</strong></td>
              <td><span class="badge ${v.status === 'supported' ? 'badge-success' : 'badge-accent'}">${escapeHtml(v.status || 'N/A')}</span></td>
              <td>${v.t1_mean_vari != null ? v.t1_mean_vari.toFixed(4) : 'N/A'}</td>
              <td>${v.t2_mean_vari != null ? v.t2_mean_vari.toFixed(4) : 'N/A'}</td>
              <td><strong>${v.delta_vari != null ? (v.delta_vari > 0 ? '+' : '') + v.delta_vari.toFixed(4) : 'N/A'}</strong></td>
              <td>${escapeHtml(v.interpretation || 'No interpretation')}</td>
            </tr>
          `).join("");
        } else {
          tuVariTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; color: var(--color-taupe); padding: 1rem;">No regional VARI greenness observations available.</td></tr>`;
        }
      }

      if (tuSemanticFindingsContainer) {
        const findings = res.semantic_findings || [];
        if (findings.length > 0) {
          tuSemanticFindingsContainer.innerHTML = findings.map(f => `
            <div class="semantic-finding-item">
              <div class="semantic-finding-header">
                <strong>Region #${f.region_id}: ${escapeHtml(f.change_category)}</strong>
                <span class="badge ${f.requires_human_review ? 'badge-accent' : 'badge-success'}">
                  ${f.requires_human_review ? 'Requires Human Review' : 'Automated Confidence'}
                </span>
              </div>
              <p style="font-size: 0.85rem; color: var(--color-charcoal-rock); margin-bottom: 0.4rem;">
                <strong>Visual Observation:</strong> ${escapeHtml(f.visual_observation || 'None')}
              </p>
              ${f.evidence_for_category && f.evidence_for_category.length > 0 ? `
                <div style="font-size: 0.8rem; color: var(--color-taupe);">
                  <strong>Evidence:</strong> ${f.evidence_for_category.map(escapeHtml).join('; ')}
                </div>
              ` : ''}
              ${f.alternative_interpretations && f.alternative_interpretations.length > 0 ? `
                <div style="font-size: 0.8rem; color: var(--color-taupe); margin-top: 0.2rem;">
                  <strong>Alternative Hypotheses:</strong> ${f.alternative_interpretations.map(escapeHtml).join('; ')}
                </div>
              ` : ''}
            </div>
          `).join("");
        } else {
          tuSemanticFindingsContainer.innerHTML = `<p class="section-caveat-text">Multimodal semantic analysis unavailable or no regions extracted.</p>`;
        }
      }

      const exp = res.temporal_explanation;
      if (exp) {
        if (tuExpSummary) {
          tuExpSummary.innerHTML = `<p>${escapeHtml(exp.overall_summary || 'Analysis completed.')}</p>`;
        }
        if (tuExpEvidence) {
          const evItems = exp.detector_evidence || [];
          tuExpEvidence.innerHTML = evItems.length > 0 ? evItems.map(e => `<li>${escapeHtml(e)}</li>`).join("") : "<li>Unaltered detector telemetry.</li>";
        }
        if (tuExpLimitations) {
          const limItems = exp.scientific_limitations || [];
          tuExpLimitations.innerHTML = limItems.length > 0 ? limItems.map(l => `<li>${escapeHtml(l)}</li>`).join("") : "<li>Resolution and sensor limits apply.</li>";
        }
        if (tuExpUnknowns) {
          const unkItems = exp.unknown_variables || [];
          tuExpUnknowns.innerHTML = unkItems.length > 0 ? unkItems.map(u => `<li>${escapeHtml(u)}</li>`).join("") : "<li>No unknown variables noted.</li>";
        }
        if (tuExpRecommendations) {
          const recItems = exp.human_review_recommendations || [];
          tuExpRecommendations.innerHTML = recItems.length > 0 ? recItems.map(r => `<li>${escapeHtml(r)}</li>`).join("") : "<li>Automated verification complete.</li>";
        }
      } else {
        if (tuExpSummary) {
          tuExpSummary.innerHTML = `<p>Groq vision/explanation service unconfigured or rate-limited. Deterministic detection masks, registration metrics, and VARI evidence are fully preserved.</p>`;
        }
      }

      if (tuResultsStage) {
        tuResultsStage.style.display = "block";
        tuResultsStage.scrollIntoView({ behavior: "smooth", block: "start" });
      }

    } catch (err) {
      showValidationMessage(`Temporal analysis failed: ${err.message}`);
    } finally {
      if (btnSubmitTU) btnSubmitTU.disabled = false;
      if (btnTextTU) btnTextTU.textContent = "Execute Temporal Analysis";
      if (btnSpinnerTU) btnSpinnerTU.classList.add("hidden");
    }
  });
}

