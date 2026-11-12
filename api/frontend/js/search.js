import { searchGallery, explainSearchResult } from "./api.js";

export function initSearch() {
  const form = document.getElementById("search-form");
  const queryInput = document.getElementById("search-query-input");
  const methodSelect = document.getElementById("search-method-select");
  const topkSelect = document.getElementById("search-topk-select");
  const submitBtn = document.getElementById("search-submit-btn");
  const btnText = submitBtn.querySelector(".btn-text");
  const btnSpinner = submitBtn.querySelector(".btn-spinner");
  
  const metaBar = document.getElementById("search-meta-bar");
  const metaMethod = document.getElementById("meta-method-name");
  const metaLatency = document.getElementById("meta-latency-val");
  const metaCount = document.getElementById("meta-results-count");
  
  const resultsContainer = document.getElementById("search-results-container");
  const chipButtons = document.querySelectorAll(".chip-btn");

  // Keep track of current query and dataset results for explain calls
  let currentSearchData = null;

  // Query chips click handler
  chipButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-query");
      if (q) {
        queryInput.value = q;
        form.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });
  });

  // Search form submit
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) {
      window.app?.showAlert("Please enter a search query.", "warning");
      return;
    }

    const method = methodSelect.value;
    const topK = parseInt(topkSelect.value, 10) || 8;

    // Loading UI state
    submitBtn.disabled = true;
    btnText.textContent = "Searching...";
    btnSpinner.classList.remove("hidden");

    try {
      const response = await searchGallery(query, method, topK);
      currentSearchData = response;
      renderResults(response);
    } catch (err) {
      window.app?.showAlert(`Semantic search failed: ${err.message}`, "error");
    } finally {
      submitBtn.disabled = false;
      btnText.textContent = "Search Gallery";
      btnSpinner.classList.add("hidden");
    }
  });

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
              <ul>${evidenceItems || "<li>Evidence derived from query and similarity ranking telemetry.</li>"}</ul>
            </div>
            <div class="explanation-detail-col col-limitations">
              <h5><span>⚠</span> Documented Limitations</h5>
              <ul>${limitationItems || "<li>Subject to general embedding representation constraints.</li>"}</ul>
            </div>
            <div class="explanation-detail-col col-unknowns">
              <h5><span>?</span> Unknowns / Missing</h5>
              <ul>${unknownItems || "<li>Sub-pixel spatial attributes unavailable.</li>"}</ul>
            </div>
          </div>
        </div>
      </div>
    `;
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

  function renderResults(data) {
    metaBar.style.display = "flex";
    
    // Format method label
    const methodNames = {
      "clip": "CLIP ViT-B/32 (Zero-Shot)",
      "prompt_ensemble": "CLIP + Domain Prompt Ensemble",
      "bm25": "BM25 Lexical Baseline"
    };
    metaMethod.textContent = methodNames[data.method] || data.method;
    metaLatency.textContent = `${data.latency_ms.toFixed(1)} ms`;
    metaCount.textContent = data.results.length;

    if (!data.results || data.results.length === 0) {
      resultsContainer.innerHTML = `
        <div class="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          <h3>No matching satellite scenes found</h3>
          <p>Try modifying your query or selecting a different retrieval algorithm.</p>
        </div>
      `;
      return;
    }

    let cardsHtml = '<div class="search-results-list">';
    data.results.forEach(item => {
      const scoreFormatted = data.method.includes("clip") || data.method.includes("ensemble")
        ? `${(item.similarity_score * 100).toFixed(1)}%`
        : item.similarity_score.toFixed(3);

      const scoreLabel = data.method.includes("clip") || data.method.includes("ensemble")
        ? "Cosine Similarity"
        : "BM25 Score";

      const firstCaption = item.captions && item.captions.length > 0
        ? item.captions[0]
        : "No caption metadata available.";

      cardsHtml += `
        <div class="search-result-card" data-rank="${item.rank}">
          <div class="search-result-main">
            <div class="search-result-thumb-box result-img-wrapper" data-src="${item.image_url}" data-caption="${escapeHtml(item.filename)} (Rank ${item.rank}, ${scoreLabel}: ${scoreFormatted})">
              <img src="${item.image_url}" alt="${escapeHtml(item.filename)}" loading="lazy">
              <div class="rank-badge">#${item.rank}</div>
              <div class="category-tag">${escapeHtml(item.category)}</div>
            </div>
            <div class="search-result-content">
              <div class="search-result-meta-row">
                <div class="search-result-filename" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</div>
                <div class="search-result-score-badge" title="${scoreLabel}">
                  <span class="score-type-label">${scoreLabel}:</span>
                  <span class="score-val">${scoreFormatted}</span>
                </div>
              </div>
              <p class="search-result-caption" title="${escapeHtml(firstCaption)}">${escapeHtml(firstCaption)}</p>
              <div class="search-result-actions">
                <button type="button" class="btn-explain btn-explain-search" data-rank="${item.rank}">
                  <span class="explain-icon">💡</span> Explain This Result (Qwen 27B)
                </button>
              </div>
            </div>
          </div>
          <div class="explanation-slot" id="explain-slot-${item.rank}"></div>
        </div>
      `;
    });
    cardsHtml += '</div>';

    resultsContainer.innerHTML = cardsHtml;

    // Attach click listener for modal preview
    resultsContainer.querySelectorAll(".result-img-wrapper").forEach(wrapper => {
      wrapper.addEventListener("click", () => {
        const src = wrapper.getAttribute("data-src");
        const caption = wrapper.getAttribute("data-caption");
        window.app?.openModal(src, caption);
      });
    });

    // Attach explain buttons
    resultsContainer.querySelectorAll(".btn-explain-search").forEach(btn => {
      btn.addEventListener("click", async () => {
        const rank = parseInt(btn.getAttribute("data-rank"), 10);
        const item = data.results.find(r => r.rank === rank);
        if (!item) return;

        const slot = document.getElementById(`explain-slot-${rank}`);
        if (!slot) return;

        // Toggle if already visible
        if (slot.innerHTML.trim() !== "") {
          slot.innerHTML = "";
          btn.innerHTML = `<span class="explain-icon">💡</span> Explain This Result (Qwen 27B)`;
          return;
        }

        const originalBtnHtml = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span>⏳ Analyzing with Qwen 27B...</span>`;

        try {
          const payload = {
            query: data.query,
            method: data.method,
            image_id: item.image_id,
            filename: item.filename,
            rank: item.rank,
            score: item.similarity_score,
            score_type: data.method.includes("clip") || data.method.includes("ensemble") ? "Cosine Similarity" : "BM25 Score",
            category: item.category,
            captions: item.captions || []
          };
          const explanation = await explainSearchResult(payload);
          slot.innerHTML = renderExplanationCard(explanation);
          btn.innerHTML = `<span class="explain-icon">✕</span> Hide Explanation`;
        } catch (err) {
          slot.innerHTML = `
            <div class="explanation-error-state">
              <span>Could not generate explanation: ${escapeHtml(err.message)}</span>
              <button type="button" class="btn-explain-retry">Retry</button>
            </div>
          `;
          slot.querySelector(".btn-explain-retry")?.addEventListener("click", () => {
            slot.innerHTML = "";
            btn.click();
          });
          btn.innerHTML = originalBtnHtml;
        } finally {
          btn.disabled = false;
        }
      });
    });
  }
}

