/**
 * Semantic Search Module for RSICD Cross-Modal Retrieval
 */

import { searchGallery } from "./api.js";

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
      renderResults(response);
    } catch (err) {
      window.app?.showAlert(`Semantic search failed: ${err.message}`, "error");
    } finally {
      submitBtn.disabled = false;
      btnText.textContent = "Search Gallery";
      btnSpinner.classList.add("hidden");
    }
  });

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

    let cardsHtml = '<div class="results-grid">';
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
        <div class="result-card">
          <div class="result-img-wrapper" data-src="${item.image_url}" data-caption="${item.filename} (Rank ${item.rank}, ${scoreLabel}: ${scoreFormatted})">
            <img src="${item.image_url}" alt="${item.filename}" loading="lazy">
            <div class="result-rank">#${item.rank}</div>
          </div>
          <div class="result-info">
            <div class="result-header">
              <span class="result-category">${item.category}</span>
              <span class="result-score" title="${scoreLabel}">${scoreFormatted}</span>
            </div>
            <div class="result-filename" title="${item.filename}">${item.filename}</div>
            <p class="result-caption" title="${firstCaption}">${firstCaption}</p>
          </div>
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
  }
}
