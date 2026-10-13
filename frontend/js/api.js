/**
 * API Client Module for Milestone 12 Web Interface
 * Communicates with the FastAPI backend endpoints under /api/
 */

const API_BASE = "";

/**
 * Handle API HTTP response with detailed error parsing.
 * @param {Response} response 
 * @returns {Promise<any>}
 */
async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch (_) {
      // Keep default statusText if parsing fails
    }
    throw new Error(`[HTTP ${response.status}] ${errorDetail}`);
  }
  return response.json();
}

/**
 * Check backend health status.
 * @returns {Promise<{status: string, models_loaded: string[], uptime_seconds: number}>}
 */
export async function checkHealth() {
  const resp = await fetch(`${API_BASE}/api/health`, {
    headers: { "Accept": "application/json" }
  });
  return handleResponse(resp);
}

/**
 * Search the RSICD gallery using natural language query.
 * @param {string} query 
 * @param {string} method ('clip', 'prompt_ensemble', 'bm25')
 * @param {number} topK 
 * @returns {Promise<{query: string, method: string, top_k: number, latency_ms: number, results: Array}>}
 */
export async function searchGallery(query, method = "clip", topK = 8) {
  const resp = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json"
    },
    body: JSON.stringify({
      query: query.trim(),
      method: method,
      top_k: parseInt(topK, 10)
    })
  });
  return handleResponse(resp);
}

/**
 * Get the list of available LEVIR-CD test scene pairs.
 * @returns {Promise<Array<{pair_id: string, name: string, has_ground_truth: boolean}>>}
 */
export async function getTestPairs() {
  const resp = await fetch(`${API_BASE}/api/pairs`, {
    headers: { "Accept": "application/json" }
  });
  return handleResponse(resp);
}

/**
 * Execute bi-temporal change detection on a scene pair.
 * @param {string} pairId 
 * @param {string} model 
 * @param {number|null} threshold 
 * @param {number} gamma 
 * @returns {Promise<Object>}
 */
export async function detectChange(pairId, model, threshold = null, gamma = 0.0) {
  const payload = {
    pair_id: pairId,
    model: model,
    threshold: threshold !== null && !isNaN(threshold) ? parseFloat(threshold) : null,
    gamma: parseFloat(gamma) || 0.0
  };

  const resp = await fetch(`${API_BASE}/api/change-detection`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json"
    },
    body: JSON.stringify(payload)
  });
  return handleResponse(resp);
}

/**
 * Fetch official Milestone M11 benchmark evaluation results.
 * @returns {Promise<Object>}
 */
export async function getBenchmarkData() {
  const resp = await fetch(`${API_BASE}/api/benchmark`, {
    headers: { "Accept": "application/json" }
  });
  return handleResponse(resp);
}

/**
 * Check Groq LLM explanation service status.
 * @param {boolean} checkLive
 * @returns {Promise<{configured: boolean, provider: string, model: string, status: string, connectivity: string, message: string}>}
 */
export async function getLLMStatus(checkLive = false) {
  const url = `${API_BASE}/api/llm/status${checkLive ? "?check=true" : ""}`;
  const resp = await fetch(url, {
    headers: { "Accept": "application/json" }
  });
  return handleResponse(resp);
}

/**
 * Request an evidence-grounded explanation for a change detection prediction.
 * @param {Object} payload 
 * @returns {Promise<Object>}
 */
export async function explainChangeDetection(payload) {
  const resp = await fetch(`${API_BASE}/api/explain/change-detection`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json"
    },
    body: JSON.stringify(payload)
  });
  return handleResponse(resp);
}

/**
 * Request an evidence-grounded explanation for a semantic search result.
 * @param {Object} payload 
 * @returns {Promise<Object>}
 */
export async function explainSearchResult(payload) {
  const resp = await fetch(`${API_BASE}/api/explain/semantic-search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json"
    },
    body: JSON.stringify(payload)
  });
  return handleResponse(resp);
}

