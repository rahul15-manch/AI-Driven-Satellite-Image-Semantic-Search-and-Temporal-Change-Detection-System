/**
 * Methodology & Limitations Module
 * Handles mathematical typesetting, equation formatting, and responsive card presentation.
 */

export function initMethodology() {
  const panel = document.getElementById("panel-methodology");
  if (!panel) return;

  // Initial render of math in methodology panel and application
  renderMath(panel);

  // Re-render when switching to methodology tab
  document.addEventListener("tab-switched", (e) => {
    if (e.detail?.tab === "methodology") {
      renderMath(panel);
    }
  });

  // Also listen for DOMContentLoaded or window load in case KaTeX loads async
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => renderMath(panel));
  }
  window.addEventListener("load", () => renderMath(panel));
}

export function renderMath(rootElement = document) {
  if (!rootElement) return;

  // 1. Preprocess bare LaTeX commands that lack delimiters
  preprocessBareMath(rootElement);

  // 2. If KaTeX auto-render is available, use official KaTeX renderer
  if (typeof window.renderMathInElement === "function") {
    try {
      window.renderMathInElement(rootElement, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true }
        ],
        throwOnError: false,
        errorColor: "#C86B4D"
      });
      return;
    } catch (err) {
      console.warn("KaTeX renderMathInElement failed, using fallback math typesetting:", err);
    }
  }

  // 3. Fallback math typesetting for offline or non-KaTeX environments
  renderFallbackMath(rootElement);
}

function preprocessBareMath(container) {
  const targets = container.querySelectorAll(
    ".info-card p, .info-card li, .methodology-card p, .methodology-card li, .panel-desc"
  );
  targets.forEach((el) => {
    let text = el.innerHTML;
    // Wrap bare \tau^* with $\tau^*$ if not already wrapped
    text = text.replace(/(?<!\$)\\tau\^\*(?!\$)/g, "$\\tau^*$");
    // Wrap bare \mathbb{R}^{...} with $\mathbb{R}^{...}$
    text = text.replace(/(?<!\$)\\mathbb\x7BR\x7D\^\x7B([^\x7D]+)\x7D(?!\$)/g, "$\\mathbb" + String.fromCharCode(123) + "R" + String.fromCharCode(125) + "^" + String.fromCharCode(123) + "$1" + String.fromCharCode(125) + "$");
    // Wrap bare \gamma^* with $\gamma^*$
    text = text.replace(/(?<!\$)\\gamma\^\*(?!\$)/g, "$\\gamma^*$");
    // Wrap bare \gamma = with $\gamma$ =
    text = text.replace(/(?<!\$)\\gamma(?!\*|\$)/g, "$\\gamma$");
    if (text !== el.innerHTML) {
      el.innerHTML = text;
    }
  });
}

function renderFallbackMath(container) {
  const targets = container.querySelectorAll(
    ".info-card p, .info-card li, .methodology-card p, .methodology-card li, .panel-desc"
  );
  targets.forEach((el) => {
    if (el.getAttribute("data-math-fallback") === "true") return;

    let html = el.innerHTML;
    if (!html.includes("$") && !html.includes("\\")) return;

    // Display math $$...$$
    html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
      return `<div class="katex-display math-display">${typesetExpression(math)}</div>`;
    });

    // Inline math $...$
    html = html.replace(/\$([^$]+?)\$/g, (_, math) => {
      return `<span class="katex math-inline">${typesetExpression(math)}</span>`;
    });

    el.innerHTML = html;
    el.setAttribute("data-math-fallback", "true");
  });
}

function typesetExpression(tex) {
  let s = tex.trim();

  // If KaTeX is directly available on window
  if (typeof window.katex !== "undefined" && typeof window.katex.renderToString === "function") {
    try {
      return window.katex.renderToString(s, { throwOnError: false });
    } catch (_) {
      // Fall through to manual typesetter
    }
  }

  // Robust semantic HTML/CSS typesetter mappings
  s = s.replace(/\\mathbb\x7BR\x7D\^\x7B([^\x7D]+)\x7D/g, '<span class="math-bb">ℝ</span><sup class="math-sup">$1</sup>');
  s = s.replace(/\\mathbb\x7BR\x7D/g, '<span class="math-bb">ℝ</span>');
  s = s.replace(/\\tau\^(\*|\x7B[^\x7D]+\x7D)/g, '<span class="math-sym">τ</span><sup class="math-sup">$1</sup>');
  s = s.replace(/\\tau/g, '<span class="math-sym">τ</span>');
  s = s.replace(/\\gamma\^(\*|\x7B[^\x7D]+\x7D)/g, '<span class="math-sym">γ</span><sup class="math-sup">$1</sup>');
  s = s.replace(/\\gamma/g, '<span class="math-sym">γ</span>');
  s = s.replace(/\\rightarrow/g, '→');
  s = s.replace(/\\times/g, '×');
  s = s.replace(/\\in/g, '∈');
  s = s.replace(/\\Delta/g, 'Δ');
  s = s.replace(/\\alpha/g, 'α');
  s = s.replace(/\\text\x7BSSIM\x7D/g, '<span class="math-text">SSIM</span>');
  s = s.replace(/\\text\x7B([^\x7D]+)\x7D/g, '<span class="math-text">$1</span>');
  s = s.replace(/L_([0-9a-zA-Z]+)/g, 'L<sub class="math-sub">$1</sub>');
  s = s.replace(/T_([0-9a-zA-Z]+)/g, 'T<sub class="math-sub">$1</sub>');
  s = s.replace(/\x7B([^\x7D]+)\x7D/g, '$1');

  if (s === "q" || s === "x") {
    return `<span class="math-var">${s}</span>`;
  }

  return `<span class="math-expr">${s}</span>`;
}
