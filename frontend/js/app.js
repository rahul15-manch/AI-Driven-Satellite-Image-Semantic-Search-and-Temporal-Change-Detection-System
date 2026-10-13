/**
 * Main Application Orchestrator for Milestone 12 Web Interface
 */

import { checkHealth } from "./api.js";
import { initSearch } from "./search.js";
import { initChangeDetection } from "./change_detection.js";
import { initResults } from "./results.js";

class App {
  constructor() {
    this.statusIndicator = document.getElementById("status-indicator");
    this.statusLabel = document.getElementById("status-label");
    this.alertBanner = document.getElementById("alert-banner");
    this.alertMessage = document.getElementById("alert-message");
    this.alertClose = document.getElementById("alert-close");
    
    this.modal = document.getElementById("image-modal");
    this.modalImage = document.getElementById("modal-image");
    this.modalCaption = document.getElementById("modal-caption");
    this.modalClose = document.getElementById("modal-close-btn");

    this.alertTimer = null;
  }

  init() {
    this.setupNavigation();
    this.setupAlertBanner();
    this.setupModal();
    this.pollHealth();

    // Initialize sub-modules
    initSearch();
    initChangeDetection();
    initResults();

    // Check health periodically every 20 seconds
    setInterval(() => this.pollHealth(), 20000);
  }

  setupNavigation() {
    const navButtons = document.querySelectorAll(".nav-btn");
    const panels = document.querySelectorAll(".content-panel");

    navButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        if (!targetTab) return;

        navButtons.forEach(b => b.classList.remove("active"));
        panels.forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        const activePanel = document.getElementById(`panel-${targetTab}`);
        if (activePanel) {
          activePanel.classList.add("active");
          window.dispatchEvent(new Event("resize"));
          document.dispatchEvent(new CustomEvent("tab-switched", { detail: { tab: targetTab } }));
        }
      });
    });
  }

  setupAlertBanner() {
    this.alertClose?.addEventListener("click", () => {
      this.alertBanner.classList.add("hidden");
    });
  }

  showAlert(message, type = "info") {
    if (this.alertTimer) {
      clearTimeout(this.alertTimer);
    }

    this.alertMessage.textContent = message;
    this.alertBanner.className = `alert-banner alert-${type}`;
    this.alertBanner.classList.remove("hidden");

    this.alertTimer = setTimeout(() => {
      this.alertBanner.classList.add("hidden");
    }, 6000);
  }

  setupModal() {
    this.modalClose?.addEventListener("click", () => this.closeModal());
    this.modal?.querySelector(".modal-backdrop")?.addEventListener("click", () => this.closeModal());

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !this.modal.classList.contains("hidden")) {
        this.closeModal();
      }
    });
  }

  openModal(src, caption = "") {
    if (!this.modal || !this.modalImage) return;
    this.modalImage.src = src;
    this.modalCaption.textContent = caption;
    this.modal.classList.remove("hidden");
  }

  closeModal() {
    if (!this.modal) return;
    this.modal.classList.add("hidden");
    this.modalImage.src = "";
    this.modalCaption.textContent = "";
  }

  async pollHealth() {
    try {
      const data = await checkHealth();
      if (data && data.status === "healthy") {
        this.statusIndicator.className = "status-indicator online";
        this.statusLabel.textContent = `Backend Online (CPU • ${data.models_loaded?.length || 0} models)`;
      } else {
        this.statusIndicator.className = "status-indicator degraded";
        this.statusLabel.textContent = "Backend Degraded";
      }
    } catch (_) {
      this.statusIndicator.className = "status-indicator offline";
      this.statusLabel.textContent = "Backend Offline";
    }
  }
}

// Bootstrap application on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  const app = new App();
  window.app = app;
  app.init();
});
