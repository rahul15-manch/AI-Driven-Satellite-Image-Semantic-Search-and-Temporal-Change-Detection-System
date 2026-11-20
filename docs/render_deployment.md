# Render Cloud Deployment Guide (CPU-Optimized)

**Project:** AI-Driven Satellite Image Semantic Search & Temporal Change Detection System  
**Deployment Target:** [Render](https://render.com) (Free / Starter Web Service)  
**Architecture:** FastAPI RESTful Backend + Static HTML/CSS/JS Frontend UI (`/`) + Swagger Docs (`/docs`)

---

## 1. Minimal Image Footprint Strategy

Render free/standard tiers enforce strict slug and memory limits (512 MB RAM on free tier). The complete raw academic datasets are over **4.4 GB** (`data/raw/levir_cd/` = 2.3 GB, `data/raw/rsicd/` = 2.0 GB).

To prevent deployment timeouts, build failures, or memory crashes:
- **Raw Datasets (`data/raw/*`) & Processed Datasets (`data/processed/*`) are excluded** via `.gitignore`.
- **Curated Necessary Demo Pack (`data/demo_samples/`) is packaged (~19 MB total):**
  - **LEVIR-CD Bi-Temporal Pairs:** 5 verified test pairs (`test_1`, `test_10`, `test_100`, `test_101`, `test_102`) across `A/` (T1), `B/` (T2), and `label/` (Ground Truth).
  - **RSICD Semantic Scenes:** 15 representative aerial scenes across diverse land-cover categories (`airport`, `baseballfield`, `denseresidential`, `port`, `railwaystation`, `storagetanks`).
- **Resilient Fallback Handling:** The API serves verified sample pairs for live inference, precomputed FAISS embeddings (2.2 MB) for the entire 1,093 RSICD gallery, and category-matched demo fallbacks with SVG badges for missing image IDs.

---

## 2. Quick Deploy: Option A (Render Blueprint via `render.yaml`)

This repository includes a preconfigured [render.yaml](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/render.yaml) blueprint:

1. Push your repository to GitHub or GitLab.
2. Log into the [Render Dashboard](https://dashboard.render.com).
3. Click **New +** &rarr; **Blueprint**.
4. Select your connected repository.
5. Render will automatically detect `render.yaml` and configure:
   - **Service Type:** Web Service
   - **Runtime:** Python 3.10
   - **Build Command:** `./build.sh`
   - **Start Command:** `python run_web.py`
   - **Health Check Endpoint:** `/api/health`
6. Click **Apply**. Render will trigger the build and provide your live URL (e.g., `https://satellite-ai-system.onrender.com`).

---

## 3. Quick Deploy: Option B (Manual Web Service Configuration)

If configuring manually via the Render web UI:

| Setting | Value |
| :--- | :--- |
| **Name** | `satellite-ai-system` |
| **Environment** | `Python` |
| **Region** | Oregon (US West) or Frankfurt (EU) |
| **Branch** | `main` |
| **Build Command** | `./build.sh` *(or `pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt`)* |
| **Start Command** | `python run_web.py` |
| **Plan** | Free (or Starter for persistent uptime) |

### Environment Variables
Under the **Environment** tab on Render, add:
- `PYTHONPATH`: `.`
- `HOST`: `0.0.0.0`
- `PYTHON_VERSION`: `3.10.14`

---

## 4. Why the Custom `build.sh` is Critical

Standard `pip install torch` downloads GPU/CUDA binaries exceeding **2.5 GB**, which causes Render free builds to terminate due to out-of-disk/memory limits.

Our [`build.sh`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/build.sh) enforces:
```bash
python -m pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -r requirements.txt
```
This reduces the PyTorch installation package from **2,500 MB to ~160 MB**, ensuring clean, deterministic builds under Render's quotas.

---

## 5. Live Service Endpoints

Once deployed:

- **Web Application & Interactive Analyst Interface:**  
  `https://<your-subdomain>.onrender.com/`
- **Interactive Swagger REST Documentation:**  
  `https://<your-subdomain>.onrender.com/docs`
- **Alternative ReDoc Documentation:**  
  `https://<your-subdomain>.onrender.com/redoc`
- **System Health & Hardware Telemetry:**  
  `https://<your-subdomain>.onrender.com/api/health`
- **Verified Test Pairs Endpoint:**  
  `https://<your-subdomain>.onrender.com/api/pairs`
