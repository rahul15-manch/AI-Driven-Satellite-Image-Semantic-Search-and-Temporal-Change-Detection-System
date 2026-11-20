#!/usr/bin/env bash
# Render Build Script for AI-Driven Satellite Image Semantic Search & Change Detection System
set -o errexit

echo "================================================================================"
echo "Starting Render Deployment Build: Satellite AI System (CPU-Optimized)"
echo "================================================================================"

# 1. Upgrade pip, setuptools, wheel
python -m pip install --upgrade pip setuptools wheel

# 2. Install CPU-only PyTorch first to prevent downloading 2GB+ CUDA wheels
echo "==> Installing CPU-only PyTorch (lightweight footprint)..."
python -m pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining production dependencies
echo "==> Installing project dependencies from requirements.txt..."
python -m pip install --no-cache-dir -r requirements.txt

# 4. Verification check
echo "==> Verifying core dependencies..."
python -c "import torch, faiss, fastapi, uvicorn; print(f'Verification SUCCESS: PyTorch {torch.__version__} (CPU: {not torch.cuda.is_available()})')"

echo "================================================================================"
echo "Render Build Completed Successfully!"
echo "================================================================================"
