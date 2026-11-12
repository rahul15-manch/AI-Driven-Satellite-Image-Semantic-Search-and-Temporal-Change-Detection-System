"""Vercel Serverless Function entrypoint for the Satellite AI Research System API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' and 'utils' packages are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set working directory to project root for consistent relative paths
try:
    os.chdir(str(PROJECT_ROOT))
except Exception:
    pass

from src.api.app import app

# Export app for Vercel's ASGI runtime
__all__ = ["app"]
