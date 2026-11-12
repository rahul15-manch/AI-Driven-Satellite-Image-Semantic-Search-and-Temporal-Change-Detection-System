"""Vercel Serverless Function entrypoint for the Satellite AI Research System API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Also ensure current working directory is in sys.path
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# Set working directory to project root for consistent relative paths
try:
    os.chdir(root_str)
except Exception:
    pass

# Explicitly ensure PYTHONPATH has project root
if "PYTHONPATH" in os.environ:
    if root_str not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = f"{root_str}:{os.environ['PYTHONPATH']}"
else:
    os.environ["PYTHONPATH"] = root_str

from src.api.app import app

__all__ = ["app"]
