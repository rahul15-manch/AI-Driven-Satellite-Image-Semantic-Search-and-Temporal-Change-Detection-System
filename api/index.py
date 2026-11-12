"""Vercel Serverless Function entrypoint for the Satellite AI Research System API."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# Set working directory to project root for consistent relative paths
try:
    os.chdir(root_str)
except Exception:
    pass

if "PYTHONPATH" in os.environ:
    if root_str not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = f"{root_str}:{os.environ['PYTHONPATH']}"
else:
    os.environ["PYTHONPATH"] = root_str

app = None
startup_error = None

try:
    from src.api.app import app as _app
    app = _app
except Exception as e:
    startup_error = traceback.format_exc()

if app is None or startup_error is not None:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Diagnostic Fallback")

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
    def catch_all(full_path: str):
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "FastAPI application failed to initialize on Vercel cold-start.",
                "python_version": sys.version,
                "current_working_dir": os.getcwd(),
                "sys_path": sys.path,
                "traceback": startup_error,
            },
        )

__all__ = ["app"]
