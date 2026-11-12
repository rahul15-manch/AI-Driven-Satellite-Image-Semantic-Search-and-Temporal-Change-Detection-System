"""Vercel Serverless Function entrypoint for the Satellite AI Research System API."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Ensure project root and api directory are in sys.path
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent

for p in [str(PROJECT_ROOT), str(API_DIR), os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)
root_str = str(PROJECT_ROOT)

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
    try:
        from src.api.server import app as _app
    except ImportError:
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
        task_dir = "/var/task"
        root_files = os.listdir(task_dir) if os.path.exists(task_dir) else []
        tree = {}
        for root, dirs, files in os.walk(task_dir):
            rel = os.path.relpath(root, task_dir)
            if rel.count(os.sep) < 2:
                tree[rel] = files[:10] + ([f"... and {len(files)-10} more"] if len(files) > 10 else [])

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "FastAPI application failed to initialize on Vercel cold-start.",
                "python_version": sys.version,
                "current_working_dir": os.getcwd(),
                "sys_path": sys.path,
                "files_in_var_task": root_files,
                "dir_tree": tree,
                "traceback": startup_error,
            },
        )

__all__ = ["app"]
