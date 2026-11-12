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

if app is not None and startup_error is None:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from urllib.parse import parse_qsl, urlencode

    @app.middleware("http")
    async def vercel_path_rewrite_middleware(request: Request, call_next):
        # 1. If Vercel rewrote /api/:match* with ?path=:match* query param
        subpath = request.query_params.get("path") or request.query_params.get("match")
        if subpath and request.url.path in ("/api/index", "/api/index.py", "/api", "/api/"):
            if not subpath.startswith("/"):
                subpath = "/" + subpath
            request.scope["path"] = f"/api{subpath}"
            # Clean 'path' and 'match' keys from query_string
            qs = request.scope.get("query_string", b"")
            if qs:
                parsed = parse_qsl(qs.decode("latin1"))
                filtered = [(k, v) for k, v in parsed if k not in ("path", "match")]
                request.scope["query_string"] = urlencode(filtered).encode("latin1")
        else:
            # 2. Check Vercel routing headers
            original_path = request.headers.get("x-matched-path") or request.headers.get("x-forwarded-uri")
            if original_path and request.url.path in ("/api/index", "/api/index.py", "/api", "/api/"):
                clean_path = original_path.split("?")[0]
                request.scope["path"] = clean_path

        return await call_next(request)

    @app.api_route("/debug", methods=["GET", "POST"])
    @app.api_route("/api/debug", methods=["GET", "POST"])
    @app.api_route("/api/index", methods=["GET", "POST"])
    @app.api_route("/api/index.py", methods=["GET", "POST"])
    def debug_route(request: Request):
        return JSONResponse(
            content={
                "url_path": request.url.path,
                "scope_path": request.scope.get("path"),
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
                "available_routes": [
                    r.path if hasattr(r, "path") else getattr(r, "name", str(type(r)))
                    for r in request.app.routes
                ],
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "RouteNotFound",
                    "requested_url_path": request.url.path,
                    "scope_path": request.scope.get("path"),
                    "headers": dict(request.headers),
                    "query_params": dict(request.query_params),
                    "available_routes": [r.path for r in request.app.routes if hasattr(r, "path")],
                },
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

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
