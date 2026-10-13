"""FastAPI application factory and middleware configuration for Milestone 12."""

from __future__ import annotations

import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router as api_router
from utils.config_loader import get_project_root

ROOT = get_project_root()


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="AI-Driven Satellite Image Semantic Search & Change Detection API",
        description="RESTful serving layer for Cross-Modal Semantic Retrieval and Bi-Temporal Change Detection under CPU-first constraints.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Middleware (allows local research dashboard access)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    # 3. Include API Router
    app.include_router(api_router)

    # 4. Mount Frontend Static Files if directory exists
    frontend_dir = ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


# Module-level instance for uvicorn run command
app = create_app()
