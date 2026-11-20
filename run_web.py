"""Launcher script for Milestone 12 Web Interface & FastAPI server.

Usage:
    python run_web.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse
import sys
from pathlib import Path
import uvicorn

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


import os


def main():
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", "8000"))

    parser = argparse.ArgumentParser(description="Launch Satellite AI Research Web Interface")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host interface (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port number (default: {default_port})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")
    args = parser.parse_args()

    print("================================================================================")
    print("AI-Driven Satellite Semantic Search & Change Detection System (Milestone 12)")
    print(f"Launching web interface at: http://{args.host}:{args.port}")
    print(f"API documentation at:     http://{args.host}:{args.port}/docs")
    print("================================================================================")

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
