"""CLI runner entry points for launching the backend and frontend servers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_api() -> None:
    """Launch the FastAPI backend server using uvicorn."""
    parser = argparse.ArgumentParser(description="Start the NACC progression FastAPI backend server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port number")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    import uvicorn
    from nacc_progression.platform.api import app

    print(f"🚀 Starting NACC Progression FastAPI server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def run_ui() -> None:
    """Launch the Streamlit clinical frontend."""
    parser = argparse.ArgumentParser(description="Start the NACC progression Streamlit frontend.")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    parser.add_argument("--server-address", default="0.0.0.0", help="Server address")
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={args.port}",
        f"--server.address={args.server_address}",
    ]
    print(f"🧠 Launching Streamlit frontend on http://localhost:{args.port}...")
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        sys.argv.pop(1)
        run_ui()
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "api":
            sys.argv.pop(1)
        run_api()
