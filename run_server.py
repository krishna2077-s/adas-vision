#!/usr/bin/env python3
"""
ADAS Vision & Scenario Lab - Unified Production Web Server
===========================================================
Hosts both the FastAPI REST backend and the React + Vite frontend on a single port.
Ready for 1-click execution on Replit, Render, Docker, or local environments.
"""

import os
import sys
import subprocess
import pathlib

# Ensure project root is in sys.path
BASE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Default to Python Kinematic Simulation Engine (0 MATLAB dependencies needed)
os.environ.setdefault("USE_PYTHON_SIM", "1")


def ensure_frontend_build():
    """Builds the frontend bundle if dist/ is missing."""
    frontend_dir = BASE_DIR / "scenario_lab" / "frontend"
    dist_dir = frontend_dir / "dist"
    index_html = dist_dir / "index.html"

    if index_html.exists():
        print(f"[*] Frontend production build found at: {dist_dir}")
        return

    print("[*] Frontend build missing. Building React + Vite frontend...")
    try:
        # Check npm availability
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("[*] Running npm run build...")
        subprocess.run(["npm", "run", "build"], cwd=str(frontend_dir), check=True)
        print("[+] Frontend build completed successfully.")
    except Exception as exc:
        print(f"[!] Warning: Could not build frontend automatically ({exc}).")
        print("[!] If running in production, run 'npm run build' inside scenario_lab/frontend.")


def main():
    ensure_frontend_build()

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 65)
    print("  ADAS Vision & Scenario Lab - Web Server")
    print(f"  Live at: http://localhost:{port}")
    print(f"  Listening on: {host}:{port}")
    print("  Engine: Pure-Python Kinematics (No MATLAB license required)")
    print("=" * 65)

    try:
        import uvicorn
    except ImportError:
        print("[!] Error: 'uvicorn' is not installed. Run: pip install uvicorn")
        sys.exit(1)

    uvicorn.run(
        "scenario_lab.api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
