"""
run.py — Universal launcher for the ADAS Vision project.

After Akshit's refactor, all modules moved into subfolders:
    core/       — main.py, config.py, decision_engine.py, frame_guard.py
    perception/ — lane_detection.py, object_detection.py, tracker.py, etc.
    simulation/ — udp_bridge.py, forward_collision_warning.py, etc.

This script adds all those folders to sys.path so the flat imports inside
each module keep working, then launches whichever entry-point you want.

Usage:
    # Run the main ADAS pipeline (same as before)
    python run.py main --video dashcam.mp4 --async

    # Run the UDP bridge (for MATLAB co-simulation)
    python run.py bridge --video dashcam.mp4 --async

    # Run the web demo
    python run.py demo
"""

import sys
from pathlib import Path

# ── Add all module directories to sys.path ────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
for folder in ("core", "perception", "simulation", "training", "export",
               "evaluation", "demo"):
    p = ROOT / folder
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Also add root itself (for any remaining top-level files)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Subcommand dispatch ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    subcmd = sys.argv.pop(1)   # remove so the target script sees its own args

    if subcmd == "main":
        import runpy
        runpy.run_path(str(ROOT / "core" / "main.py"), run_name="__main__")

    elif subcmd == "bridge":
        import runpy
        runpy.run_path(str(ROOT / "simulation" / "udp_bridge.py"),
                       run_name="__main__")

    elif subcmd == "demo":
        import runpy
        runpy.run_path(str(ROOT / "demo" / "web_demo.py"),
                       run_name="__main__")

    elif subcmd == "test":
        import runpy
        runpy.run_path(str(ROOT / "evaluation" / "test_udp_bridge.py"),
                       run_name="__main__")

    elif subcmd == "audit":
        import runpy
        runpy.run_path(str(ROOT / "evaluation" / "safety_audit.py"),
                       run_name="__main__")

    elif subcmd == "lidar":
        import runpy
        runpy.run_path(str(ROOT / "simulation" / "lidar_camera_fusion.py"),
                       run_name="__main__")

    else:
        print(f"Unknown subcommand: {subcmd!r}")
        print("Valid subcommands: main | bridge | demo | test | audit | lidar")
        sys.exit(1)
