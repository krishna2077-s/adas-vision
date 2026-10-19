"""
run_esmini.py -- Launch OpenSCENARIO and OpenDRIVE in esmini standalone 3D viewer.

Usage:
    python run_esmini.py --scenario template_twinx_nashik_ringroad
    python run_esmini.py --scenario template_monsoon_expressway
    python run_esmini.py --osc scenario_lab/generated/twinx_ringroad.xosc
"""

import os
import sys
import subprocess
import argparse
from typing import Optional, Tuple


def find_esmini_exe() -> Optional[str]:
    """Locate the esmini.exe executable inside the project or system path."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.join(project_root, "tools", "esmini")
    if os.path.isdir(tools_dir):
        for root, _, files in os.walk(tools_dir):
            if "esmini.exe" in files:
                return os.path.join(root, "esmini.exe")

    # Check PATH
    import shutil
    path_exe = shutil.which("esmini") or shutil.which("esmini.exe")
    if path_exe:
        return path_exe

    return None


def ensure_scenario_files(scenario_name: str) -> Tuple[str, str]:
    """Ensure .xodr and .xosc exist for the given scenario or template ID."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(project_root, "scenario_lab", "generated")
    os.makedirs(gen_dir, exist_ok=True)

    clean_name = scenario_name.replace("template_", "").replace(" ", "_").lower()
    xodr_path = os.path.join(gen_dir, f"{clean_name}.xodr")
    xosc_path = os.path.join(gen_dir, f"{clean_name}.xosc")

    if not os.path.isfile(xodr_path):
        from scenario_lab.export.opendrive_exporter import export_to_opendrive
        print(f"[*] Generating OpenDRIVE road network -> {xodr_path}")
        export_to_opendrive(scenario_name, output_path=xodr_path)

    if not os.path.isfile(xosc_path):
        from scenario_lab.export.openscenario_exporter import export_to_openscenario
        print(f"[*] Generating OpenSCENARIO triggers and actors -> {xosc_path}")
        xml_out = export_to_openscenario(scenario_name, xodr_filename=os.path.basename(xodr_path))
        with open(xosc_path, "w", encoding="utf-8") as f:
            f.write(xml_out)

    return xodr_path, xosc_path


def launch(xodr_path: Optional[str], xosc_path: str, window_w: int = 980, window_h: int = 720,
           headless: bool = False, cockpit: bool = True,
           weather: str = "dry", vehicle: str = "hector_ev",
           weight: Optional[float] = None, friction: Optional[float] = None,
           fcw: str = "comfort", tja: bool = True):
    exe = find_esmini_exe()
    if not exe:
        raise FileNotFoundError(
            "esmini.exe was not found. Please check tools/esmini."
        )

    bin_dir = os.path.dirname(exe)
    esmini_root = os.path.dirname(bin_dir)  # tools/esmini/esmini-demo
    gen_dir = os.path.dirname(os.path.abspath(xosc_path))

    cmd = [
        exe,
        "--path", gen_dir,
        "--path", esmini_root,
        "--osc", os.path.abspath(xosc_path),
    ]
    if xodr_path and os.path.isfile(xodr_path):
        cmd.extend(["--odr", os.path.abspath(xodr_path)])

    cockpit_proc = None
    if headless:
        cmd.append("--headless")
    else:
        cmd.extend([
            "--window", "20", "40", str(window_w), str(window_h),
            "--follow_object", "0",
            "--info_text", "3",
            "--sensors",
            "--view_mode", "both",
        ])
        if cockpit:
            cockpit_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hackathon_cockpit.py")
            if os.path.isfile(cockpit_script):
                print(f"[*] Launching ADAS Telemetry Cockpit ({vehicle.upper()}, {weather.upper()})...")
                cockpit_cmd = [sys.executable, cockpit_script, "--weather", weather, "--vehicle", vehicle, "--fcw", fcw]
                if not tja:
                    cockpit_cmd.append("--no-tja")
                if weight is not None:
                    cockpit_cmd.extend(["--weight", str(weight)])
                if friction is not None:
                    cockpit_cmd.extend(["--friction", str(friction)])
                cockpit_proc = subprocess.Popen(cockpit_cmd)

    print(f"\n{'=' * 60}")
    print(f"  Launching ADAS Vision - Hackathon Simulation Suite")
    print(f"  3D Simulator: {exe}")
    if xodr_path:
        print(f"  OpenDRIVE:    {xodr_path}")
    print(f"  Scenario:     {xosc_path}")
    print(f"  Vehicle:      {vehicle.upper()} (Custom Weight: {weight} kg)" if weight else f"  Vehicle:      {vehicle.upper()}")
    print(f"  Road/Weather: {weather.upper()} (Custom Friction mu: {friction})" if friction else f"  Road/Weather: {weather.upper()}")
    print(f"  ADAS Profile: FCW={fcw.upper()}, TJA={'ON' if tja else 'OFF'}, LKA=Gentle Nudge")
    print(f"{'=' * 60}")
    print("3D Simulator Controls:")
    print("  [Space]       Pause / Play")
    print("  [Tab]         Cycle Cameras (Follow Ego, Orbit, Driver POV, Top-down)")
    print("  [Up] / [Down] Speed Up / Slow Down Simulation")
    print("  [Left]/[Right]Scrub Timeline")
    print("  [Esc]         Exit\n")
    print("Cockpit HUD Telemetry Controls:")
    print("  [W]           Cycle Weather (Dry -> Monsoon -> Heavy Rain -> Gravel)")
    print("  [V]           Cycle Vehicle (MG Hector EV -> Hector PHEV -> Sedan -> SUV)")
    print("  [A]           Cycle FCW Sensitivity (Comfort/Dialed-Back -> Normal -> Aggressive)")
    print("  [T]           Toggle Traffic Jam Assist (TJA Curve/Bend Control)")
    print("  [L]           Toggle Gentle LKA Tactile Nudge")
    print("  [R]           Reset Telemetry Timeline to 0.0s\n")

    try:
        return subprocess.run(cmd, cwd=esmini_root)
    finally:
        if cockpit_proc:
            try:
                cockpit_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch scenario in esmini 3D simulator")
    parser.add_argument("--scenario", default="real_world",
                        help="Scenario template ID or name ('real_world', 'template_twinx_nashik_ringroad', etc.)")
    parser.add_argument("--osc", default=None, help="Path to .xosc file")
    parser.add_argument("--odr", default=None, help="Path to .xodr file")
    parser.add_argument("--width", type=int, default=980, help="Viewer window width")
    parser.add_argument("--height", type=int, default=720, help="Viewer window height")
    parser.add_argument("--headless", action="store_true", help="Run headlessly without graphical window")
    parser.add_argument("--no-cockpit", action="store_true", help="Disable the secondary telemetry cockpit HUD")
    parser.add_argument("--weather", default="dry",
                        choices=["dry", "wet", "heavy_rain", "gravel"],
                        help="Weather & road friction profile")
    parser.add_argument("--vehicle", default="hector_ev",
                        choices=["hector_ev", "hector_phev", "sedan", "hatchback", "compact_suv", "heavy_suv"],
                        help="Vehicle class preset (Default: hector_ev)")
    parser.add_argument("--weight", type=float, default=None,
                        help="Custom total vehicle mass (kg)")
    parser.add_argument("--friction", type=float, default=None,
                        help="Custom road surface peak friction coefficient mu")
    parser.add_argument("--fcw", default="comfort",
                        choices=["comfort", "normal", "aggressive"],
                        help="Driver-adjustable FCW sensitivity mode (Default: comfort)")
    parser.add_argument("--no-tja", action="store_true",
                        help="Disable Traffic Jam Assist (TJA)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(project_root, "scenario_lab", "generated")

    if args.osc and args.odr:
        xodr, xosc = args.odr, args.osc
    elif args.osc:
        xosc = args.osc
        xodr = args.osc.replace(".xosc", ".xodr")
        if not os.path.isfile(xodr):
            xodr = None
    elif args.scenario in ("real_world", "real_world_indian_highway"):
        xosc = os.path.join(gen_dir, "real_world_indian_highway.xosc")
        xodr = None
    else:
        xodr, xosc = ensure_scenario_files(args.scenario)

    launch(xodr, xosc, window_w=args.width, window_h=args.height,
           headless=args.headless, cockpit=not args.no_cockpit,
           weather=args.weather, vehicle=args.vehicle,
           weight=args.weight, friction=args.friction,
           fcw=args.fcw, tja=not args.no_tja)

