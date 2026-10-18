"""
dsd_exporter.py -- MATLAB Driving Scenario Designer (DSD) Exporter.

Converts Scenario objects or OSMRoadNetwork road graphs into executable MATLAB
scripts (.m) utilizing the MATLAB Automated Driving Toolbox API:
  - scenario = drivingScenario('SampleTime', dt, 'StopTime', t_stop);
  - road(scenario, roadCenters, roadWidth, 'Lanes', lanesSpec);
  - vehicle(scenario, 'ClassID', ..., 'Length', ..., 'Width', ...);
  - trajectory(actor, waypoints, speeds);
  - pedestrian(scenario, ...) and actor(scenario, ...) for diverse traffic entities
    (pedestrians, two-wheelers, auto-rickshaws, cattle, pushcarts, barricades).
  - Weather and environment annotations plus visionDetectionGenerator simulation.

CLI Usage:
  python dsd_exporter.py --scenario <path_or_id> --output <path.m>
  python dsd_exporter.py --osm-lat 19.0760 --osm-lon 72.8777 --output <path.m>
"""

import os
import sys
import math
import json
import re
import argparse
from typing import Optional, Dict, List, Tuple, Union, Any

# Ensure scenario_lab and project root are in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCENARIO_LAB = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SCENARIO_LAB not in sys.path:
    sys.path.insert(0, _SCENARIO_LAB)

try:
    from scenario_lab.schema.scenario_schema import (
        Scenario, Road, Environment, EgoVehicle, Actor,
        Metadata, SimulationParameters,
    )
    _HAS_SCHEMA = True
except ImportError:
    _HAS_SCHEMA = False

try:
    from scenario_lab.osm.osm_importer import (
        OSMRoadNetwork, RoadSegment, OSMNode,
        import_from_coordinates, network_to_scenario,
    )
    _HAS_OSM = True
except ImportError:
    _HAS_OSM = False

try:
    from scenario_lab.storage.store import ScenarioStore
    _HAS_STORE = True
except ImportError:
    _HAS_STORE = False


# ---------------------------------------------------------------------------
# Actor Specifications for MATLAB Automated Driving Toolbox
# ---------------------------------------------------------------------------
# Maps actor_class to:
# (matlab_type, class_id, length_m, width_m, height_m, plot_color_rgb, mesh_type)
ACTOR_MATLAB_SPECS: Dict[str, Tuple[str, int, float, float, float, Tuple[float, float, float], Optional[str]]] = {
    # Passenger Vehicles (ClassID 1)
    "car":                  ("vehicle",    1, 4.7, 1.8, 1.4, (0.000, 0.447, 0.741), "driving.scenario.carMesh"),
    "suv":                  ("vehicle",    1, 4.8, 1.9, 1.7, (0.200, 0.400, 0.800), "driving.scenario.carMesh"),
    # Heavy Commercial Vehicles (ClassID 2)
    "truck":                ("vehicle",    2, 8.5, 2.5, 3.4, (0.850, 0.325, 0.098), "driving.scenario.truckMesh"),
    "bus":                  ("vehicle",    2, 10.5, 2.5, 3.2, (0.929, 0.694, 0.125), "driving.scenario.truckMesh"),
    "tractor":              ("vehicle",    2, 5.0, 2.2, 2.4, (0.466, 0.674, 0.188), None),
    # Two-Wheelers & Cycles (ClassID 3)
    "motorcycle":           ("vehicle",    3, 2.0, 0.8, 1.3, (0.635, 0.078, 0.184), "driving.scenario.bicycleMesh"),
    "scooter":              ("vehicle",    3, 1.9, 0.7, 1.2, (0.494, 0.184, 0.556), "driving.scenario.bicycleMesh"),
    "bicycle":              ("vehicle",    3, 1.8, 0.6, 1.1, (0.301, 0.745, 0.933), "driving.scenario.bicycleMesh"),
    "person":               ("pedestrian", 4, 0.5, 0.5, 1.7, (0.850, 0.325, 0.098), "driving.scenario.pedestrianMesh"),
    "pedestrian":           ("pedestrian", 4, 0.5, 0.5, 1.7, (0.850, 0.325, 0.098), "driving.scenario.pedestrianMesh"),
    "child":                ("pedestrian", 4, 0.4, 0.4, 1.2, (0.900, 0.500, 0.200), "driving.scenario.pedestrianMesh"),
    # Three-Wheelers / Para-transit (ClassID 5)
    "auto_rickshaw":        ("vehicle",    5, 2.8, 1.4, 1.8, (0.950, 0.800, 0.100), None),
    "e_rickshaw":           ("vehicle",    5, 2.8, 1.3, 1.8, (0.200, 0.700, 0.400), None),
    # Animals & Cattle (ClassID 6)
    "cow":                  ("actor",      6, 2.2, 1.5, 1.5, (0.450, 0.280, 0.150), None),
    "buffalo":              ("actor",      6, 2.4, 1.6, 1.6, (0.150, 0.150, 0.150), None),
    "dog":                  ("actor",      6, 0.8, 0.4, 0.6, (0.750, 0.550, 0.350), None),
    "goat":                 ("actor",      6, 0.9, 0.4, 0.75, (0.984, 0.443, 0.522), None),
    # Slow Non-Motorized / Informal Transport (ClassID 7)
    "pushcart":             ("actor",      7, 2.2, 1.4, 1.2, (0.550, 0.400, 0.250), None),
    "thela":                ("actor",      7, 2.2, 1.4, 1.2, (0.550, 0.400, 0.250), None),
    # Barricades & Road Hazards (ClassID 8)
    "barricade":            ("actor",      8, 2.0, 0.5, 1.0, (0.850, 0.325, 0.098), None),
    "barrier":              ("actor",      8, 2.0, 0.5, 1.0, (0.850, 0.325, 0.098), None),
    "police_barricade":     ("actor",      8, 2.0, 0.5, 1.0, (0.850, 0.325, 0.098), None),
    "construction_barrier": ("actor",      8, 2.0, 0.5, 1.0, (0.850, 0.325, 0.098), None),
    "pothole":              ("actor",      8, 1.2, 1.2, 0.1, (0.200, 0.200, 0.200), None),
    "obstacle":             ("actor",      8, 1.5, 1.5, 1.0, (0.500, 0.500, 0.500), None),
}

DEFAULT_ACTOR_SPEC = ("actor", 0, 2.0, 1.0, 1.2, (0.500, 0.500, 0.500), None)


# ---------------------------------------------------------------------------
# Sensor & Environment Parameter Computations
# ---------------------------------------------------------------------------
def _compute_sensor_params(env: Optional[Environment]) -> Tuple[float, float, float]:
    """
    Computes visionDetectionGenerator specifications based on weather conditions.
    Returns: (max_range_m, detection_probability, false_positives_per_image)
    """
    if env is None:
        return 120.0, 0.95, 0.05

    weather = env.weather or "clear"
    visibility = env.visibility_m if env.visibility_m is not None else 300.0

    if weather == "heavy_rain":
        max_range = min(50.0, visibility * 0.8)
        p_det = 0.70
        fppi = 0.50
    elif weather == "light_rain":
        max_range = min(90.0, visibility * 0.9)
        p_det = 0.88
        fppi = 0.15
    elif weather == "fog":
        max_range = min(40.0, visibility * 0.75)
        p_det = 0.60
        fppi = 1.00
    elif weather == "overcast":
        max_range = min(120.0, visibility)
        p_det = 0.92
        fppi = 0.08
    else:  # clear
        max_range = min(150.0, visibility)
        p_det = 0.98
        fppi = 0.05

    if env.time_of_day == "night":
        max_range = min(max_range, 70.0)
        p_det = max(0.40, p_det - 0.15)
        fppi *= 2.0
    elif env.time_of_day == "dusk":
        max_range = min(max_range, 100.0)
        p_det = max(0.50, p_det - 0.05)
        fppi *= 1.2

    return max_range, p_det, round(fppi, 2)


# ---------------------------------------------------------------------------
# Coordinate Projection Helper (WGS84 -> Cartesian ENU)
# ---------------------------------------------------------------------------
def _gps_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Convert (lat, lon) to local Cartesian (x: East, y: North) in meters."""
    dlat = lat - lat0
    dlon = lon - lon0
    y = dlat * 111320.0
    x = dlon * (111320.0 * math.cos(math.radians(lat0)))
    return x, y


def _sanitize_func_name(name: str) -> str:
    """Sanitize a string into a valid MATLAB function identifier."""
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "scenario_" + cleaned
    return cleaned[:48]


# ---------------------------------------------------------------------------
# Trajectory Generation for DSD
# ---------------------------------------------------------------------------
def _generate_actor_trajectory(
    actor: Actor,
    max_time_s: float
) -> Tuple[List[Tuple[float, float, float]], List[float], Optional[List[float]]]:
    """
    Computes 3D waypoints [x, y, z], speeds [mps], and optional waypoint arrival times
    based on the actor's initial state, velocity, and Indian traffic behaviour.
    """
    x0 = float(actor.initial_x_m)
    y0 = float(actor.initial_y_m)
    vx = float(actor.initial_vx_mps)
    vy = float(actor.initial_vy_mps)
    beh = actor.behaviour or "constant_velocity"
    trig_s = float(actor.behaviour_trigger_s) if actor.behaviour_trigger_s is not None else 2.5
    y_tgt = float(actor.behaviour_y_target_m) if actor.behaviour_y_target_m is not None else (0.0 if y0 != 0.0 else 3.5)

    initial_speed = math.hypot(vx, vy)

    if beh == "static" or (initial_speed < 0.05 and beh not in ("crossing", "cut_in")):
        # Static entity: Fixed position, zero speed
        return [(x0, y0, 0.0)], [0.0], None

    if beh == "constant_velocity":
        t_end = max(1.0, max_time_s)
        x1 = x0 + vx * t_end
        y1 = y0 + vy * t_end
        waypoints = [(x0, y0, 0.0), (x1, y1, 0.0)]
        speeds = [initial_speed, initial_speed]
        return waypoints, speeds, None

    elif beh == "sudden_stop":
        t_stop = min(max(0.5, trig_s), max_time_s - 0.5)
        x_trig = x0 + vx * t_stop
        y_trig = y0 + vy * t_stop
        waypoints = [
            (x0, y0, 0.0),
            (x_trig, y_trig, 0.0),
            (x_trig + 0.01, y_trig, 0.0),
        ]
        times = [0.0, t_stop, max_time_s]
        speeds = [initial_speed, initial_speed, 0.0]
        return waypoints, speeds, times

    elif beh == "crossing":
        t_start = min(trig_s, max_time_s * 0.4)
        cross_vy = vy if abs(vy) > 0.1 else (1.2 if y_tgt > y0 else -1.2)
        dist_y = y_tgt - y0
        t_cross = max(1.0, abs(dist_y) / abs(cross_vy))
        t_done = min(t_start + t_cross, max_time_s)

        x_wait = x0 + vx * t_start
        y_wait = y0
        x_cross = x_wait + vx * (t_done - t_start)
        y_cross = y_tgt

        t_rem = max_time_s - t_done
        x_end = x_cross + vx * t_rem
        y_end = y_cross

        waypoints = [
            (x0, y0, 0.0),
            (x_wait, y_wait, 0.0),
            (x_cross, y_cross, 0.0),
            (x_end, y_end, 0.0),
        ]
        times = [0.0, t_start, t_done, max_time_s]
        v_cross = math.hypot(vx, cross_vy)
        speeds = [math.hypot(vx, 0), math.hypot(vx, 0), v_cross, math.hypot(vx, 0)]
        return waypoints, speeds, times

    elif beh == "cut_in":
        t_start = min(trig_s, max_time_s * 0.4)
        t_cut = 2.0
        t_done = min(t_start + t_cut, max_time_s)

        x_wait = x0 + vx * t_start
        x_cut = x_wait + vx * t_cut
        t_rem = max_time_s - t_done
        x_end = x_cut + vx * t_rem

        waypoints = [
            (x0, y0, 0.0),
            (x_wait, y0, 0.0),
            (x_cut, y_tgt, 0.0),
            (x_end, y_tgt, 0.0),
        ]
        times = [0.0, t_start, t_done, max_time_s]
        speeds = [initial_speed, initial_speed, initial_speed, initial_speed]
        return waypoints, speeds, times

    elif beh == "wrong_side":
        # Moving against traffic direction (typically negative x)
        speed = initial_speed if initial_speed > 0.5 else 8.0
        x1 = x0 - speed * max_time_s
        waypoints = [(x0, y0, 0.0), (x1, y0, 0.0)]
        speeds = [speed, speed]
        return waypoints, speeds, None

    else:  # gradual or general fallback
        t_end = max_time_s
        x1 = x0 + vx * t_end
        y1 = y_tgt
        speed = math.hypot(vx, (y_tgt - y0) / t_end)
        waypoints = [(x0, y0, 0.0), (x1, y1, 0.0)]
        speeds = [speed, speed]
        return waypoints, speeds, None


# ---------------------------------------------------------------------------
# Resolver Helper
# ---------------------------------------------------------------------------
def _load_scenario_from_input(target: Union[Scenario, Dict[str, Any], str]) -> Scenario:
    """Resolves scenario from Scenario object, dict, file path, storage DB, or template ID."""
    if isinstance(target, Scenario):
        return target

    if isinstance(target, dict):
        if "scenario" in target and isinstance(target["scenario"], dict):
            return Scenario.model_validate(target["scenario"])
        return Scenario.model_validate(target)

    if isinstance(target, str):
        # 1. File path
        if os.path.isfile(target):
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "scenario" in data and isinstance(data["scenario"], dict):
                    return Scenario.model_validate(data["scenario"])
                return Scenario.model_validate(data)

        # 2. Check Storage DB
        storage_db_dir = os.path.join(_SCENARIO_LAB, "storage", "db", "scenarios")
        direct_path = os.path.join(storage_db_dir, f"{target}.json")
        if os.path.isfile(direct_path):
            with open(direct_path, "r", encoding="utf-8") as f:
                return Scenario.model_validate(json.load(f))

        # 3. Check Templates JSON
        templates_path = os.path.join(_SCENARIO_LAB, "templates", "indian_road_templates.json")
        if os.path.isfile(templates_path):
            with open(templates_path, "r", encoding="utf-8") as f:
                templates = json.load(f)
                for t in templates:
                    if t.get("id") == target:
                        return Scenario.model_validate(t["scenario"])

    raise FileNotFoundError(f"Could not resolve Scenario from target: '{target}'")


def _load_osm_network_from_input(target: Union[Any, Dict[str, Any], str]) -> Any:
    """Resolves OSMRoadNetwork from OSMRoadNetwork instance, dict, or json file path."""
    if hasattr(target, "segments"):
        return target

    if isinstance(target, str) and os.path.isfile(target):
        with open(target, "r", encoding="utf-8") as f:
            target = json.load(f)

    if isinstance(target, dict) and "segments" in target:
        segments = [
            RoadSegment(
                osm_way_id=s.get("osm_way_id", 0),
                name=s.get("name", "Road"),
                highway_type=s.get("highway_type", "residential"),
                indian_road_type=s.get("indian_road_type", "urban_street"),
                indian_description=s.get("indian_description", ""),
                lanes=s.get("lanes", 2),
                speed_limit_kmh=s.get("speed_limit_kmh", 50),
                is_oneway=s.get("is_oneway", False),
                surface=s.get("surface", "tarmac"),
                length_m=s.get("length_m", 100.0),
                nodes=[OSMNode(**n) for n in s.get("nodes", [])],
            )
            for s in target.get("segments", [])
        ]
        return OSMRoadNetwork(
            segments=segments,
            bbox=tuple(target.get("bbox", (0, 0, 0, 0))),
            center_lat=target.get("center", {}).get("lat", 0.0),
            center_lon=target.get("center", {}).get("lon", 0.0),
        )

    raise ValueError(f"Could not resolve OSMRoadNetwork from input: {type(target)}")


# ---------------------------------------------------------------------------
# MATLAB DSD Script Builder from Scenario
# ---------------------------------------------------------------------------
def export_to_dsd_script(
    scenario: Union[Scenario, Dict[str, Any], str],
    output_path: Optional[str] = None
) -> str:
    """
    Exports a Scenario dataclass instance to an executable MATLAB script (.m)
    compatible with MATLAB Driving Scenario Designer and Automated Driving Toolbox.

    Args:
        scenario: Scenario dataclass instance, dict, file path, or template ID.
        output_path: Optional filesystem path to write the .m script.

    Returns:
        Generated MATLAB script code as string.
    """
    sc = _load_scenario_from_input(scenario)

    s_id = sc.scenario_id or "dsd_scenario"
    if output_path:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        func_name = _sanitize_func_name(base_name)
    else:
        func_name = _sanitize_func_name(f"dsd_{s_id}")

    sc_name = sc.metadata.name if sc.metadata and sc.metadata.name else s_id
    sim_dt = sc.simulation_parameters.dt_s if sc.simulation_parameters else 0.033
    t_stop = sc.simulation_parameters.max_time_s if sc.simulation_parameters else 35.0

    # Environment
    env = sc.environment
    env_weather = env.weather if (env and env.weather) else "clear"
    env_time = env.time_of_day if (env and env.time_of_day) else "day"
    env_vis = env.visibility_m if (env and env.visibility_m is not None) else 300.0

    sensor_range, sensor_pdet, sensor_fppi = _compute_sensor_params(env)

    # Road specifications
    road = sc.road
    lane_half_w = road.lane_half_width_m if road else 3.5
    road_len = road.length_m if road else 200.0
    total_road_width = round(2.0 * lane_half_w, 2)
    single_lane_w = round(lane_half_w, 2)

    # Determine lane layout based on road type and markings
    road_type = road.type if road else "urban_street"
    markings = road.markings if road else "broken_center"

    if road_type == "highway":
        lane_spec_str = f"lanespec([2 2], 'Width', {round(total_road_width / 4.0, 2)})"
    elif markings == "none" or road_type == "village_road":
        lane_spec_str = f"lanespec([1 1], 'Width', {single_lane_w})"
    else:
        lane_spec_str = f"lanespec([1 1], 'Width', {single_lane_w})"

    # Ego vehicle
    ego = sc.ego_vehicle
    ego_speed_mps = round(ego.initial_speed_kmh / 3.6, 4) if ego else 11.1111
    ego_x0 = ego.initial_x_m if ego else 0.0
    ego_y0 = ego.initial_y_m if ego else 0.0
    ego_yaw = ego.initial_heading_deg if ego and ego.initial_heading_deg else 0.0
    ego_goal_x = ego.goal_x_m if ego else road_len
    ego_goal_y = ego.goal_y_m if ego else 0.0

    ego_yaw_line = f"\n    'Yaw', {ego_yaw:.2f}," if abs(ego_yaw) > 1e-3 else ""

    # Build MATLAB code lines
    lines = [
        f"function [scenario, egoVehicle, visionSensor] = {func_name}()",
        f"% {func_name.upper()} Driving Scenario Designer Export",
        f"% Scenario:     {sc_name}",
        f"% Scenario ID:  {s_id}",
        f"% Generated by: ADAS Scenario Lab (SIH 2025 Winning Solution - Team TwinX)",
        f"% Architecture: MATLAB Automated Driving Toolbox (drivingScenario API)",
        f"%",
        f"% Environment Profile:",
        f"%   Weather:     {env_weather}",
        f"%   Time of Day: {env_time}",
        f"%   Visibility:  {env_vis} m",
        f"",
        f"% Construct Driving Scenario object",
        f"scenario = drivingScenario('SampleTime', {sim_dt}, 'StopTime', {t_stop});",
        f"",
        f"% -------------------------------------------------------------------------",
        f"% Road Network Construction",
        f"% -------------------------------------------------------------------------",
        f"roadCenters = [0.0 0.0 0.0; {road_len} 0.0 0.0];",
        f"roadWidth = {total_road_width};",
        f"lanesSpec = {lane_spec_str};",
        f"road(scenario, roadCenters, roadWidth, 'Lanes', lanesSpec);",
        f"",
        f"% -------------------------------------------------------------------------",
        f"% Ego Vehicle Setup",
        f"% -------------------------------------------------------------------------",
        f"egoVehicle = vehicle(scenario, ...",
        f"    'ClassID', 1, ...",
        f"    'Length', 4.7, ...",
        f"    'Width', 1.8, ...",
        f"    'Height', 1.4, ...",
        f"    'Position', [{ego_x0} {ego_y0} 0.0], ...{ego_yaw_line}",
        f"    'PlotColor', [0.000 0.447 0.741], ...",
        f"    'Mesh', driving.scenario.carMesh);",
        f"",
    ]

    # Ego trajectory
    if ego_speed_mps > 0.01:
        lines.extend([
            f"egoWaypoints = [{ego_x0} {ego_y0} 0.0; {ego_goal_x} {ego_goal_y} 0.0];",
            f"egoSpeeds = [{ego_speed_mps}; {ego_speed_mps}];",
            f"trajectory(egoVehicle, egoWaypoints, egoSpeeds);",
            f"",
        ])

    # Actors
    lines.extend([
        f"% -------------------------------------------------------------------------",
        f"% Traffic Entities & Indian Road Actors",
        f"% -------------------------------------------------------------------------",
    ])

    actors = sc.actors or []
    for idx, act in enumerate(actors, 1):
        spec = ACTOR_MATLAB_SPECS.get(act.actor_class, DEFAULT_ACTOR_SPEC)
        m_type, class_id, def_len, def_wid, def_hgt, color, mesh = spec

        act_len = act.length_m if act.length_m is not None else def_len
        act_wid = act.width_m if act.width_m is not None else def_wid
        act_hgt = def_hgt
        act_var = f"actor{idx}"

        c_str = f"[{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}]"
        mesh_str = f", ...\n    'Mesh', {mesh}" if mesh else ""

        # Heading alignment from initial velocities
        act_yaw = 0.0
        if abs(act.initial_vx_mps) > 0.05 or abs(act.initial_vy_mps) > 0.05:
            act_yaw = round(math.degrees(math.atan2(act.initial_vy_mps, act.initial_vx_mps)), 2)
        act_yaw_line = f", ...\n    'Yaw', {act_yaw}" if abs(act_yaw) > 1e-3 else ""

        lines.extend([
            f"% Actor {idx}: {act.id} ({act.actor_class}) - Behaviour: {act.behaviour}",
            f"{act_var} = {m_type}(scenario, ...",
            f"    'ClassID', {class_id}, ...",
            f"    'Length', {act_len}, ...",
            f"    'Width', {act_wid}, ...",
            f"    'Height', {act_hgt}, ...",
            f"    'Position', [{act.initial_x_m} {act.initial_y_m} 0.0], ...",
            f"    'PlotColor', {c_str}{act_yaw_line}{mesh_str});",
        ])

        waypoints, speeds, times = _generate_actor_trajectory(act, t_stop)

        if len(waypoints) > 1:
            wp_str = "; ".join(f"{round(p[0], 2)} {round(p[1], 2)} {round(p[2], 2)}" for p in waypoints)
            spd_str = "; ".join(f"{round(s, 2)}" for s in speeds)
            lines.extend([
                f"trajectory({act_var}, [{wp_str}], [{spd_str}]);",
                f"",
            ])
        else:
            lines.extend([
                f"% Static entity - position fixed at [{act.initial_x_m} {act.initial_y_m} 0.0]",
                f"",
            ])

    # Vision detection generator simulation configuration (Automated Driving Toolbox)
    lines.extend([
        f"% -------------------------------------------------------------------------",
        f"% Vision Sensor Simulation (ADAS Robustness Suite)",
        f"% -------------------------------------------------------------------------",
        f"% Sensor parameters conditioned on weather: {env_weather}, time: {env_time}",
        f"visionSensor = visionDetectionGenerator(...",
        f"    'SensorIndex', 1, ...",
        f"    'SensorLocation', [1.9 0.0], ...",
        f"    'Height', 1.2, ...",
        f"    'Pitch', 1.0, ...",
        f"    'MaxRange', {sensor_range:.1f}, ...",
        f"    'DetectionProbability', {sensor_pdet:.2f}, ...",
        f"    'FalsePositivesPerImage', {sensor_fppi:.2f});",
        f"",
        f"% -------------------------------------------------------------------------",
        f"% Visualizer / Standalone Execution Loop",
        f"% -------------------------------------------------------------------------",
        f"% Note: To inspect in GUI, run: drivingScenarioDesigner(scenario);",
        f"if nargout == 0",
        f"    figure('Name', '{func_name}', 'NumberTitle', 'off');",
        f"    plot(scenario);",
        f"    while advance(scenario)",
        f"        pause(scenario.SampleTime);",
        f"    end",
        f"end",
        f"end",
        f"",
    ])

    content = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    return content


# ---------------------------------------------------------------------------
# MATLAB DSD Script Builder from OSMRoadNetwork
# ---------------------------------------------------------------------------
def export_osm_to_dsd(
    osm_network: Union[Any, Dict[str, Any], str],
    output_path: Optional[str] = None
) -> str:
    """
    Exports an OSMRoadNetwork graph to an executable MATLAB script (.m)
    with accurate multi-segment road geometry and lane specifications.

    Args:
        osm_network: OSMRoadNetwork instance, dict, or file path.
        output_path: Optional filesystem path to write the .m script.

    Returns:
        Generated MATLAB script code as string.
    """
    net = _load_osm_network_from_input(osm_network)

    if not hasattr(net, "segments") or not net.segments:
        raise ValueError("Cannot export empty OSMRoadNetwork: no road segments present.")

    lat0 = float(net.center_lat)
    lon0 = float(net.center_lon)

    if output_path:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        func_name = _sanitize_func_name(base_name)
    else:
        func_name = _sanitize_func_name(f"dsd_osm_{int(abs(lat0 * 100))}_{int(abs(lon0 * 100))}")

    sim_dt = 0.033
    t_stop = 35.0

    lines = [
        f"function [scenario, egoVehicle, visionSensor] = {func_name}()",
        f"% {func_name.upper()} OpenStreetMap to MATLAB Driving Scenario Designer Export",
        f"% Center:       Lat {lat0:.5f}, Lon {lon0:.5f}",
        f"% Segments:     {len(net.segments)}",
        f"% Total Length: {round(net.total_length_m, 1)} m",
        f"% Generated by: ADAS Scenario Lab (SIH 2025 Winning Solution - Team TwinX)",
        f"",
        f"% Construct Driving Scenario object",
        f"scenario = drivingScenario('SampleTime', {sim_dt}, 'StopTime', {t_stop});",
        f"",
        f"% -------------------------------------------------------------------------",
        f"% Road Network Construction from OpenStreetMap",
        f"% -------------------------------------------------------------------------",
    ]

    valid_roads = 0
    dominant_segment: Optional[RoadSegment] = None
    max_len = -1.0
    dominant_road_centers: List[Tuple[float, float, float]] = []

    for seg_idx, seg in enumerate(net.segments, 1):
        if not seg.nodes or len(seg.nodes) < 2:
            continue

        raw_centers: List[Tuple[float, float, float]] = []
        for node in seg.nodes:
            x, y = _gps_to_enu(node.lat, node.lon, lat0, lon0)
            raw_centers.append((round(x, 2), round(y, 2), 0.0))

        # Filter out consecutive duplicate points (< 0.05m)
        filtered_centers: List[Tuple[float, float, float]] = [raw_centers[0]]
        for pt in raw_centers[1:]:
            last_pt = filtered_centers[-1]
            if math.hypot(pt[0] - last_pt[0], pt[1] - last_pt[1]) >= 0.05:
                filtered_centers.append(pt)

        if len(filtered_centers) < 2:
            continue

        valid_roads += 1
        lanes = max(1, seg.lanes)
        lane_w = 3.5
        total_w = round(lanes * lane_w, 2)

        if seg.is_oneway or lanes == 1:
            lane_spec_str = f"lanespec({lanes}, 'Width', {lane_w})"
        else:
            num_fwd = max(1, lanes // 2)
            num_rev = max(1, lanes - num_fwd)
            lane_spec_str = f"lanespec([{num_fwd} {num_rev}], 'Width', {lane_w})"

        pts_str = "; ".join(f"{p[0]} {p[1]} {p[2]}" for p in filtered_centers)
        safe_name = re.sub(r'[^a-zA-Z0-9 ]', '', seg.name or 'Road')

        lines.extend([
            f"% Road {seg_idx}: {safe_name} ({seg.indian_road_type}) - Lanes: {lanes}",
            f"roadCenters_{seg_idx} = [{pts_str}];",
            f"roadWidth_{seg_idx} = {total_w};",
            f"lanesSpec_{seg_idx} = {lane_spec_str};",
            f"road(scenario, roadCenters_{seg_idx}, roadWidth_{seg_idx}, 'Lanes', lanesSpec_{seg_idx});",
            f"",
        ])

        if seg.length_m > max_len:
            max_len = seg.length_m
            dominant_segment = seg
            dominant_road_centers = filtered_centers

    if valid_roads == 0:
        raise ValueError("Could not extract any valid road segments with at least 2 distinct vertices.")

    # Ego vehicle placement along dominant road
    ego_speed_mps = round(min(60.0, float(dominant_segment.speed_limit_kmh)) / 3.6, 4)
    start_pt = dominant_road_centers[0]

    ego_waypoints_str = "; ".join(f"{p[0]} {p[1]} {p[2]}" for p in dominant_road_centers)
    speeds_str = "; ".join(f"{ego_speed_mps}" for _ in dominant_road_centers)

    lines.extend([
        f"% -------------------------------------------------------------------------",
        f"% Ego Vehicle Setup (Navigating dominant road: {dominant_segment.name})",
        f"% -------------------------------------------------------------------------",
        f"egoVehicle = vehicle(scenario, ...",
        f"    'ClassID', 1, ...",
        f"    'Length', 4.7, ...",
        f"    'Width', 1.8, ...",
        f"    'Height', 1.4, ...",
        f"    'Position', [{start_pt[0]} {start_pt[1]} {start_pt[2]}], ...",
        f"    'PlotColor', [0.000 0.447 0.741], ...",
        f"    'Mesh', driving.scenario.carMesh);",
        f"",
        f"egoWaypoints = [{ego_waypoints_str}];",
        f"egoSpeeds = [{speeds_str}];",
        f"trajectory(egoVehicle, egoWaypoints, egoSpeeds);",
        f"",
        f"% -------------------------------------------------------------------------",
        f"% Standard Vision Sensor Simulation Configuration",
        f"% -------------------------------------------------------------------------",
        f"visionSensor = visionDetectionGenerator(...",
        f"    'SensorIndex', 1, ...",
        f"    'SensorLocation', [1.9 0.0], ...",
        f"    'Height', 1.2, ...",
        f"    'Pitch', 1.0, ...",
        f"    'MaxRange', 120.0, ...",
        f"    'DetectionProbability', 0.95, ...",
        f"    'FalsePositivesPerImage', 0.05);",
        f"",
        f"% -------------------------------------------------------------------------",
        f"% Visualizer / Standalone Execution Loop",
        f"% -------------------------------------------------------------------------",
        f"if nargout == 0",
        f"    figure('Name', '{func_name}', 'NumberTitle', 'off');",
        f"    plot(scenario);",
        f"    while advance(scenario)",
        f"        pause(scenario.SampleTime);",
        f"    end",
        f"end",
        f"end",
        f"",
    ])

    content = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    return content


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MATLAB Driving Scenario Designer (DSD) Exporter -- ADAS Scenario Lab"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=str, help="Scenario file path, ID, or template ID")
    group.add_argument("--osm-lat", type=float, help="Latitude for OSM road import")

    parser.add_argument("--osm-lon", type=float, help="Longitude for OSM road import")
    parser.add_argument("--osm-radius", type=float, default=500.0, help="Search radius in meters for OSM (default: 500)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .m script file path")

    args = parser.parse_args()

    try:
        if args.scenario:
            out_file = args.output
            export_to_dsd_script(args.scenario, out_file)
            print(f"[OK] Exported scenario '{args.scenario}' to MATLAB DSD script: {out_file}")
        elif args.osm_lat is not None:
            if args.osm_lon is None:
                parser.error("--osm-lon is required when --osm-lat is specified.")
            print(f"[INFO] Fetching OpenStreetMap network at ({args.osm_lat}, {args.osm_lon}, r={args.osm_radius}m)...")
            osm_net = import_from_coordinates(args.osm_lat, args.osm_lon, args.osm_radius)
            out_file = args.output
            export_osm_to_dsd(osm_net, out_file)
            print(f"[OK] Exported {len(osm_net.segments)} OSM road segments to MATLAB DSD script: {out_file}")

    except Exception as e:
        print(f"[ERROR] Export failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
