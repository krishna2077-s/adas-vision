"""
opendrive_exporter.py -- RoadRunner / OpenDRIVE (.xodr) Exporter.

Converts Scenario objects or OSMRoadNetwork road graphs into ASAM OpenDRIVE
XML format (.xodr) directly importable by MathWorks RoadRunner, CARLA, and VIRES VTD.

Produces valid OpenDRIVE structures conforming to ASAM OpenDRIVE v1.4+:
  - <header> with geographic reference (Proj4 / WGS84) and bounding box in inertial meters.
  - <road> with ID, name, length, and junction properties.
  - <type> with road categorization and speed limit.
  - <planView> containing <geometry> reference lines with <line> elements.
  - <elevationProfile> with flat/graded elevations.
  - <lateralProfile> with superelevation specifications.
  - <lanes> with <laneSection>, <left>, <center>, and <right> lane definitions,
    lane widths (polynomial standard), and standard roadMarks (solid, broken, solid solid).
  - <objects> embedding stationary actors, vehicles, cattle, and pedestrians with proper types.

CLI Usage:
  python opendrive_exporter.py --scenario <path_or_id> --output <path.xodr>
  python opendrive_exporter.py --osm-lat 19.0760 --osm-lon 72.8777 --output <path.xodr>
"""

import os
import sys
import math
import json
import re
import argparse
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
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
    from scenario_lab.export.dsd_exporter import ACTOR_MATLAB_SPECS, DEFAULT_ACTOR_SPEC
except ImportError:
    ACTOR_MATLAB_SPECS = {}
    DEFAULT_ACTOR_SPEC = ("actor", 0, 2.0, 1.0, 1.2, (0.5, 0.5, 0.5), None)


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


# Road mark mapping from schema to OpenDRIVE standard e_roadMarkType
ROAD_MARK_MAP = {
    "none": "none",
    "broken_center": "broken",
    "double_center": "solid solid",
    "full": "solid",
}

# Road type mapping from schema to OpenDRIVE type
ROAD_TYPE_MAP = {
    "village_road": "rural",
    "urban_street": "town",
    "highway": "motorway",
    "market_street": "pedestrian",
    "intersection": "town",
    "custom": "unknown",
}

# OpenDRIVE standard object types (e_objectType)
ACTOR_OPENDRIVE_TYPE_MAP: Dict[str, str] = {
    # Passenger vehicles
    "car":                  "car",
    "suv":                  "car",
    # Commercial vehicles
    "truck":                "truck",
    "bus":                  "bus",
    "tractor":              "obstacle",
    # Two-wheelers
    "motorcycle":           "motorbike",
    "scooter":              "motorbike",
    "bicycle":              "bike",
    # Vulnerable road users
    "person":               "pedestrian",
    "pedestrian":           "pedestrian",
    "child":                "pedestrian",
    # Three-wheelers & informal
    "auto_rickshaw":        "obstacle",
    "e_rickshaw":           "obstacle",
    "pushcart":             "obstacle",
    "thela":                "obstacle",
    # Cattle & animals
    "cow":                  "obstacle",
    "buffalo":              "obstacle",
    "dog":                  "obstacle",
    "goat":                 "obstacle",
    # Barriers & barricades
    "barricade":            "barrier",
    "barrier":              "barrier",
    "police_barricade":     "barrier",
    "construction_barrier": "barrier",
    "pothole":              "obstacle",
    "obstacle":             "obstacle",
}


# ---------------------------------------------------------------------------
# XML Construction Helpers
# ---------------------------------------------------------------------------
def _create_header(
    root: ET.Element,
    name: str,
    lat0: float = 0.0,
    lon0: float = 0.0,
    north: float = 0.0,
    south: float = 0.0,
    east: float = 0.0,
    west: float = 0.0,
) -> ET.Element:
    """
    Builds standard OpenDRIVE <header> tag with projection info.
    Note: In ASAM OpenDRIVE standard, north, south, east, west specify
    the spatial bounding box in inertial meters (y_max, y_min, x_max, x_min).
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    header = ET.SubElement(
        root,
        "header",
        {
            "revMajor": "1",
            "revMinor": "4",
            "name": name,
            "version": "1.0",
            "date": now_str,
            "north": f"{north:.4f}",
            "south": f"{south:.4f}",
            "east": f"{east:.4f}",
            "west": f"{west:.4f}",
            "vendor": "TwinX_ADAS_Vision",
        },
    )
    # Standard Proj4 transverse mercator string centered on lat0, lon0
    proj_str = (
        f"+proj=tmerc +lat_0={lat0:.6f} +lon_0={lon0:.6f} "
        f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    geo_ref = ET.SubElement(header, "geoReference")
    geo_ref.text = proj_str
    return header


def _add_plan_view_polyline(
    road_elem: ET.Element,
    points: List[Tuple[float, float, float]]
) -> float:
    """
    Constructs <planView> with piecewise <geometry> segments along reference line.
    Returns total road length in meters.
    """
    plan_view = ET.SubElement(road_elem, "planView")

    if len(points) < 2:
        geom = ET.SubElement(
            plan_view,
            "geometry",
            {"s": "0.0000", "x": "0.0000", "y": "0.0000", "hdg": "0.000000", "length": "100.0000"},
        )
        ET.SubElement(geom, "line")
        return 100.0

    current_s = 0.0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-4:
            continue
        hdg = math.atan2(dy, dx)

        geom = ET.SubElement(
            plan_view,
            "geometry",
            {
                "s": f"{current_s:.4f}",
                "x": f"{p1[0]:.4f}",
                "y": f"{p1[1]:.4f}",
                "hdg": f"{hdg:.6f}",
                "length": f"{seg_len:.4f}",
            },
        )
        ET.SubElement(geom, "line")
        current_s += seg_len

    if current_s < 1e-4:
        geom = ET.SubElement(
            plan_view,
            "geometry",
            {"s": "0.0000", "x": "0.0000", "y": "0.0000", "hdg": "0.000000", "length": "50.0000"},
        )
        ET.SubElement(geom, "line")
        current_s = 50.0

    return round(current_s, 4)


def _add_elevation_and_lateral_profiles(road_elem: ET.Element):
    """Adds standard flat elevation and lateral superelevation profiles."""
    elev_prof = ET.SubElement(road_elem, "elevationProfile")
    ET.SubElement(
        elev_prof,
        "elevation",
        {"s": "0.0", "a": "0.0", "b": "0.0", "c": "0.0", "d": "0.0"},
    )
    lat_prof = ET.SubElement(road_elem, "lateralProfile")
    ET.SubElement(
        lat_prof,
        "superelevation",
        {"s": "0.0", "a": "0.0", "b": "0.0", "c": "0.0", "d": "0.0"},
    )


def _add_lanes(
    road_elem: ET.Element,
    total_lanes: int,
    lane_width: float,
    is_oneway: bool = False,
    marking_type: str = "broken",
    has_shoulder: bool = False,
):
    """
    Constructs OpenDRIVE standard <lanes> block with <laneSection>,
    including left, center (id 0), and right lanes, width polynomials, and roadMarks.
    """
    lanes_elem = ET.SubElement(road_elem, "lanes")
    lane_sec = ET.SubElement(lanes_elem, "laneSection", {"s": "0.0000"})

    center_mark = marking_type
    outer_mark = "broken" if marking_type != "none" else "none"

    if is_oneway or total_lanes == 1:
        # All driving lanes positioned on the right (id: -1, -2, ...)
        center = ET.SubElement(lane_sec, "center")
        center_lane = ET.SubElement(center, "lane", {"id": "0", "type": "none", "level": "false"})
        ET.SubElement(
            center_lane,
            "roadMark",
            {"sOffset": "0.0", "type": center_mark, "weight": "standard", "color": "standard", "width": "0.15"},
        )

        right = ET.SubElement(lane_sec, "right")
        for i in range(1, total_lanes + 1):
            lane_id = -i
            r_lane = ET.SubElement(right, "lane", {"id": str(lane_id), "type": "driving", "level": "false"})
            ET.SubElement(
                r_lane,
                "width",
                {"sOffset": "0.0", "a": f"{lane_width:.2f}", "b": "0.0", "c": "0.0", "d": "0.0"},
            )
            mark = outer_mark if i < total_lanes else "solid"
            ET.SubElement(
                r_lane,
                "roadMark",
                {"sOffset": "0.0", "type": mark, "weight": "standard", "color": "standard", "width": "0.15"},
            )

        if has_shoulder:
            sh_id = -(total_lanes + 1)
            sh_lane = ET.SubElement(right, "lane", {"id": str(sh_id), "type": "shoulder", "level": "false"})
            ET.SubElement(
                sh_lane,
                "width",
                {"sOffset": "0.0", "a": "1.50", "b": "0.0", "c": "0.0", "d": "0.0"},
            )
            ET.SubElement(
                sh_lane,
                "roadMark",
                {"sOffset": "0.0", "type": "solid", "weight": "standard", "color": "standard", "width": "0.15"},
            )

    else:
        # Two-way bidirectional road: Left (positive IDs) and Right (negative IDs)
        num_left = max(1, total_lanes // 2)
        num_right = max(1, total_lanes - num_left)

        # Left lanes (outer to inner: num_left down to 1)
        left = ET.SubElement(lane_sec, "left")
        for i in range(num_left, 0, -1):
            l_lane = ET.SubElement(left, "lane", {"id": str(i), "type": "driving", "level": "false"})
            ET.SubElement(
                l_lane,
                "width",
                {"sOffset": "0.0", "a": f"{lane_width:.2f}", "b": "0.0", "c": "0.0", "d": "0.0"},
            )
            mark = outer_mark if i < num_left else "solid"
            ET.SubElement(
                l_lane,
                "roadMark",
                {"sOffset": "0.0", "type": mark, "weight": "standard", "color": "standard", "width": "0.15"},
            )

        # Center lane (id = 0)
        center = ET.SubElement(lane_sec, "center")
        center_lane = ET.SubElement(center, "lane", {"id": "0", "type": "none", "level": "false"})
        ET.SubElement(
            center_lane,
            "roadMark",
            {"sOffset": "0.0", "type": center_mark, "weight": "standard", "color": "standard", "width": "0.15"},
        )

        # Right lanes (id = -1, -2, ... from center outwards)
        right = ET.SubElement(lane_sec, "right")
        for i in range(1, num_right + 1):
            lane_id = -i
            r_lane = ET.SubElement(right, "lane", {"id": str(lane_id), "type": "driving", "level": "false"})
            ET.SubElement(
                r_lane,
                "width",
                {"sOffset": "0.0", "a": f"{lane_width:.2f}", "b": "0.0", "c": "0.0", "d": "0.0"},
            )
            mark = outer_mark if i < num_right else "solid"
            ET.SubElement(
                r_lane,
                "roadMark",
                {"sOffset": "0.0", "type": mark, "weight": "standard", "color": "standard", "width": "0.15"},
            )

        if has_shoulder:
            sh_id = -(num_right + 1)
            sh_lane = ET.SubElement(right, "lane", {"id": str(sh_id), "type": "shoulder", "level": "false"})
            ET.SubElement(
                sh_lane,
                "width",
                {"sOffset": "0.0", "a": "1.50", "b": "0.0", "c": "0.0", "d": "0.0"},
            )
            ET.SubElement(
                sh_lane,
                "roadMark",
                {"sOffset": "0.0", "type": "solid", "weight": "standard", "color": "standard", "width": "0.15"},
            )


def _add_objects(road_elem: ET.Element, actors: List[Actor]):
    """Embeds Scenario actors into OpenDRIVE <objects> block with typed mapping."""
    objects_elem = ET.SubElement(road_elem, "objects")
    for act in actors:
        spec = ACTOR_MATLAB_SPECS.get(act.actor_class, DEFAULT_ACTOR_SPEC)
        _, _, def_len, def_wid, def_hgt, _, _ = spec

        length = act.length_m if act.length_m is not None else def_len
        width = act.width_m if act.width_m is not None else def_wid
        height = def_hgt

        # In OpenDRIVE Frenet coordinates:
        # s is longitudinal distance along reference line (x),
        # t is lateral offset (y: positive left, negative right).
        s_coord = float(act.initial_x_m)
        t_coord = float(act.initial_y_m)

        # Heading angle from initial velocity if present
        hdg = 0.0
        if abs(act.initial_vx_mps) > 0.05 or abs(act.initial_vy_mps) > 0.05:
            hdg = math.atan2(act.initial_vy_mps, act.initial_vx_mps)

        obj_type = ACTOR_OPENDRIVE_TYPE_MAP.get(act.actor_class, "obstacle")

        ET.SubElement(
            objects_elem,
            "object",
            {
                "id": str(act.id),
                "name": str(act.actor_class),
                "type": obj_type,
                "s": f"{s_coord:.2f}",
                "t": f"{t_coord:.2f}",
                "zOffset": "0.00",
                "validLength": "0.00",
                "orientation": "none",
                "length": f"{length:.2f}",
                "width": f"{width:.2f}",
                "height": f"{height:.2f}",
                "hdg": f"{hdg:.6f}",
                "pitch": "0.000000",
                "roll": "0.000000",
            },
        )


# ---------------------------------------------------------------------------
# OpenDRIVE Exporter from Scenario
# ---------------------------------------------------------------------------
def _scenario_to_opendrive_xml(scenario: Scenario) -> ET.Element:
    """Builds OpenDRIVE XML ElementTree from a Scenario dataclass instance."""
    root = ET.Element("OpenDRIVE")

    sc_name = scenario.metadata.name if scenario.metadata and scenario.metadata.name else scenario.scenario_id
    road_len = scenario.road.length_m if scenario.road else 200.0
    lane_half_w = scenario.road.lane_half_width_m if scenario.road else 3.5
    road_type = scenario.road.type if scenario.road else "urban_street"
    markings = scenario.road.markings if scenario.road else "broken_center"
    has_shoulder = scenario.road.has_shoulder if scenario.road else False

    # Compute bounding box in inertial meters (X: East-West, Y: North-South)
    min_x = 0.0
    max_x = road_len
    min_y = -lane_half_w
    max_y = lane_half_w
    if scenario.actors:
        for act in scenario.actors:
            min_x = min(min_x, float(act.initial_x_m))
            max_x = max(max_x, float(act.initial_x_m))
            min_y = min(min_y, float(act.initial_y_m))
            max_y = max(max_y, float(act.initial_y_m))

    # Header with inertial meter bounds
    _create_header(
        root,
        name=f"Scenario_{scenario.scenario_id}",
        north=round(max_y + 2.0, 4),
        south=round(min_y - 2.0, 4),
        east=round(max_x + 5.0, 4),
        west=round(min_x - 5.0, 4),
    )

    # Road
    od_road_type = ROAD_TYPE_MAP.get(road_type, "town")
    speed_limit = int(scenario.ego_vehicle.initial_speed_kmh) if scenario.ego_vehicle else 60
    if speed_limit <= 0:
        speed_limit = 50

    road_elem = ET.SubElement(
        root,
        "road",
        {
            "name": sc_name,
            "length": f"{road_len:.4f}",
            "id": "1",
            "junction": "-1",
        },
    )

    type_elem = ET.SubElement(road_elem, "type", {"s": "0.0", "type": od_road_type})
    ET.SubElement(type_elem, "speed", {"max": str(speed_limit), "unit": "km/h"})

    # Straight road points: (0, 0, 0) to (road_len, 0, 0)
    points = [(0.0, 0.0, 0.0), (road_len, 0.0, 0.0)]
    _add_plan_view_polyline(road_elem, points)

    _add_elevation_and_lateral_profiles(road_elem)

    # Lanes
    num_lanes = 4 if road_type == "highway" else (1 if road_type == "village_road" and lane_half_w <= 2.5 else 2)
    single_lane_w = lane_half_w if num_lanes == 2 else round(2.0 * lane_half_w / num_lanes, 2)
    od_mark = ROAD_MARK_MAP.get(markings, "broken")

    _add_lanes(
        road_elem,
        total_lanes=num_lanes,
        lane_width=single_lane_w,
        is_oneway=False,
        marking_type=od_mark,
        has_shoulder=has_shoulder,
    )

    # Surface & Pavement Friction for RoadRunner Physics
    try:
        from scenario_lab.weather.sensor_degradation import calculate_sensor_degradation
        env_weather = getattr(scenario.environment, "weather", "clear") if scenario.environment else "clear"
        env_tod = getattr(scenario.environment, "time_of_day", "day") if scenario.environment else "day"
        deg = calculate_sensor_degradation(weather=env_weather, time_of_day=env_tod, surface=surface)
        friction_val = deg.friction_coefficient
    except Exception:
        friction_val = 0.85

    surface_elem = ET.SubElement(road_elem, "surface")
    ET.SubElement(surface_elem, "friction", {"s": "0.0", "friction": f"{friction_val:.2f}"})

    # Objects
    _add_objects(road_elem, scenario.actors or [])

    # Signals
    ET.SubElement(road_elem, "signals")

    return root


# ---------------------------------------------------------------------------
# OpenDRIVE Exporter from OSMRoadNetwork
# ---------------------------------------------------------------------------
def _osm_network_to_opendrive_xml(osm_network: Any) -> ET.Element:
    """Builds OpenDRIVE XML ElementTree from an OSMRoadNetwork instance."""
    if not hasattr(osm_network, "segments") or not osm_network.segments:
        raise ValueError("Cannot export empty OSMRoadNetwork: no road segments present.")

    root = ET.Element("OpenDRIVE")

    lat0 = float(osm_network.center_lat)
    lon0 = float(osm_network.center_lon)

    # Project all node coordinates to local inertial ENU meters to compute bounds
    all_x = []
    all_y = []
    for seg in osm_network.segments:
        if seg.nodes:
            for node in seg.nodes:
                x, y = _gps_to_enu(node.lat, node.lon, lat0, lon0)
                all_x.append(x)
                all_y.append(y)

    min_x = min(all_x) if all_x else -100.0
    max_x = max(all_x) if all_x else 100.0
    min_y = min(all_y) if all_y else -100.0
    max_y = max(all_y) if all_y else 100.0

    _create_header(
        root,
        name=f"OSM_Network_{int(abs(lat0*100))}_{int(abs(lon0*100))}",
        lat0=lat0,
        lon0=lon0,
        north=round(max_y + 10.0, 4),
        south=round(min_y - 10.0, 4),
        east=round(max_x + 10.0, 4),
        west=round(min_x - 10.0, 4),
    )

    valid_roads = 0
    for idx, seg in enumerate(osm_network.segments, 1):
        if not seg.nodes or len(seg.nodes) < 2:
            continue

        raw_centers: List[Tuple[float, float, float]] = []
        for node in seg.nodes:
            x, y = _gps_to_enu(node.lat, node.lon, lat0, lon0)
            raw_centers.append((round(x, 2), round(y, 2), 0.0))

        # Filter out points that are too close
        filtered_centers: List[Tuple[float, float, float]] = [raw_centers[0]]
        for pt in raw_centers[1:]:
            last_pt = filtered_centers[-1]
            if math.hypot(pt[0] - last_pt[0], pt[1] - last_pt[1]) >= 0.05:
                filtered_centers.append(pt)

        if len(filtered_centers) < 2:
            continue

        valid_roads += 1
        road_id = str(seg.osm_way_id if seg.osm_way_id else idx)
        road_name = seg.name or f"Road_{idx}"
        od_road_type = ROAD_TYPE_MAP.get(seg.indian_road_type, "town")

        road_elem = ET.SubElement(
            root,
            "road",
            {
                "name": road_name,
                "length": f"{seg.length_m:.4f}",
                "id": road_id,
                "junction": "-1",
            },
        )

        type_elem = ET.SubElement(road_elem, "type", {"s": "0.0", "type": od_road_type})
        ET.SubElement(type_elem, "speed", {"max": str(seg.speed_limit_kmh), "unit": "km/h"})

        # Reference line polyline
        actual_len = _add_plan_view_polyline(road_elem, filtered_centers)
        road_elem.set("length", f"{actual_len:.4f}")

        _add_elevation_and_lateral_profiles(road_elem)

        lanes = max(1, seg.lanes)
        lane_w = 3.5
        marking = "solid" if lanes >= 4 else ("broken" if lanes >= 2 else "none")

        _add_lanes(
            road_elem,
            total_lanes=lanes,
            lane_width=lane_w,
            is_oneway=seg.is_oneway,
            marking_type=marking,
            has_shoulder=(seg.indian_road_type == "highway"),
        )

        ET.SubElement(road_elem, "objects")
        ET.SubElement(road_elem, "signals")

    if valid_roads == 0:
        raise ValueError("Could not extract any valid road segments with at least 2 vertices.")

    return root


# ---------------------------------------------------------------------------
# Public Exporter Function
# ---------------------------------------------------------------------------
def export_to_opendrive(
    scenario_or_network: Union[Scenario, Any, Dict[str, Any], str],
    output_path: Optional[str] = None,
) -> str:
    """
    Exports a Scenario or OSMRoadNetwork object to standard ASAM OpenDRIVE XML (.xodr).

    Args:
        scenario_or_network: Scenario instance, OSMRoadNetwork instance, dict, or path to json/id.
        output_path: Optional filesystem path to write the .xodr file.

    Returns:
        Generated OpenDRIVE XML string.
    """
    target = scenario_or_network

    # Handle string filepath or ID
    if isinstance(target, str):
        if os.path.isfile(target):
            with open(target, "r", encoding="utf-8") as f:
                target = json.load(f)
        else:
            # Check templates or scenario DB
            storage_path = os.path.join(_SCENARIO_LAB, "storage", "db", "scenarios", f"{target}.json")
            if os.path.isfile(storage_path):
                with open(storage_path, "r", encoding="utf-8") as f:
                    target = json.load(f)
            else:
                templates_path = os.path.join(_SCENARIO_LAB, "templates", "indian_road_templates.json")
                if os.path.isfile(templates_path):
                    with open(templates_path, "r", encoding="utf-8") as f:
                        templates = json.load(f)
                        for t in templates:
                            if t.get("id") == target:
                                target = t["scenario"]
                                break

    # Convert dict to model if needed
    if isinstance(target, dict):
        if "segments" in target:
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
            target = OSMRoadNetwork(
                segments=segments,
                bbox=tuple(target.get("bbox", (0, 0, 0, 0))),
                center_lat=target.get("center", {}).get("lat", 0.0),
                center_lon=target.get("center", {}).get("lon", 0.0),
            )
        elif "scenario" in target:
            target = Scenario.model_validate(target["scenario"])
        else:
            target = Scenario.model_validate(target)

    # Dispatch to appropriate builder
    if hasattr(target, "segments"):
        root = _osm_network_to_opendrive_xml(target)
    elif isinstance(target, Scenario) or hasattr(target, "road"):
        root = _scenario_to_opendrive_xml(target)
    else:
        raise TypeError(f"Unsupported input type for export_to_opendrive: {type(target)}")

    # Pretty-print XML
    raw_xml = ET.tostring(root, encoding="utf-8")
    dom = minidom.parseString(raw_xml)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    # Ensure clean XML declaration
    lines = pretty_xml.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    pretty_xml = "\n".join(lines) + "\n"

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

    return pretty_xml


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RoadRunner / ASAM OpenDRIVE (.xodr) Exporter -- ADAS Scenario Lab"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=str, help="Scenario file path, ID, or template ID")
    group.add_argument("--osm-lat", type=float, help="Latitude for OSM road import")

    parser.add_argument("--osm-lon", type=float, help="Longitude for OSM road import")
    parser.add_argument("--osm-radius", type=float, default=500.0, help="Search radius in meters for OSM (default: 500)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .xodr file path")

    args = parser.parse_args()

    try:
        if args.scenario:
            out_file = args.output
            export_to_opendrive(args.scenario, out_file)
            print(f"[OK] Exported scenario '{args.scenario}' to OpenDRIVE XML: {out_file}")
        elif args.osm_lat is not None:
            if args.osm_lon is None:
                parser.error("--osm-lon is required when --osm-lat is specified.")
            print(f"[INFO] Fetching OpenStreetMap network at ({args.osm_lat}, {args.osm_lon}, r={args.osm_radius}m)...")
            osm_net = import_from_coordinates(args.osm_lat, args.osm_lon, args.osm_radius)
            out_file = args.output
            export_to_opendrive(osm_net, out_file)
            print(f"[OK] Exported {len(osm_net.segments)} OSM road segments to OpenDRIVE XML: {out_file}")

    except Exception as e:
        print(f"[ERROR] OpenDRIVE export failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
