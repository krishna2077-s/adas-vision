"""
opendrive_importer.py — ASAM OpenDRIVE (.xodr) Importer.

Parses ASAM OpenDRIVE XML road definitions (.xodr) into canonical ADAS Scenario
schema objects, enabling execution across MATLAB and Python simulation engines.
"""

import os
import re
import datetime
import xml.etree.ElementTree as ET
from typing import Optional, Union

from scenario_lab.schema.scenario_schema import (
    Scenario, Road, Environment, EgoVehicle, Actor, Metadata, SimulationParameters
)


CLASS_MAPPING = {
    "auto_rickshaw": "auto_rickshaw",
    "rickshaw": "auto_rickshaw",
    "cow": "cow",
    "cattle": "cow",
    "bull": "cow",
    "buffalo": "buffalo",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "scooter": "scooter",
    "bicycle": "bicycle",
    "barricade": "barricade",
    "barrier": "barrier",
    "police_barricade": "police_barricade",
    "pothole": "pothole",
    "pushcart": "pushcart",
    "thela": "thela",
    "pedestrian": "pedestrian",
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "obstacle": "obstacle",
}


def import_from_opendrive(xodr_input: Union[str, os.PathLike]) -> Scenario:
    """
    Parses OpenDRIVE .xodr XML file path or raw XML string into a Scenario object.
    """
    if os.path.exists(xodr_input):
        scenario_id_fallback = os.path.splitext(os.path.basename(xodr_input))[0]
        tree = ET.parse(xodr_input)
        root = tree.getroot()
    else:
        scenario_id_fallback = f"scen-xodr-{int(datetime.datetime.now().timestamp())}"
        root = ET.fromstring(xodr_input)

    if root.tag != "OpenDRIVE":
        raise ValueError(f"Root tag is '{root.tag}', expected 'OpenDRIVE'.")

    # 1. Header parsing
    header = root.find("header")
    name_attr = header.attrib.get("name", scenario_id_fallback) if header is not None else scenario_id_fallback
    clean_name = re.sub(r"^Scenario_", "", name_attr).strip()
    scenario_id = clean_name if clean_name.startswith("scen-") else scenario_id_fallback

    # 2. Road parsing
    road = root.find("road")
    if road is None:
        raise ValueError("No <road> element found in OpenDRIVE XML.")

    raw_road_name = road.attrib.get("name", "Imported OpenDRIVE Road")
    road_name = raw_road_name.strip(' "\'')
    road_length = float(road.attrib.get("length", 200.0))

    # Road type & speed limit
    road_type_elem = road.find("type")
    od_type = road_type_elem.attrib.get("type", "rural") if road_type_elem is not None else "rural"
    type_map = {
        "rural": "village_road",
        "motorway": "highway",
        "town": "urban_street",
        "pedestrian": "market_street",
        "unknown": "village_road",
    }
    schema_road_type = type_map.get(od_type, "village_road")

    speed_elem = road_type_elem.find("speed") if road_type_elem is not None else None
    speed_kmh = float(speed_elem.attrib.get("max", 45.0)) if speed_elem is not None else 45.0

    # Friction / surface
    surface_elem = road.find("surface")
    friction_val = 0.85
    if surface_elem is not None:
        f_elem = surface_elem.find("friction")
        if f_elem is not None:
            friction_val = float(f_elem.attrib.get("friction", 0.85))

    if "broken" in road_name.lower() or "rough" in road_name.lower() or "tarmac" in road_name.lower():
        surface_type = "broken_tarmac"
    elif friction_val < 0.6:
        surface_type = "dirt"
    else:
        surface_type = "tarmac"

    # Lanes
    lane_hw = 3.5
    has_shoulder = False
    markings = "none" if schema_road_type == "village_road" else "broken_center"

    lanes_elem = road.find("lanes")
    if lanes_elem is not None:
        for lane in lanes_elem.iter("lane"):
            l_type = lane.attrib.get("type", "")
            if l_type == "shoulder":
                has_shoulder = True
            width_elem = lane.find("width")
            if width_elem is not None:
                w_val = float(width_elem.attrib.get("a", 3.5))
                if w_val >= 2.0:
                    lane_hw = w_val
            rm_elem = lane.find("roadMark")
            if rm_elem is not None:
                rm_type = rm_elem.attrib.get("type", "none")
                if rm_type == "solid":
                    markings = "full"
                elif rm_type == "broken":
                    markings = "broken_center"

    # 3. Objects / Dynamic & Static Actors
    actors = []
    objects_elem = road.find("objects")
    if objects_elem is not None:
        for idx, obj in enumerate(objects_elem.findall("object")):
            obj_id = obj.attrib.get("id", f"actor_{idx+1}")
            obj_name = obj.attrib.get("name", "obstacle").lower()
            obj_type = obj.attrib.get("type", "").lower()

            # Map to canonical class
            actor_cls = CLASS_MAPPING.get(obj_name, CLASS_MAPPING.get(obj_type, "obstacle"))

            s_pos = float(obj.attrib.get("s", 30.0 + idx * 25.0))
            t_pos = float(obj.attrib.get("t", 0.0))
            length_m = float(obj.attrib.get("length", 2.0))
            width_m = float(obj.attrib.get("width", 1.0))
            hdg = float(obj.attrib.get("hdg", 0.0))

            # Behavior & velocity inference
            if actor_cls in ("barricade", "barrier", "police_barricade", "pothole"):
                beh = "static"
                vx = 0.0
                vy = 0.0
                trig_s = 0.0
                y_tgt = None
            elif actor_cls in ("cow", "buffalo"):
                beh = "crossing"
                vx = 0.2
                vy = 0.9 if t_pos < 0 else -0.9
                trig_s = 1.0
                y_tgt = 2.0 if t_pos < 0 else -2.0
            elif abs(hdg - 3.14159) < 0.5:
                # Facing opposite direction (oncoming)
                beh = "cut_in" if actor_cls == "auto_rickshaw" else "wrong_side"
                vx = -6.0
                vy = -1.2 if t_pos > 0 else 1.2
                trig_s = 1.2
                y_tgt = 0.0
            else:
                beh = "constant_velocity"
                vx = min(12.0, (speed_kmh / 3.6) * 0.45)
                vy = 0.0
                trig_s = 0.0
                y_tgt = None

            actor = Actor(
                id=obj_id,
                **{"class": actor_cls},
                initial_x_m=s_pos,
                initial_y_m=t_pos,
                initial_vx_mps=round(vx, 2),
                initial_vy_mps=round(vy, 2),
                width_m=width_m,
                length_m=length_m,
                behaviour=beh,
                behaviour_trigger_s=trig_s,
                behaviour_y_target_m=y_tgt
            )
            actors.append(actor)

    if not actors:
        # Default placeholder actor if road has no objects
        actors.append(Actor(
            id="obs_1",
            **{"class": "auto_rickshaw"},
            initial_x_m=45.0,
            initial_y_m=0.0,
            initial_vx_mps=0.0,
            initial_vy_mps=0.0,
            behaviour="static"
        ))

    # 4. Ego Vehicle Configuration
    ego = EgoVehicle(
        initial_speed_kmh=speed_kmh,
        initial_x_m=0.0,
        initial_y_m=0.0,
        initial_heading_deg=0.0,
        goal_x_m=round(road_length * 0.9, 1),
        goal_y_m=0.0,
        goal_tolerance_m=4.0,
        mode="adaptive"
    )

    # 5. Metadata & Environment
    meta = Metadata(
        name=road_name.title() if len(road_name) > 3 else "ASAM OpenDRIVE Imported Scenario",
        description=f"Imported from ASAM OpenDRIVE XML (.xodr): {scenario_id}",
        source="builder",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tags=["xodr", "opendrive", schema_road_type, surface_type] + [a.actor_class for a in actors]
    )

    env = Environment(
        weather="clear",
        time_of_day="day",
        visibility_m=300.0
    )

    return Scenario(
        scenario_id=scenario_id,
        metadata=meta,
        road=Road(
            type=schema_road_type,
            lane_half_width_m=lane_hw,
            length_m=road_length,
            markings=markings,
            surface=surface_type,
            has_shoulder=has_shoulder
        ),
        environment=env,
        ego_vehicle=ego,
        actors=actors,
        simulation_parameters=SimulationParameters(
            dt_s=0.033,
            max_time_s=40.0
        )
    )
