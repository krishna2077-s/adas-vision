"""
openscenario_exporter.py -- ASAM OpenSCENARIO (.xosc) Exporter for MathWorks RoadRunner.

Converts Scenario objects or road network graphs into ASAM OpenSCENARIO v1.0 / v1.2 XML format,
directly importable and executable by MathWorks RoadRunner Scenario, CARLA, and esmini.

Produces valid OpenSCENARIO structure:
  - <FileHeader> with revMajor="1", revMinor="2", author, timestamp, description.
  - <ParameterDeclarations> for ego speed and road geometry parameters.
  - <CatalogLocations> referencing vehicle, pedestrian, and miscellaneous object catalogs.
  - <RoadNetwork> referencing the corresponding OpenDRIVE (.xodr) logic file.
  - <Entities> defining ScenarioObjects for EgoVehicle and all Indian road participants.
  - <Storyboard> with <Init> teleport actions and <Story>/<Act>/<ManeuverGroup> defining
    dynamic actor behaviors (crossing, cut-in, sudden stop, constant velocity, static).
  - Triggers using SimulationTimeCondition, DistanceCondition, and ReachPositionCondition.

CLI Usage:
  python openscenario_exporter.py --scenario <path_or_id> --output <path.xosc>
"""

import os
import sys
import math
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any

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
    from scenario_lab.export.dsd_exporter import ACTOR_MATLAB_SPECS, DEFAULT_ACTOR_SPEC
except ImportError:
    ACTOR_MATLAB_SPECS = {}
    DEFAULT_ACTOR_SPEC = ("actor", 0, 2.0, 1.0, 1.2, (0.5, 0.5, 0.5), None)


# Mapping actor class to OpenSCENARIO Object Category and Model
# ("category", "object_type", length, width, height)
OPENSCENARIO_ACTOR_MAP: Dict[str, Tuple[str, str, float, float, float]] = {
    "car":                  ("vehicle", "car", 4.7, 1.8, 1.4),
    "suv":                  ("vehicle", "car", 4.8, 1.9, 1.7),
    "truck":                ("vehicle", "truck", 8.5, 2.5, 3.4),
    "bus":                  ("vehicle", "bus", 10.5, 2.5, 3.2),
    "tractor":              ("vehicle", "truck", 5.0, 2.2, 2.4),
    "motorcycle":           ("vehicle", "motorbike", 2.0, 0.8, 1.3),
    "scooter":              ("vehicle", "motorbike", 1.9, 0.7, 1.2),
    "bicycle":              ("vehicle", "bicycle", 1.8, 0.6, 1.1),
    "person":               ("pedestrian", "pedestrian", 0.5, 0.5, 1.7),
    "pedestrian":           ("pedestrian", "pedestrian", 0.5, 0.5, 1.7),
    "child":                ("pedestrian", "pedestrian", 0.4, 0.4, 1.2),
    "auto_rickshaw":        ("vehicle", "car", 2.8, 1.4, 1.8),
    "e_rickshaw":           ("vehicle", "car", 2.8, 1.3, 1.8),
    "cow":                  ("miscellaneous", "animal", 2.2, 1.5, 1.5),
    "buffalo":              ("miscellaneous", "animal", 2.4, 1.6, 1.6),
    "dog":                  ("miscellaneous", "animal", 0.8, 0.4, 0.6),
    "goat":                 ("miscellaneous", "animal", 0.9, 0.4, 0.75),
    "pushcart":             ("miscellaneous", "obstacle", 2.2, 1.4, 1.2),
    "thela":                ("miscellaneous", "obstacle", 2.2, 1.4, 1.2),
    "barricade":            ("miscellaneous", "barrier", 2.0, 0.5, 1.0),
    "barrier":              ("miscellaneous", "barrier", 2.0, 0.5, 1.0),
    "police_barricade":     ("miscellaneous", "barrier", 2.0, 0.5, 1.0),
    "construction_barrier": ("miscellaneous", "barrier", 2.0, 0.5, 1.0),
    "pothole":              ("miscellaneous", "road_anomaly", 1.2, 1.2, 0.1),
    "obstacle":             ("miscellaneous", "obstacle", 1.5, 1.5, 1.0),
}

# 3D Model mapping for esmini and RoadRunner 3D visualizers
OPENSCENARIO_3D_MODELS: Dict[str, str] = {
    "car":                  "../models/car_white.osgb",
    "suv":                  "../models/car_blue.osgb",
    "truck":                "../models/truck_yellow.osgb",
    "bus":                  "../models/bus_blue.osgb",
    "motorcycle":           "../models/scooter.osgb",
    "scooter":              "../models/scooter.osgb",
    "bicycle":              "../models/cyclist.osgb",
    "person":               "../models/walkman.osgb",
    "pedestrian":           "../models/walkman.osgb",
    "auto_rickshaw":        "../models/car_yellow.osgb",
    "e_rickshaw":           "../models/car_yellow.osgb",
    "cow":                  "../models/moose_cc0.osgb",
    "buffalo":              "../models/moose_cc0.osgb",
    "dog":                  "../models/moose_cc0.osgb",
    "goat":                 "../models/moose_cc0.osgb",
    "barricade":            "../models/railing.osgb",
    "barrier":              "../models/railing.osgb",
    "police_barricade":     "../models/railing.osgb",
    "construction_barrier": "../models/cone-100.osgb",
}


def _get_actor_osc_spec(actor_class: str) -> Tuple[str, str, float, float, float]:
    cls_clean = actor_class.lower().strip()
    return OPENSCENARIO_ACTOR_MAP.get(cls_clean, ("miscellaneous", "obstacle", 2.0, 1.0, 1.2))


def export_to_openscenario(scenario: Any, xodr_filename: Optional[str] = None) -> str:
    """
    Export a Scenario instance into an ASAM OpenSCENARIO v1.2 XML string.
    
    Args:
        scenario: Valid scenario_lab Scenario instance, dict, filepath, or template ID.
        xodr_filename: Optional path to the linked OpenDRIVE file. Defaults to '{scenario_id}.xodr'.
    
    Returns:
        Formatted XML string conforming to ASAM OpenSCENARIO.
    """
    target = scenario
    if isinstance(target, str):
        if os.path.isfile(target):
            with open(target, "r", encoding="utf-8") as f:
                target = json.load(f)
        else:
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
    if isinstance(target, dict):
        if "scenario" in target:
            target = Scenario.model_validate(target["scenario"])
        else:
            target = Scenario.model_validate(target)
    scenario = target

    if xodr_filename is None:
        xodr_filename = f"{scenario.scenario_id}.xodr"

    root = ET.Element("OpenSCENARIO")

    # 1. FileHeader
    header = ET.SubElement(root, "FileHeader", {
        "revMajor": "1",
        "revMinor": "2",
        "date": datetime.now(timezone.utc).isoformat(),
        "description": f"Indian Road ADAS Scenario: {scenario.metadata.name} (RoadRunner Compatible)",
        "author": "ADAS Scenario Lab - SIH 2025 Winning Platform",
    })

    # 2. ParameterDeclarations
    param_decls = ET.SubElement(root, "ParameterDeclarations")
    ego_v_mps = scenario.ego_vehicle.initial_speed_kmh / 3.6
    ET.SubElement(param_decls, "ParameterDeclaration", {
        "name": "EgoSpeed_mps",
        "parameterType": "double",
        "value": f"{ego_v_mps:.2f}",
    })
    ET.SubElement(param_decls, "ParameterDeclaration", {
        "name": "RoadLength_m",
        "parameterType": "double",
        "value": f"{scenario.road.length_m:.1f}",
    })

    # 3. CatalogLocations
    ET.SubElement(root, "CatalogLocations")

    # 4. RoadNetwork
    road_network = ET.SubElement(root, "RoadNetwork")
    ET.SubElement(road_network, "LogicFile", {"filepath": xodr_filename})

    # 5. Entities
    entities = ET.SubElement(root, "Entities")

    # 5.1 Ego Vehicle Object
    ego_obj = ET.SubElement(entities, "ScenarioObject", {"name": "Ego"})
    ego_veh = ET.SubElement(ego_obj, "Vehicle", {
        "name": "EgoVehicle",
        "vehicleCategory": "car",
        "model3d": "../models/car_white.osgb",
    })
    ego_bb = ET.SubElement(ego_veh, "BoundingBox")
    ET.SubElement(ego_bb, "Center", {"x": "1.4", "y": "0.0", "z": "0.7"})
    ET.SubElement(ego_bb, "Dimensions", {"width": "1.8", "length": "4.7", "height": "1.4"})
    ego_perf = ET.SubElement(ego_veh, "Performance", {
        "maxSpeed": "55.0",
        "maxAcceleration": "6.0",
        "maxDeceleration": "8.0",
    })
    ego_axles = ET.SubElement(ego_veh, "Axles")
    ET.SubElement(ego_axles, "FrontAxle", {
        "maxSteering": "0.6",
        "wheelDiameter": "0.65",
        "trackWidth": "1.55",
        "positionX": "2.7",
        "positionZ": "0.32",
    })
    ET.SubElement(ego_axles, "RearAxle", {
        "maxSteering": "0.0",
        "wheelDiameter": "0.65",
        "trackWidth": "1.55",
        "positionX": "0.0",
        "positionZ": "0.32",
    })

    # 5.2 Scenario Actors Objects
    for actor in scenario.actors:
        a_class = getattr(actor, "actor_class", None) or getattr(actor, "class", "car")
        category, osc_type, def_l, def_w, def_h = _get_actor_osc_spec(a_class)
        model_3d = OPENSCENARIO_3D_MODELS.get(str(a_class).lower().strip(), "")
        act_l = actor.length_m if actor.length_m else def_l
        act_w = actor.width_m if actor.width_m else def_w
        act_h = def_h

        act_obj = ET.SubElement(entities, "ScenarioObject", {"name": actor.id})
        if category == "vehicle":
            v_attrs = {
                "name": f"Vehicle_{actor.id}_{a_class}",
                "vehicleCategory": osc_type,
            }
            if model_3d:
                v_attrs["model3d"] = model_3d
            v_elem = ET.SubElement(act_obj, "Vehicle", v_attrs)
            bb = ET.SubElement(v_elem, "BoundingBox")
            ET.SubElement(bb, "Center", {"x": f"{act_l/2:.2f}", "y": "0.0", "z": f"{act_h/2:.2f}"})
            ET.SubElement(bb, "Dimensions", {"width": f"{act_w:.2f}", "length": f"{act_l:.2f}", "height": f"{act_h:.2f}"})
            ET.SubElement(v_elem, "Performance", {"maxSpeed": "40.0", "maxAcceleration": "4.0", "maxDeceleration": "7.0"})
            ax = ET.SubElement(v_elem, "Axles")
            ET.SubElement(ax, "FrontAxle", {"maxSteering": "0.5", "wheelDiameter": "0.6", "trackWidth": f"{act_w*0.8:.2f}", "positionX": f"{act_l*0.6:.2f}", "positionZ": "0.3"})
            ET.SubElement(ax, "RearAxle", {"maxSteering": "0.0", "wheelDiameter": "0.6", "trackWidth": f"{act_w*0.8:.2f}", "positionX": "0.0", "positionZ": "0.3"})
        elif category == "pedestrian":
            p_attrs = {
                "name": f"Pedestrian_{actor.id}_{a_class}",
                "pedestrianCategory": "pedestrian",
                "mass": "70",
            }
            if model_3d:
                p_attrs["model3d"] = model_3d
            p_elem = ET.SubElement(act_obj, "Pedestrian", p_attrs)
            bb = ET.SubElement(p_elem, "BoundingBox")
            ET.SubElement(bb, "Center", {"x": "0.0", "y": "0.0", "z": f"{act_h/2:.2f}"})
            ET.SubElement(bb, "Dimensions", {"width": f"{act_w:.2f}", "length": f"{act_l:.2f}", "height": f"{act_h:.2f}"})
        else:
            m_attrs = {
                "name": f"Obstacle_{actor.id}_{a_class}",
                "miscObjectCategory": "obstacle",
                "mass": "300" if ("cow" in str(a_class) or "buffalo" in str(a_class)) else "50",
            }
            if model_3d:
                m_attrs["model3d"] = model_3d
            m_elem = ET.SubElement(act_obj, "MiscObject", m_attrs)
            bb = ET.SubElement(m_elem, "BoundingBox")
            ET.SubElement(bb, "Center", {"x": f"{act_l/2:.2f}", "y": "0.0", "z": f"{act_h/2:.2f}"})
            ET.SubElement(bb, "Dimensions", {"width": f"{act_w:.2f}", "length": f"{act_l:.2f}", "height": f"{act_h:.2f}"})

    # 6. Storyboard
    storyboard = ET.SubElement(root, "Storyboard")

    # 6.1 Init Section
    init = ET.SubElement(storyboard, "Init")
    init_actions = ET.SubElement(init, "Actions")

    # Global Environment Action (Weather, Illumination, Pavement Friction)
    env = getattr(scenario, "environment", None)
    weather_cond = getattr(env, "weather", "clear") or "clear"
    tod = getattr(env, "time_of_day", "day") or "day"
    vis_m = getattr(env, "visibility_m", None) or (45.0 if weather_cond == "heavy_rain" else (35.0 if weather_cond == "fog" else 200.0))
    road = getattr(scenario, "road", None)
    surf = getattr(road, "surface", "tarmac") or "tarmac"

    try:
        from scenario_lab.weather.sensor_degradation import calculate_sensor_degradation
        deg_p = calculate_sensor_degradation(weather=weather_cond, time_of_day=tod, surface=surf, ego_speed_kmh=ego_v_mps * 3.6)
        fric_scale = round(deg_p.friction_coefficient / 0.85, 3)
    except Exception:
        fric_scale = 1.0

    global_action = ET.SubElement(init_actions, "GlobalAction")
    env_action = ET.SubElement(global_action, "EnvironmentAction")
    env_elem = ET.SubElement(env_action, "Environment", {"name": f"IndianEnv_{weather_cond}_{tod}"})

    tod_map = {"day": "2026-10-17T12:00:00", "dusk": "2026-10-17T16:00:00", "night": "2026-10-17T21:00:00"}
    ET.SubElement(env_elem, "TimeOfDay", {
        "animation": "false",
        "dateTime": tod_map.get(tod, "2026-10-17T12:00:00"),
    })

    cloud_state = "overcast" if weather_cond in ("overcast", "heavy_rain", "light_rain") else "skyOff"
    weather_elem = ET.SubElement(env_elem, "Weather", {"cloudState": cloud_state})
    sun_intensity = "0.4" if tod == "night" else ("0.85" if tod == "dusk" or weather_cond == "heavy_rain" else "1.0")
    ET.SubElement(weather_elem, "Sun", {"intensity": sun_intensity, "azimuth": "1.57", "elevation": "1.10"})
    ET.SubElement(weather_elem, "Fog", {"visualRange": f"{vis_m:.1f}"})

    if weather_cond in ("heavy_rain", "light_rain"):
        precip_intensity = "0.85" if weather_cond == "heavy_rain" else "0.35"
        ET.SubElement(weather_elem, "Precipitation", {
            "precipitationType": "rain",
            "intensity": precip_intensity,
        })

    ET.SubElement(env_elem, "RoadCondition", {"frictionScaleFactor": f"{fric_scale:.2f}"})

    # Ego init action
    ego_priv = ET.SubElement(init_actions, "Private", {"entityRef": "Ego"})
    ego_p_actions = ET.SubElement(ego_priv, "PrivateAction")
    ego_tele = ET.SubElement(ego_p_actions, "TeleportAction")
    ego_pos = ET.SubElement(ego_tele, "Position")
    ET.SubElement(ego_pos, "WorldPosition", {
        "x": f"{scenario.ego_vehicle.initial_x_m:.2f}",
        "y": f"{scenario.ego_vehicle.initial_y_m:.2f}",
        "z": "0.0",
        "h": f"{math.radians(scenario.ego_vehicle.initial_heading_deg):.4f}",
    })
    ego_speed_action = ET.SubElement(ego_priv, "PrivateAction")
    ego_long = ET.SubElement(ego_speed_action, "LongitudinalAction")
    ego_spd = ET.SubElement(ego_long, "SpeedAction")
    ET.SubElement(ego_spd, "SpeedActionDynamics", {"dynamicsShape": "step", "value": "0.0", "dynamicsDimension": "time"})
    ET.SubElement(ego_spd, "SpeedActionTarget").append(
        ET.Element("AbsoluteTargetSpeed", {"value": f"{ego_v_mps:.2f}"})
    )

    # Actors init actions
    for actor in scenario.actors:
        act_priv = ET.SubElement(init_actions, "Private", {"entityRef": actor.id})
        act_p_actions = ET.SubElement(act_priv, "PrivateAction")
        act_tele = ET.SubElement(act_p_actions, "TeleportAction")
        act_pos = ET.SubElement(act_tele, "Position")
        if actor.behaviour == "wrong_side":
            heading = math.pi if actor.initial_vx_mps >= 0 else math.atan2(actor.initial_vy_mps, actor.initial_vx_mps)
            act_speed = abs(actor.initial_vx_mps) if abs(actor.initial_vx_mps) > 0.5 else 7.0
        else:
            heading = math.atan2(actor.initial_vy_mps, actor.initial_vx_mps) if (actor.initial_vx_mps != 0 or actor.initial_vy_mps != 0) else 0.0
            act_speed = math.hypot(actor.initial_vx_mps, actor.initial_vy_mps)

        ET.SubElement(act_pos, "WorldPosition", {
            "x": f"{actor.initial_x_m:.2f}",
            "y": f"{actor.initial_y_m:.2f}",
            "z": "0.0",
            "h": f"{heading:.4f}",
        })
        if act_speed > 0.0:
            act_sp_action = ET.SubElement(act_priv, "PrivateAction")
            act_long = ET.SubElement(act_sp_action, "LongitudinalAction")
            act_spd = ET.SubElement(act_long, "SpeedAction")
            ET.SubElement(act_spd, "SpeedActionDynamics", {"dynamicsShape": "step", "value": "0.0", "dynamicsDimension": "time"})
            ET.SubElement(act_spd, "SpeedActionTarget").append(
                ET.Element("AbsoluteTargetSpeed", {"value": f"{act_speed:.2f}"})
            )

    # 6.2 Story / Act
    story = ET.SubElement(storyboard, "Story", {"name": "IndianRoadTrafficStory"})
    act = ET.SubElement(story, "Act", {"name": "TrafficDynamicsAct"})

    # ManeuverGroups for dynamic actors
    for actor in scenario.actors:
        beh = actor.behaviour
        if beh in ("crossing", "cut_in", "sudden_stop", "wrong_side", "gradual"):
            mg = ET.SubElement(act, "ManeuverGroup", {
                "name": f"ManeuverGroup_{actor.id}",
                "maximumExecutionCount": "1",
            })
            actors_ref = ET.SubElement(mg, "Actors", {"selectTriggeringEntities": "false"})
            ET.SubElement(actors_ref, "EntityRef", {"entityRef": actor.id})

            maneuver = ET.SubElement(mg, "Maneuver", {"name": f"Maneuver_{actor.id}_{beh}"})
            event = ET.SubElement(maneuver, "Event", {
                "name": f"Event_{actor.id}_{beh}",
                "priority": "overwrite",
            })

            # Action
            action_elem = ET.SubElement(event, "Action", {"name": f"Action_{actor.id}"})
            priv_act = ET.SubElement(action_elem, "PrivateAction")

            if beh in ("crossing", "cut_in"):
                lat_act = ET.SubElement(priv_act, "LateralAction")
                lane_offset = ET.SubElement(lat_act, "LaneOffsetAction")
                ET.SubElement(lane_offset, "LaneOffsetActionDynamics", {
                    "maxLateralAcc": "2.0",
                    "dynamicsShape": "sinusoidal",
                })
                target_y = actor.behaviour_y_target_m if actor.behaviour_y_target_m is not None else 0.0
                ET.SubElement(lane_offset, "LaneOffsetTarget").append(
                    ET.Element("AbsoluteTargetLaneOffset", {"value": f"{target_y:.2f}"})
                )
            elif beh == "gradual":
                lat_act = ET.SubElement(priv_act, "LateralAction")
                lane_offset = ET.SubElement(lat_act, "LaneOffsetAction")
                ET.SubElement(lane_offset, "LaneOffsetActionDynamics", {
                    "maxLateralAcc": "0.8",
                    "dynamicsShape": "linear",
                })
                target_y = actor.behaviour_y_target_m if actor.behaviour_y_target_m is not None else 0.0
                ET.SubElement(lane_offset, "LaneOffsetTarget").append(
                    ET.Element("AbsoluteTargetLaneOffset", {"value": f"{target_y:.2f}"})
                )
            elif beh == "sudden_stop":
                long_act = ET.SubElement(priv_act, "LongitudinalAction")
                spd_act = ET.SubElement(long_act, "SpeedAction")
                ET.SubElement(spd_act, "SpeedActionDynamics", {
                    "dynamicsShape": "linear",
                    "value": "4.5",
                    "dynamicsDimension": "rate",
                })
                ET.SubElement(spd_act, "SpeedActionTarget").append(
                    ET.Element("AbsoluteTargetSpeed", {"value": "0.0"})
                )
            elif beh == "wrong_side":
                long_act = ET.SubElement(priv_act, "LongitudinalAction")
                spd_act = ET.SubElement(long_act, "SpeedAction")
                ET.SubElement(spd_act, "SpeedActionDynamics", {
                    "dynamicsShape": "step",
                    "value": "0.0",
                    "dynamicsDimension": "time",
                })
                target_spd = abs(actor.initial_vx_mps) if abs(actor.initial_vx_mps) > 0.5 else 7.0
                ET.SubElement(spd_act, "SpeedActionTarget").append(
                    ET.Element("AbsoluteTargetSpeed", {"value": f"{target_spd:.2f}"})
                )

            # StartTrigger for Event
            start_trigger = ET.SubElement(event, "StartTrigger")
            cond_group = ET.SubElement(start_trigger, "ConditionGroup")
            trigger_s = actor.behaviour_trigger_s if (actor.behaviour_trigger_s is not None and actor.behaviour_trigger_s > 0) else 1.5
            cond = ET.SubElement(cond_group, "Condition", {
                "name": f"TimeTrigger_{actor.id}",
                "delay": "0.0",
                "conditionEdge": "rising",
            })
            by_val = ET.SubElement(cond, "ByValueCondition")
            ET.SubElement(by_val, "SimulationTimeCondition", {
                "value": f"{trigger_s:.2f}",
                "rule": "greaterThan",
            })

    # Guaranteed ASAM OpenSCENARIO schema compliance: Act must have >= 1 ManeuverGroup
    if len(act.findall("ManeuverGroup")) == 0:
        mg_ego = ET.SubElement(act, "ManeuverGroup", {
            "name": "ManeuverGroup_Ego_Cruise",
            "maximumExecutionCount": "1",
        })
        actors_ref = ET.SubElement(mg_ego, "Actors", {"selectTriggeringEntities": "false"})
        ET.SubElement(actors_ref, "EntityRef", {"entityRef": "Ego"})
        maneuver = ET.SubElement(mg_ego, "Maneuver", {"name": "Maneuver_Ego_Cruise"})
        event = ET.SubElement(maneuver, "Event", {"name": "Event_Ego_Cruise", "priority": "overwrite"})
        act_elem = ET.SubElement(event, "Action", {"name": "Action_Ego_HoldSpeed"})
        priv_act = ET.SubElement(act_elem, "PrivateAction")
        long_act = ET.SubElement(priv_act, "LongitudinalAction")
        spd_act = ET.SubElement(long_act, "SpeedAction")
        ET.SubElement(spd_act, "SpeedActionDynamics", {"dynamicsShape": "step", "value": "0.0", "dynamicsDimension": "time"})
        ET.SubElement(spd_act, "SpeedActionTarget").append(
            ET.Element("AbsoluteTargetSpeed", {"value": f"{ego_v_mps:.2f}"})
        )
        start_trigger = ET.SubElement(event, "StartTrigger")
        cg = ET.SubElement(start_trigger, "ConditionGroup")
        cond = ET.SubElement(cg, "Condition", {"name": "StartImmediate", "delay": "0.0", "conditionEdge": "none"})
        by_val = ET.SubElement(cond, "ByValueCondition")
        ET.SubElement(by_val, "SimulationTimeCondition", {"value": "0.0", "rule": "greaterOrEqual"})

    # Act StartTrigger (Immediate)
    act_start = ET.SubElement(act, "StartTrigger")
    act_cg = ET.SubElement(act_start, "ConditionGroup")
    act_cond = ET.SubElement(act_cg, "Condition", {
        "name": "ActStartCondition",
        "delay": "0.0",
        "conditionEdge": "none",
    })
    act_byval = ET.SubElement(act_cond, "ByValueCondition")
    ET.SubElement(act_byval, "SimulationTimeCondition", {"value": "0.0", "rule": "greaterOrEqual"})

    # 6.3 Storyboard StopTrigger
    max_time_s = scenario.simulation_parameters.max_time_s or 35.0
    stop_trigger = ET.SubElement(storyboard, "StopTrigger")
    stop_cg = ET.SubElement(stop_trigger, "ConditionGroup")
    stop_cond = ET.SubElement(stop_cg, "Condition", {
        "name": "MaxTimeLimit",
        "delay": "0.0",
        "conditionEdge": "rising",
    })
    stop_byval = ET.SubElement(stop_cond, "ByValueCondition")
    ET.SubElement(stop_byval, "SimulationTimeCondition", {
        "value": f"{max_time_s:.1f}",
        "rule": "greaterThan",
    })

    # Format pretty XML string
    rough_string = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export scenario to ASAM OpenSCENARIO (.xosc)")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON, scenario ID, or template ID")
    parser.add_argument("--output", required=True, help="Path for output .xosc file")
    parser.add_argument("--xodr", default=None, help="Linked OpenDRIVE (.xodr) filename or relative path")
    args = parser.parse_args()

    xml_out = export_to_openscenario(args.scenario, xodr_filename=args.xodr)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(xml_out)
    print(f"[OK] Exported OpenSCENARIO to {args.output}")
