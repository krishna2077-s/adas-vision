"""
test_twinx_features.py -- Comprehensive test suite for TwinX SIH 2025 Winning Enhancements:
  - ASAM OpenSCENARIO (.xosc) Exporter for MathWorks RoadRunner
  - Physics-Informed Sensor Perception Degradation & Pavement Friction Model
  - Head-to-Head Comparative Benchmark Engine (Adaptive vs Baseline ADAS)
  - New API endpoints: /export/openscenario, /benchmark, /sensor-degradation
"""

import os
import json
import xml.etree.ElementTree as ET
import pytest
from fastapi.testclient import TestClient

from scenario_lab.schema.scenario_schema import (
    Scenario, Road, Environment, EgoVehicle, Actor, Metadata, SimulationParameters
)
from scenario_lab.export.openscenario_exporter import export_to_openscenario
from scenario_lab.weather.sensor_degradation import (
    calculate_sensor_degradation, SensorDegradationProfile
)
from scenario_lab.stress.benchmark_engine import BenchmarkEngine
from scenario_lab.api.main import app

client = TestClient(app)


@pytest.fixture
def sample_indian_scenario():
    """Provides a realistic Indian road scenario with cattle and auto-rickshaw."""
    return Scenario(
        scenario_id="test_twinx_nashik",
        metadata=Metadata(
            name="Nashik Ring Road TwinX Test",
            description="Testing bovine crossing and auto-rickshaw cut-in",
            source="builder",
            created_at="2026-10-17T00:00:00Z",
            tags=["twinx", "cattle", "auto-rickshaw"]
        ),
        road=Road(
            type="village_road",
            lane_half_width_m=3.5,
            length_m=200.0,
            markings="none",
            surface="broken_tarmac",
            has_shoulder=True
        ),
        environment=Environment(
            time_of_day="dusk",
            weather="clear",
            visibility_m=120.0
        ),
        ego_vehicle=EgoVehicle(
            initial_speed_kmh=40.0,
            initial_x_m=0.0,
            initial_y_m=0.0,
            goal_x_m=180.0,
            mode="adaptive"
        ),
        actors=[
            Actor(
                id="desi_cow",
                **{"class": "cow"},
                initial_x_m=60.0,
                initial_y_m=-3.0,
                initial_vx_mps=0.1,
                initial_vy_mps=0.8,
                behaviour="crossing",
                behaviour_trigger_s=1.5,
                behaviour_y_target_m=2.0
            ),
            Actor(
                id="auto_rickshaw_1",
                **{"class": "auto_rickshaw"},
                initial_x_m=40.0,
                initial_y_m=3.0,
                initial_vx_mps=6.0,
                initial_vy_mps=-1.2,
                behaviour="cut_in",
                behaviour_trigger_s=1.0,
                behaviour_y_target_m=0.0
            )
        ],
        simulation_parameters=SimulationParameters(dt_s=0.033, max_time_s=35.0)
    )


# ---------------------------------------------------------------------------
# 1. OpenSCENARIO (.xosc) Exporter Tests
# ---------------------------------------------------------------------------
def test_openscenario_export_validity(sample_indian_scenario):
    """Verify that export_to_openscenario generates well-formed ASAM OpenSCENARIO XML."""
    xosc_xml = export_to_openscenario(sample_indian_scenario)
    assert xosc_xml.startswith("<?xml")
    
    # Parse XML structure
    root = ET.fromstring(xosc_xml)
    assert root.tag == "OpenSCENARIO"

    # Header check
    header = root.find("FileHeader")
    assert header is not None
    assert header.attrib.get("revMajor") == "1"
    assert header.attrib.get("revMinor") == "2"

    # RoadNetwork check
    logic_file = root.find("RoadNetwork/LogicFile")
    assert logic_file is not None
    assert logic_file.attrib.get("filepath") == "test_twinx_nashik.xodr"

    # Entities check: Ego + 2 actors
    entities = root.find("Entities")
    assert entities is not None
    entity_names = [e.attrib.get("name") for e in entities.findall("ScenarioObject")]
    assert "Ego" in entity_names
    assert "desi_cow" in entity_names
    assert "auto_rickshaw_1" in entity_names

    # Storyboard check
    storyboard = root.find("Storyboard")
    assert storyboard is not None
    init = storyboard.find("Init")
    assert init is not None

    # Teleport actions exist for Ego and Actors
    privates = init.findall("Actions/Private")
    assert len(privates) == 3


def test_openscenario_maneuvers_and_triggers(sample_indian_scenario):
    """Verify dynamic maneuvers (crossing, cut_in) generate corresponding actions and triggers."""
    xosc_xml = export_to_openscenario(sample_indian_scenario)
    root = ET.fromstring(xosc_xml)

    maneuver_groups = root.findall("Storyboard/Story/Act/ManeuverGroup")
    assert len(maneuver_groups) == 2  # desi_cow (crossing) and auto_rickshaw_1 (cut_in)

    # Validate condition group simulation time triggers
    for mg in maneuver_groups:
        event = mg.find("Maneuver/Event")
        assert event is not None
        trigger = event.find("StartTrigger/ConditionGroup/Condition/ByValueCondition/SimulationTimeCondition")
        assert trigger is not None
        assert float(trigger.attrib.get("value")) > 0.0


# ---------------------------------------------------------------------------
# 2. Physics-Informed Sensor Degradation Tests
# ---------------------------------------------------------------------------
def test_sensor_degradation_clear_vs_heavy_rain():
    """Verify physical degradation under monsoon rain compared to clear day."""
    clear_profile = calculate_sensor_degradation(weather="clear", time_of_day="day", surface="tarmac", ego_speed_kmh=50.0)
    rain_profile = calculate_sensor_degradation(weather="heavy_rain", time_of_day="day", surface="tarmac", ego_speed_kmh=50.0)

    # Pavement friction must drop significantly
    assert rain_profile.friction_coefficient < clear_profile.friction_coefficient
    assert rain_profile.max_braking_decel_mps2 < clear_profile.max_braking_decel_mps2

    # Stopping distance must increase
    assert rain_profile.stopping_distance_m > clear_profile.stopping_distance_m
    assert rain_profile.safe_headway_buffer_m > clear_profile.safe_headway_buffer_m

    # Optical camera range must degrade
    assert rain_profile.camera_effective_range_m < clear_profile.camera_effective_range_m
    assert rain_profile.overall_perception_health_pct < clear_profile.overall_perception_health_pct


def test_sensor_degradation_fog_optical_contrast():
    """Verify fog attenuation severely impacts camera and lidar visibility."""
    fog_profile = calculate_sensor_degradation(weather="fog", time_of_day="day", surface="tarmac", ego_speed_kmh=40.0)
    assert fog_profile.camera_fog_contrast_loss_pct > 50.0
    assert fog_profile.camera_effective_range_m <= 45.0
    assert fog_profile.lidar_backscatter_noise_pct > 30.0


# ---------------------------------------------------------------------------
# 3. Head-to-Head Comparative Benchmark Engine Tests
# ---------------------------------------------------------------------------
def test_benchmark_engine_execution(sample_indian_scenario):
    """Verify BenchmarkEngine runs both adaptive and baseline modes on identical geometry."""
    results = BenchmarkEngine.run_benchmark(sample_indian_scenario)

    assert "adaptive" in results
    assert "baseline" in results
    assert "comparison" in results
    assert "sensor_degradation" in results

    adaptive = results["adaptive"]
    baseline = results["baseline"]
    comp = results["comparison"]

    # Both modes must produce complete trajectory data
    assert len(adaptive["traj_x"]) > 10
    assert len(baseline["traj_x"]) > 10

    # Adaptive mode must arrive successfully and win benchmark
    assert adaptive["arrived"] is True
    assert comp["winner"] == "adaptive"

    # Key comparative metrics
    assert "winner" in comp
    assert "verdict" in comp
    assert "key_differentiators" in comp
    assert len(comp["key_differentiators"]) >= 4


# ---------------------------------------------------------------------------
# 4. API Endpoints Integration Tests
# ---------------------------------------------------------------------------
def test_api_openscenario_export_endpoint(sample_indian_scenario):
    """Test POST /api/v1/scenarios/{id}/export/openscenario endpoint."""
    resp = client.post(
        f"/api/v1/scenarios/{sample_indian_scenario.scenario_id}/export/openscenario",
        json=sample_indian_scenario.model_dump()
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "xosc" in data
    assert "filename" in data
    assert data["filename"].endswith(".xosc")
    assert "<OpenSCENARIO" in data["xosc"]


def test_api_benchmark_endpoint(sample_indian_scenario):
    """Test POST /api/v1/scenarios/{id}/benchmark endpoint."""
    resp = client.post(
        f"/api/v1/scenarios/{sample_indian_scenario.scenario_id}/benchmark",
        json=sample_indian_scenario.model_dump()
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "adaptive" in data
    assert "baseline" in data
    assert "comparison" in data
    assert data["comparison"]["winner"] == "adaptive"


def test_api_sensor_degradation_endpoint():
    """Test GET /api/v1/weather/sensor-degradation endpoint."""
    resp = client.get("/api/v1/weather/sensor-degradation?weather=heavy_rain&speed_kmh=60")
    assert resp.status_code == 200
    data = resp.json()
    assert "friction_coefficient" in data
    assert "stopping_distance_m" in data
    assert "camera_effective_range_m" in data
    assert data["weather_condition"] == "heavy_rain"


def test_api_health_check_updated_features():
    """Verify health check lists all new RoadRunner & benchmark features."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    features = data.get("features", [])
    assert "openscenario_roadrunner" in features
    assert "head_to_head_benchmark" in features
    assert "sensor_degradation_physics" in features


# ---------------------------------------------------------------------------
# 5. Advanced RoadRunner & Edge Cases Tests
# ---------------------------------------------------------------------------
def test_openscenario_environment_action_and_friction(sample_indian_scenario):
    """Verify OpenSCENARIO Init Actions contain GlobalAction EnvironmentAction with Weather and Friction."""
    sample_indian_scenario.environment.weather = "heavy_rain"
    sample_indian_scenario.environment.visibility_m = 45.0
    xosc_xml = export_to_openscenario(sample_indian_scenario)
    root = ET.fromstring(xosc_xml)

    env_action = root.find("Storyboard/Init/Actions/GlobalAction/EnvironmentAction")
    assert env_action is not None

    env = env_action.find("Environment")
    assert env is not None
    assert "heavy_rain" in env.attrib.get("name", "")

    tod = env.find("TimeOfDay")
    assert tod is not None
    assert "18:15:00" in tod.attrib.get("dateTime", "")

    weather = env.find("Weather")
    assert weather is not None
    assert weather.attrib.get("cloudState") == "overcast"

    precip = weather.find("Precipitation")
    assert precip is not None
    assert precip.attrib.get("precipitationType") == "rain"
    assert float(precip.attrib.get("intensity")) > 0.5

    fog = weather.find("Fog")
    assert fog is not None
    assert float(fog.attrib.get("visualRange")) == 45.0

    road_cond = env.find("RoadCondition")
    assert road_cond is not None
    fric_factor = float(road_cond.attrib.get("frictionScaleFactor", "1.0"))
    assert 0.3 <= fric_factor <= 0.8  # Wet road reduced friction factor


def test_openscenario_static_actors_schema_compliance():
    """Verify that a scenario with ONLY static actors still produces >= 1 ManeuverGroup (ASAM schema requirement)."""
    static_sc = Scenario(
        scenario_id="static_only_test",
        metadata=Metadata(name="Static Pothole Test", source="builder", created_at="2026-10-17T00:00:00Z"),
        road=Road(type="urban_street", lane_half_width_m=3.5, length_m=100.0),
        ego_vehicle=EgoVehicle(initial_speed_kmh=40.0, goal_x_m=90.0, mode="adaptive"),
        actors=[Actor(id="pot_1", **{"class": "pothole"}, initial_x_m=40.0, initial_y_m=0.0, behaviour="static")]
    )
    xosc_xml = export_to_openscenario(static_sc)
    root = ET.fromstring(xosc_xml)

    maneuver_groups = root.findall("Storyboard/Story/Act/ManeuverGroup")
    assert len(maneuver_groups) >= 1  # Guaranteed schema compliance


def test_openscenario_wrong_side_and_gradual_actors():
    """Verify wrong_side and gradual behaviors generate corresponding actions and headings."""
    sc = Scenario(
        scenario_id="wrong_side_test",
        metadata=Metadata(name="Wrong Side Test", source="builder", created_at="2026-10-17T00:00:00Z"),
        road=Road(type="urban_street", lane_half_width_m=3.5, length_m=120.0),
        ego_vehicle=EgoVehicle(initial_speed_kmh=40.0, goal_x_m=100.0, mode="adaptive"),
        actors=[
            Actor(
                id="oncoming_bike",
                **{"class": "motorcycle"},
                initial_x_m=80.0,
                initial_y_m=0.5,
                initial_vx_mps=-7.5,
                behaviour="wrong_side"
            ),
            Actor(
                id="drifting_cow",
                **{"class": "cow"},
                initial_x_m=60.0,
                initial_y_m=-2.5,
                initial_vx_mps=0.2,
                behaviour="gradual",
                behaviour_y_target_m=1.0
            )
        ]
    )
    xosc_xml = export_to_openscenario(sc)
    root = ET.fromstring(xosc_xml)

    mgs = root.findall("Storyboard/Story/Act/ManeuverGroup")
    assert len(mgs) == 2
    mg_names = [mg.attrib.get("name") for mg in mgs]
    assert "ManeuverGroup_oncoming_bike" in mg_names
    assert "ManeuverGroup_drifting_cow" in mg_names


def test_opendrive_surface_friction_export(sample_indian_scenario):
    """Verify OpenDRIVE .xodr export embeds road surface friction."""
    from scenario_lab.export.opendrive_exporter import export_to_opendrive
    xodr_xml = export_to_opendrive(sample_indian_scenario)
    root = ET.fromstring(xodr_xml)

    surface_fric = root.find("road/surface/friction")
    assert surface_fric is not None
    fric_val = float(surface_fric.attrib.get("friction"))
    assert 0.3 <= fric_val <= 0.95


def test_nlp_offline_text_to_scenario_generation():
    """Verify text-to-scenario generation functions deterministically offline without an external API key."""
    from scenario_lab.nlp.llm_parser import LLMParser
    parser = LLMParser()
    sc, assumptions = parser.parse_text("An auto-rickshaw aggressively cuts in at 40 km/h on a wet Nashik ring road while a cow crosses")

    assert sc is not None
    assert sc.scenario_id.startswith("scen-nlp-")
    assert len(sc.actors) >= 2
    actor_classes = [getattr(a, "actor_class", None) or getattr(a, "class") for a in sc.actors]
    assert "auto_rickshaw" in actor_classes
    assert "cow" in actor_classes
    assert any("offline fallback" in a for a in assumptions)

