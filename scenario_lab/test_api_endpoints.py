"""
test_api_endpoints.py -- Unit and Integration Tests for Scenario Lab API Endpoints.

Tests:
  1. POST /api/v1/scenarios/from-osm (OSM import with mock & error cases)
  2. GET /api/v1/weather (Live/fallback and seasonal parameter simulation)
  3. POST /api/v1/scenarios/{id}/export/dsd (MATLAB DSD .m script export)
  4. POST /api/v1/scenarios/{id}/export/opendrive (OpenDRIVE .xodr XML export)
  5. GET /api/v1/assets/library (Indian 3D road assets catalogue)
  6. Schema validation for new Indian hazard actors (barricades, potholes)
  7. Windows cp1252 ASCII-clean execution.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET

# Ensure project root is in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi.testclient import TestClient
from scenario_lab.api.main import app, scenario_store
from scenario_lab.schema.scenario_schema import (
    Scenario, Road, Environment, EgoVehicle, Actor,
    Metadata, SimulationParameters
)
from scenario_lab.osm.osm_importer import OSMRoadNetwork, RoadSegment, OSMNode
from scenario_lab.api.assets_library import INDIAN_ROAD_ASSETS, get_asset_library


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Create a test scenario in scenario_store
        self.test_scenario = Scenario(
            scenario_id="test_api_scenario_001",
            metadata=Metadata(
                name="Test API Scenario",
                description="Scenario created for API unit testing",
                source="builder",
                created_at="2026-10-16T12:00:00Z",
                tags=["test", "api"],
            ),
            road=Road(
                type="urban_street",
                lane_half_width_m=3.5,
                length_m=200.0,
                markings="broken_center",
                surface="tarmac",
                has_shoulder=False,
            ),
            environment=Environment(
                time_of_day="day",
                weather="clear",
                visibility_m=300.0,
            ),
            ego_vehicle=EgoVehicle(
                initial_speed_kmh=40.0,
                initial_x_m=0.0,
                initial_y_m=0.0,
                goal_x_m=180.0,
                goal_y_m=0.0,
            ),
            actors=[
                Actor(
                    id="pothole_1",
                    actor_class="pothole",
                    initial_x_m=50.0,
                    initial_y_m=1.0,
                    initial_vx_mps=0.0,
                    initial_vy_mps=0.0,
                    behaviour="static",
                ),
                Actor(
                    id="police_barricade_1",
                    actor_class="police_barricade",
                    initial_x_m=80.0,
                    initial_y_m=-1.0,
                    initial_vx_mps=0.0,
                    initial_vy_mps=0.0,
                    behaviour="static",
                ),
                Actor(
                    id="auto_1",
                    actor_class="auto_rickshaw",
                    initial_x_m=30.0,
                    initial_y_m=0.0,
                    initial_vx_mps=8.0,
                    initial_vy_mps=0.0,
                    behaviour="constant_velocity",
                ),
            ],
            simulation_parameters=SimulationParameters(dt_s=0.033, max_time_s=30.0),
        )
        scenario_store.save(self.test_scenario)

    # -------------------------------------------------------------------------
    # 1. Weather Endpoint Tests
    # -------------------------------------------------------------------------
    def test_get_weather_live_or_fallback(self):
        """Test GET /api/v1/weather returns weather and environment objects."""
        resp = self.client.get("/api/v1/weather?lat=19.0760&lon=72.8777")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("weather", data)
        self.assertIn("environment", data)
        self.assertIn("condition", data["weather"])
        self.assertIn("temperature_c", data["weather"])
        self.assertIn("rain_intensity", data["weather"])
        self.assertIn("time_of_day", data["environment"])
        self.assertIn("weather", data["environment"])

    def test_get_weather_seasonal_simulation(self):
        """Test GET /api/v1/weather with month and hour params."""
        # July (Monsoon) in Mumbai at noon
        resp = self.client.get("/api/v1/weather?lat=19.0760&lon=72.8777&month=7&hour=12")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["weather"]["condition"], ["light_rain", "heavy_rain", "overcast", "clear"])
        self.assertEqual(data["environment"]["time_of_day"], "day")

        # January (Winter night / fog) in Delhi at 4am
        resp_delhi = self.client.get("/api/v1/weather?lat=28.6139&lon=77.2090&month=1&hour=4")
        self.assertEqual(resp_delhi.status_code, 200)
        delhi_data = resp_delhi.json()
        self.assertEqual(delhi_data["environment"]["time_of_day"], "night")
        self.assertTrue(delhi_data["weather"]["is_night"])

    def test_get_weather_invalid_coords(self):
        """Test GET /api/v1/weather with invalid non-numeric coords."""
        resp = self.client.get("/api/v1/weather?lat=invalid&lon=invalid")
        self.assertEqual(resp.status_code, 422)

    # -------------------------------------------------------------------------
    # 2. OSM Scenario Import Endpoint Tests
    # -------------------------------------------------------------------------
    @patch("scenario_lab.api.main.import_from_coordinates")
    def test_import_from_osm_success(self, mock_import):
        """Test POST /api/v1/scenarios/from-osm generates and saves Scenario."""
        # Setup mock road network
        nodes = [
            OSMNode(id=1, lat=19.0657, lon=72.8687),
            OSMNode(id=2, lat=19.0665, lon=72.8695),
            OSMNode(id=3, lat=19.0672, lon=72.8702),
        ]
        seg = RoadSegment(
            osm_way_id=999,
            name="BKC Avenue 1",
            highway_type="primary",
            indian_road_type="highway",
            indian_description="State Highway / Arterial",
            lanes=4,
            speed_limit_kmh=60,
            is_oneway=False,
            surface="tarmac",
            length_m=180.0,
            nodes=nodes,
        )
        mock_network = OSMRoadNetwork(
            segments=[seg],
            bbox=(19.065, 72.868, 19.068, 72.871),
            center_lat=19.0657,
            center_lon=72.8687,
            query_timestamp="2026-10-16T12:00:00Z",
        )
        mock_import.return_value = mock_network

        payload = {
            "lat": 19.0657,
            "lon": 72.8687,
            "radius": 300.0,
            "attach_weather": True,
        }
        resp = self.client.post("/api/v1/scenarios/from-osm", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("scenario", data)
        self.assertIn("summary", data)

        sc = data["scenario"]
        self.assertTrue(sc["scenario_id"].startswith("osm-"))
        self.assertEqual(sc["metadata"]["source"], "osm")
        self.assertEqual(sc["road"]["type"], "highway")
        self.assertEqual(sc["road"]["length_m"], 180.0)
        self.assertIsNotNone(sc.get("environment"))

        summary = data["summary"]
        self.assertEqual(summary["segment_count"], 1)
        self.assertEqual(summary["total_length_m"], 180.0)

        # Verify saved in scenario_store
        saved_scenario = scenario_store.get(sc["scenario_id"])
        self.assertIsNotNone(saved_scenario)
        self.assertEqual(saved_scenario.scenario_id, sc["scenario_id"])

    @patch("scenario_lab.api.main.import_from_coordinates")
    def test_import_from_osm_no_roads_error(self, mock_import):
        """Test POST /api/v1/scenarios/from-osm returns 400 when no roads found."""
        mock_network = OSMRoadNetwork(
            segments=[],
            bbox=(0, 0, 0, 0),
            center_lat=0.0,
            center_lon=0.0,
        )
        mock_import.return_value = mock_network

        payload = {"lat": 0.0, "lon": 0.0, "radius": 100.0}
        resp = self.client.post("/api/v1/scenarios/from-osm", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No valid road network", resp.json()["detail"])

    @patch("scenario_lab.api.main.import_from_coordinates")
    def test_import_from_osm_overpass_failure(self, mock_import):
        """Test POST /api/v1/scenarios/from-osm returns 200 with fallback digital twin when Overpass query raises."""
        mock_import.side_effect = RuntimeError("Overpass API timeout")
        payload = {"lat": 19.0, "lon": 72.0}
        resp = self.client.post("/api/v1/scenarios/from-osm", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("scenario", data)
        self.assertEqual(data["scenario"]["metadata"]["source"], "osm")

    # -------------------------------------------------------------------------
    # 3. MATLAB DSD Export Endpoint Tests
    # -------------------------------------------------------------------------
    def test_export_dsd_success(self):
        """Test POST /api/v1/scenarios/{id}/export/dsd returns valid .m script."""
        resp = self.client.post(f"/api/v1/scenarios/{self.test_scenario.scenario_id}/export/dsd")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("script", data)
        self.assertIn("filename", data)
        self.assertTrue(data["filename"].endswith(".m"))

        script = data["script"]
        self.assertIn("drivingScenario", script)
        self.assertIn("lanespec", script)
        self.assertIn("egoVehicle", script)
        self.assertIn("visionDetectionGenerator", script)
        self.assertIn("pothole", script)
        self.assertIn("police_barricade", script)
        self.assertIn("auto_rickshaw", script)

        # MATLAB function name must match filename (without .m)
        func_name = os.path.splitext(data["filename"])[0]
        self.assertIn(f"function [scenario, egoVehicle, visionSensor] = {func_name}()", script)

    def test_export_dsd_not_found(self):
        """Test POST /api/v1/scenarios/{id}/export/dsd with non-existent id returns 404."""
        resp = self.client.post("/api/v1/scenarios/non_existent_scenario_xyz/export/dsd")
        self.assertEqual(resp.status_code, 404)

    def test_export_dsd_get_method(self):
        """Test GET /api/v1/scenarios/{id}/export/dsd is also supported."""
        resp = self.client.get(f"/api/v1/scenarios/{self.test_scenario.scenario_id}/export/dsd")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["filename"].endswith(".m"))

    # -------------------------------------------------------------------------
    # 4. OpenDRIVE Export Endpoint Tests
    # -------------------------------------------------------------------------
    def test_export_opendrive_success(self):
        """Test POST /api/v1/scenarios/{id}/export/opendrive returns valid XML."""
        resp = self.client.post(f"/api/v1/scenarios/{self.test_scenario.scenario_id}/export/opendrive")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("xodr", data)
        self.assertIn("filename", data)
        self.assertTrue(data["filename"].endswith(".xodr"))

        xml_str = data["xodr"]
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "OpenDRIVE")
        self.assertIsNotNone(root.find("header"))
        self.assertIsNotNone(root.find("road"))
        self.assertIsNotNone(root.find("road/lanes"))
        self.assertIsNotNone(root.find("road/objects"))

    def test_export_opendrive_not_found(self):
        """Test POST /api/v1/scenarios/{id}/export/opendrive returns 404 for unknown id."""
        resp = self.client.post("/api/v1/scenarios/non_existent_opendrive_xyz/export/opendrive")
        self.assertEqual(resp.status_code, 404)

    def test_export_opendrive_get_method(self):
        """Test GET /api/v1/scenarios/{id}/export/opendrive is also supported."""
        resp = self.client.get(f"/api/v1/scenarios/{self.test_scenario.scenario_id}/export/opendrive")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["filename"].endswith(".xodr"))

    # -------------------------------------------------------------------------
    # 5. Assets Library Endpoint Tests
    # -------------------------------------------------------------------------
    def test_get_assets_library(self):
        """Test GET /api/v1/assets/library returns catalogue of Indian road assets."""
        resp = self.client.get("/api/v1/assets/library")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("assets", data)
        self.assertIn("categories", data)
        self.assertIn("total_count", data)

        assets = data["assets"]
        categories = data["categories"]

        self.assertIn("Cattle", categories)
        self.assertIn("Barriers", categories)
        self.assertIn("Surface Anomalies", categories)
        self.assertIn("Indian Vehicles", categories)

        cattle_ids = [a["id"] for a in categories["Cattle"]]
        self.assertIn("cow", cattle_ids)
        self.assertIn("buffalo", cattle_ids)
        self.assertIn("goat", cattle_ids)
        self.assertIn("dog", cattle_ids)

        barrier_ids = [a["id"] for a in categories["Barriers"]]
        self.assertIn("police_barricade", barrier_ids)
        self.assertIn("construction_barrier", barrier_ids)
        self.assertIn("traffic_cone", barrier_ids)

        surface_ids = [a["id"] for a in categories["Surface Anomalies"]]
        self.assertIn("pothole", surface_ids)
        self.assertIn("broken_tarmac", surface_ids)
        self.assertIn("speed_bump", surface_ids)

        vehicle_ids = [a["id"] for a in categories["Indian Vehicles"]]
        self.assertIn("auto_rickshaw", vehicle_ids)
        self.assertIn("e_rickshaw", vehicle_ids)
        self.assertIn("pushcart", vehicle_ids)
        self.assertIn("motorcycle", vehicle_ids)

        # Check each asset structure has required keys
        for a in assets:
            self.assertIn("id", a)
            self.assertIn("name", a)
            self.assertIn("category", a)
            self.assertIn("dimensions", a)
            self.assertIn("length_m", a["dimensions"])
            self.assertIn("width_m", a["dimensions"])
            self.assertIn("height_m", a["dimensions"])
            self.assertIn("collision_classification", a)
            self.assertIn("description", a)

    # -------------------------------------------------------------------------
    # 6. Schema Hazards Validation
    # -------------------------------------------------------------------------
    def test_schema_actor_classes(self):
        """Test Actor schema allows barricade, police_barricade, construction_barrier, pothole, obstacle, goat, pedestrian, barrier."""
        hazard_classes = [
            "barricade", "barrier", "police_barricade", "construction_barrier",
            "pothole", "obstacle", "goat", "pedestrian"
        ]
        for hc in hazard_classes:
            actor = Actor(
                id=f"test_{hc}",
                actor_class=hc,
                initial_x_m=10.0,
                initial_y_m=0.0,
            )
            self.assertEqual(actor.actor_class, hc)

    # -------------------------------------------------------------------------
    # 7. Comprehensive Export and Persistence Tests
    # -------------------------------------------------------------------------
    def test_export_scenario_with_goat_and_hazards(self):
        """Verify scenario with goat and hazard actors exports cleanly to both DSD and OpenDRIVE."""
        sc = Scenario(
            scenario_id="test_goat_hazards_001",
            metadata=Metadata(
                name="Goat and Hazards Test",
                description="Scenario with goat and hazards",
                source="builder",
                created_at="2026-10-16T12:00:00Z",
            ),
            road=Road(
                type="village_road",
                lane_half_width_m=3.0,
                length_m=150.0,
                markings="none",
            ),
            ego_vehicle=EgoVehicle(
                initial_speed_kmh=30.0,
                goal_x_m=120.0,
            ),
            actors=[
                Actor(
                    id="goat_1",
                    actor_class="goat",
                    initial_x_m=35.0,
                    initial_y_m=1.0,
                    initial_vx_mps=0.0,
                    initial_vy_mps=1.5,
                    behaviour="crossing",
                ),
                Actor(
                    id="barrier_1",
                    actor_class="construction_barrier",
                    initial_x_m=70.0,
                    initial_y_m=0.0,
                    behaviour="static",
                ),
            ],
            simulation_parameters=SimulationParameters(dt_s=0.033, max_time_s=25.0),
        )
        scenario_store.save(sc)

        # 1. DSD export
        resp_dsd = self.client.post("/api/v1/scenarios/test_goat_hazards_001/export/dsd")
        self.assertEqual(resp_dsd.status_code, 200)
        dsd_script = resp_dsd.json()["script"]
        self.assertIn("goat", dsd_script)
        self.assertIn("construction_barrier", dsd_script)
        # Goat is ClassID 6 (cattle/animals)
        self.assertIn("'ClassID', 6", dsd_script)

        # 2. OpenDRIVE export
        resp_od = self.client.post("/api/v1/scenarios/test_goat_hazards_001/export/opendrive")
        self.assertEqual(resp_od.status_code, 200)
        od_xml = resp_od.json()["xodr"]
        self.assertIn("goat", od_xml)
        self.assertIn("construction_barrier", od_xml)

    def test_export_with_scenario_body(self):
        """Verify export endpoints accept an in-memory scenario in the POST body without pre-saving."""
        in_memory_sc = Scenario(
            scenario_id="in_memory_unsaved_sc_999",
            metadata=Metadata(
                name="In-Memory Unsaved Scenario",
                description="Testing body export",
                source="builder",
                created_at="2026-10-16T12:00:00Z",
            ),
            road=Road(type="urban_street", lane_half_width_m=3.5, length_m=100.0),
            ego_vehicle=EgoVehicle(initial_speed_kmh=40.0, goal_x_m=80.0),
            actors=[
                Actor(id="cone_1", actor_class="barricade", initial_x_m=40.0, initial_y_m=0.0, behaviour="static")
            ],
        )
        body = in_memory_sc.model_dump(by_alias=True)

        # POST DSD with body
        resp = self.client.post(
            "/api/v1/scenarios/in_memory_unsaved_sc_999/export/dsd",
            json=body
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["filename"].endswith(".m"))
        self.assertIn("drivingScenario", resp.json()["script"])

        # POST OpenDRIVE with body
        resp_od = self.client.post(
            "/api/v1/scenarios/in_memory_unsaved_sc_999/export/opendrive",
            json=body
        )
        self.assertEqual(resp_od.status_code, 200)
        self.assertTrue(resp_od.json()["filename"].endswith(".xodr"))
        self.assertIn("OpenDRIVE", resp_od.json()["xodr"])

    @patch("scenario_lab.api.main.llm_parser.parse_text")
    def test_from_text_persists_to_store(self, mock_parse):
        """Verify POST /api/v1/scenarios/from-text saves scenario to scenario_store."""
        mock_sc = Scenario(
            scenario_id="text_generated_sc_001",
            metadata=Metadata(
                name="Text Generated Scenario",
                description="From LLM",
                source="nlp",
                created_at="2026-10-16T12:00:00Z",
            ),
            road=Road(type="village_road", lane_half_width_m=3.0, length_m=120.0),
            ego_vehicle=EgoVehicle(initial_speed_kmh=35.0, goal_x_m=100.0),
            actors=[
                Actor(id="a_cow", actor_class="cow", initial_x_m=50.0, initial_y_m=0.0, behaviour="static")
            ],
        )
        mock_parse.return_value = (mock_sc, ["Assumption 1"])

        resp = self.client.post("/api/v1/scenarios/from-text", json={"text": "Cow on village road"})
        self.assertEqual(resp.status_code, 200)

        # Check it is now in scenario_store
        stored = scenario_store.get("text_generated_sc_001")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.scenario_id, "text_generated_sc_001")

        # Check export works immediately by ID
        exp_resp = self.client.post("/api/v1/scenarios/text_generated_sc_001/export/dsd")
        self.assertEqual(exp_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
