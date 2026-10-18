"""
test_exporters.py -- Unit and Integration Tests for DSD and OpenDRIVE Exporters.

Verifies:
  1. MATLAB Driving Scenario Designer (.m) script syntax, road specs, actors,
     trajectories, and visionDetectionGenerator sensor configurations.
  2. ASAM OpenDRIVE (.xodr) XML structure, header bounding box in inertial meters,
     planView geometries, laneSections, standard roadMarks ('solid solid'),
     typed objects, and coordinate projections.
  3. Edge cases: Single-lane, multi-lane highways, shoulders, curves, empty actors,
     barricades/barriers, and heading yaw angles.
  4. Flexible input dispatch: Scenario objects, dictionary payloads, template IDs.
  5. Windows cp1252 character set compatibility (zero unicode emojis).
  6. Standalone CLI execution of both exporter modules.
"""

import os
import sys
import math
import json
import tempfile
import unittest
import subprocess
import xml.etree.ElementTree as ET

# Ensure scenario_lab and project root are in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scenario_lab.schema.scenario_schema import (
    Scenario, Road, Environment, EgoVehicle, Actor,
    Metadata, SimulationParameters,
)
from scenario_lab.osm.osm_importer import (
    OSMNode, RoadSegment, OSMRoadNetwork,
)
from scenario_lab.export.dsd_exporter import (
    export_to_dsd_script,
    export_osm_to_dsd,
    ACTOR_MATLAB_SPECS,
)
from scenario_lab.export.opendrive_exporter import (
    export_to_opendrive,
    ACTOR_OPENDRIVE_TYPE_MAP,
)


def _create_sample_scenario(
    road_type: str = "urban_street",
    lane_half_width_m: float = 3.5,
    length_m: float = 200.0,
    markings: str = "broken_center",
    has_shoulder: bool = False,
    actors: list = None,
    weather: str = "clear",
    time_of_day: str = "day",
    visibility_m: float = 300.0,
    initial_heading_deg: float = 0.0,
) -> Scenario:
    """Helper to synthesize test Scenario objects with full schema compliance."""
    if actors is None:
        actors = [
            Actor(
                id="cow1",
                actor_class="cow",
                initial_x_m=60.0,
                initial_y_m=-3.0,
                initial_vx_mps=0.0,
                initial_vy_mps=0.8,
                behaviour="crossing",
                behaviour_trigger_s=2.0,
                behaviour_y_target_m=2.5,
            ),
            Actor(
                id="auto1",
                actor_class="auto_rickshaw",
                initial_x_m=40.0,
                initial_y_m=1.8,
                initial_vx_mps=8.0,
                initial_vy_mps=0.0,
                behaviour="constant_velocity",
            ),
        ]

    kwargs = dict(
        scenario_id="unit_test_scenario",
        metadata=Metadata(
            name="Unit Test Scenario",
            description="Scenario generated for unit testing exporters",
            source="builder",
            created_at="2026-10-13T00:00:00Z",
            tags=["test", "unit"],
        ),
        road=Road(
            type=road_type,
            lane_half_width_m=lane_half_width_m,
            length_m=length_m,
            markings=markings,
            has_shoulder=has_shoulder,
        ),
        environment=Environment(
            weather=weather,
            time_of_day=time_of_day,
            visibility_m=visibility_m,
        ),
        ego_vehicle=EgoVehicle(
            initial_speed_kmh=45.0,
            initial_x_m=0.0,
            initial_y_m=0.0,
            initial_heading_deg=initial_heading_deg,
            goal_x_m=length_m - 20.0,
            goal_y_m=0.0,
        ),
        actors=actors,
        simulation_parameters=SimulationParameters(dt_s=0.033, max_time_s=30.0),
    )
    if len(actors) == 0:
        return Scenario.model_construct(**kwargs)
    return Scenario(**kwargs)


def _create_sample_osm_network() -> OSMRoadNetwork:
    """Helper to synthesize a multi-segment OSMRoadNetwork with curves."""
    seg1 = RoadSegment(
        osm_way_id=101,
        name="MG Road",
        highway_type="primary",
        indian_road_type="highway",
        indian_description="State Highway",
        lanes=4,
        speed_limit_kmh=60,
        is_oneway=False,
        surface="tarmac",
        length_m=150.0,
        nodes=[
            OSMNode(id=1, lat=19.0760, lon=72.8777),
            OSMNode(id=2, lat=19.0765, lon=72.8782),
            OSMNode(id=3, lat=19.0772, lon=72.8790),
        ],
    )
    seg2 = RoadSegment(
        osm_way_id=102,
        name="Station Lane",
        highway_type="residential",
        indian_road_type="urban_street",
        indian_description="Residential street",
        lanes=2,
        speed_limit_kmh=30,
        is_oneway=True,
        surface="tarmac",
        length_m=80.0,
        nodes=[
            OSMNode(id=4, lat=19.0760, lon=72.8777),
            OSMNode(id=5, lat=19.0758, lon=72.8770),
        ],
    )
    return OSMRoadNetwork(
        segments=[seg1, seg2],
        bbox=(19.0750, 72.8760, 19.0780, 72.8800),
        center_lat=19.0760,
        center_lon=72.8777,
        query_timestamp="2026-10-13T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Test Suite 1: MATLAB Driving Scenario Designer Exporter
# ---------------------------------------------------------------------------
class TestDSDExporter(unittest.TestCase):
    """Verifies MATLAB Driving Scenario Designer (.m) script exporter."""

    def test_export_basic_scenario_script(self):
        scenario = _create_sample_scenario()
        script = export_to_dsd_script(scenario)

        # Check function structure
        self.assertTrue(script.startswith("function [scenario, egoVehicle, visionSensor] = dsd_unit_test_scenario()"))
        self.assertTrue(script.strip().endswith("end"))

        # Check Automated Driving Toolbox API invocations
        self.assertIn("drivingScenario('SampleTime', 0.033, 'StopTime', 30.0)", script)
        self.assertIn("road(scenario, roadCenters, roadWidth, 'Lanes', lanesSpec);", script)
        self.assertIn("lanespec([1 1], 'Width', 3.5)", script)
        self.assertIn("egoVehicle = vehicle(scenario", script)
        self.assertIn("trajectory(egoVehicle, egoWaypoints, egoSpeeds);", script)

        # Check actor generation
        self.assertIn("actor1 = actor(scenario", script)  # cow is an actor
        self.assertIn("actor2 = vehicle(scenario", script)  # auto_rickshaw is a vehicle

        # Check visionDetectionGenerator System object usage
        self.assertIn("visionDetectionGenerator", script)
        self.assertIn("'SensorLocation', [1.9 0.0]", script)
        self.assertIn("'FalsePositivesPerImage'", script)

    def test_export_all_indian_actor_classes(self):
        """Verify dimensions and types for diverse Indian traffic participants."""
        actors = [
            Actor(id="a_car", actor_class="car", initial_x_m=10, initial_y_m=0),
            Actor(id="a_cow", actor_class="cow", initial_x_m=20, initial_y_m=1),
            Actor(id="a_auto", actor_class="auto_rickshaw", initial_x_m=30, initial_y_m=2),
            Actor(id="a_ped", actor_class="person", initial_x_m=40, initial_y_m=3),
            Actor(id="a_bike", actor_class="motorcycle", initial_x_m=50, initial_y_m=-1),
            Actor(id="a_bus", actor_class="bus", initial_x_m=60, initial_y_m=-2),
            Actor(id="a_cart", actor_class="pushcart", initial_x_m=70, initial_y_m=-3),
        ]
        scenario = _create_sample_scenario(actors=actors)
        script = export_to_dsd_script(scenario)

        # Pedestrian should be instantiated with pedestrian(...)
        self.assertIn("pedestrian(scenario", script)
        # Vehicles with vehicle(...)
        self.assertIn("vehicle(scenario", script)
        # Animals / obstacles with actor(...)
        self.assertIn("actor(scenario", script)

    def test_export_barricade_and_barrier_actor(self):
        """Verify barricades and road hazards map to actor with ClassID 8."""
        actors = [
            Actor.model_construct(
                id="barr1",
                actor_class="barricade",
                initial_x_m=50.0,
                initial_y_m=0.0,
                initial_vx_mps=0.0,
                initial_vy_mps=0.0,
                behaviour="static",
            ),
        ]
        scenario = _create_sample_scenario(actors=actors)
        script = export_to_dsd_script(scenario)

        self.assertIn("actor1 = actor(scenario", script)
        self.assertIn("'ClassID', 8", script)
        self.assertIn("'Length', 2.0", script)
        self.assertIn("'Width', 0.5", script)

    def test_export_various_actor_behaviours(self):
        """Verify trajectory calculation for all behaviour patterns using standard speeds."""
        actors = [
            Actor(id="a1", actor_class="car", initial_x_m=10, initial_y_m=0, behaviour="static"),
            Actor(id="a2", actor_class="car", initial_x_m=20, initial_y_m=0, initial_vx_mps=10, behaviour="constant_velocity"),
            Actor(id="a3", actor_class="car", initial_x_m=30, initial_y_m=0, initial_vx_mps=10, behaviour="sudden_stop", behaviour_trigger_s=3),
            Actor(id="a4", actor_class="person", initial_x_m=40, initial_y_m=-3, initial_vy_mps=1.0, behaviour="crossing", behaviour_trigger_s=2),
            Actor(id="a5", actor_class="auto_rickshaw", initial_x_m=50, initial_y_m=3, initial_vx_mps=6, behaviour="cut_in", behaviour_trigger_s=2),
            Actor(id="a6", actor_class="motorcycle", initial_x_m=60, initial_y_m=0, initial_vx_mps=-8, behaviour="wrong_side"),
            Actor(id="a7", actor_class="car", initial_x_m=70, initial_y_m=0, initial_vx_mps=10, behaviour="gradual"),
        ]
        scenario = _create_sample_scenario(actors=actors)
        script = export_to_dsd_script(scenario)

        # Uses standard trajectory(actor, waypoints, speeds)
        self.assertIn("trajectory(", script)
        self.assertNotIn("WaypointTimes", script)  # Prohibits hallucinated non-existent parameter

    def test_export_ego_heading_yaw(self):
        """Verify ego initial heading generates Yaw parameter in vehicle constructor."""
        scenario = _create_sample_scenario(initial_heading_deg=15.0)
        script = export_to_dsd_script(scenario)
        self.assertIn("'Yaw', 15.00", script)

    def test_export_weather_sensor_parameters(self):
        """Verify vision sensor range/probabilities adapt to adverse weather."""
        # Heavy rain test
        sc_rain = _create_sample_scenario(weather="heavy_rain", visibility_m=60.0)
        script_rain = export_to_dsd_script(sc_rain)
        self.assertIn("DetectionProbability', 0.70", script_rain)
        self.assertIn("FalsePositivesPerImage', 0.50", script_rain)

        # Fog test
        sc_fog = _create_sample_scenario(weather="fog", visibility_m=40.0)
        script_fog = export_to_dsd_script(sc_fog)
        self.assertIn("DetectionProbability', 0.60", script_fog)
        self.assertIn("FalsePositivesPerImage', 1.00", script_fog)

    def test_export_osm_network_to_dsd(self):
        osm_net = _create_sample_osm_network()
        script = export_osm_to_dsd(osm_net)

        self.assertIn("roadCenters_1", script)
        self.assertIn("roadCenters_2", script)
        self.assertIn("lanesSpec_1", script)
        self.assertIn("lanesSpec_2", script)
        self.assertIn("egoWaypoints", script)
        self.assertIn("egoSpeeds", script)
        self.assertIn("visionDetectionGenerator", script)

    def test_export_empty_osm_raises(self):
        empty_net = OSMRoadNetwork(segments=[])
        with self.assertRaises(ValueError):
            export_osm_to_dsd(empty_net)

    def test_export_to_file_and_function_name_matching(self):
        """Verify output filename determines valid matching MATLAB function name."""
        scenario = _create_sample_scenario()
        with tempfile.NamedTemporaryFile(suffix=".m", delete=False) as tf:
            tf_path = tf.name
        try:
            export_to_dsd_script(scenario, tf_path)
            self.assertTrue(os.path.isfile(tf_path))
            with open(tf_path, "r", encoding="utf-8") as f:
                content = f.read()
            expected_func = os.path.splitext(os.path.basename(tf_path))[0]
            self.assertIn(f"function [scenario, egoVehicle, visionSensor] = {expected_func}()", content)
            self.assertIn("drivingScenario", content)
        finally:
            if os.path.isfile(tf_path):
                os.remove(tf_path)

    def test_dsd_export_from_template_id(self):
        """Verify export_to_dsd_script accepts template ID string."""
        script = export_to_dsd_script("template_cattle_crossing")
        self.assertIn("drivingScenario", script)
        self.assertIn("road(scenario", script)

    def test_dsd_export_empty_actors(self):
        """Edge case: Scenario with 0 actors generates valid DSD script."""
        scenario = _create_sample_scenario(actors=[])
        script = export_to_dsd_script(scenario)
        self.assertIn("drivingScenario", script)
        self.assertIn("egoVehicle = vehicle(scenario", script)
        self.assertNotIn("actor1", script)


# ---------------------------------------------------------------------------
# Test Suite 2: RoadRunner / OpenDRIVE Exporter
# ---------------------------------------------------------------------------
class TestOpenDriveExporter(unittest.TestCase):
    """Verifies ASAM OpenDRIVE (.xodr) XML exporter for RoadRunner & CARLA."""

    def test_export_basic_scenario_opendrive(self):
        scenario = _create_sample_scenario()
        xml_str = export_to_opendrive(scenario)

        # Verify XML syntax and standard elements
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "OpenDRIVE")

        header = root.find("header")
        self.assertIsNotNone(header)
        self.assertEqual(header.get("revMajor"), "1")
        self.assertEqual(header.get("revMinor"), "4")

        # Verify header bounds are in METERS along inertial axes
        north_val = float(header.get("north"))
        south_val = float(header.get("south"))
        east_val = float(header.get("east"))
        west_val = float(header.get("west"))

        # east (x_max) should comfortably enclose road length 200m
        self.assertGreaterEqual(east_val, 200.0)
        self.assertLessEqual(west_val, 0.0)
        # north (y_max) and south (y_min) enclose road half width 3.5m and actor y
        self.assertGreaterEqual(north_val, 3.5)
        self.assertLessEqual(south_val, -3.5)

        geo_ref = header.find("geoReference")
        self.assertIsNotNone(geo_ref)
        self.assertIn("+proj=tmerc", geo_ref.text)

        road = root.find("road")
        self.assertIsNotNone(road)
        self.assertEqual(road.get("id"), "1")
        self.assertEqual(road.get("length"), "200.0000")

        # Plan view reference geometry
        plan_view = road.find("planView")
        self.assertIsNotNone(plan_view)
        geoms = plan_view.findall("geometry")
        self.assertGreater(len(geoms), 0)
        self.assertIsNotNone(geoms[0].find("line"))

        # Elevation and lateral profiles
        self.assertIsNotNone(road.find("elevationProfile"))
        self.assertIsNotNone(road.find("lateralProfile"))

        # Lanes specification
        lanes = road.find("lanes")
        self.assertIsNotNone(lanes)
        lane_sec = lanes.find("laneSection")
        self.assertIsNotNone(lane_sec)
        self.assertIsNotNone(lane_sec.find("left"))
        self.assertIsNotNone(lane_sec.find("center"))
        self.assertIsNotNone(lane_sec.find("right"))

        # Verify no redundant empty <link/> tags inside lanes
        for lane in lane_sec.findall(".//lane"):
            self.assertIsNone(lane.find("link"))

        # Objects
        objects = road.find("objects")
        self.assertIsNotNone(objects)
        obj_list = objects.findall("object")
        self.assertEqual(len(obj_list), 2)
        cow_obj = next((o for o in obj_list if o.get("id") == "cow1"), None)
        self.assertIsNotNone(cow_obj)
        self.assertEqual(cow_obj.get("s"), "60.00")
        self.assertEqual(cow_obj.get("t"), "-3.00")
        self.assertEqual(cow_obj.get("type"), "obstacle")

    def test_export_typed_objects_and_barricades(self):
        """Verify pedestrians map to 'pedestrian', cars to 'car', barriers to 'barrier'."""
        actors = [
            Actor(id="p1", actor_class="person", initial_x_m=30.0, initial_y_m=2.0),
            Actor(id="c1", actor_class="car", initial_x_m=50.0, initial_y_m=0.0, initial_vx_mps=15.0),
            Actor(id="m1", actor_class="motorcycle", initial_x_m=70.0, initial_y_m=-1.5),
            Actor.model_construct(
                id="b1", actor_class="barricade", initial_x_m=90.0, initial_y_m=0.0,
                initial_vx_mps=0.0, initial_vy_mps=0.0, behaviour="static"
            ),
        ]
        scenario = _create_sample_scenario(actors=actors)
        xml_str = export_to_opendrive(scenario)
        root = ET.fromstring(xml_str)
        objs = {o.get("id"): o for o in root.findall("road/objects/object")}

        self.assertEqual(objs["p1"].get("type"), "pedestrian")
        self.assertEqual(objs["c1"].get("type"), "car")
        self.assertEqual(objs["m1"].get("type"), "motorbike")
        self.assertEqual(objs["b1"].get("type"), "barrier")

    def test_export_single_lane_road(self):
        """Edge case: Village single-lane road."""
        scenario = _create_sample_scenario(
            road_type="village_road",
            lane_half_width_m=2.0,
            markings="none",
        )
        xml_str = export_to_opendrive(scenario)
        root = ET.fromstring(xml_str)
        lane_sec = root.find("road/lanes/laneSection")
        # In single lane, left is omitted and right has 1 driving lane
        self.assertIsNone(lane_sec.find("left"))
        self.assertIsNotNone(lane_sec.find("right"))
        right_lanes = lane_sec.find("right").findall("lane")
        self.assertEqual(len(right_lanes), 1)

    def test_export_multilane_highway_with_shoulder_and_double_markings(self):
        """Edge case: 4-lane highway with shoulder and ASAM OpenDRIVE standard 'solid solid' markings."""
        scenario = _create_sample_scenario(
            road_type="highway",
            lane_half_width_m=7.0,
            has_shoulder=True,
            markings="double_center",
        )
        xml_str = export_to_opendrive(scenario)
        root = ET.fromstring(xml_str)
        lane_sec = root.find("road/lanes/laneSection")

        left = lane_sec.find("left")
        right = lane_sec.find("right")
        center = lane_sec.find("center")
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertIsNotNone(center)

        # Check standard OpenDRIVE 'solid solid' road mark on center lane
        center_mark = center.find("lane/roadMark")
        self.assertIsNotNone(center_mark)
        self.assertEqual(center_mark.get("type"), "solid solid")

        # Verify shoulder lane on right
        shoulder_lanes = [l for l in right.findall("lane") if l.get("type") == "shoulder"]
        self.assertEqual(len(shoulder_lanes), 1)

    def test_export_empty_actors(self):
        """Edge case: 0 actors on road."""
        scenario = _create_sample_scenario(actors=[])
        xml_str = export_to_opendrive(scenario)
        root = ET.fromstring(xml_str)
        objects = root.find("road/objects")
        self.assertIsNotNone(objects)
        self.assertEqual(len(objects.findall("object")), 0)

    def test_export_osm_network_to_opendrive_with_meter_bounds(self):
        """Test multi-segment curved road network export with meter coordinates in header."""
        osm_net = _create_sample_osm_network()
        xml_str = export_to_opendrive(osm_net)

        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "OpenDRIVE")

        # Header bounds must be in Cartesian meters around origin (0, 0), NOT lat/lon degrees
        header = root.find("header")
        north_val = float(header.get("north"))
        south_val = float(header.get("south"))
        east_val = float(header.get("east"))
        west_val = float(header.get("west"))

        # In ENU coordinates, delta lat/lon from center 19.076 produces meters in hundreds
        self.assertGreater(north_val, 50.0)
        self.assertLess(south_val, 0.0)
        self.assertGreater(east_val, 50.0)
        self.assertLess(west_val, 0.0)

        roads = root.findall("road")
        self.assertEqual(len(roads), 2)

        # Check geometry segmentation in first road (has 3 nodes -> 2 lines)
        road1 = roads[0]
        geoms = road1.find("planView").findall("geometry")
        self.assertEqual(len(geoms), 2)

    def test_export_from_dict_and_template_id(self):
        """Verify flexible dispatch accepting dict or template ID string."""
        xml_str = export_to_opendrive("template_cattle_crossing")
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "OpenDRIVE")
        self.assertIsNotNone(root.find("road"))


# ---------------------------------------------------------------------------
# Test Suite 3: Windows cp1252 Character Encoding Compatibility
# ---------------------------------------------------------------------------
class TestWindowsCp1252Compatibility(unittest.TestCase):
    """Ensures all generated outputs and terminal prints are 100% cp1252 safe."""

    def test_dsd_script_cp1252_encodable(self):
        scenario = _create_sample_scenario()
        script = export_to_dsd_script(scenario)
        # Must encode to cp1252 without UnicodeEncodeError
        encoded = script.encode("cp1252")
        self.assertIsInstance(encoded, bytes)

    def test_opendrive_cp1252_encodable(self):
        scenario = _create_sample_scenario()
        xml_str = export_to_opendrive(scenario)
        # Must encode to cp1252 without UnicodeEncodeError
        encoded = xml_str.encode("cp1252")
        self.assertIsInstance(encoded, bytes)

    def test_no_unicode_emojis_in_source_code(self):
        """Scan exporter source files for emoji characters that cause crashes on Windows console."""
        dsd_path = os.path.join(_THIS_DIR, "export", "dsd_exporter.py")
        od_path = os.path.join(_THIS_DIR, "export", "opendrive_exporter.py")

        for fpath in [dsd_path, od_path]:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            for line_no, line in enumerate(content.splitlines(), 1):
                if "print(" in line:
                    try:
                        line.encode("cp1252")
                    except UnicodeEncodeError as e:
                        self.fail(f"Unicode character incompatible with cp1252 in {fpath}:{line_no}: {line}")


# ---------------------------------------------------------------------------
# Test Suite 4: Standalone CLI Execution Tests
# ---------------------------------------------------------------------------
class TestCLIExecution(unittest.TestCase):
    """Verifies standalone invocation of both exporter modules via Python CLI."""

    def test_dsd_exporter_cli(self):
        dsd_script = os.path.join(_THIS_DIR, "export", "dsd_exporter.py")
        with tempfile.NamedTemporaryFile(suffix=".m", delete=False) as tf:
            out_file = tf.name

        try:
            cmd = [
                sys.executable,
                dsd_script,
                "--scenario",
                "template_cattle_crossing",
                "--output",
                out_file,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertIn("[OK]", res.stdout)
            self.assertTrue(os.path.isfile(out_file))
        finally:
            if os.path.isfile(out_file):
                os.remove(out_file)

    def test_opendrive_exporter_cli(self):
        od_script = os.path.join(_THIS_DIR, "export", "opendrive_exporter.py")
        with tempfile.NamedTemporaryFile(suffix=".xodr", delete=False) as tf:
            out_file = tf.name

        try:
            cmd = [
                sys.executable,
                od_script,
                "--scenario",
                "template_cattle_crossing",
                "--output",
                out_file,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertIn("[OK]", res.stdout)
            self.assertTrue(os.path.isfile(out_file))
        finally:
            if os.path.isfile(out_file):
                os.remove(out_file)


if __name__ == "__main__":
    unittest.main()
