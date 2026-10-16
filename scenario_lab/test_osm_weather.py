"""
test_osm_weather.py -- Unit and integration tests for OSM Importer and Weather Provider.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scenario_lab.schema.scenario_schema import Scenario, Road, Environment
from scenario_lab.osm.osm_importer import (
    OSMNode,
    RoadSegment,
    OSMRoadNetwork,
    haversine_m,
    bbox_from_point,
    parse_osm_response,
    network_to_road,
    network_to_scenario,
    _parse_lanes,
    _parse_speed_limit,
    _parse_oneway,
    _parse_surface,
    INDIAN_ROAD_TYPE_MAP,
)
from scenario_lab.weather.weather_provider import (
    WeatherCondition,
    get_weather,
    get_weather_for_coordinates,
    estimate_weather,
    weather_to_environment,
    _parse_owm_response,
)


class TestOSMImporter(unittest.TestCase):
    """Test OSM road network parsing and schema conversion."""

    def test_haversine_same_point(self):
        dist = haversine_m(19.0760, 72.8777, 19.0760, 72.8777)
        self.assertAlmostEqual(dist, 0.0, places=3)

    def test_haversine_known_distance(self):
        # 1 degree of latitude is ~111.1-111.3 km
        dist = haversine_m(19.0, 72.0, 20.0, 72.0)
        self.assertTrue(110_000 < dist < 112_000)

    def test_bbox_from_point(self):
        s, w, n, e = bbox_from_point(19.0760, 72.8777, 500)
        self.assertLess(s, n)
        self.assertLess(w, e)
        self.assertAlmostEqual((s + n) / 2, 19.0760, places=4)
        self.assertAlmostEqual((w + e) / 2, 72.8777, places=4)

    def test_tag_parsers(self):
        # Lanes
        self.assertEqual(_parse_lanes({"lanes": "4"}, "motorway"), 4)
        self.assertEqual(_parse_lanes({"lanes": "2;3"}, "trunk"), 2)
        self.assertEqual(_parse_lanes({"lanes": "invalid"}, "primary"), 2)
        self.assertEqual(_parse_lanes({}, "motorway"), 4)

        # Speed limit
        self.assertEqual(_parse_speed_limit({"maxspeed": "80"}, "highway"), 80)
        self.assertEqual(_parse_speed_limit({"maxspeed": "60 km/h"}, "highway"), 60)
        self.assertEqual(_parse_speed_limit({"maxspeed": "50 mph"}, "highway"), 80)
        self.assertEqual(_parse_speed_limit({}, "motorway"), 120)

        # One-way
        self.assertTrue(_parse_oneway({"oneway": "yes"}, "secondary"))
        self.assertTrue(_parse_oneway({"oneway": "1"}, "secondary"))
        self.assertFalse(_parse_oneway({"oneway": "no"}, "motorway"))
        self.assertTrue(_parse_oneway({}, "motorway"))
        self.assertFalse(_parse_oneway({}, "residential"))

        # Surface
        self.assertEqual(_parse_surface({"surface": "asphalt"}), "tarmac")
        self.assertEqual(_parse_surface({"surface": "concrete"}), "concrete")
        self.assertEqual(_parse_surface({"surface": "gravel"}), "dirt")
        self.assertEqual(_parse_surface({"surface": "cobblestone"}), "broken_tarmac")
        self.assertEqual(_parse_surface({}), "tarmac")

    def test_indian_road_type_mapping(self):
        # Verify all mapped types produce valid Scenario Road.type literals
        valid_schema_road_types = {
            "village_road", "urban_street", "highway", "market_street", "intersection", "custom"
        }
        for osm_tag, (schema_type, desc) in INDIAN_ROAD_TYPE_MAP.items():
            self.assertIn(
                schema_type, valid_schema_road_types,
                f"Mapping for {osm_tag} -> {schema_type} not in schema literals!"
            )

    def test_parse_osm_response_mock(self):
        mock_data = {
            "elements": [
                {"type": "node", "id": 1, "lat": 19.0760, "lon": 72.8777},
                {"type": "node", "id": 2, "lat": 19.0770, "lon": 72.8787},
                {
                    "type": "way",
                    "id": 101,
                    "nodes": [1, 2],
                    "tags": {
                        "highway": "trunk",
                        "name": "Eastern Express Highway",
                        "lanes": "6",
                        "maxspeed": "80",
                        "surface": "asphalt",
                        "oneway": "yes"
                    }
                },
                {
                    "type": "way",
                    "id": 102,
                    "nodes": [1, 2],
                    "tags": {
                        "highway": "footway",  # should be ignored
                        "name": "Sidewalk"
                    }
                }
            ]
        }
        bbox = (19.07, 72.87, 19.08, 72.88)
        network = parse_osm_response(mock_data, bbox, center_lat=19.076, center_lon=72.877)
        self.assertEqual(len(network.segments), 1)
        seg = network.segments[0]
        self.assertEqual(seg.osm_way_id, 101)
        self.assertEqual(seg.name, "Eastern Express Highway")
        self.assertEqual(seg.indian_road_type, "highway")
        self.assertEqual(seg.lanes, 6)
        self.assertEqual(seg.speed_limit_kmh, 80)
        self.assertTrue(seg.is_oneway)
        self.assertEqual(seg.surface, "tarmac")
        self.assertGreater(seg.length_m, 0)

    def test_network_to_scenario(self):
        mock_data = {
            "elements": [
                {"type": "node", "id": 1, "lat": 19.0760, "lon": 72.8777},
                {"type": "node", "id": 2, "lat": 19.0780, "lon": 72.8797},
                {
                    "type": "way",
                    "id": 101,
                    "nodes": [1, 2],
                    "tags": {
                        "highway": "secondary",
                        "name": "LBS Marg",
                        "lanes": "2",
                        "surface": "asphalt"
                    }
                }
            ]
        }
        bbox = (19.07, 72.87, 19.08, 72.88)
        network = parse_osm_response(mock_data, bbox)
        scenario = network_to_scenario(network)
        self.assertIsNotNone(scenario)
        self.assertIsInstance(scenario, Scenario)
        self.assertTrue(scenario.scenario_id.startswith("osm-"))
        self.assertEqual(scenario.road.type, "urban_street")
        self.assertGreaterEqual(scenario.road.length_m, 50.0)
        self.assertLessEqual(scenario.road.length_m, 500.0)
        self.assertGreaterEqual(len(scenario.actors), 1)
        self.assertEqual(scenario.ego_vehicle.mode, "adaptive")

    def test_tag_parsers_edge_cases(self):
        # Null / None handling
        self.assertEqual(_parse_surface({"surface": None}), "tarmac")
        self.assertTrue(_parse_oneway({"oneway": None}, "motorway"))
        self.assertFalse(_parse_oneway({"oneway": None}, "residential"))
        self.assertTrue(_parse_oneway({"oneway": True}, "primary"))
        self.assertTrue(_parse_oneway({"oneway": 1}, "primary"))
        self.assertEqual(_parse_lanes({"lanes": 4}, "primary"), 4)
        self.assertEqual(_parse_lanes({"lanes": None}, "primary"), 2)
        self.assertEqual(_parse_speed_limit({"maxspeed": 80}, "highway"), 80)
        self.assertEqual(_parse_speed_limit({"maxspeed": None}, "highway"), 30)
        self.assertEqual(_parse_speed_limit({"maxspeed": "signals"}, "highway"), 30)

    def test_haversine_precision_clamping(self):
        # Antipodal points should not trigger math domain error
        dist = haversine_m(0.0, 0.0, 0.0, 180.0)
        self.assertGreater(dist, 20_000_000)

    def test_bbox_from_point_polar(self):
        s, w, n, e = bbox_from_point(90.0, 0.0, 500)
        self.assertGreaterEqual(s, -90.0)
        self.assertLessEqual(n, 90.0)
        self.assertGreaterEqual(w, -180.0)
        self.assertLessEqual(e, 180.0)

    def test_network_convenience_methods(self):
        seg = RoadSegment(
            osm_way_id=1, name="Village Track", highway_type="track",
            indian_road_type="village_road", indian_description="Track",
            lanes=1, speed_limit_kmh=20, is_oneway=False, surface="dirt",
            length_m=100.0, nodes=[]
        )
        net = OSMRoadNetwork(segments=[seg])
        road = net.to_road()
        self.assertIsNotNone(road)
        self.assertEqual(road.type, "village_road")
        self.assertEqual(road.surface, "dirt")

        scenario = net.to_scenario()
        self.assertIsNotNone(scenario)
        # Ensure actor obstacle is positioned inside the lane
        self.assertLessEqual(scenario.actors[0].initial_y_m, road.lane_half_width_m)


class TestWeatherProvider(unittest.TestCase):
    """Test Weather API integration and climate estimation fallback."""

    def test_weather_condition_dataclass(self):
        # Test 6-argument positional creation
        wc1 = WeatherCondition("clear", 0.0, 500.0, 2.0, False, 25.0)
        self.assertEqual(wc1.condition, "clear")
        self.assertEqual(wc1.rain_intensity, 0.0)
        self.assertEqual(wc1.fog_visibility_m, 500.0)
        self.assertEqual(wc1.wind_speed_ms, 2.0)
        self.assertFalse(wc1.is_night)
        self.assertEqual(wc1.temperature_c, 25.0)
        self.assertEqual(wc1.road_wetness, 0.0)

        # Test keyword arguments with road_wetness
        wc2 = WeatherCondition(
            condition="heavy_rain",
            rain_intensity=0.8,
            fog_visibility_m=200.0,
            wind_speed_ms=7.0,
            is_night=True,
            temperature_c=27.0,
            road_wetness=0.9,
            source="api",
        )
        self.assertEqual(wc2.road_wetness, 0.9)
        self.assertEqual(wc2.source, "api")
        d = wc2.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["condition"], "heavy_rain")

    def test_weather_to_environment_mapping(self):
        # 1. Clear day
        wc_clear = WeatherCondition("clear", 0.0, 500.0, 2.0, False, 28.0)
        env_clear = wc_clear.to_environment()
        self.assertIsInstance(env_clear, Environment)
        self.assertEqual(env_clear.weather, "clear")
        self.assertEqual(env_clear.time_of_day, "day")
        self.assertEqual(env_clear.visibility_m, 500.0)

        # 2. Night light rain
        wc_rain = WeatherCondition("light_rain", 0.3, 400.0, 4.0, True, 24.0)
        env_rain = wc_rain.to_environment()
        self.assertEqual(env_rain.weather, "light_rain")
        self.assertEqual(env_rain.time_of_day, "night")
        self.assertEqual(env_rain.visibility_m, 400.0)

        # 3. Dense fog
        wc_fog = WeatherCondition("fog", 0.0, 100.0, 1.5, False, 12.0)
        env_fog = wc_fog.to_environment()
        self.assertEqual(env_fog.weather, "fog")
        self.assertEqual(env_fog.visibility_m, 100.0)

        # 4. Heavy rain
        wc_heavy = WeatherCondition("heavy_rain", 0.8, 300.0, 6.0, False, 26.0)
        env_heavy = wc_heavy.to_environment()
        self.assertEqual(env_heavy.weather, "heavy_rain")

        # 5. Overcast
        wc_overcast = WeatherCondition("overcast", 0.0, 480.0, 3.0, False, 22.0)
        env_overcast = wc_overcast.to_environment()
        self.assertEqual(env_overcast.weather, "overcast")

    def test_environment_visibility_clamping(self):
        # Schema visibility_m has Field(None, ge=20.0, le=500.0)
        wc_low = WeatherCondition("fog", 0.0, 5.0, 1.0, False, 10.0)
        env_low = weather_to_environment(wc_low)
        self.assertEqual(env_low.visibility_m, 20.0)

        wc_high = WeatherCondition("clear", 0.0, 10000.0, 1.0, False, 25.0)
        env_high = weather_to_environment(wc_high)
        self.assertEqual(env_high.visibility_m, 500.0)

    def test_estimate_weather_indian_monsoon(self):
        # Mumbai in July (peak monsoon)
        wc = estimate_weather(lat=19.0760, lon=72.8777, month=7, hour=12.0)
        self.assertEqual(wc.condition, "heavy_rain")
        self.assertGreater(wc.rain_intensity, 0.5)
        self.assertGreater(wc.road_wetness, 0.7)
        self.assertFalse(wc.is_night)
        self.assertEqual(wc.source, "fallback_climate_model")

    def test_estimate_weather_indian_winter_fog(self):
        # Delhi in January at 6:00 AM (winter morning fog)
        wc = estimate_weather(lat=28.6139, lon=77.2090, month=1, hour=6.0)
        self.assertEqual(wc.condition, "fog")
        self.assertLessEqual(wc.fog_visibility_m, 150.0)
        self.assertLessEqual(wc.temperature_c, 15.0)
        self.assertGreater(wc.road_wetness, 0.0)

    def test_estimate_weather_indian_summer(self):
        # Jaipur in May (hot summer)
        wc = estimate_weather(lat=26.9124, lon=75.7873, month=5, hour=14.0)
        self.assertEqual(wc.condition, "clear")
        self.assertEqual(wc.rain_intensity, 0.0)
        self.assertGreaterEqual(wc.temperature_c, 32.0)
        self.assertFalse(wc.is_night)

    def test_estimate_weather_day_night(self):
        # Night at 23:00
        wc_night = estimate_weather(lat=19.0760, lon=72.8777, month=3, hour=23.0)
        self.assertTrue(wc_night.is_night)

        # Day at 12:00
        wc_day = estimate_weather(lat=19.0760, lon=72.8777, month=3, hour=12.0)
        self.assertFalse(wc_day.is_night)

    def test_parse_owm_response_rain(self):
        mock_resp = {
            "coord": {"lat": 19.076, "lon": 72.877},
            "weather": [{"id": 502, "main": "Rain", "description": "heavy intensity rain", "icon": "10d"}],
            "main": {"temp": 27.5, "humidity": 85},
            "visibility": 3500,
            "wind": {"speed": 5.4},
            "rain": {"1h": 12.0},
            "dt": 1690000000,
            "sys": {"sunrise": 1689980000, "sunset": 1690025000},
        }
        wc = _parse_owm_response(mock_resp)
        self.assertEqual(wc.condition, "heavy_rain")
        self.assertGreater(wc.rain_intensity, 0.7)
        self.assertGreater(wc.road_wetness, 0.7)
        self.assertFalse(wc.is_night)
        self.assertEqual(wc.temperature_c, 27.5)
        self.assertEqual(wc.source, "api")

    def test_parse_owm_response_fog(self):
        mock_resp = {
            "coord": {"lat": 28.61, "lon": 77.20},
            "weather": [{"id": 741, "main": "Fog", "description": "dense fog", "icon": "50n"}],
            "main": {"temp": 9.0, "humidity": 98},
            "visibility": 150,
            "wind": {"speed": 1.2},
            "dt": 1673000000,
            "sys": {"sunrise": 1673050000, "sunset": 1673080000},
        }
        wc = _parse_owm_response(mock_resp)
        self.assertEqual(wc.condition, "fog")
        self.assertEqual(wc.fog_visibility_m, 150.0)
        self.assertTrue(wc.is_night)
        self.assertEqual(wc.road_wetness, 0.15)

    def test_get_weather_fallback_when_no_api_key(self):
        # Without OPENWEATHER_API_KEY, should fall back without error
        with patch.dict(os.environ, {}, clear=True):
            wc = get_weather(19.0760, 72.8777)
            self.assertIsInstance(wc, WeatherCondition)
            self.assertEqual(wc.source, "fallback_climate_model")

    def test_get_weather_fallback_on_network_error(self):
        # When requests fails, should fall back gracefully
        with patch("requests.get", side_effect=Exception("Connection timed out")):
            wc = get_weather(19.0760, 72.8777, api_key="dummy_key")
            self.assertIsInstance(wc, WeatherCondition)
            self.assertEqual(wc.source, "fallback_climate_model")

    def test_parse_owm_response_none_fields(self):
        # Handle responses where keys exist with None values
        mock_null_resp = {
            "weather": None,
            "main": None,
            "wind": None,
            "rain": None,
            "sys": None,
            "visibility": None,
        }
        wc = _parse_owm_response(mock_null_resp)
        self.assertIsInstance(wc, WeatherCondition)
        self.assertEqual(wc.condition, "clear")
        self.assertEqual(wc.temperature_c, 25.0)
        self.assertEqual(wc.rain_intensity, 0.0)

        # Handle inner dicts with None values
        mock_inner_null = {
            "weather": [{"id": None, "main": None}],
            "main": {"temp": None},
            "wind": {"speed": None},
            "rain": {"1h": None},
            "visibility": None,
        }
        wc2 = _parse_owm_response(mock_inner_null)
        self.assertIsInstance(wc2, WeatherCondition)
        self.assertEqual(wc2.condition, "clear")

    def test_parse_owm_response_snow(self):
        # Snow conditions (weather id 601)
        mock_snow = {
            "weather": [{"id": 601, "main": "Snow"}],
            "main": {"temp": -2.0},
            "visibility": 800,
        }
        wc = _parse_owm_response(mock_snow)
        self.assertEqual(wc.condition, "fog")
        self.assertEqual(wc.road_wetness, 0.50)

    def test_weather_to_environment_none_fields(self):
        # WeatherCondition with None fields should convert cleanly
        wc_none = WeatherCondition(
            condition=None,
            rain_intensity=None,
            fog_visibility_m=None,
            wind_speed_ms=1.0,
            is_night=False,
            temperature_c=20.0,
        )
        env = weather_to_environment(wc_none)
        self.assertIsInstance(env, Environment)
        self.assertEqual(env.weather, "clear")
        self.assertEqual(env.visibility_m, 500.0)

    def test_estimate_weather_clamped_coordinates(self):
        # Out-of-bounds coordinates should not throw
        wc_polar = estimate_weather(lat=150.0, lon=250.0)
        self.assertIsInstance(wc_polar, WeatherCondition)
        self.assertIsNotNone(wc_polar.condition)


if __name__ == "__main__":
    unittest.main()

