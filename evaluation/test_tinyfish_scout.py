"""
test_tinyfish_scout.py — Unit & Integration tests for TinyFish Web Hazard Scout.

Verifies:
1. Scout initialization (enabled, disabled, mock modes).
2. Data structure integrity of RoadHazardAlert.
3. HUD chip rendering on OpenCV frames (resolution, dtype, bounds check).
4. Asynchronous worker thread lifecycle (start, poll, graceful stop).
5. Error containment (bad keys or network dropouts never crash the ADAS main loop).
"""

import sys
import time
from pathlib import Path
import unittest

import numpy as np

# Ensure project roots are in path
ROOT = Path(__file__).resolve().parent.parent
for folder in ("core", "perception", "simulation"):
    p = ROOT / folder
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinyfish_scout import TinyFishRoadScout, RoadHazardAlert, MOCK_HAZARD_DATABASE


class TestTinyFishScout(unittest.TestCase):

    def test_mock_scout_instantiation(self):
        """Verify scout can instantiate in mock mode without any API key."""
        scout = TinyFishRoadScout(mock_mode=True, location="NH48 Test Highway")
        self.assertTrue(scout.is_available())
        self.assertTrue(scout.mock_mode)

    def test_hazard_alert_schema(self):
        """Verify that scout returns valid RoadHazardAlert items with correct types."""
        scout = TinyFishRoadScout(mock_mode=True, location="NH48 Delhi-Gurgaon")
        alerts = scout.scout_hazards("NH48 Delhi-Gurgaon")
        self.assertIsInstance(alerts, list)
        self.assertGreater(len(alerts), 0)

        alert = alerts[0]
        self.assertIsInstance(alert, RoadHazardAlert)
        self.assertIn(alert.severity, ("LOW", "MEDIUM", "HIGH"))
        self.assertIsInstance(alert.headline, str)
        self.assertIsInstance(alert.hazard_type, str)
        d = alert.to_dict()
        self.assertIn("headline", d)
        self.assertIn("severity", d)

    def test_hud_draw_chip(self):
        """Verify HUD chip overlay correctly modifies a 720p frame without altering dimensions."""
        scout = TinyFishRoadScout(mock_mode=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Baseline: frame is completely black
        self.assertEqual(frame.sum(), 0)

        # Draw chip
        annotated = scout.draw_chip(frame)
        self.assertEqual(annotated.shape, (720, 1280, 3))
        self.assertEqual(annotated.dtype, np.uint8)

        # Pixels where the badge was drawn should now be non-zero
        self.assertGreater(annotated.sum(), 0)

    def test_background_thread_lifecycle(self):
        """Verify background daemon thread starts, provides thread-safe alert access, and shuts down."""
        scout = TinyFishRoadScout(mock_mode=True, poll_interval_s=10.0)
        self.assertIsNone(scout._thread)

        scout.start_background_scout()
        self.assertIsNotNone(scout._thread)
        self.assertTrue(scout._thread.is_alive())

        alerts = scout.get_latest_alerts()
        self.assertIsInstance(alerts, list)

        scout.stop_background_scout()
        self.assertIsNone(scout._thread)

    def test_graceful_fallback_on_invalid_endpoint(self):
        """Verify that an invalid API key / bad network does not raise an unhandled exception."""
        scout = TinyFishRoadScout(api_key="sk-tinyfish-invalid-key-for-test", mock_mode=False)
        # Should gracefully return fallback without throwing
        alerts = scout.scout_hazards("NH48 Test")
        self.assertIsInstance(alerts, list)
        self.assertGreater(len(alerts), 0)

    def test_malformed_llm_payload_sanitization(self):
        """Verify strict schema validation sanitizes malformed/hallucinated LLM agent output."""
        # 1. Hallucinated "CRITICAL" severity -> mapped to HIGH
        raw1 = {"hazard_type": "FOG", "severity": "CRITICAL", "headline": "Zero visibility", "recommended_speed_kmh": 35}
        al1 = RoadHazardAlert.from_dict(raw1)
        self.assertEqual(al1.severity, "HIGH")
        self.assertEqual(al1.recommended_speed_kmh, 35)

        # 2. Out-of-bounds speed clamping (speed 180 clamped to 100, speed 5 clamped to 20)
        raw2 = {"hazard_type": "WATERLOGGING", "severity": "HIGH", "recommended_speed_kmh": 180}
        al2 = RoadHazardAlert.from_dict(raw2)
        self.assertEqual(al2.recommended_speed_kmh, 100)

        raw3 = {"hazard_type": "WATERLOGGING", "severity": "HIGH", "recommended_speed_kmh": 5}
        al3 = RoadHazardAlert.from_dict(raw3)
        self.assertEqual(al3.recommended_speed_kmh, 20)

        # 3. Completely invalid types / missing fields
        al4 = RoadHazardAlert.from_dict("not-a-dict")
        self.assertEqual(al4.hazard_type, "CLEAR")
        self.assertEqual(al4.severity, "LOW")

    def test_ttl_expiry_and_stale_flag(self):
        """Verify that alerts older than ttl_s are flagged as is_stale."""
        alert = RoadHazardAlert(
            hazard_type="FOG",
            severity="HIGH",
            headline="Dense fog",
            location="NH48",
            recommended_speed_kmh=40,
            timestamp=time.time() - 400.0,
            ttl_s=300.0,
        )
        self.assertTrue(alert.is_stale)
        self.assertGreater(alert.age_s, 300.0)

        fresh_alert = RoadHazardAlert(
            hazard_type="FOG",
            severity="HIGH",
            headline="Dense fog",
            location="NH48",
            recommended_speed_kmh=40,
            timestamp=time.time(),
            ttl_s=300.0,
        )
        self.assertFalse(fresh_alert.is_stale)

    def test_scoped_corridor_telemetry(self):
        """Verify that vehicle heading and KM progression are correctly incorporated into queries."""
        scout = TinyFishRoadScout(location="NH48", mock_mode=True)
        scout.update_telemetry(heading="Southbound", km_marker=42.0)
        query = scout._build_scoped_query("NH48")
        self.assertIn("NH48", query)
        self.assertIn("Southbound", query)
        self.assertIn("km 42-52", query)

    def test_planner_scout_speed_cap(self):
        """Verify that prediction planner caps cruise target speed to scout recommended speed."""
        from prediction_planning import Planner
        planner = Planner(1280, 720)
        scout_alert = RoadHazardAlert(
            hazard_type="WATERLOGGING",
            severity="HIGH",
            headline="Severe waterlogging",
            location="NH48",
            recommended_speed_kmh=36,  # 36 km/h = 10 m/s
            timestamp=time.time(),
        )
        # Without scout alert, default cruise is ~20 m/s (~72 km/h)
        p_normal = planner.process([], None)
        self.assertGreater(p_normal.target_speed_mps, 10.0)

        # With scout alert, target speed is capped to 10.0 m/s
        p_capped = planner.process([], None, scout_hazard=scout_alert)
        self.assertAlmostEqual(p_capped.target_speed_mps, 10.0, delta=0.1)
        self.assertIn("scout WATERLOGGING cap 36 km/h", p_capped.reason)

    def test_drive_logger_serializes_scout_alert(self):
        """Verify that DriveLogger logs scout alert metadata cleanly in JSONL."""
        import tempfile, json
        from drive_logger import DriveLogger
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            log_path = tf.name

        scout_alert = RoadHazardAlert(
            hazard_type="FOG",
            severity="HIGH",
            headline="Dense fog advisory",
            location="NH48",
            recommended_speed_kmh=40,
            timestamp=time.time(),
        )
        logger = DriveLogger(log_path, source="test", frame_width=1280, frame_height=720, fps=30.0)
        logger.log(0, None, scout_alert=scout_alert)
        logger.close()

        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        frame_recs = [l for l in lines if l.get("type") == "frame"]
        self.assertEqual(len(frame_recs), 1)
        scout_rec = frame_recs[0].get("scout")
        self.assertIsNotNone(scout_rec)
        self.assertEqual(scout_rec["type"], "FOG")
        self.assertEqual(scout_rec["sev"], "HIGH")
        self.assertEqual(scout_rec["spd"], 40)
        self.assertFalse(scout_rec["stale"])


if __name__ == "__main__":
    unittest.main()
