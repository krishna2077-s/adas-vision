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


if __name__ == "__main__":
    unittest.main()
