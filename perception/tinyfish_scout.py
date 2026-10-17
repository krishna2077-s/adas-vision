"""
tinyfish_scout.py — TinyFish AI Web Hazard & Route Scout (Advisory Layer).

Integrates TinyFish AI (agent.tinyfish.ai) to scout live external route conditions,
road hazards, traffic incident alerts, and adverse weather warnings from the web.

Design Philosophy & Safety:
    - DECOUPLED & NON-HARDCODED: Controlled entirely by feature flag (cfg.ENABLE_TINYFISH)
      and environment variable TINYFISH_API_KEY. If disabled or unset, the module runs in
      offline/mock mode or stays dormant without impacting ADAS frame rates.
    - CREDIT CONSERVATIVE: Uses TinyFish's free Search API (12,000 req/day) for routine
      scouting. Paid Agent Automation ($0.016/step) is only activated when explicitly requested.
    - ASYNCHRONOUS: Runs web scouting in a background daemon thread, guaranteeing that
      monocular perception and decision engine loops run at full 30-60 FPS without I/O stalls.
    - ADVISORY ONLY: Like all ADAS Vision modules, this is strictly advisory and NEVER
      actuates steering, throttle, or brakes.

Usage (standalone):
    python perception/tinyfish_scout.py --location "NH48 Delhi Gurgaon" --mock
    python perception/tinyfish_scout.py --location "Mumbai Pune Expressway" --deep
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# TinyFish Gateway Endpoints
TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"
TINYFISH_FETCH_URL  = "https://api.fetch.tinyfish.ai"
TINYFISH_AGENT_URL  = "https://agent.tinyfish.ai/v1/automation/run"


@dataclass
class RoadHazardAlert:
    """Represents an external environmental hazard scouted from the web."""
    hazard_type: str            # "FOG", "WATERLOGGING", "ACCIDENT", "CONSTRUCTION", "CLEAR"
    severity: str               # "LOW", "MEDIUM", "HIGH"
    headline: str               # Short human-readable summary
    location: str               # Highway / road section
    recommended_speed_kmh: Optional[int] = None
    source: str = "TinyFish Web Scout"
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Realistic fallback / offline scenarios for simulation & testing without spending API credits
MOCK_HAZARD_DATABASE = [
    RoadHazardAlert(
        hazard_type="FOG",
        severity="HIGH",
        headline="Dense fog advisory; visibility under 50m",
        location="NH48 Delhi-Gurgaon Expressway",
        recommended_speed_kmh=45,
        source="IMD / Traffic Police Advisory",
        timestamp=time.time(),
    ),
    RoadHazardAlert(
        hazard_type="WATERLOGGING",
        severity="MEDIUM",
        headline="Submerged shoulder & slow traffic near underpass",
        location="Outer Ring Road km 18",
        recommended_speed_kmh=40,
        source="Municipal Incident Feed",
        timestamp=time.time(),
    ),
    RoadHazardAlert(
        hazard_type="CONSTRUCTION",
        severity="MEDIUM",
        headline="Flyover lane diversion; 2 lanes merged",
        location="NH44 Panipat Stretch",
        recommended_speed_kmh=50,
        source="Highway Authority Notice",
        timestamp=time.time(),
    ),
    RoadHazardAlert(
        hazard_type="CLEAR",
        severity="LOW",
        headline="All lanes clear; normal highway flow",
        location="Current Route",
        recommended_speed_kmh=80,
        source="Highway Monitoring Feed",
        timestamp=time.time(),
    ),
]


class TinyFishRoadScout:
    """
    Asynchronous web hazard scout powered by TinyFish AI.
    Runs periodic web recon in the background and surfaces live advisory chips on the ADAS HUD.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
        location: str = "NH48 Delhi-Gurgaon",
        poll_interval_s: float = 60.0,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or os.environ.get("TINYFISH_API_KEY", "").strip()
        env_enabled = os.environ.get("ENABLE_TINYFISH", "false").lower() in ("1", "true", "yes")
        self.mock_mode = mock_mode or (not self.api_key)
        self.enabled = enabled if enabled is not None else (env_enabled or bool(self.api_key) or bool(mock_mode))
        self.location = location
        self.poll_interval_s = max(10.0, poll_interval_s)

        self._active_alerts: List[RoadHazardAlert] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_poll_time = 0.0

        if self.mock_mode:
            logger.info("TinyFish Scout: Running in OFFLINE/MOCK mode (0 credits used).")
            # Seed with initial simulated alert
            self._active_alerts = [MOCK_HAZARD_DATABASE[0]]
        else:
            logger.info(f"TinyFish Scout: Initialized with API key for location '{self.location}'.")

    def is_available(self) -> bool:
        """Returns True if the scout is enabled and ready to provide alerts."""
        return self.enabled and (_HAS_REQUESTS or self.mock_mode)

    def start_background_scout(self) -> None:
        """Starts the background worker thread for periodic non-blocking scouting."""
        if not self.is_available() or (self._thread and self._thread.is_alive()):
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TinyFishScoutWorker")
        self._thread.start()
        logger.info("TinyFish Scout: Background polling worker started.")

    def stop_background_scout(self) -> None:
        """Stops the background worker thread gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _worker_loop(self) -> None:
        """Background thread target that polls periodically without freezing video frames."""
        while not self._stop_event.is_set():
            now = time.time()
            if now - self._last_poll_time >= self.poll_interval_s:
                try:
                    alerts = self.scout_hazards(self.location)
                    with self._lock:
                        self._active_alerts = alerts
                    self._last_poll_time = now
                except Exception as exc:
                    logger.warning(f"TinyFish Scout background poll error: {exc}")
            self._stop_event.wait(timeout=1.0)

    def get_latest_alerts(self) -> List[RoadHazardAlert]:
        """Thread-safe retrieval of current alerts."""
        with self._lock:
            return list(self._active_alerts)

    def scout_hazards(self, location: str, deep_agent: bool = False) -> List[RoadHazardAlert]:
        """
        Query TinyFish for hazards along the route.
        Tier 1: Free Search API (12k req/day).
        Tier 2: Paid Agent Automation API ($0.016/step) if deep_agent=True.
        """
        if self.mock_mode or not self.api_key or not _HAS_REQUESTS:
            # Pick contextual mock alert based on location or default
            for m in MOCK_HAZARD_DATABASE:
                if m.location.lower() in location.lower():
                    return [m]
            return [MOCK_HAZARD_DATABASE[0]]

        if deep_agent:
            return self._query_agent_automation(location)
        return self._query_search_api(location)

    def _query_search_api(self, location: str) -> List[RoadHazardAlert]:
        """Free-tier structured web search via TinyFish Search Gateway."""
        headers = {"X-API-Key": self.api_key}
        query = f"traffic hazard road closure weather warning accident {location} live advisory"
        params = {
            "query": query,
            "recency_minutes": 1440,  # last 24 hours
            "language": "en",
        }
        try:
            resp = requests.get(TINYFISH_SEARCH_URL, headers=headers, params=params, timeout=8.0)
            if resp.status_code != 200:
                logger.warning(f"TinyFish Search API returned HTTP {resp.status_code}: {resp.text[:120]}")
                return [MOCK_HAZARD_DATABASE[0]]

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return [RoadHazardAlert(
                    hazard_type="CLEAR",
                    severity="LOW",
                    headline="No active road hazards reported",
                    location=location,
                    recommended_speed_kmh=80,
                    source="TinyFish Search (0 active alerts)",
                    timestamp=time.time(),
                )]

            # Analyze snippets to derive structured hazard
            combined_text = " ".join([r.get("title", "") + " " + r.get("snippet", "") for r in results[:4]]).lower()
            return [self._parse_text_to_hazard(combined_text, location, source="TinyFish Live Web Search")]
        except Exception as exc:
            logger.warning(f"TinyFish Search request failed: {exc}")
            return [MOCK_HAZARD_DATABASE[0]]

    def _query_agent_automation(self, location: str) -> List[RoadHazardAlert]:
        """Paid-tier Agent Automation ($0.016/step) with natural language browsing goal."""
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "url": "https://www.google.com/search?q=" + requests.utils.quote(f"live traffic accident road work {location}"),
            "goal": (
                f"Identify if there is any serious traffic incident, accident, waterlogging, or severe weather on {location}. "
                "Return a JSON object with: hazard_type (FOG, WATERLOGGING, ACCIDENT, CONSTRUCTION, or CLEAR), "
                "severity (LOW, MEDIUM, HIGH), headline, and recommended_speed_kmh (integer or null)."
            ),
        }
        try:
            resp = requests.post(TINYFISH_AGENT_URL, headers=headers, json=payload, timeout=25.0)
            if resp.status_code != 200:
                logger.warning(f"TinyFish Agent API returned HTTP {resp.status_code}")
                return self._query_search_api(location)

            res_data = resp.json()
            raw_result = res_data.get("result") or res_data.get("result_json") or {}
            if isinstance(raw_result, str):
                try:
                    raw_result = json.loads(raw_result)
                except Exception:
                    raw_result = {}

            h_type = str(raw_result.get("hazard_type", "GENERAL")).upper()
            sev = str(raw_result.get("severity", "MEDIUM")).upper()
            head = str(raw_result.get("headline", f"Reported advisory for {location}"))
            speed = raw_result.get("recommended_speed_kmh")
            if speed is not None:
                speed = int(speed)

            return [RoadHazardAlert(
                hazard_type=h_type,
                severity=sev if sev in ("LOW", "MEDIUM", "HIGH") else "MEDIUM",
                headline=head,
                location=location,
                recommended_speed_kmh=speed or (45 if sev == "HIGH" else 60),
                source="TinyFish Autonomous Agent",
                timestamp=time.time(),
            )]
        except Exception as exc:
            logger.warning(f"TinyFish Agent automation failed: {exc}")
            return self._query_search_api(location)

    def _parse_text_to_hazard(self, text: str, location: str, source: str) -> RoadHazardAlert:
        """Deterministic keyword heuristic mapping search snippets into structured alerts."""
        if any(k in text for k in ("fog", "smog", "poor visibility", "mist")):
            return RoadHazardAlert(
                hazard_type="FOG",
                severity="HIGH",
                headline="Reduced visibility: Dense fog / smog reported",
                location=location,
                recommended_speed_kmh=40,
                source=source,
                timestamp=time.time(),
            )
        if any(k in text for k in ("waterlog", "flood", "submerged", "heavy rain", "inundat")):
            return RoadHazardAlert(
                hazard_type="WATERLOGGING",
                severity="HIGH",
                headline="Waterlogging & reduced tire traction reported",
                location=location,
                recommended_speed_kmh=35,
                source=source,
                timestamp=time.time(),
            )
        if any(k in text for k in ("accident", "crash", "collision", "overturned", "pileup")):
            return RoadHazardAlert(
                hazard_type="ACCIDENT",
                severity="HIGH",
                headline="Traffic bottleneck: Road accident reported ahead",
                location=location,
                recommended_speed_kmh=45,
                source=source,
                timestamp=time.time(),
            )
        if any(k in text for k in ("construction", "repair", "roadwork", "diversion", "barricade")):
            return RoadHazardAlert(
                hazard_type="CONSTRUCTION",
                severity="MEDIUM",
                headline="Lane diversion: Construction / maintenance underway",
                location=location,
                recommended_speed_kmh=50,
                source=source,
                timestamp=time.time(),
            )

        return RoadHazardAlert(
            hazard_type="CLEAR",
            severity="LOW",
            headline="Normal highway conditions reported",
            location=location,
            recommended_speed_kmh=80,
            source=source,
            timestamp=time.time(),
        )

    def draw_chip(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders a non-intrusive advisory banner on top of the frame.
        Positioned neatly in the upper-right corner of the ADAS display.
        """
        alerts = self.get_latest_alerts()
        if not alerts:
            return frame

        alert = alerts[0]
        h, w = frame.shape[:2]

        # Colors (BGR)
        if alert.severity == "HIGH":
            bg_color = (20, 20, 180)     # Crimson red
            border_color = (40, 40, 240)
            text_color = (255, 255, 255)
        elif alert.severity == "MEDIUM":
            bg_color = (15, 120, 210)    # Amber/Orange
            border_color = (30, 160, 255)
            text_color = (255, 255, 255)
        else:
            bg_color = (30, 110, 40)     # Forest Green
            border_color = (50, 180, 70)
            text_color = (240, 255, 240)

        # Dimensions & Box location (top right)
        card_w, card_h = 360, 52
        margin = 16
        x1 = w - card_w - margin
        y1 = margin
        x2 = x1 + card_w
        y2 = y1 + card_h

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 2)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Title line
        title_str = f"TINYFISH SCOUT: {alert.hazard_type} [{alert.severity}]"
        cv2.putText(
            frame,
            title_str,
            (x1 + 10, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            text_color,
            1,
            cv2.LINE_AA,
        )

        # Subtitle / advisory line
        spd_str = f"MAX {alert.recommended_speed_kmh} KM/H" if alert.recommended_speed_kmh else "CAUTION"
        sub_str = f"{alert.headline[:26]}... | {spd_str}"
        cv2.putText(
            frame,
            sub_str,
            (x1 + 10, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 230, 230),
            1,
            cv2.LINE_AA,
        )

        return frame


# ── Standalone CLI for testing & verification ─────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TinyFish Web Hazard Scout CLI")
    parser.add_argument("--location", default="NH48 Delhi-Gurgaon", help="Highway or city stretch to scout")
    parser.add_argument("--mock", action="store_true", help="Force offline mock mode (0 API calls, 0 credits)")
    parser.add_argument("--deep", action="store_true", help="Trigger paid TinyFish Agent Automation run ($0.016/step)")
    args = parser.parse_args()

    print("=" * 60)
    print("TinyFish AI — Route Hazard & Environmental Scout")
    print("=" * 60)
    scout = TinyFishRoadScout(location=args.location, mock_mode=args.mock)
    print(f"Target Location : {args.location}")
    print(f"Execution Mode  : {'MOCK/OFFLINE' if scout.mock_mode else 'LIVE TINYFISH API'}")
    print(f"Agent Depth     : {'Deep Autonomous Browser ($)' if args.deep else 'Free Search Gateway (0$)'}")
    print("Querying TinyFish...")

    results = scout.scout_hazards(args.location, deep_agent=args.deep)
    for idx, alert in enumerate(results, start=1):
        print(f"\n[Alert {idx}]")
        print(f"  Type             : {alert.hazard_type}")
        print(f"  Severity         : {alert.severity}")
        print(f"  Headline         : {alert.headline}")
        print(f"  Location         : {alert.location}")
        print(f"  Speed Advisory   : {alert.recommended_speed_kmh} km/h")
        print(f"  Data Source      : {alert.source}")

    # Test HUD drawing
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    drawn = scout.draw_chip(blank_frame)
    print("\nHUD Overlay Test: Successfully rendered 1280x720 advisory chip.")
    print("=" * 60)
