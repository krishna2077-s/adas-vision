"""
weather_provider.py -- Weather API Integration for ADAS Scenario Lab.

Integrates real-time weather data from OpenWeatherMap (or a robust climate-based
estimation fallback when no API key is set or network is unavailable) and maps it
to ADAS simulation parameters and the ScenarioLab Environment schema.

Usage (standalone):
    python weather_provider.py --lat 19.0760 --lon 72.8777
    python weather_provider.py --lat 28.6139 --lon 77.2090 --month 1 --hour 6
    python weather_provider.py --lat 19.0760 --lon 72.8777 --json

Requires: requests (standard project dependency)
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# -- sys.path fix for standalone testing (flat-import style) --
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCENARIO_LAB = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SCENARIO_LAB not in sys.path:
    sys.path.insert(0, _SCENARIO_LAB)

import requests

# ---------------------------------------------------------------------------
# Import existing schema models
# ---------------------------------------------------------------------------
try:
    from scenario_lab.schema.scenario_schema import Environment
    _HAS_SCHEMA = True
except ImportError:
    _HAS_SCHEMA = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class WeatherCondition:
    """Weather condition dataclass mapped to ADAS simulation parameters.

    Fields:
        condition: Descriptive weather label ('clear', 'overcast', 'light_rain',
                   'heavy_rain', 'fog').
        rain_intensity: Float in [0.0, 1.0] representing precipitation severity.
        fog_visibility_m: Meteorological optical range / visibility in metres.
        wind_speed_ms: Wind velocity in metres per second.
        is_night: True if solar elevation indicates nighttime, False for daytime.
        temperature_c: Ambient surface temperature in degrees Celsius.
        road_wetness: Estimated road friction degradation factor in [0.0, 1.0]
                      (0.0 = completely dry, 1.0 = flooded/standing water).
        source: Provenance indicator ('api' or 'fallback_climate_model').
    """
    condition: str
    rain_intensity: float
    fog_visibility_m: float
    wind_speed_ms: float
    is_night: bool
    temperature_c: float
    road_wetness: float = 0.0
    source: str = "fallback_climate_model"

    def to_dict(self) -> Dict[str, Any]:
        """Convert WeatherCondition to a serializable dictionary."""
        return asdict(self)

    def to_environment(self) -> Optional["Environment"]:
        """Convert this WeatherCondition into a scenario_lab Environment instance.

        Returns None if schema is not available.
        """
        return weather_to_environment(self)


# ---------------------------------------------------------------------------
# Schema Integration
# ---------------------------------------------------------------------------
def weather_to_environment(weather: WeatherCondition) -> Optional["Environment"]:
    """Map a WeatherCondition to the scenario_lab Environment model.

    Ensures that values strictly adhere to the schema's Literal and Field
    constraints:
        time_of_day: Literal["day", "dusk", "night"]
        weather: Literal["clear", "overcast", "light_rain", "heavy_rain", "fog"]
        visibility_m: Field(ge=20.0, le=500.0)
    """
    if not _HAS_SCHEMA:
        return None

    # Time of day mapping
    time_of_day = "night" if weather.is_night else "day"

    # Schema weather Literal mapping
    cond = str(weather.condition or "clear").lower()
    rain_int = float(weather.rain_intensity) if weather.rain_intensity is not None else 0.0
    if any(k in cond for k in ("fog", "mist", "haze", "smoke")):
        schema_weather = "fog"
    elif "heavy" in cond or rain_int > 0.5:
        schema_weather = "heavy_rain"
    elif "rain" in cond or "drizzle" in cond or rain_int > 0.05:
        schema_weather = "light_rain"
    elif any(k in cond for k in ("cloud", "overcast")):
        schema_weather = "overcast"
    else:
        schema_weather = "clear"

    # Visibility clamping to [20.0, 500.0] as enforced by scenario_schema
    vis = weather.fog_visibility_m if weather.fog_visibility_m is not None else 500.0
    try:
        clamped_visibility = max(20.0, min(500.0, float(vis)))
    except (TypeError, ValueError):
        clamped_visibility = 500.0

    return Environment(
        time_of_day=time_of_day,
        weather=schema_weather,
        visibility_m=round(clamped_visibility, 1),
    )


# ---------------------------------------------------------------------------
# Fallback Climate & Physical Model
# ---------------------------------------------------------------------------
def estimate_weather(lat: float, lon: float,
                     month: Optional[int] = None,
                     hour: Optional[float] = None) -> WeatherCondition:
    """Estimate weather conditions from latitude, longitude, month, and solar time.

    Uses seasonal and geographical heuristics calibrated for India and general
    tropics/subtropics, with robust fallbacks for global coordinates.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        month: Month of year (1-12). Defaults to current UTC month.
        hour: Local solar hour (0.0-24.0). Defaults to calculated solar time.

    Returns:
        WeatherCondition instance.
    """
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))
    now = datetime.now(timezone.utc)
    if month is None:
        month = now.month
    month = max(1, min(12, int(month)))

    # Compute approximate local solar time from longitude if hour not supplied
    if hour is None:
        tz_offset_hours = lon / 15.0
        utc_hours = now.hour + now.minute / 60.0 + now.second / 3600.0
        local_hour = (utc_hours + tz_offset_hours) % 24.0
    else:
        local_hour = float(hour) % 24.0

    # Solar elevation heuristic: day if 06:00 <= hour < 18:30
    is_night = local_hour < 6.0 or local_hour >= 18.5

    # Check whether coordinates lie in the Indian subcontinent region
    is_india = (6.0 <= lat <= 38.0) and (68.0 <= lon <= 98.0)

    if is_india:
        condition, rain, vis, wind, temp, wetness = _estimate_india_weather(
            lat, lon, month, local_hour
        )
    else:
        condition, rain, vis, wind, temp, wetness = _estimate_global_weather(
            lat, lon, month, local_hour
        )

    return WeatherCondition(
        condition=condition,
        rain_intensity=round(rain, 2),
        fog_visibility_m=round(vis, 1),
        wind_speed_ms=round(wind, 1),
        is_night=is_night,
        temperature_c=round(temp, 1),
        road_wetness=round(wetness, 2),
        source="fallback_climate_model",
    )


def _estimate_india_weather(lat: float, lon: float, month: int,
                            local_hour: float) -> tuple:
    """Seasonal climate logic for Indian subcontinent."""
    # 1. South-West Monsoon Season (June - September: months 6, 7, 8, 9)
    if 6 <= month <= 9:
        if month in (7, 8):
            condition = "heavy_rain"
            rain = 0.75
            vis = 300.0
            wind = 6.5
            temp = 28.0
            wetness = 0.85
        else:  # June or September
            condition = "light_rain"
            rain = 0.40
            vis = 420.0
            wind = 5.0
            temp = 30.0
            wetness = 0.55

    # 2. Winter Season (December - February: months 12, 1, 2)
    elif month in (12, 1, 2):
        # Northern India (lat >= 22.0N: Delhi, Punjab, UP, Bihar, WB plains)
        if lat >= 22.0:
            # Dense winter morning/night fog
            if local_hour < 10.0 or local_hour >= 20.0:
                condition = "fog"
                rain = 0.0
                vis = 120.0
                wind = 1.8
                temp = 12.0
                wetness = 0.20
            else:
                condition = "clear"
                rain = 0.0
                vis = 450.0
                wind = 2.5
                temp = 21.0
                wetness = 0.0
        else:
            # Peninsular / Southern India: mild, clear winter
            condition = "clear"
            rain = 0.0
            vis = 500.0
            wind = 3.0
            temp = 26.0
            wetness = 0.0

    # 3. Summer / Pre-Monsoon Season (March - May: months 3, 4, 5)
    elif 3 <= month <= 5:
        condition = "clear"
        rain = 0.0
        vis = 500.0
        wind = 4.0
        temp = 35.0 if lat >= 20.0 else 32.0
        wetness = 0.0

    # 4. Post-Monsoon / Retreating Monsoon (October - November: months 10, 11)
    else:
        # Southern coastal regions (lat < 16.0: Tamil Nadu / coastal AP)
        # experience North-East monsoon rains in Oct-Nov
        if lat < 16.0:
            condition = "light_rain"
            rain = 0.35
            vis = 400.0
            wind = 4.5
            temp = 27.0
            wetness = 0.40
        else:
            condition = "clear"
            rain = 0.0
            vis = 500.0
            wind = 2.8
            temp = 28.0
            wetness = 0.0

    return condition, rain, vis, wind, temp, wetness


def _estimate_global_weather(lat: float, lon: float, month: int,
                             local_hour: float) -> tuple:
    """Latitude-dependent fallback for locations outside India."""
    abs_lat = abs(lat)
    is_north = lat >= 0

    # Northern hemisphere winter = Dec-Feb; Southern hemisphere winter = Jun-Aug
    is_winter = month in (12, 1, 2) if is_north else month in (6, 7, 8)

    if abs_lat < 23.5:  # Tropical zone
        temp = 27.0
        condition = "overcast" if month in (6, 7, 8) else "clear"
        rain = 0.2 if condition == "overcast" else 0.0
        vis = 450.0 if rain > 0 else 500.0
        wind = 3.5
        wetness = rain * 0.8
    elif abs_lat < 50.0:  # Temperate zone
        if is_winter:
            temp = 5.0
            condition = "fog" if (local_hour < 9.0 or local_hour >= 21.0) else "overcast"
            rain = 0.0
            vis = 180.0 if condition == "fog" else 420.0
            wind = 4.0
            wetness = 0.15 if condition == "fog" else 0.0
        else:
            temp = 22.0
            condition = "clear"
            rain = 0.0
            vis = 500.0
            wind = 3.0
            wetness = 0.0
    else:  # High latitude / Polar zone
        temp = -5.0 if is_winter else 12.0
        condition = "fog" if is_winter else "clear"
        rain = 0.0
        vis = 200.0 if is_winter else 480.0
        wind = 5.0
        wetness = 0.1 if is_winter else 0.0

    return condition, rain, vis, wind, temp, wetness


# ---------------------------------------------------------------------------
# OpenWeatherMap API Integration
# ---------------------------------------------------------------------------
def _parse_owm_response(data: dict) -> WeatherCondition:
    """Parse JSON response from OpenWeatherMap into a WeatherCondition."""
    weather_list = data.get("weather") or []
    weather_info = weather_list[0] if (isinstance(weather_list, list) and weather_list and isinstance(weather_list[0], dict)) else {}

    raw_id = weather_info.get("id")
    try:
        weather_id = int(raw_id) if raw_id is not None else 800
    except (ValueError, TypeError):
        weather_id = 800

    weather_main = str(weather_info.get("main") or "").lower()
    icon = str(weather_info.get("icon") or "")

    main_data = data.get("main") or {}
    raw_temp = main_data.get("temp") if isinstance(main_data, dict) else None
    try:
        temp_c = float(raw_temp) if raw_temp is not None else 25.0
    except (ValueError, TypeError):
        temp_c = 25.0

    wind_data = data.get("wind") or {}
    raw_wind = wind_data.get("speed") if isinstance(wind_data, dict) else None
    try:
        wind_speed_ms = float(raw_wind) if raw_wind is not None else 2.0
    except (ValueError, TypeError):
        wind_speed_ms = 2.0

    # OWM reports visibility in metres, capped at 10,000m
    raw_vis = data.get("visibility")
    try:
        raw_visibility_m = float(raw_vis) if raw_vis is not None else 10000.0
    except (ValueError, TypeError):
        raw_visibility_m = 10000.0

    # Day / night calculation
    sys_data = data.get("sys") or {}
    dt = data.get("dt") or 0
    sunrise = sys_data.get("sunrise") if isinstance(sys_data, dict) else 0
    sunset = sys_data.get("sunset") if isinstance(sys_data, dict) else 0

    if sunrise and sunset and dt:
        is_night = dt < sunrise or dt > sunset
    elif icon:
        is_night = icon.endswith("n")
    else:
        is_night = False

    # Rain intensity calculation
    rain_data = data.get("rain") or {}
    raw_1h = rain_data.get("1h") if isinstance(rain_data, dict) else 0.0
    try:
        rain_1h_mm = float(raw_1h) if raw_1h is not None else 0.0
    except (ValueError, TypeError):
        rain_1h_mm = 0.0

    if rain_1h_mm > 0:
        rain_intensity = min(1.0, rain_1h_mm / 15.0)
    elif 200 <= weather_id < 300:  # Thunderstorm
        rain_intensity = 0.85
    elif 300 <= weather_id < 400:  # Drizzle
        rain_intensity = 0.25
    elif 500 <= weather_id < 600:  # Rain
        intensity_map = {500: 0.20, 501: 0.50, 502: 0.80, 503: 0.95, 504: 1.0}
        rain_intensity = intensity_map.get(weather_id, 0.40)
    elif "rain" in weather_main or "drizzle" in weather_main:
        rain_intensity = 0.35
    else:
        rain_intensity = 0.0

    # Condition mapping
    if 200 <= weather_id < 300:
        condition = "heavy_rain"
    elif 300 <= weather_id < 400:
        condition = "light_rain"
    elif 500 <= weather_id < 600:
        condition = "heavy_rain" if rain_intensity > 0.5 else "light_rain"
    elif 600 <= weather_id < 700:
        condition = "fog" if raw_visibility_m < 1000 else "overcast"
    elif 700 <= weather_id < 800:
        condition = "fog"
    elif weather_id == 800:
        condition = "clear"
    elif weather_id in (801, 802):
        condition = "clear"  # few / scattered clouds
    elif weather_id in (803, 804):
        condition = "overcast"  # broken / overcast clouds
    else:
        condition = "clear"

    # Fog visibility in simulation range (capped at 500m for ADAS schema compatibility)
    if condition == "fog":
        fog_visibility_m = min(500.0, max(20.0, raw_visibility_m))
    elif condition == "heavy_rain":
        fog_visibility_m = min(350.0, max(50.0, raw_visibility_m))
    elif condition == "light_rain":
        fog_visibility_m = min(450.0, max(100.0, raw_visibility_m))
    else:
        fog_visibility_m = min(500.0, max(20.0, raw_visibility_m))

    # Road wetness calculation
    if rain_intensity > 0.0:
        road_wetness = min(1.0, round(rain_intensity * 1.15, 2))
    elif 600 <= weather_id < 700:
        road_wetness = 0.50
    elif condition == "fog":
        road_wetness = 0.15
    else:
        road_wetness = 0.0

    return WeatherCondition(
        condition=condition,
        rain_intensity=round(rain_intensity, 2),
        fog_visibility_m=round(fog_visibility_m, 1),
        wind_speed_ms=round(wind_speed_ms, 1),
        is_night=is_night,
        temperature_c=round(temp_c, 1),
        road_wetness=round(road_wetness, 2),
        source="api",
    )


def get_weather(lat: float, lon: float,
                api_key: Optional[str] = None,
                timeout_s: int = 5) -> WeatherCondition:
    """Fetch current weather for GPS coordinates.

    Uses OpenWeatherMap if an API key is available (passed directly or in the
    OPENWEATHER_API_KEY environment variable). If no key is set or the API
    request fails (timeout, network error, invalid key), automatically falls
    back to the climate estimation model.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        api_key: Optional OpenWeatherMap API key.
        timeout_s: HTTP request timeout in seconds.

    Returns:
        WeatherCondition instance.
    """
    effective_key = api_key or os.environ.get("OPENWEATHER_API_KEY", "").strip()

    if effective_key:
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": effective_key,
                "units": "metric",
            }
            resp = requests.get(OPENWEATHER_BASE_URL, params=params,
                                timeout=timeout_s)
            if resp.status_code == 200:
                return _parse_owm_response(resp.json())
            else:
                # E.g., 401 Unauthorized, 429 Rate Limit
                pass
        except Exception:
            # Network failure or timeout -> proceed to fallback
            pass

    # Seamless fallback to climate-based estimation
    return estimate_weather(lat, lon)


# Alias for backward and caller flexibility
get_weather_for_coordinates = get_weather


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def _print_summary(lat: float, lon: float, wc: WeatherCondition) -> None:
    """Print an ASCII human-readable summary of weather conditions."""
    env = wc.to_environment()
    print("=" * 60)
    print("  ADAS Vision - Weather Provider")
    print("=" * 60)
    print(f"  Coordinates   : ({lat:.4f}, {lon:.4f})")
    print(f"  Source        : {wc.source}")
    print(f"  Condition     : {wc.condition}")
    print(f"  Time of Day   : {'night' if wc.is_night else 'day'} (is_night={wc.is_night})")
    print(f"  Temperature   : {wc.temperature_c:.1f} C")
    print(f"  Rain Intensity: {wc.rain_intensity:.2f} (0.0=none, 1.0=max)")
    print(f"  Road Wetness  : {wc.road_wetness:.2f} (0.0=dry, 1.0=flooded)")
    print(f"  Visibility    : {wc.fog_visibility_m:.1f} m")
    print(f"  Wind Speed    : {wc.wind_speed_ms:.1f} m/s")
    print()

    if env:
        print("  [OK] Environment schema mapped successfully:")
        print(f"    Environment.weather      : {env.weather}")
        print(f"    Environment.time_of_day  : {env.time_of_day}")
        print(f"    Environment.visibility_m : {env.visibility_m:.1f} m")
    else:
        print("  [!] Environment schema could not be loaded.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch or estimate weather conditions for ADAS Scenario Lab."
    )
    parser.add_argument("--lat", type=float, required=True,
                        help="Latitude in decimal degrees")
    parser.add_argument("--lon", type=float, required=True,
                        help="Longitude in decimal degrees")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Optional OpenWeatherMap API key (or set OPENWEATHER_API_KEY)")
    parser.add_argument("--month", type=int, default=None,
                        help="Override month (1-12) for climate estimation")
    parser.add_argument("--hour", type=float, default=None,
                        help="Override hour (0-23) for solar time calculation")
    parser.add_argument("--force-fallback", action="store_true",
                        help="Bypass API call and use climate estimation model")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of human-readable summary")

    args = parser.parse_args()

    if not (-90.0 <= args.lat <= 90.0):
        parser.error("--lat must be between -90.0 and 90.0 degrees")
    if not (-180.0 <= args.lon <= 180.0):
        parser.error("--lon must be between -180.0 and 180.0 degrees")

    if args.force_fallback or (args.month is not None) or (args.hour is not None):
        wc = estimate_weather(args.lat, args.lon, month=args.month, hour=args.hour)
    else:
        wc = get_weather(args.lat, args.lon, api_key=args.api_key)

    if args.json:
        out = wc.to_dict()
        env = wc.to_environment()
        if env:
            out["environment"] = env.model_dump()
        print(json.dumps(out, indent=2))
    else:
        _print_summary(args.lat, args.lon, wc)


if __name__ == "__main__":
    main()
