"""
sensor_degradation.py -- Physics-Informed Sensor Attenuation & Friction Safety Model.

Quantifies the degradation of autonomous driving perception and braking envelopes
under diverse Indian meteorological conditions (Monsoon deluge, dense fog/smog,
night glare, wet waterlogged pavement).

Provides:
  - Camera optical contrast attenuation (Beer-Lambert law) and effective detection range
  - LiDAR droplet backscattering loss and point density decay
  - Radar multipath surface clutter and RCS fluctuation
  - Road friction coefficient (mu) and minimum dynamic stopping distance safety envelope
  - Safe headway margin calculation for the ADAS decision engine
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class SensorDegradationProfile:
    """Quantitative sensor degradation metrics under current environmental conditions."""
    weather_condition: str
    time_of_day: str
    road_surface: str
    road_wetness: float  # [0.0, 1.0]
    
    # Pavement Friction & Dynamics
    friction_coefficient: float  # mu in [0.35, 0.90]
    max_braking_decel_mps2: float  # mu * g
    stopping_distance_m: float  # at reference speed
    safe_headway_buffer_m: float
    
    # Camera Perception Degradation
    camera_attenuation_factor: float  # [0.0, 1.0], 1.0 = optimal clear day
    camera_effective_range_m: float   # Max confident detection range
    camera_fog_contrast_loss_pct: float
    
    # LiDAR Perception Degradation
    lidar_attenuation_factor: float   # [0.0, 1.0]
    lidar_effective_range_m: float
    lidar_backscatter_noise_pct: float
    
    # Radar Perception Degradation
    radar_attenuation_factor: float   # [0.0, 1.0]
    radar_multipath_clutter_pct: float
    
    # Overall Environmental Risk Assessment
    overall_perception_health_pct: float
    recommended_speed_limit_kmh: float
    safety_advisory: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Base friction values for different Indian road surfaces (dry condition)
SURFACE_BASE_FRICTION: Dict[str, float] = {
    "tarmac": 0.85,
    "concrete": 0.88,
    "broken_tarmac": 0.65,
    "dirt": 0.55,
}

# Weather impact multipliers on surface friction
WEATHER_FRICTION_MULTIPLIER: Dict[str, float] = {
    "clear": 1.0,
    "overcast": 0.96,
    "light_rain": 0.78,
    "heavy_rain": 0.52,  # Waterlogging / aquaplaning risk
    "fog": 0.88,        # Condensation / moisture film
}


def calculate_sensor_degradation(
    weather: str = "clear",
    time_of_day: str = "day",
    surface: str = "tarmac",
    road_wetness: Optional[float] = None,
    ego_speed_kmh: float = 40.0,
    visibility_m: Optional[float] = None,
) -> SensorDegradationProfile:
    """
    Compute comprehensive sensor attenuation and braking envelope.

    Utilizes:
      - Koschmieder's Law: gamma_opt = 3.912 / V
      - Beer-Lambert Optical Transmittance: T(d) = exp(-gamma_opt * d)
      - Kim/Kruse NIR LiDAR Atmospheric Extinction & Droplet Backscatter
      - Pavement tire-road adhesion & dynamic stopping envelopes

    Args:
        weather: 'clear', 'overcast', 'light_rain', 'heavy_rain', 'fog'
        time_of_day: 'day', 'dusk', 'night'
        surface: 'tarmac', 'concrete', 'dirt', 'broken_tarmac'
        road_wetness: Optional float [0.0, 1.0]. If None, inferred from weather.
        ego_speed_kmh: Current ego vehicle speed in km/h.
        visibility_m: Optional meteorological optical range in meters.
    """
    w_clean = weather.lower().strip()
    tod_clean = time_of_day.lower().strip()
    surf_clean = surface.lower().strip()

    # Determine road wetness if not provided
    if road_wetness is None:
        if w_clean == "heavy_rain":
            wetness = 0.95
        elif w_clean == "light_rain":
            wetness = 0.50
        elif w_clean == "fog":
            wetness = 0.30
        else:
            wetness = 0.05
    else:
        wetness = max(0.0, min(1.0, float(road_wetness)))

    # Determine meteorological visibility V (meters)
    if visibility_m is not None and visibility_m > 0:
        vis_v = max(15.0, min(1000.0, float(visibility_m)))
    else:
        vis_map = {
            "clear": 320.0,
            "overcast": 220.0,
            "light_rain": 130.0,
            "heavy_rain": 45.0,
            "fog": 32.0,
        }
        vis_v = vis_map.get(w_clean, 250.0)

    # 1. Pavement Friction & Dynamics
    base_mu = SURFACE_BASE_FRICTION.get(surf_clean, 0.85)
    w_mult = WEATHER_FRICTION_MULTIPLIER.get(w_clean, 1.0)
    # Wetness non-linearly degrades friction (adhesion loss + hydrodynamic uplift)
    mu = base_mu * (w_mult * (1.0 - 0.40 * wetness))
    mu = max(0.32, min(0.92, round(mu, 3)))

    g = 9.80665
    max_decel = round(mu * g, 2)

    v_mps = max(0.0, ego_speed_kmh / 3.6)
    t_react = 0.35  # Autonomous ADAS perception-to-brake latency (s)
    stopping_dist = round((v_mps ** 2) / (2.0 * max_decel) + v_mps * t_react, 2)
    safe_buffer = round(max(6.0, stopping_dist * 1.35), 2)

    # 2. Camera Attenuation (Beer-Lambert optical extinction & Koschmieder's Law)
    # Koschmieder extinction coefficient gamma = 3.912 / V
    gamma_opt = 3.912 / vis_v

    # Optical contrast loss at nominal detection distance (40m)
    # C_loss = 1 - exp(-gamma * d)
    fog_contrast_loss = round((1.0 - math.exp(-gamma_opt * 40.0)) * 100.0, 1)

    # Solar illumination factor based on time of day
    tod_scale = 1.0 if tod_clean == "day" else (0.80 if tod_clean == "dusk" else 0.52)

    # Maximum confident optical detection range where transmittance >= 5% (ln(0.05) ~ -3.0)
    d_eff_clear = 3.0 / gamma_opt
    cam_effective_range = round(max(15.0, min(120.0, d_eff_clear * tod_scale)), 1)
    cam_factor = round(max(0.18, min(1.0, (cam_effective_range / 120.0))), 2)

    # 3. LiDAR Attenuation (Kim/Kruse NIR 905nm Extinction & Rain Droplet Backscatter)
    # Wavelength scaling for 905nm: q = 0.585 for medium aerosol/fog
    lambda_scale = (0.905 / 0.550) ** (-0.585)
    gamma_lidar_fog = gamma_opt * lambda_scale

    # Rain droplet backscattering rate: gamma_rain = 0.01 * R^0.6 (R = mm/h rain rate)
    rain_rate_mmh = 28.0 if w_clean == "heavy_rain" else (4.5 if w_clean == "light_rain" else 0.0)
    gamma_rain = 0.01 * (rain_rate_mmh ** 0.6) if rain_rate_mmh > 0 else 0.0
    gamma_lidar_total = gamma_lidar_fog + gamma_rain

    d_lidar_eff = 3.0 / max(0.005, gamma_lidar_total)
    lidar_effective_range = round(max(18.0, min(100.0, d_lidar_eff)), 1)
    lidar_factor = round(max(0.20, min(1.0, lidar_effective_range / 100.0)), 2)
    lidar_noise = round(min(70.0, max(0.0, (1.0 - lidar_factor) * 75.0 + (rain_rate_mmh * 0.4))), 1)

    # 4. Radar Attenuation (Multipath ground bounce & wet pavement clutter)
    radar_clutter = round(min(70.0, 5.0 + wetness * 52.0), 1)
    radar_factor = round(max(0.70, min(1.0, 1.0 - 0.28 * wetness)), 2)

    # 5. Composite Perception Health
    health = round((cam_factor * 0.45 + lidar_factor * 0.35 + radar_factor * 0.20) * 100.0, 1)

    # 6. Recommended Speed Limit & Advisory
    if w_clean == "heavy_rain" or mu < 0.45:
        rec_speed = 30.0
        advisory = "SEVERE: Monsoon deluge & low surface adhesion. Expand following distance by 2.5x; avoid aggressive lane changes."
    elif w_clean == "fog" or cam_effective_range < 35.0:
        rec_speed = 35.0
        advisory = "WARNING: Dense fog optical occlusion. Camera contrast degraded by >60%; rely on radar confirmation."
    elif tod_clean == "night" and surf_clean in ("dirt", "broken_tarmac"):
        rec_speed = 40.0
        advisory = "CAUTION: Unpaved rural road at night. Unpainted road boundaries require drivable area segmentation fallback."
    elif wetness > 0.3:
        rec_speed = 50.0
        advisory = "MODERATE: Wet road surface. Modulate brake onset early to prevent wheel lock."
    else:
        rec_speed = 65.0
        advisory = "NOMINAL: Favorable driving conditions. All sensor modalities operating at peak fidelity."

    return SensorDegradationProfile(
        weather_condition=w_clean,
        time_of_day=tod_clean,
        road_surface=surf_clean,
        road_wetness=round(wetness, 2),
        friction_coefficient=mu,
        max_braking_decel_mps2=max_decel,
        stopping_distance_m=stopping_dist,
        safe_headway_buffer_m=safe_buffer,
        camera_attenuation_factor=round(cam_factor, 2),
        camera_effective_range_m=cam_effective_range,
        camera_fog_contrast_loss_pct=round(fog_contrast_loss, 1),
        lidar_attenuation_factor=round(lidar_factor, 2),
        lidar_effective_range_m=lidar_effective_range,
        lidar_backscatter_noise_pct=round(lidar_noise, 1),
        radar_attenuation_factor=round(radar_factor, 2),
        radar_multipath_clutter_pct=round(radar_clutter, 1),
        overall_perception_health_pct=health,
        recommended_speed_limit_kmh=rec_speed,
        safety_advisory=advisory,
    )
