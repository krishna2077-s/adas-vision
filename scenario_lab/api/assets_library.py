"""
assets_library.py -- Catalogue of Indian 3D Road Assets for ADAS Scenario Lab.

Contains geometric dimensions, collision profiles, and context descriptions
for typical Indian road entities (Cattle, Barriers, Surface Anomalies, Vehicles).
"""

from typing import List, Dict, Any

INDIAN_ROAD_ASSETS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # Cattle & Animals
    # -------------------------------------------------------------------------
    {
        "id": "cow",
        "class": "cow",
        "category": "Cattle",
        "name": "Indian Desi Cow (Bos indicus)",
        "dimensions": {"length_m": 2.2, "width_m": 1.2, "height_m": 1.5},
        "collision_classification": "dynamic_obstacle",
        "radar_cross_section": "medium",
        "mass_kg": 350,
        "typical_speed_kmh": 3.0,
        "description": "Unpredictable livestock roaming or sitting on rural and semi-urban carriageways; slow lateral crossing dynamics and reflective eye shine.",
    },
    {
        "id": "buffalo",
        "class": "buffalo",
        "category": "Cattle",
        "name": "Indian Water Buffalo (Murrah)",
        "dimensions": {"length_m": 2.4, "width_m": 1.4, "height_m": 1.6},
        "collision_classification": "dynamic_obstacle",
        "radar_cross_section": "large",
        "mass_kg": 500,
        "typical_speed_kmh": 2.5,
        "description": "Heavy dark-coated bovine with low optical contrast at dusk/night, highly stable mass requiring emergency braking.",
    },
    {
        "id": "goat",
        "class": "goat",
        "category": "Cattle",
        "name": "Desi Goat / Herd Member",
        "dimensions": {"length_m": 0.9, "width_m": 0.4, "height_m": 0.75},
        "collision_classification": "vulnerable_road_user",
        "radar_cross_section": "low",
        "mass_kg": 35,
        "typical_speed_kmh": 6.0,
        "description": "Small, agile animal prone to erratic darting movements across village roads in packs.",
    },
    {
        "id": "dog",
        "class": "dog",
        "category": "Cattle",
        "name": "Indian Stray Pariah Dog",
        "dimensions": {"length_m": 0.8, "width_m": 0.35, "height_m": 0.55},
        "collision_classification": "vulnerable_road_user",
        "radar_cross_section": "low",
        "mass_kg": 18,
        "typical_speed_kmh": 12.0,
        "description": "Ubiquitous agile street canine with sudden high-acceleration lateral dash behaviour.",
    },

    # -------------------------------------------------------------------------
    # Barriers & Road Hazards
    # -------------------------------------------------------------------------
    {
        "id": "police_barricade",
        "class": "police_barricade",
        "category": "Barriers",
        "name": "Police Metal Barricade",
        "dimensions": {"length_m": 2.0, "width_m": 0.5, "height_m": 1.2},
        "collision_classification": "static_barrier",
        "radar_cross_section": "high_metallic",
        "mass_kg": 75,
        "typical_speed_kmh": 0.0,
        "description": "Yellow/blue painted structural iron mesh barrier deployed in zigzag configurations for nakabandi security checkpoints.",
    },
    {
        "id": "construction_barrier",
        "class": "construction_barrier",
        "category": "Barriers",
        "name": "Construction / Metro Barrier",
        "dimensions": {"length_m": 2.2, "width_m": 0.6, "height_m": 1.4},
        "collision_classification": "static_barrier",
        "radar_cross_section": "high_metallic",
        "mass_kg": 120,
        "typical_speed_kmh": 0.0,
        "description": "Corrugated GI sheet or heavy modular barrier used around flyover construction sites and lane closures.",
    },
    {
        "id": "traffic_cone",
        "class": "barricade",
        "category": "Barriers",
        "name": "Reflective Traffic Cone",
        "dimensions": {"length_m": 0.4, "width_m": 0.4, "height_m": 0.75},
        "collision_classification": "static_barrier",
        "radar_cross_section": "low",
        "mass_kg": 4,
        "typical_speed_kmh": 0.0,
        "description": "High-visibility orange cone with retro-reflective bands marking temporary road maintenance or diversions.",
    },

    # -------------------------------------------------------------------------
    # Surface Anomalies
    # -------------------------------------------------------------------------
    {
        "id": "pothole",
        "class": "pothole",
        "category": "Surface Anomalies",
        "name": "Monsoon Pothole (Khadha)",
        "dimensions": {"length_m": 1.2, "width_m": 1.0, "height_m": 0.15},
        "collision_classification": "surface_hazard",
        "radar_cross_section": "negative_relief",
        "mass_kg": 0,
        "typical_speed_kmh": 0.0,
        "description": "Severe asphalt cavity formed by water ingress and heavy axle loads, causing sudden driver swerves and suspension impact.",
    },
    {
        "id": "broken_tarmac",
        "class": "obstacle",
        "category": "Surface Anomalies",
        "name": "Broken Tarmac Section",
        "dimensions": {"length_m": 5.0, "width_m": 3.0, "height_m": 0.08},
        "collision_classification": "surface_hazard",
        "radar_cross_section": "diffuse",
        "mass_kg": 0,
        "typical_speed_kmh": 0.0,
        "description": "Stripped asphalt and loose gravel patchwork degrading vehicle braking grip and generating high acoustic chassis noise.",
    },
    {
        "id": "speed_bump",
        "class": "obstacle",
        "category": "Surface Anomalies",
        "name": "Unmarked Speed Bump / Breaker",
        "dimensions": {"length_m": 1.5, "width_m": 3.5, "height_m": 0.18},
        "collision_classification": "surface_hazard",
        "radar_cross_section": "geometric_bump",
        "mass_kg": 0,
        "typical_speed_kmh": 0.0,
        "description": "Steep asphalt/concrete speed hump lacking IRC standard reflective zebra paint, requiring sharp deceleration.",
    },

    # -------------------------------------------------------------------------
    # Indian Vehicles
    # -------------------------------------------------------------------------
    {
        "id": "auto_rickshaw",
        "class": "auto_rickshaw",
        "category": "Indian Vehicles",
        "name": "CNG Three-Wheeled Auto-Rickshaw",
        "dimensions": {"length_m": 2.8, "width_m": 1.4, "height_m": 1.8},
        "collision_classification": "vehicle",
        "radar_cross_section": "medium",
        "mass_kg": 400,
        "typical_speed_kmh": 35.0,
        "description": "Iconic three-wheeled commercial paratransit known for agile lateral weaving, abrupt stops, and soft canvas side curtains.",
    },
    {
        "id": "e_rickshaw",
        "class": "e_rickshaw",
        "category": "Indian Vehicles",
        "name": "Battery Electric Rickshaw (Toto)",
        "dimensions": {"length_m": 2.7, "width_m": 1.3, "height_m": 1.8},
        "collision_classification": "vehicle",
        "radar_cross_section": "medium",
        "mass_kg": 320,
        "typical_speed_kmh": 22.0,
        "description": "Low-speed electric passenger carrier with high passenger occupancy and slow acceleration in mixed urban traffic.",
    },
    {
        "id": "pushcart",
        "class": "pushcart",
        "category": "Indian Vehicles",
        "name": "Handcart / Street Vendor Pushcart (Thela)",
        "dimensions": {"length_m": 2.2, "width_m": 1.4, "height_m": 1.2},
        "collision_classification": "vulnerable_road_user",
        "radar_cross_section": "low_wood_metal",
        "mass_kg": 150,
        "typical_speed_kmh": 4.0,
        "description": "Non-motorized four-wheeled wooden cart pushed by street vendor, occupying road shoulders and pedestrian lanes.",
    },
    {
        "id": "motorcycle",
        "class": "motorcycle",
        "category": "Indian Vehicles",
        "name": "Commuter Two-Wheeler / Scooter (100cc-150cc)",
        "dimensions": {"length_m": 2.0, "width_m": 0.75, "height_m": 1.25},
        "collision_classification": "vehicle",
        "radar_cross_section": "low",
        "mass_kg": 130,
        "typical_speed_kmh": 45.0,
        "description": "Ubiquitous commuter bike performing frequent lane-filtering and rapid cut-in manoeuvres between larger vehicles.",
    },
]


def get_asset_library() -> Dict[str, Any]:
    """Return categorized Indian road assets catalogue."""
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for asset in INDIAN_ROAD_ASSETS:
        cat = asset["category"]
        categories.setdefault(cat, []).append(asset)
    return {
        "assets": INDIAN_ROAD_ASSETS,
        "categories": categories,
        "total_count": len(INDIAN_ROAD_ASSETS),
    }
