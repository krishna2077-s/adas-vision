import uuid
from typing import Dict, Any

class DefaultFiller:
    ROAD_WIDTHS = {
        "village_road": 3.0,
        "urban_street": 3.5,
        "highway": 5.0,
        "market_street": 2.5,
        "intersection": 4.0,
        "custom": 3.5
    }

    ROAD_SPEEDS = {
        "village_road": 30.0,
        "urban_street": 40.0,
        "highway": 80.0,
        "market_street": 15.0,
        "intersection": 30.0,
        "custom": 40.0
    }

    @classmethod
    def fill_defaults(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fills standard Indian road defaults into raw LLM JSON output."""

        # UUID
        if 'scenario_id' not in data:
            data['scenario_id'] = f"scen-{uuid.uuid4().hex[:12]}"

        # Metadata
        if 'metadata' not in data:
            data['metadata'] = {}
        data['metadata']['source'] = data['metadata'].get('source', 'nlp')
        if 'name' not in data['metadata']:
            data['metadata']['name'] = "Generated Scenario"
        if 'description' not in data['metadata']:
            data['metadata']['description'] = ""

        # Road — normalize type BEFORE using it for defaults
        if 'road' not in data:
            data['road'] = {}
        road_type = data['road'].get('type', 'urban_street')
        data['road']['type'] = cls._normalize_road_type(road_type)
        road_type = data['road']['type']

        if 'lane_half_width_m' not in data['road']:
            data['road']['lane_half_width_m'] = cls.ROAD_WIDTHS.get(road_type, 3.5)

        if 'length_m' not in data['road']:
            data['road']['length_m'] = 200.0

        # Ego
        if 'ego_vehicle' not in data:
            data['ego_vehicle'] = {}

        if 'initial_speed_kmh' not in data['ego_vehicle']:
            data['ego_vehicle']['initial_speed_kmh'] = cls.ROAD_SPEEDS.get(road_type, 40.0)

        if 'goal_x_m' not in data['ego_vehicle']:
            data['ego_vehicle']['goal_x_m'] = data['road']['length_m'] * 0.9

        # Actors
        if 'actors' not in data or not data['actors']:
            data['actors'] = [{
                "id": "dummy1",
                "class": "cow",
                "initial_x_m": 30.0,
                "initial_y_m": 0.0,
                "behaviour": "static"
            }]

        for i, actor in enumerate(data['actors']):
            if 'id' not in actor:
                actor['id'] = f"act_{i}"
            if 'class' not in actor:
                actor['class'] = "motorcycle"
            if 'initial_x_m' not in actor:
                actor['initial_x_m'] = 25.0
            if 'initial_y_m' not in actor:
                actor['initial_y_m'] = 0.0

        # Sim params
        if 'simulation_parameters' not in data:
            data['simulation_parameters'] = {}
        if 'dt_s' not in data['simulation_parameters']:
            data['simulation_parameters']['dt_s'] = 0.033
        if 'max_time_s' not in data['simulation_parameters']:
            data['simulation_parameters']['max_time_s'] = 35.0

        return data

    @classmethod
    def _normalize_road_type(cls, road_type: str) -> str:
        """Map any LLM-invented road type string to the nearest valid enum value."""
        VALID = {"village_road", "urban_street", "highway", "market_street", "intersection", "custom"}
        if road_type in VALID:
            return road_type

        road_lower = road_type.lower().replace("-", "_").replace(" ", "_")
        if road_lower in VALID:
            return road_lower

        # Keyword mapping for common LLM hallucinations
        KEYWORD_MAP = {
            "village": "village_road",
            "rural": "village_road",
            "narrow": "village_road",
            "bridge": "village_road",
            "dirt": "village_road",
            "highway": "highway",
            "expressway": "highway",
            "motorway": "highway",
            "freeway": "highway",
            "market": "market_street",
            "bazaar": "market_street",
            "crowded": "market_street",
            "intersection": "intersection",
            "junction": "intersection",
            "crossroad": "intersection",
            "urban": "urban_street",
            "city": "urban_street",
            "street": "urban_street",
            "road": "village_road",
        }
        for keyword, mapped in KEYWORD_MAP.items():
            if keyword in road_lower:
                print(f"[DefaultFiller] Normalized road type '{road_type}' -> '{mapped}'")
                return mapped

        print(f"[DefaultFiller] Unknown road type '{road_type}', defaulting to 'custom'")
        return "custom"
