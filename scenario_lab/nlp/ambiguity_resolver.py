from typing import Dict, Any, List

class AmbiguityResolver:
    @classmethod
    def extract_assumptions(cls, original_text: str, data: Dict[str, Any]) -> List[str]:
        """Detects implicit assumptions made during NLP generation."""
        assumptions = []
        text_lower = original_text.lower()
        
        # Road assumptions
        road = data.get('road', {})
        if road.get('type') and road.get('type').replace('_', ' ') not in text_lower:
            if "road" not in text_lower and "street" not in text_lower and "highway" not in text_lower:
                assumptions.append(f"Road type inferred as '{road.get('type')}'")
                
        # Ego assumptions
        ego = data.get('ego_vehicle', {})
        speed = str(ego.get('initial_speed_kmh', ''))
        if speed and speed not in text_lower:
            assumptions.append(f"Ego speed assumed to be {speed} km/h (not stated)")
            
        # Actor distance assumptions
        for actor in data.get('actors', []):
            dist = str(actor.get('initial_x_m', ''))
            if dist and dist not in text_lower:
                assumptions.append(f"Actor '{actor.get('class')}' distance assumed as {dist}m")
                
        return assumptions
