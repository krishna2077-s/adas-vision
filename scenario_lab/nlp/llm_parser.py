import os
import json
import time
import datetime

try:
    from google import genai
except (ImportError, AttributeError, Exception):
    genai = None

from scenario_lab.schema.scenario_schema import Scenario
from scenario_lab.nlp.default_filler import DefaultFiller
from scenario_lab.nlp.ambiguity_resolver import AmbiguityResolver

# Models to try in order — fallback if one is overloaded or missing
MODELS = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash']
MAX_RETRIES = 3

class LLMParser:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[LLMParser] Warning: Failed to initialize genai client: {e}")
                self.client = None
        else:
            self.client = None

        prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_templates', 'system_prompt.txt')
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = ""

    def _generate_with_retry(self, text: str) -> str:
        """Try each model with exponential backoff on 503 overload errors."""
        last_error = None
        for model in MODELS:
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=text,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=self.system_prompt,
                            response_mime_type="application/json",
                        )
                    )
                    print(f"[LLMParser] Success with model: {model}")
                    return response.text
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    is_overload = '503' in err_str or 'UNAVAILABLE' in err_str or 'high demand' in err_str
                    is_not_found = '404' in err_str or 'NOT_FOUND' in err_str or 'no longer available' in err_str
                    if is_not_found:
                        print(f"[LLMParser] {model} not found, trying next model...")
                        break  # Skip to next model
                    if is_overload and attempt < MAX_RETRIES - 1:
                        wait = 2 ** attempt  # 1s, 2s
                        print(f"[LLMParser] {model} overloaded (attempt {attempt+1}), retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    print(f"[LLMParser] {model} failed: {err_str[:120]}, trying next...")
                    break
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def _parse_text_offline(self, text: str) -> tuple[Scenario, list[str]]:
        """
        Deterministic rule-based Indian ADAS semantic parser.
        Extracts road geometry, climate/weather, ego kinematics, and dynamic actors
        from natural language prompts without requiring an external LLM API key.
        """
        import re
        txt_low = text.lower()
        assumptions = ["Generated via deterministic Indian ADAS semantic NLP parser (offline fallback)."]

        # 1. Road classification
        if any(w in txt_low for w in ["highway", "expressway", "nh48", "nh-48", "nh", "flyover", "toll"]):
            road_type = "highway"
            lane_w = 7.0
            r_len = 260.0
            def_speed = 80.0
        elif any(w in txt_low for w in ["village", "rural", "dirt", "unpaved", "nashik", "campus"]):
            road_type = "village_road"
            lane_w = 3.5
            r_len = 200.0
            def_speed = 35.0
        elif any(w in txt_low for w in ["market", "bazaar", "chandni", "crowd", "narrow"]):
            road_type = "market_street"
            lane_w = 3.0
            r_len = 160.0
            def_speed = 25.0
        elif any(w in txt_low for w in ["junction", "intersection", "silk board", "roundabout", "chowk"]):
            road_type = "intersection"
            lane_w = 4.0
            r_len = 180.0
            def_speed = 30.0
        else:
            road_type = "urban_street"
            lane_w = 3.5
            r_len = 200.0
            def_speed = 40.0

        # Surface
        if any(w in txt_low for w in ["pothole", "broken", "rough", "crater"]):
            surface = "broken_tarmac"
        elif any(w in txt_low for w in ["dirt", "mud", "unpaved"]):
            surface = "dirt"
        elif "concrete" in txt_low:
            surface = "concrete"
        else:
            surface = "tarmac"

        # 2. Weather & Illumination
        if any(w in txt_low for w in ["heavy rain", "monsoon", "deluge", "torrential", "downpour", "waterlog"]):
            weather = "heavy_rain"
            vis_m = 45.0
        elif any(w in txt_low for w in ["rain", "drizzle", "wet", "puddle"]):
            weather = "light_rain"
            vis_m = 120.0
        elif any(w in txt_low for w in ["fog", "smog", "mist", "haze"]):
            weather = "fog"
            vis_m = 35.0
        elif any(w in txt_low for w in ["cloudy", "overcast"]):
            weather = "overcast"
            vis_m = 220.0
        else:
            weather = "clear"
            vis_m = 300.0

        if any(w in txt_low for w in ["night", "dark", "midnight"]):
            tod = "night"
        elif any(w in txt_low for w in ["dusk", "evening", "sunset", "twilight"]):
            tod = "dusk"
        else:
            tod = "day"

        # 3. Ego Speed
        spd_match = re.search(r"(\d+)\s*(?:km/?h|kmph)", txt_low)
        ego_speed = float(spd_match.group(1)) if spd_match else def_speed

        # 4. Extract Actor Entities & Behaviors
        actors_data = []

        # Helper mapping of classes to search keywords
        class_patterns = [
            ("auto_rickshaw", ["auto", "rickshaw", "tuk-tuk", "three-wheeler"]),
            ("cow", ["cow", "cattle", "bull", "bovine", "stray cow"]),
            ("buffalo", ["buffalo"]),
            ("pedestrian", ["pedestrian", "person", "child", "man", "woman", "crossing person"]),
            ("motorcycle", ["motorcycle", "bike", "scooter", "two-wheeler", "biker"]),
            ("truck", ["truck", "lorry", "container"]),
            ("bus", ["bus"]),
            ("barricade", ["barricade", "barrier", "nakabandi"]),
            ("pothole", ["pothole", "road anomaly"]),
            ("thela", ["thela", "pushcart", "handcart", "vendor cart"]),
        ]

        found_classes = []
        for cls_name, keywords in class_patterns:
            if any(k in txt_low for k in keywords):
                found_classes.append(cls_name)

        if not found_classes:
            found_classes = ["auto_rickshaw", "cow"]
            assumptions.append("Defaulted to standard Indian road mixed traffic (auto-rickshaw and cow).")

        for idx, act_cls in enumerate(found_classes[:4]):
            act_id = f"{act_cls}_{idx+1}"
            x_pos = 35.0 + idx * 25.0

            # Behavior inference
            if any(w in txt_low for w in ["wrong side", "wrong-way", "opposite", "oncoming", "against"]):
                beh = "wrong_side" if idx == 0 else "constant_velocity"
            elif any(w in txt_low for w in ["cut in", "cut-in", "swerving", "lane change"]):
                beh = "cut_in" if idx == 0 else "crossing"
            elif any(w in txt_low for w in ["cross", "crossing", "jaywalk", "stray"]):
                beh = "crossing"
            elif any(w in txt_low for w in ["sudden stop", "brakes", "braking", "halt"]):
                beh = "sudden_stop"
            elif any(w in txt_low for w in ["parked", "standing", "stationary", "stalled", "static"]):
                beh = "static"
            elif act_cls in ("barricade", "pothole"):
                beh = "static"
            elif act_cls in ("cow", "buffalo"):
                beh = "crossing"
            elif act_cls == "auto_rickshaw":
                beh = "cut_in"
            else:
                beh = "constant_velocity"

            # Position & velocity parametrization based on behavior
            if beh == "cut_in":
                y_pos = 2.8 if idx % 2 == 0 else -2.8
                vx = min(12.0, ego_speed * 0.4)
                vy = -1.2 if y_pos > 0 else 1.2
                y_tgt = 0.0
                trig_s = 1.2
            elif beh == "crossing":
                y_pos = -3.2 if idx % 2 == 0 else 3.2
                vx = 0.2
                vy = 0.9 if y_pos < 0 else -0.9
                y_tgt = -y_pos * 0.6
                trig_s = 1.8
            elif beh == "wrong_side":
                y_pos = 0.5
                vx = -7.5
                vy = 0.0
                y_tgt = None
                trig_s = 0.0
            elif beh == "sudden_stop":
                y_pos = 0.0
                vx = min(14.0, ego_speed * 0.5)
                vy = 0.0
                y_tgt = None
                trig_s = 1.5
            elif beh == "static":
                y_pos = 0.2 if idx == 0 else -1.5
                vx = 0.0
                vy = 0.0
                y_tgt = None
                trig_s = 0.0
            else:
                y_pos = 1.2 if idx % 2 == 0 else -1.2
                vx = min(10.0, ego_speed * 0.45)
                vy = 0.0
                y_tgt = None
                trig_s = 0.0

            act_dict = {
                "id": act_id,
                "class": act_cls,
                "initial_x_m": round(x_pos, 1),
                "initial_y_m": round(y_pos, 1),
                "initial_vx_mps": round(vx, 1),
                "initial_vy_mps": round(vy, 1),
                "behaviour": beh,
            }
            if trig_s is not None:
                act_dict["behaviour_trigger_s"] = trig_s
            if y_tgt is not None:
                act_dict["behaviour_y_target_m"] = round(y_tgt, 1)

            actors_data.append(act_dict)

        # Build payload
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        raw_data = {
            "scenario_id": f"scen-nlp-{int(time.time())}",
            "metadata": {
                "name": text[:45].strip().title() if len(text) > 5 else "Indian Road NLP Scenario",
                "description": f"Generated from text: {text}",
                "source": "nlp",
                "created_at": now_utc,
                "tags": [road_type, weather, surface, tod] + found_classes,
                "original_text": text,
            },
            "road": {
                "type": road_type,
                "lane_half_width_m": lane_w,
                "length_m": r_len,
                "surface": surface,
                "markings": "none" if road_type == "village_road" else "broken_center",
                "has_shoulder": True,
            },
            "environment": {
                "weather": weather,
                "time_of_day": tod,
                "visibility_m": vis_m,
            },
            "ego_vehicle": {
                "initial_speed_kmh": round(ego_speed, 1),
                "initial_x_m": 0.0,
                "initial_y_m": 0.0,
                "goal_x_m": round(r_len * 0.9, 1),
                "mode": "adaptive",
            },
            "actors": actors_data,
            "simulation_parameters": {
                "dt_s": 0.033,
                "max_time_s": 35.0,
            },
        }

        raw_data = DefaultFiller.fill_defaults(raw_data)
        scenario = Scenario.model_validate(raw_data)
        return scenario, assumptions

    def parse_text(self, text: str) -> tuple[Scenario, list[str]]:
        # If no client / API key, immediately use offline deterministic semantic parser
        if not self.client:
            return self._parse_text_offline(text)

        try:
            raw_json = self._generate_with_retry(text)
            data = json.loads(raw_json)
            if 'metadata' not in data:
                data['metadata'] = {}
            data['metadata']['source'] = 'nlp'
            data['metadata']['original_text'] = text
            data['metadata']['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            assumptions = AmbiguityResolver.extract_assumptions(text, data)
            data = DefaultFiller.fill_defaults(data)
            scenario = Scenario.model_validate(data)
            return scenario, assumptions
        except Exception as e:
            print(f"[LLMParser] LLM online generation failed: {e}. Falling back to offline semantic parser.")
            return self._parse_text_offline(text)
