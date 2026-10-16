import os
import json
import time
import datetime
from google import genai

from scenario_lab.schema.scenario_schema import Scenario
from scenario_lab.nlp.default_filler import DefaultFiller
from scenario_lab.nlp.ambiguity_resolver import AmbiguityResolver

# Models to try in order — fallback if one is overloaded or missing
MODELS = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash']
MAX_RETRIES = 3

class LLMParser:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

        prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_templates', 'system_prompt.txt')
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()

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

    def parse_text(self, text: str) -> tuple[Scenario, list[str]]:
        if not self.client:
            raise ValueError("LLM Parsing requires GEMINI_API_KEY")

        raw_json = self._generate_with_retry(text)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\n{raw_json}")

        # Add explicit metadata
        if 'metadata' not in data:
            data['metadata'] = {}
        data['metadata']['source'] = 'nlp'
        data['metadata']['original_text'] = text
        data['metadata']['created_at'] = datetime.datetime.utcnow().isoformat() + "Z"

        # Resolve ambiguities and apply defaults
        assumptions = AmbiguityResolver.extract_assumptions(text, data)
        data = DefaultFiller.fill_defaults(data)

        # Validate with Pydantic
        scenario = Scenario.model_validate(data)
        return scenario, assumptions
