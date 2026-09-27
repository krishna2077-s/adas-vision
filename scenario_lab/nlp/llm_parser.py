import os
import json
import uuid
import datetime
from google import genai

from scenario_lab.schema.scenario_schema import Scenario
from scenario_lab.nlp.default_filler import DefaultFiller
from scenario_lab.nlp.ambiguity_resolver import AmbiguityResolver

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

    def parse_text(self, text: str) -> tuple[Scenario, list[str]]:
        if not self.client:
            raise ValueError("LLM Parsing requires GEMINI_API_KEY")

        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=text,
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
            )
        )
        
        raw_json = response.text
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\\n{raw_json}")

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
