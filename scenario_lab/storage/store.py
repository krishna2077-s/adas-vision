import os
import json
from scenario_lab.schema.scenario_schema import Scenario, SimulationResult

class BaseStore:
    def __init__(self, base_dir: str, entity_name: str):
        self.dir = os.path.join(base_dir, entity_name)
        os.makedirs(self.dir, exist_ok=True)
        self.index_path = os.path.join(self.dir, 'index.json')
        if not os.path.exists(self.index_path):
            self._save_index([])

    def _load_index(self):
        with open(self.index_path, 'r') as f:
            return json.load(f)

    def _save_index(self, index):
        with open(self.index_path, 'w') as f:
            json.dump(index, f)

    def _get_path(self, entity_id: str):
        return os.path.join(self.dir, f"{entity_id}.json")

class ScenarioStore(BaseStore):
    def __init__(self, base_dir: str):
        super().__init__(base_dir, "scenarios")

    def save(self, scenario: Scenario):
        data = scenario.model_dump(by_alias=True)
        with open(self._get_path(scenario.scenario_id), 'w') as f:
            json.dump(data, f, indent=2)
            
        index = self._load_index()
        # Remove if exists
        index = [i for i in index if i["id"] != scenario.scenario_id]
        index.append({
            "id": scenario.scenario_id,
            "name": scenario.metadata.name,
            "tags": scenario.metadata.tags,
            "source": scenario.metadata.source,
            "created_at": scenario.metadata.created_at
        })
        self._save_index(index)

    def get(self, scenario_id: str) -> Scenario:
        path = self._get_path(scenario_id)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            data = json.load(f)
            # Pydantic v2 loading
            return Scenario.model_validate(data)

    def list_all(self, tag=None, source=None):
        index = self._load_index()
        if tag:
            index = [i for i in index if tag in i["tags"]]
        if source:
            index = [i for i in index if i["source"] == source]
        return index

class ResultStore(BaseStore):
    def __init__(self, base_dir: str):
        super().__init__(base_dir, "results")

    def save(self, result: SimulationResult):
        data = result.model_dump()
        with open(self._get_path(result.result_id), 'w') as f:
            json.dump(data, f, indent=2)
            
        index = self._load_index()
        index = [i for i in index if i["id"] != result.result_id]
        index.append({
            "id": result.result_id,
            "scenario_id": result.scenario_id,
            "status": result.status,
            "arrived": result.arrived,
            "executed_at": result.executed_at
        })
        self._save_index(index)

    def get(self, result_id: str) -> SimulationResult:
        path = self._get_path(result_id)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return SimulationResult.model_validate(json.load(f))
