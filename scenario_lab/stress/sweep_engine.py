import copy
from typing import List, Tuple
from scenario_lab.schema.scenario_schema import Scenario, SweepSpec
from scenario_lab.schema.scenario_schema import SimulationResult

class SweepEngine:
    @classmethod
    def apply_parameter(cls, scenario: Scenario, path: str, value: float) -> Scenario:
        """Applies a value to a scenario via simple JSON path dot notation."""
        data = scenario.model_dump(by_alias=True)
        keys = path.split('.')
        
        current = data
        for k in keys[:-1]:
            # Simple array index parsing e.g. "actors[0]" -> k="actors", index=0
            if '[' in k and k.endswith(']'):
                name, idx_str = k[:-1].split('[')
                idx = int(idx_str)
                current = current[name][idx]
            else:
                current = current[k]
                
        final_k = keys[-1]
        current[final_k] = value
        
        return Scenario.model_validate(data)

    @classmethod
    def find_failure_boundary(cls, runs: List[Tuple[float, SimulationResult]]) -> float:
        """Find the parameter value where collisions first occur."""
        for val, res in sorted(runs, key=lambda x: x[0]):
            if res.collisions > 0:
                return val
        return None
