import os
import sys
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

# Ensure the parent directory of scenario_lab is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scenario_lab.schema.scenario_schema import Scenario, SweepSpec
from scenario_lab.storage.store import ScenarioStore, ResultStore
from scenario_lab.adapter.matlab_adapter import MatlabAdapter
from scenario_lab.adapter.matlab_runner import MatlabRunner
from scenario_lab.adapter.explanations import build_explanation

from scenario_lab.nlp.llm_parser import LLMParser
from pydantic import BaseModel
from typing import List

app = FastAPI(title="ADAS Scenario Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, 'storage', 'db')
MATLAB_DIR = os.path.join(os.path.dirname(BASE_DIR), 'matlab')
GENERATED_DIR = os.path.join(BASE_DIR, 'generated')

scenario_store = ScenarioStore(DB_DIR)
result_store = ResultStore(DB_DIR)
adapter = MatlabAdapter(MATLAB_DIR, GENERATED_DIR)
runner = MatlabRunner(MATLAB_DIR)
llm_parser = LLMParser()  # expects GEMINI_API_KEY env var

# In-memory execution state
execution_states: Dict[str, str] = {}

class TextScenarioRequest(BaseModel):
    text: str

class TextScenarioResponse(BaseModel):
    scenario: Scenario
    assumptions: List[str]

@app.post("/api/v1/scenarios/from-text", response_model=TextScenarioResponse)
async def generate_scenario_from_text(req: TextScenarioRequest):
    try:
        scenario, assumptions = llm_parser.parse_text(req.text)
        return TextScenarioResponse(scenario=scenario, assumptions=assumptions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def run_simulation_task(scenario: Scenario, result_id: str):

    try:
        script_path, out_json_path = adapter.generate_script(scenario)
        raw_result = runner.run(script_path, out_json_path)
        
        sim_result = build_explanation(raw_result, scenario.scenario_id, scenario.ego_vehicle.mode)
        sim_result.result_id = result_id  # override to map to API request
        result_store.save(sim_result)
        
        execution_states[result_id] = "complete"
    except Exception as e:
        execution_states[result_id] = f"error: {str(e)}"

@app.post("/api/v1/scenarios/run")
async def run_scenario(scenario: Scenario, background_tasks: BackgroundTasks):
    scenario_store.save(scenario)
    
    import uuid
    result_id = str(uuid.uuid4())
    execution_states[result_id] = "running"
    
    background_tasks.add_task(run_simulation_task, scenario, result_id)
    return {"result_id": result_id, "status": "running"}

@app.get("/api/v1/scenarios")
async def list_scenarios(tag: str = None, source: str = None):
    return scenario_store.list_all(tag=tag, source=source)

@app.get("/api/v1/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str):
    scenario = scenario_store.get(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@app.get("/api/v1/results/{result_id}")
async def get_result(result_id: str):
    result = result_store.get(result_id)
    if result:
        return result.model_dump()
        
    state = execution_states.get(result_id)
    if not state:
        raise HTTPException(status_code=404, detail="Result not found")
        
    if state == "running":
        return {"status": "running"}
    elif state.startswith("error"):
        return {"status": "error", "message": state}

@app.get("/api/v1/templates")
async def list_templates():
    templates_path = os.path.join(BASE_DIR, 'templates', 'indian_road_templates.json')
    if os.path.exists(templates_path):
        with open(templates_path, 'r') as f:
            return json.load(f)
    return []

@app.post("/api/v1/scenarios/stress-test")
async def run_stress_test(sweep_spec: SweepSpec, background_tasks: BackgroundTasks):
    from scenario_lab.stress.sweep_engine import SweepEngine
    
    base_scenario = scenario_store.get(sweep_spec.base_scenario_id)
    if not base_scenario:
        raise HTTPException(status_code=404, detail="Base scenario not found")
        
    sweep_id = str(uuid.uuid4())
    execution_states[sweep_id] = "running"
    
    # Normally we'd run this asynchronously, for the hackathon we just kick off individual runs
    run_ids = []
    for val in sweep_spec.values:
        variant = SweepEngine.apply_parameter(base_scenario, sweep_spec.parameter_path, val)
        variant.scenario_id = f"{base_scenario.scenario_id}_sweep_{val}"
        variant.parent_id = base_scenario.scenario_id
        
        run_id = str(uuid.uuid4())
        run_ids.append(run_id)
        
        scenario_store.save(variant)
        background_tasks.add_task(run_simulation_task, variant, run_id)
        
    return {"sweep_id": sweep_id, "run_ids": run_ids}


