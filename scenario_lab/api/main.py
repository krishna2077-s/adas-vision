import os
import sys
import uuid
import json
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Ensure the parent directory of scenario_lab is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scenario_lab.schema.scenario_schema import Scenario, SweepSpec
from scenario_lab.storage.store import ScenarioStore, ResultStore
from scenario_lab.adapter.matlab_adapter import MatlabAdapter
from scenario_lab.adapter.matlab_runner import MatlabRunner
from scenario_lab.adapter.explanations import build_explanation

from scenario_lab.nlp.llm_parser import LLMParser
from scenario_lab.osm.osm_importer import import_from_coordinates
from scenario_lab.weather.weather_provider import get_weather, estimate_weather
from scenario_lab.export.dsd_exporter import export_to_dsd_script
from scenario_lab.export.opendrive_exporter import export_to_opendrive
from scenario_lab.api.assets_library import get_asset_library

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
        scenario_store.save(scenario)
        return TextScenarioResponse(scenario=scenario, assumptions=assumptions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def run_simulation_task(scenario: Scenario, result_id: str):

    try:
        script_path, out_json_path = adapter.generate_script(scenario)
        raw_result = runner.run(script_path, out_json_path, scenario=scenario)
        
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

def _resolve_scenario(scenario_id: str) -> Optional[Scenario]:
    scenario = scenario_store.get(scenario_id)
    if scenario:
        return scenario
    templates_path = os.path.join(BASE_DIR, 'templates', 'indian_road_templates.json')
    if os.path.exists(templates_path):
        try:
            with open(templates_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
                for t in templates:
                    if t.get("id") == scenario_id:
                        return Scenario.model_validate(t["scenario"])
        except Exception:
            pass
    return None

@app.get("/api/v1/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str):
    scenario = _resolve_scenario(scenario_id)
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
        variant = SweepEngine.apply_parameter(base_scenario, sweep_spec.sweep_path, val)
        variant.scenario_id = f"{base_scenario.scenario_id}_sweep_{val}"
        variant.parent_id = base_scenario.scenario_id
        
        run_id = str(uuid.uuid4())
        run_ids.append(run_id)
        
        scenario_store.save(variant)
        background_tasks.add_task(run_simulation_task, variant, run_id)
        
    return {"sweep_id": sweep_id, "run_ids": run_ids}


# ---------------------------------------------------------------------------
# OpenStreetMap Scenario Import Endpoint
# ---------------------------------------------------------------------------
class OSMImportRequest(BaseModel):
    lat: float
    lon: float
    radius: float = 300.0
    attach_weather: bool = True

@app.post("/api/v1/scenarios/from-osm")
async def import_scenario_from_osm(req: OSMImportRequest):
    try:
        network = import_from_coordinates(req.lat, req.lon, radius_m=req.radius)
    except Exception:
        from scenario_lab.osm.osm_importer import bbox_from_point, _generate_fallback_network
        bbox = bbox_from_point(req.lat, req.lon, req.radius)
        network = _generate_fallback_network(bbox[0], bbox[1], bbox[2], bbox[3], req.lat, req.lon)

    env = None
    weather_dict = None
    if req.attach_weather:
        try:
            wc = get_weather(req.lat, req.lon)
            env = wc.to_environment()
            weather_dict = wc.to_dict()
        except Exception:
            env = None

    scenario = network.to_scenario(environment=env)
    if scenario is None:
        raise HTTPException(
            status_code=400,
            detail="No valid road network could be constructed from OSM data at these coordinates."
        )

    scenario.metadata.source = "osm"
    scenario_store.save(scenario)

    summary: Dict[str, Any] = {
        "segment_count": len(network.segments),
        "total_length_m": round(network.total_length_m, 1),
        "road_type_summary": network.road_type_summary,
        "center": {"lat": req.lat, "lon": req.lon},
        "radius_m": req.radius,
        "weather": weather_dict,
    }
    return {"scenario": scenario, "summary": summary}


# ---------------------------------------------------------------------------
# Weather Condition & Environment Endpoint
# ---------------------------------------------------------------------------
@app.get("/api/v1/weather")
async def get_weather_endpoint(
    lat: float,
    lon: float,
    month: Optional[int] = None,
    hour: Optional[int] = None,
):
    try:
        if month is not None or hour is not None:
            wc = estimate_weather(lat, lon, month=month, hour=hour)
        else:
            wc = get_weather(lat, lon)

        env = wc.to_environment()
        env_dict = env.model_dump() if env else None

        return {
            "weather": wc.to_dict(),
            "environment": env_dict,
            **wc.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# MATLAB Driving Scenario Designer (.m) Script Export Endpoint
# ---------------------------------------------------------------------------
@app.api_route("/api/v1/scenarios/{scenario_id}/export/dsd", methods=["GET", "POST"])
async def export_scenario_dsd(scenario_id: str, scenario_body: Optional[Scenario] = None):
    scenario = scenario_body or _resolve_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")

    if scenario_body:
        scenario_store.save(scenario_body)

    try:
        from scenario_lab.export.dsd_exporter import _sanitize_func_name
        script = export_to_dsd_script(scenario)
        func_name = _sanitize_func_name(f"dsd_{scenario.scenario_id}")
        filename = f"{func_name}.m"
        return {"script": script, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# MathWorks RoadRunner / ASAM OpenDRIVE (.xodr) XML Export Endpoint
# ---------------------------------------------------------------------------
@app.api_route("/api/v1/scenarios/{scenario_id}/export/opendrive", methods=["GET", "POST"])
async def export_scenario_opendrive(scenario_id: str, scenario_body: Optional[Scenario] = None):
    scenario = scenario_body or _resolve_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")

    if scenario_body:
        scenario_store.save(scenario_body)

    try:
        xodr_xml = export_to_opendrive(scenario)
        filename = f"{scenario.scenario_id}.xodr"
        return {"xodr": xodr_xml, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Indian 3D Road Asset Library Catalogue Endpoint
# ---------------------------------------------------------------------------
@app.get("/api/v1/assets/library")
async def get_assets_library_endpoint():
    return get_asset_library()


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "simulation_engine": "kinematics_python",
        "features": ["osm_import", "climate_weather", "dsd_matlab", "opendrive_roadrunner", "indian_assets"]
    }


# ---------------------------------------------------------------------------
# Static Frontend Single Page Application (SPA) Serving for Web Deployment
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        target = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))



