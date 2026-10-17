# ADAS Scenario Lab

This is the standalone **ADAS Scenario Lab** module, an additive layer over the existing MATLAB-based ADAS-Vision project.

## Features

- **NLP to Scenario:** Translates natural language into canonical JSON Scenario schemas via Google GenAI (Gemini) with automatic offline deterministic semantic rule-based parser fallback (100% uptime with zero external dependencies).
- **OpenStreetMap Live GPS Import:** Ingests live road network geometry directly from Overpass API with Indian roadway classification.
- **Dynamic Climate & Sensor Degradation:** Physics-informed models for Koschmieder optical extinction, Beer-Lambert exponential transmittance, Kruse LiDAR rain backscatter, radar surface clutter, and tire-pavement friction ($\mu$).
- **Triple Simulator Interoperability:** Exports to MATLAB Driving Scenario Designer (`.m`), MathWorks RoadRunner OpenDRIVE (`.xodr` with friction), and ASAM OpenSCENARIO (`.xosc` with `<EnvironmentAction>` weather, illumination, precipitation, fog, and friction scaling).
- **Head-to-Head Comparative Benchmarking:** Evaluates Adaptive ADAS against baseline Western ADAS on identical road hazards.
- **14 Authentic Indian 3D Road Assets:** Detailed bounding boxes, radar cross-sections, and collision profiles for cows, water buffaloes, thelas, police barricades, e-rickshaws, and potholes.
- **Interactive Web Application:** React + Vite UI with live animated scrub player (animating all 7 actor behaviors), dual-trajectory benchmark canvas with hazard overlays, dynamic telemetry HUD, and comparative metric matrices.
- **Headless Execution & Kinematic Fallback:** Runs MATLAB in `-batch` mode or high-fidelity Python kinematic simulation with sub-millisecond execution.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Google API key (for Gemini):
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   ```

## Running the API

Start the FastAPI server:
```bash
uvicorn api.main:app --reload --port 8000
```

### Key API Endpoints

- `POST /api/v1/scenarios/from-text` - Convert text to scenario JSON (Gemini AI)
- `POST /api/v1/scenarios/from-osm` - Ingest live road geometry and weather from GPS coordinates
- `POST /api/v1/scenarios/run` - Queue a scenario for simulation execution
- `GET /api/v1/results/{id}` - Poll for execution results
- `POST /api/v1/scenarios/{id}/benchmark` - Run Head-to-Head Comparative Benchmark (Adaptive vs Baseline)
- `GET /api/v1/weather/sensor-degradation` - Calculate optical/EM attenuation & friction stopping envelope
- `POST /api/v1/scenarios/{id}/export/dsd` - Export MATLAB Driving Scenario Designer (`.m`) script
- `POST /api/v1/scenarios/{id}/export/opendrive` - Export MathWorks RoadRunner OpenDRIVE (`.xodr`) XML
- `POST /api/v1/scenarios/{id}/export/openscenario` - Export MathWorks RoadRunner OpenSCENARIO (`.xosc`) XML
- `GET /api/v1/assets/library` - Retrieve Indian 3D Road Asset Catalogue
- `GET /api/v1/templates` - Get built-in SIH 2025 competition scenarios

## Architecture Note
This module operates in complete isolation from the existing `matlab/` codebase. All auto-generated scripts are written to `scenario_lab/generated/` and cleaned up automatically.
