# ADAS Scenario Lab

This is the standalone **ADAS Scenario Lab** module, an additive layer over the existing MATLAB-based ADAS-Vision project.

## Features

- **NLP to Scenario:** Translates natural language into a canonical JSON Scenario schema using Google GenAI (Gemini).
- **MATLAB Adapter:** Uses Jinja2 templates to convert Canonical JSON into a valid MATLAB script containing the `cfg` struct.
- **Headless Execution:** Runs MATLAB in `-batch` mode in the background.
- **Deterministic Explanations:** Extracts results via `export_result_json.m` and provides data-backed explanations without LLM hallucination.
- **Stress Testing:** Parameter sweep engine to detect failure boundaries.
-
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

- `POST /api/v1/scenarios/from-text` - Convert text to scenario JSON
- `POST /api/v1/scenarios/run` - Queue a scenario for MATLAB execution
- `GET /api/v1/results/{id}` - Poll for execution results
- `POST /api/v1/scenarios/stress-test` - Run a parameter sweep
- `GET /api/v1/templates` - Get built-in templates

## Architecture Note
This module operates in complete isolation from the existing `matlab/` codebase. The only modification to the original code was the addition of `matlab/export_result_json.m`. All auto-generated scripts are written to `scenario_lab/generated/` and cleaned up automatically.
