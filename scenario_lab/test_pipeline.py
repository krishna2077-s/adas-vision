import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenario_lab.nlp.llm_parser import LLMParser
from scenario_lab.adapter.matlab_adapter import MatlabAdapter
from scenario_lab.adapter.matlab_runner import MatlabRunner
from scenario_lab.adapter.explanations import build_explanation

def test_full_pipeline():
    print("=== ADAS Scenario Lab End-to-End Test ===")
    
    # 1. NLP Parsing
    print("\\n[1] Parsing Natural Language...")
    try:
        parser = LLMParser()
        text = "A motorcycle approaches head-on on the wrong side of the village road at 30 km/h, about 50m ahead."
        scenario, assumptions = parser.parse_text(text)
        print("✅ Scenario parsed successfully.")
        print(f"Scenario Name: {scenario.metadata.name}")
        print("Assumptions made:")
        for a in assumptions:
            print(f" - {a}")
    except ValueError as e:
        print(f"❌ NLP Parsing failed (Check GEMINI_API_KEY): {e}")
        return

    # 2. MATLAB Script Generation
    print("\\n[2] Generating MATLAB Script...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matlab_dir = os.path.join(base_dir, "matlab")
    generated_dir = os.path.join(base_dir, "scenario_lab", "generated")
    
    adapter = MatlabAdapter(matlab_dir, generated_dir)
    script_path, result_path = adapter.generate_script(scenario)
    print(f"✅ Script generated at: {os.path.basename(script_path)}")

    # 3. MATLAB Execution
    print("\\n[3] Executing MATLAB in batch mode (this takes ~15-30s)...")
    runner = MatlabRunner(matlab_dir)
    try:
        raw_result = runner.run(script_path, result_path)
        print("✅ MATLAB Execution completed.")
    except Exception as e:
        print(f"❌ MATLAB Execution failed: {e}")
        return

    # 4. Result Explanation
    print("\\n[4] Building Deterministic Explanation...")
    sim_result = build_explanation(raw_result, scenario.scenario_id, scenario.ego_vehicle.mode)
    print(f"Outcome: {sim_result.explanation.outcome_summary}")
    print("Critical Events:")
    for ev in sim_result.explanation.critical_events:
        print(f" - {ev}")
        
    print("\\n🎉 End-to-End Pipeline Successful!")

if __name__ == "__main__":
    test_full_pipeline()
