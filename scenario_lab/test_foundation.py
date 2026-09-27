import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenario_lab.schema.scenario_schema import Scenario, Metadata, Road, EgoVehicle, Actor
from scenario_lab.adapter.matlab_adapter import MatlabAdapter
from scenario_lab.adapter.matlab_runner import MatlabRunner

def test_integration():
    print("Testing Foundation Phase...")
    
    scenario = Scenario(
        scenario_id="test-cattle-crossing",
        metadata=Metadata(
            name="Sudden Cattle Crossing",
            description="A cow suddenly enters the road",
            source="template",
            created_at="2026-09-27T00:00:00Z"
        ),
        road=Road(type="village_road", lane_half_width_m=3.5, length_m=180.0),
        ego_vehicle=EgoVehicle(initial_speed_kmh=30, goal_x_m=160.0),
        actors=[
            Actor(
                id="cow1",
                actor_class="cow",
                initial_x_m=75.0,
                initial_y_m=-4.5,
                initial_vx_mps=0.05,
                initial_vy_mps=0.85,
                behaviour="crossing",
                behaviour_trigger_s=2.5,
                behaviour_y_target_m=3.8
            )
        ]
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matlab_dir = os.path.join(base_dir, "matlab")
    generated_dir = os.path.join(base_dir, "scenario_lab", "generated")

    print(f"Generating MATLAB script...")
    adapter = MatlabAdapter(matlab_dir=matlab_dir, generated_dir=generated_dir)
    script_path, result_path = adapter.generate_script(scenario)
    print(f"Script created at: {script_path}")
    
    print(f"Running MATLAB... (this might take a few seconds)")
    runner = MatlabRunner(matlab_dir=matlab_dir)
    try:
        # Note: adjust matlab_executable if 'matlab' is not in PATH
        result = runner.run(script_path, result_path)
        print("Success! Result keys:", result.keys())
        print(f"Arrived: {result.get('arrived')}")
        print(f"Collisions: {result.get('collisions')}")
        print(f"Simulation Time: {result.get('t_total_s')}s")
    except Exception as e:
        print(f"Integration Test Failed: {e}")

if __name__ == "__main__":
    test_integration()
