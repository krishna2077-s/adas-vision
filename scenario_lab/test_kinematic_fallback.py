import unittest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

from scenario_lab.schema.scenario_schema import Scenario, Road, EgoVehicle, Actor, Metadata
from scenario_lab.adapter.matlab_runner import MatlabRunner, simulate_kinematics
from scenario_lab.adapter.explanations import build_explanation

class TestKinematicFallback(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.scenario = Scenario(
            scenario_id="test_sim_fallback",
            metadata=Metadata(name="Cattle Crossing Test", source="builder", created_at="2026-10-16T00:00:00Z"),
            road=Road(type="village_road", lane_half_width_m=3.5, length_m=200.0),
            ego_vehicle=EgoVehicle(initial_speed_kmh=40.0, goal_x_m=120.0, mode="adaptive"),
            actors=[
                Actor(id="cow_1", **{"class": "cow"}, initial_x_m=60.0, initial_y_m=3.0,
                      initial_vx_mps=0.0, initial_vy_mps=-1.0, behaviour="crossing")
            ]
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_direct_kinematic_simulation(self):
        """Verify Python kinematic simulation returns realistic metrics and trajectories."""
        res = simulate_kinematics(self.scenario)
        self.assertIn("traj_x", res)
        self.assertIn("traj_y", res)
        self.assertGreater(len(res["traj_x"]), 10)
        self.assertIn("decision_proceed", res)
        self.assertIn("decision_emergency_stop", res)
        self.assertIn("min_clearance_m", res)
        self.assertIn("mean_speed_kmh", res)

        # Check explanation builder compatibility
        sim_res = build_explanation(res, self.scenario.scenario_id, "adaptive")
        self.assertEqual(sim_res.scenario_id, "test_sim_fallback")
        self.assertTrue(len(sim_res.traj_x) > 0)
        self.assertIsNotNone(sim_res.explanation.outcome_summary)

    @patch("subprocess.run")
    def test_matlab_runner_license_fallback(self, mock_subproc):
        """Verify MatlabRunner activates Python engine when MATLAB returns license error."""
        # Mock MATLAB failing with exit code 1 (licensing error)
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "MathWorks Licensing Error 10 Your license for MATLAB has expired."
        mock_proc.stdout = "ERROR: MATLAB error Exit Status: 0x00000001"
        mock_subproc.return_value = mock_proc

        runner = MatlabRunner(self.test_dir, matlab_executable="matlab")
        script_path = os.path.join(self.test_dir, "test_script.m")
        result_path = os.path.join(self.test_dir, "test_out.json")

        with open(script_path, "w") as f:
            f.write("% dummy")

        res = runner.run(script_path, result_path, scenario=self.scenario)
        self.assertIn("traj_x", res)
        self.assertIn("arrived", res)
        self.assertTrue(os.path.exists(result_path))

if __name__ == "__main__":
    unittest.main()
