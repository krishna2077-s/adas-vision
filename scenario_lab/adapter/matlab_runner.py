import subprocess
import os
import json
import pathlib

class MatlabRunner:
    def __init__(self, matlab_dir: str, matlab_executable: str = "matlab"):
        self.matlab_dir = matlab_dir
        self.matlab_exe = matlab_executable

    def run(self, script_path: str, result_path: str, timeout_s: int = 90) -> dict:
        """
        Executes a generated MATLAB script in batch mode.
        Reads the resulting JSON file and returns it as a dict.
        """
        func_name = pathlib.Path(script_path).stem
        generated_dir = os.path.dirname(script_path)
        
        batch_cmd = (
            f"addpath('{self.matlab_dir}'); "
            f"addpath('{generated_dir}'); "
            f"{func_name}(); exit;"
        )
        
        try:
            proc = subprocess.run(
                [self.matlab_exe, "-batch", batch_cmd],
                capture_output=True, text=True,
                timeout=timeout_s, cwd=self.matlab_dir
            )
        except subprocess.TimeoutExpired:
            # Clean up on timeout
            try: os.remove(script_path)
            except Exception: pass
            raise RuntimeError(f"MATLAB execution timed out after {timeout_s}s.")
        
        # Cleanup generated script regardless of outcome
        try: 
            os.remove(script_path)
        except Exception: 
            pass
        
        if proc.returncode != 0:
            raise RuntimeError(f"MATLAB error:\\n{proc.stderr[:2000]}\\nStdout:\\n{proc.stdout[:2000]}")
        
        if not os.path.exists(result_path):
            raise RuntimeError(f"MATLAB ran but produced no result JSON.\\nStdout:\\n{proc.stdout[-2000:]}")
        
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        # Cleanup result file after reading
        try:
            os.remove(result_path)
        except Exception:
            pass
            
        return result
