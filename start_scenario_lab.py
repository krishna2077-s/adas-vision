import subprocess
import os
import sys
import time
import threading

def run_api():
    print("Starting API Server on port 8000...")
    subprocess.run([sys.executable, "-m", "uvicorn", "scenario_lab.api.main:app", "--reload", "--port", "8000"])

def run_frontend():
    print("Starting Frontend on port 5173...")
    frontend_dir = os.path.join(os.path.dirname(__file__), "scenario_lab", "frontend")
    if os.path.exists(frontend_dir):
        # We need shell=True on windows for npm
        subprocess.run("npm run dev", shell=True, cwd=frontend_dir)
    else:
        print("Frontend directory not found.")

if __name__ == "__main__":
    # Ensure dependencies are available
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()
    
    time.sleep(2) # Give API a moment to start
    
    frontend_thread = threading.Thread(target=run_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    print("\\n=== ADAS Scenario Lab ===")
    print("API: http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("Press Ctrl+C to exit.\\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\nShutting down...")
