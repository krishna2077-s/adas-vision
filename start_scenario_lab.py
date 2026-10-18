import subprocess
import os
import sys
import time
import threading
import webbrowser

def run_api():
    print("[Scenario Lab] Starting API & Application Server on http://localhost:8000 ...")
    subprocess.run([sys.executable, "-m", "uvicorn", "scenario_lab.api.main:app", "--reload", "--port", "8000"])

def run_frontend():
    frontend_dir = os.path.join(os.path.dirname(__file__), "scenario_lab", "frontend")
    node_modules = os.path.join(frontend_dir, "node_modules")

    if os.path.exists(frontend_dir):
        if not os.path.exists(node_modules):
            print("[Scenario Lab] Note: node_modules not detected. Installing frontend dev dependencies...")
            res = subprocess.run("npm install", shell=True, cwd=frontend_dir)
            if res.returncode != 0:
                print("[Scenario Lab] npm install skipped/failed. Serving pre-compiled production build on http://localhost:8000")
                return

        print("[Scenario Lab] Starting Vite Dev Server on http://localhost:5173 ...")
        subprocess.run("npm run dev", shell=True, cwd=frontend_dir)
    else:
        print("[Scenario Lab] Frontend directory not found.")

if __name__ == "__main__":
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    time.sleep(2)  # Give API a moment to bind

    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()
    
    print("\n" + "=" * 60)
    print("  🚀 ADAS Scenario Lab & TwinX Simulation Hub Active!")
    print("=" * 60)
    print("  ► Web Application : http://localhost:8000  (Pre-compiled SPA + API)")
    print("  ► Vite Dev Server : http://localhost:5173  (HMR Live Code Edit)")
    print("  ► API Swagger Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("Press Ctrl+C to exit.\n")
    
    # Auto-open browser
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down ADAS Scenario Lab...")
