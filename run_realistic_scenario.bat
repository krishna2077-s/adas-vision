@echo off
title ADAS Vision - Hackathon-Ready Dual-Screen Driving Simulator
echo ======================================================================
echo   ADAS Vision - Hackathon Simulation Suite
echo ======================================================================
echo   [Window 1] 3D Highway Simulator (esmini)
echo              - 1.46 km Textured Indian Highway (Daylight)
echo              - 13 Entities (Ego, Lead Car, Overtaker, Trucks, Cattle, Cones)
echo              - 3D Floating Labels, Sensor Detection Cones, Bounding Boxes
echo              - Zero-Collision Trajectory Choreography
echo.
echo   [Window 2] ADAS Mission Control HUD (hackathon_cockpit.py)
echo              - Circular Analog & Digital Speedometer (0-140 km/h)
echo              - Real-Time Bird's-Eye View (BEV) Radar Scope with Sweep Cone
echo              - ADAS Decision Engine Banner (ACC Cruise, AEB Active)
echo              - Live Threat Assessment Matrix (ID, Distance, Speed, Risk)
echo              - Vehicle Dynamics: Weight, Weather Friction (mu), Load Transfer, Stop Distance
echo ======================================================================
echo   Controls:
echo       [Space]       Pause / Resume Simulation (3D Viewer)
echo       [Tab]         Cycle Cameras (Follow Ego, Orbit, Driver POV, Top-down)
echo       [Up] / [Down] Speed Up / Slow Down Playback
echo       [Left]/[Right]Scrub Timeline
echo       [W]           Cycle Weather (Dry mu=0.88 -> Monsoon mu=0.55 -> Heavy Rain mu=0.38 -> Gravel)
echo       [V]           Cycle Vehicle (MG Hector EV 1940kg -> Hector PHEV 1880kg -> Sedan -> SUV)
echo       [A]           Cycle FCW Sensitivity (Comfort/Dialed-Back -> Normal -> Aggressive)
echo       [T]           Toggle Traffic Jam Assist (TJA Curve/Bend Control)
echo       [L]           Toggle Gentle LKA Tactile Nudge
echo       [R]           Reset Scenario & Cockpit Timeline
echo       [Esc]         Exit both windows cleanly
echo ======================================================================
echo.

cd /d "%~dp0"
python run_esmini.py --scenario real_world
pause
