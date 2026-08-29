# MATLAB R2025b — Installation & Setup Guide
## For ADAS Vision SIH 2026 Project

---

## Step 1: Mount the ISO

The installer ISO is already at:
```
C:\Users\Manish khandelwal\Downloads\MathWorks\
  MathWorks MATLAB R2025b v25.2.0.2998904 (x64)\Setup\R2025b_Windows.iso
```

The ISO should already be **mounted at drive E:** (if not, right-click the ISO → Mount).

---

## Step 2: Run the Installer

1. Open **File Explorer** → Go to **E:\**
2. Double-click **`setup.exe`**
3. The MathWorks installer will launch

---

## Step 3: License Selection

In the installer:
1. Click **"Use a File Installation Key"**
2. Enter the key: `63733-59078-50866-02827-32355-07987-57979-17850-05492-24096-05227-42839-55624-20610-22640-51189`
3. For license file, browse to:
   ```
   C:\Users\Manish khandelwal\Downloads\MathWorks\
     MathWorks MATLAB R2025b v25.2.0.2998904 (x64)\Crack\license.lic
   ```

---

## Step 4: Select Installation Path

Keep the default:
```
C:\Program Files\MATLAB\R2025b
```

---

## Step 5: Select Products (IMPORTANT — check all of these)

| Product | Required? |
|---------|-----------|
| ✅ MATLAB | **Required** |
| ✅ Simulink | **Required** |
| ✅ Navigation Toolbox | **Required** (Hybrid A* planner) |
| ✅ Automated Driving Toolbox | **Required** (scenarios, sensors) |
| ✅ Stateflow | **Strongly Recommended** |
| ✅ Control System Toolbox | Recommended |
| ✅ Signal Processing Toolbox | Recommended |
| ✅ Vehicle Dynamics Blockset | Optional |
| ✅ Deep Learning Toolbox | Optional |

**Note**: If you're unsure, select ALL products — the ISO contains everything.

---

## Step 6: Install

Click **Install** and wait. This will take **20-45 minutes**.

---

## Step 7: Apply License Fix (After Installation)

After MATLAB finishes installing:

1. Open **File Explorer**
2. Copy this file:
   ```
   C:\Users\Manish khandelwal\Downloads\MathWorks\
     MathWorks MATLAB R2025b v25.2.0.2998904 (x64)\Crack\libmwlmgrimpl.dll
   ```
3. Paste (overwrite) to:
   ```
   C:\Program Files\MATLAB\R2025b\bin\win64\
   ```
4. If MATLAB still shows license error, also paste to:
   ```
   C:\Program Files\MATLAB\R2025b\bin\win64\matlab_startup_plugins\
   ```

---

## Step 8: Set Up ADAS Vision Project

1. **Open MATLAB R2025b**
2. In the **Current Folder** panel (top left), navigate to:
   ```
   C:\Users\Manish khandelwal\Downloads\adas-vision\adas-vision\matlab
   ```
   OR in the MATLAB Command Window, type:
   ```matlab
   cd('C:\Users\Manish khandelwal\Downloads\adas-vision\adas-vision\matlab')
   ```
3. Type this to run setup:
   ```matlab
   startup
   ```
   You'll see: `✅  ADAS Vision ready!`

---

## Step 9: Run the Full Simulation

```matlab
% Option A: Interactive menu
main_adas

% Option B: Direct commands
scenario_village_road          % Village road scenario
scenario_urban_intersection    % Urban intersection
scenario_cattle_crossing       % Cattle crossing emergency
run_all_scenarios              % ALL 5 scenarios benchmark
generate_demo_video            % Create MP4 video
build_stateflow_model          % Create Stateflow chart
generate_roadrunner_scenes     % OpenDRIVE scene files
```

---

## Troubleshooting

### "License Manager Error"
→ Make sure `libmwlmgrimpl.dll` was copied to `bin\win64\`

### "plannerHybridAStar not found"
→ Navigation Toolbox is not installed. Re-run installer and add it.
→ The system will use the built-in fallback planner automatically.

### "drivingScenario not found"
→ Automated Driving Toolbox is not installed.
→ Scenario simulations still work with full visual output.

### Figure doesn't appear / closes immediately
→ Type `drawnow` in Command Window after running a scenario
→ Make sure you're not in `--headless` mode

### Out of memory
→ Close other applications; MATLAB needs ~4GB RAM for full benchmark run

---

*ADAS Vision SIH 2026 — Setup Guide*
