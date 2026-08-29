% startup.m — Auto-runs when MATLAB starts from the adas-vision/matlab folder.
%
% HOW TO USE:
%   1. Open MATLAB R2024a or newer
%   2. Set Current Folder to:
%      C:\Users\Manish khandelwal\Downloads\adas-vision\adas-vision\matlab
%   3. This file runs automatically — you'll see "ADAS Vision ready" in the Command Window
%
% After this runs, you can immediately type:
%   main_adas                    → Interactive menu (START HERE)
%   run_all_scenarios            → Benchmark all 5 Indian road scenarios
%   build_stateflow_model        → Build Simulink Stateflow decision chart
%   generate_roadrunner_scenes   → Export OpenDRIVE scene files

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║       ADAS VISION — Indian Road Autonomous Driving           ║\n');
fprintf('║       Adaptive Path Planning & Collision Avoidance           ║\n');
fprintf('║       SIH 2026                                               ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ── Add the matlab/ folder to the path ─────────────────────────────────────
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir)
    this_dir = pwd;
end
addpath(this_dir);
fprintf('[1/3] Added matlab/ folder to MATLAB path.\n');

% ── Check for required toolboxes ───────────────────────────────────────────
required_tb = {
    'Navigation Toolbox',        'plannerHybridAStar (Hybrid A* planner)';
    'Automated Driving Toolbox', 'drivingScenario (RoadRunner integration)';
    'Stateflow',                 'Stateflow chart (decision engine)';
};

fprintf('[2/3] Toolbox check:\n');
missing_tb = {};
for i = 1:size(required_tb, 1)
    info = ver(required_tb{i,1});
    if ~isempty(info)
        fprintf('       ✓  %-35s  v%s\n', required_tb{i,1}, info.Version);
    else
        fprintf('       ✗  %-35s  (fallback active)\n', required_tb{i,1});
        missing_tb{end+1} = required_tb{i,1}; %#ok<AGROW>
    end
end
if ~isempty(missing_tb)
    fprintf('       NOTE: Missing toolboxes use built-in fallbacks.\n');
    fprintf('             Full functionality requires: Navigation + Automated Driving Toolbox.\n');
end

% ── Load vehicle model parameters into workspace ───────────────────────────
fprintf('[3/3] Loading vehicle model parameters...\n');
try
    setup_vehicle_model;
    fprintf('       Vehicle model loaded (bicycle model, 1450 kg, 2.7m wheelbase).\n');
catch ME
    fprintf('       [WARN] setup_vehicle_model: %s. Default params will be used.\n', ME.message);
end

fprintf('\n✅  ADAS Vision ready!  Commands you can run now:\n\n');
fprintf('    main_adas                   → Interactive menu (recommended START)\n');
fprintf('    scenario_village_road       → Scenario 1: Unmarked village road\n');
fprintf('    scenario_urban_intersection → Scenario 2: Busy urban intersection\n');
fprintf('    scenario_highway_merge      → Scenario 3: Highway merge\n');
fprintf('    scenario_dense_market       → Scenario 4: Dense market street\n');
fprintf('    scenario_cattle_crossing    → Scenario 5: Cattle crossing\n');
fprintf('    run_all_scenarios           → Benchmark all 5 scenarios\n');
fprintf('    build_stateflow_model       → Simulink Stateflow chart\n');
fprintf('    generate_roadrunner_scenes  → OpenDRIVE scene files\n');
fprintf('    generate_demo_video         → MP4 demo video\n');
fprintf('    run_cosimulation([200,0])   → Live co-sim (Python bridge first)\n');
fprintf('\n════════════════════════════════════════════════════════════════\n\n');
