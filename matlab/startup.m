% startup.m — Auto-runs when MATLAB starts from the adas-vision/matlab folder.
%
% ADAS Vision — SIH Grand Jury Technical Environment Setup

fprintf('\n=========================================================\n');
fprintf('  ADAS Vision Co-Simulation & Multi-Sensor ADAS Suite\n');
fprintf('=========================================================\n');

% ── Add matlab/ and project root to MATLAB search path ───────────────────────
this_dir = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
addpath(this_dir);
addpath(fullfile(parent_dir, 'roadrunner', 'scripts'));
fprintf('[1/3] Added ADAS Vision toolboxes and scripts to MATLAB path.\n');

% ── Check for required and recommended toolboxes ─────────────────────────────
required = {'Automated Driving Toolbox', 'Navigation Toolbox', 'Stateflow'};
recommended = {'Lidar Toolbox', 'Sensor Fusion and Tracking Toolbox', 'Computer Vision Toolbox'};

missing_req  = {};
missing_rec  = {};

for i = 1:numel(required)
    info = ver(required{i});
    if isempty(info)
        missing_req{end+1} = required{i}; %#ok<AGROW>
    else
        fprintf('[2/3] [REQUIRED]    %-32s v%s  ✓\n', required{i}, info.Version);
    end
end

for i = 1:numel(recommended)
    info = ver(recommended{i});
    if isempty(info)
        missing_rec{end+1} = recommended{i}; %#ok<AGROW>
    else
        fprintf('      [RECOMMENDED] %-32s v%s  ✓\n', recommended{i}, info.Version);
    end
end

if ~isempty(missing_req)
    warning('Missing core toolboxes: %s. Some features may require fallback math.', strjoin(missing_req, ', '));
end

% ── Load vehicle model parameters into workspace ───────────────────────────
fprintf('[3/3] Loading vehicle dynamics parameters...\n');
try
    setup_vehicle_model;
catch ME
    fprintf('Note: setup_vehicle_model notice: %s\n', ME.message);
end

fprintf('\n✅  ADAS Vision ready! Top commands you can run now:\n');
fprintf('    demo_lidar_3d                  → 3D LiDAR Point Cloud & Sensor Fusion Demo\n');
fprintf('    run_all_scenarios              → Run all 5 mandatory SIH scenarios\n');
fprintf('    run_baseline_vs_adaptive       → Safety Benchmark: Baseline vs Adaptive\n');
fprintf('    scenario_dense_market          → Scenario 4: Dense market (Congested)\n');
fprintf('    scenario_cattle_crossing       → Scenario 5: Sudden cattle crossing\n');
fprintf('    scenario_village_road          → Scenario 1: Unmarked village road\n');
fprintf('    scenario_highway_merge         → Scenario 3: Highway merge & overtake\n');
fprintf('    scenario_urban_intersection    → Scenario 2: Busy 4-way intersection\n');
fprintf('=========================================================\n\n');
