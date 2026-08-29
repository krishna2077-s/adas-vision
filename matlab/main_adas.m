% main_adas.m  —  ADAS Vision Master Entry Point
%
% Single command to run the complete ADAS Vision simulation project.
% Adaptive Path Planning & Collision Avoidance for Autonomous Vehicles
% on Unstructured Indian Roads.
%
% Usage (in MATLAB Command Window):
%   >> main_adas
%
% Requirements:
%   MATLAB R2024a or newer
%   Navigation Toolbox (recommended for Hybrid A*)
%   Automated Driving Toolbox (recommended for sensor models)
%   Stateflow (recommended for decision chart)
%
% The simulation runs WITHOUT toolboxes — built-in fallbacks are provided.

clc;
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║       ADAS VISION — Indian Road Autonomous Driving           ║\n');
fprintf('║       Adaptive Path Planning & Collision Avoidance           ║\n');
fprintf('║       SIH 2026 — Problem Statement PS Simulation             ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% Add matlab folder to path
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir), this_dir = pwd; end
addpath(this_dir);

% Setup vehicle model
fprintf('[INIT] Setting up vehicle parameters...\n');
try
    setup_vehicle_model;
catch
    fprintf('[INIT] Using built-in default vehicle parameters.\n');
end

% Toolbox availability check
check_toolboxes();

fprintf('\n═══════════════════════ MAIN MENU ═══════════════════════════\n');
fprintf('  [1]  Run Single Scenario\n');
fprintf('  [2]  Run All 5 Scenarios — Benchmark Suite\n');
fprintf('  [3]  Baseline vs Adaptive Comparison\n');
fprintf('  [4]  Build Stateflow Decision Chart\n');
fprintf('  [5]  Generate RoadRunner Scene Files\n');
fprintf('  [6]  Live Co-Simulation (Python bridge required)\n');
fprintf('  [7]  Generate Demo Video\n');
fprintf('═════════════════════════════════════════════════════════════\n\n');

choice = input('Enter your choice (1-7): ');

switch choice
    case 1
        run_single_scenario_menu();
    case 2
        fprintf('\n[BENCHMARK] Running all 5 Indian road scenarios...\n\n');
        run_all_scenarios;
    case 3
        fprintf('\n[COMPARE] Adaptive vs Baseline across all scenarios...\n\n');
        run_baseline_vs_adaptive;
    case 4
        fprintf('\n[STATEFLOW] Building Simulink Stateflow decision chart...\n\n');
        build_stateflow_model;
    case 5
        fprintf('\n[ROADRUNNER] Generating scene files...\n\n');
        generate_roadrunner_scenes;
    case 6
        fprintf('\n[COSIM] Starting live co-simulation...\n');
        fprintf('  Make sure the Python bridge is running:\n');
        fprintf('  >> python udp_bridge.py --scenario village_road\n\n');
        goal = input('Enter goal position [x, y] (e.g. [200, 0]): ');
        if isempty(goal), goal = [200, 0]; end
        run_cosimulation(goal);
    case 7
        fprintf('\n[VIDEO] Generating simulation demo video...\n\n');
        generate_demo_video;
    otherwise
        fprintf('[INFO] Invalid choice. Running benchmark suite by default.\n\n');
        run_all_scenarios;
end

fprintf('\n✅  ADAS Vision session complete.\n');


% ─────────────────────────────────────────────────────────────────────────────
function run_single_scenario_menu()
    fprintf('\n  [1]  Scenario 1 — Unmarked Village Road\n');
    fprintf('  [2]  Scenario 2 — Urban Intersection (No Signal)\n');
    fprintf('  [3]  Scenario 3 — Highway Merge\n');
    fprintf('  [4]  Scenario 4 — Dense Market Street\n');
    fprintf('  [5]  Scenario 5 — Cattle Crossing\n\n');
    s = input('  Select scenario (1-5): ');
    switch s
        case 1, result = scenario_village_road;
        case 2, result = scenario_urban_intersection;
        case 3, result = scenario_highway_merge;
        case 4, result = scenario_dense_market;
        case 5, result = scenario_cattle_crossing;
        otherwise
            fprintf('[WARN] Invalid. Running village road.\n');
            result = scenario_village_road;
    end
    if ~isempty(result)
        collect_metrics(result, result.name, 'PlotSummary', true, 'SaveCSV', true);
    end
end


% ─────────────────────────────────────────────────────────────────────────────
function check_toolboxes()
    tb_list = {
        'Navigation Toolbox',        'plannerHybridAStar';
        'Automated Driving Toolbox', 'drivingScenario';
        'Stateflow',                 'sfroot';
        'Vehicle Dynamics Blockset', 'vdblocks';
    };
    fprintf('\n[TOOLBOX CHECK]\n');
    for k = 1:size(tb_list,1)
        info = ver(tb_list{k,1});
        if ~isempty(info)
            fprintf('  ✓  %-35s  v%s\n', tb_list{k,1}, info.Version);
        else
            fprintf('  ✗  %-35s  (not found — fallback will be used)\n', tb_list{k,1});
        end
    end
    fprintf('\n');
end
