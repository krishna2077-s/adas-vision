% startup.m — Auto-runs when MATLAB starts from the adas-vision folder.
%
% HOW TO USE:
%   1. Open MATLAB
%   2. Set Current Folder to:  C:\Users\Rakesh Sharma\OneDrive\Desktop\adas-vision\matlab
%   3. This file runs automatically — you'll see "ADAS Vision ready" in the Command Window
%
% After this runs, you can immediately type any of:
%   setup_vehicle_model
%   scenario_village_road
%   scenario_urban_intersection
%   scenario_cattle_crossing
%   run_cosimulation([200, 0])     % (needs Python bridge running first)

fprintf('\n=========================================================\n');
fprintf('  ADAS Vision Co-Simulation — MATLAB R2026a\n');
fprintf('=========================================================\n');

% ── Add the matlab/ folder and its parent to the path ─────────────────────
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
fprintf('[1/3] Added matlab/ folder to path.\n');

% ── Check for required toolboxes ───────────────────────────────────────────
required = {'Automated Driving Toolbox', 'Navigation Toolbox', 'Stateflow'};
missing  = {};
for i = 1:numel(required)
    info = ver(required{i});
    if isempty(info)
        missing{end+1} = required{i}; %#ok<AGROW>
    else
        fprintf('[2/3] %-35s v%s  ✓\n', required{i}, info.Version);
    end
end
if ~isempty(missing)
    warning('Missing toolboxes: %s', strjoin(missing, ', '));
    fprintf('      Some features may not work without these toolboxes.\n');
end

% ── Load vehicle model parameters into workspace ───────────────────────────
fprintf('[3/3] Loading vehicle model parameters...\n');
setup_vehicle_model;

fprintf('\n✅  ADAS Vision ready! Commands you can run now:\n');
fprintf('    scenario_village_road          → Scenario 1: unmarked village road\n');
fprintf('    scenario_urban_intersection    → Scenario 2: busy intersection\n');
fprintf('    scenario_cattle_crossing       → Scenario 5: cattle crossing\n');
fprintf('    run_cosimulation([200, 0])     → Live co-sim (start Python bridge first)\n');
fprintf('=========================================================\n\n');
