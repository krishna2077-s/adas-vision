% setup_toolbox_check.m
% ADAS Vision — MATLAB Toolbox Diagnostic and Fallback Configurator
%
% Checks all required and optional toolboxes, prints a clear status table,
% and configures fallback flags in the workspace for modules that use them.
%
% Usage:
%   >> setup_toolbox_check
%   >> [nav_ok, adt_ok, sf_ok] = setup_toolbox_check

function [nav_ok, adt_ok, sf_ok, vdb_ok] = setup_toolbox_check()

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║         ADAS Vision — Toolbox Diagnostic Report             ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ─────────────────────────────────────────────────────────────────────────────
% Check each toolbox
% ─────────────────────────────────────────────────────────────────────────────
tb_checks = {
    'Navigation Toolbox',          'plannerHybridAStar', 'REQUIRED — Hybrid A* path planner';
    'Automated Driving Toolbox',   'drivingScenario',    'REQUIRED — Scenario/RoadRunner integration';
    'Stateflow',                   'sfroot',             'RECOMMENDED — Stateflow decision chart';
    'Vehicle Dynamics Blockset',   'vdblocks',           'OPTIONAL — Advanced vehicle dynamics';
    'Control System Toolbox',      'pid',                'OPTIONAL — Advanced PID tuning';
    'Signal Processing Toolbox',   'bandpass',           'OPTIONAL — Sensor signal processing';
    'Deep Learning Toolbox',       'dlnetwork',          'OPTIONAL — Neural trajectory prediction';
    'Parallel Computing Toolbox',  'parfor',             'OPTIONAL — Faster batch scenario runs';
};

results = struct();
nav_ok  = false;
adt_ok  = false;
sf_ok   = false;
vdb_ok  = false;

fprintf('%-35s  %-8s  %-10s  %s\n', 'Toolbox', 'Status', 'Version', 'Purpose');
fprintf('%s\n', repmat('─', 1, 90));

for k = 1:size(tb_checks, 1)
    tb_name = tb_checks{k, 1};
    fn_check = tb_checks{k, 2};
    purpose  = tb_checks{k, 3};

    tb_ver  = ver(tb_name);
    fn_avail = exist(fn_check, 'builtin') || exist(fn_check, 'file');

    if ~isempty(tb_ver)
        status  = '✓ OK';
        ver_str = tb_ver(1).Version;
        status_color = 'present';
    elseif fn_avail
        status  = '~ FN';
        ver_str = 'func-only';
        status_color = 'partial';
    else
        status  = '✗ MISS';
        ver_str = 'N/A';
        status_color = 'missing';
    end

    is_present = ~isempty(tb_ver) || fn_avail;
    fprintf('  %-35s  %-8s  %-10s  %s\n', tb_name, status, ver_str, purpose);

    switch tb_name
        case 'Navigation Toolbox',         nav_ok  = is_present;
        case 'Automated Driving Toolbox',   adt_ok  = is_present;
        case 'Stateflow',                   sf_ok   = is_present;
        case 'Vehicle Dynamics Blockset',   vdb_ok  = is_present;
    end
end

fprintf('%s\n\n', repmat('─', 1, 90));

% ─────────────────────────────────────────────────────────────────────────────
% Summary
% ─────────────────────────────────────────────────────────────────────────────
fprintf('SUMMARY:\n');
if nav_ok && adt_ok
    fprintf('  ✅  Full ADAS Vision functionality available.\n');
    fprintf('      Hybrid A* planner + drivingScenario + OpenDRIVE export all active.\n');
elseif nav_ok && ~adt_ok
    fprintf('  ⚠️  Navigation Toolbox OK. Automated Driving Toolbox missing.\n');
    fprintf('      Hybrid A* active. RoadRunner/drivingScenario → fallback mode.\n');
elseif ~nav_ok && adt_ok
    fprintf('  ⚠️  Automated Driving Toolbox OK. Navigation Toolbox missing.\n');
    fprintf('      Path planning → built-in greedy Hybrid A* fallback.\n');
else
    fprintf('  ⚠️  Core toolboxes missing. Running in PURE FALLBACK mode.\n');
    fprintf('      All 5 scenarios still run with built-in fallback algorithms.\n');
end

if sf_ok
    fprintf('  ✅  Stateflow available — adas_decision_stateflow.slx will be built.\n');
else
    fprintf('  ℹ️   Stateflow missing — standalone state diagram will be shown instead.\n');
end

fprintf('\n');

% ─────────────────────────────────────────────────────────────────────────────
% Write flags to base workspace for other scripts to use
% ─────────────────────────────────────────────────────────────────────────────
assignin('base', 'ADAS_NAV_TOOLBOX_OK',  nav_ok);
assignin('base', 'ADAS_ADT_TOOLBOX_OK',  adt_ok);
assignin('base', 'ADAS_SF_TOOLBOX_OK',   sf_ok);
assignin('base', 'ADAS_VDB_TOOLBOX_OK',  vdb_ok);

fprintf('  Workspace flags set:\n');
fprintf('    ADAS_NAV_TOOLBOX_OK = %d\n', nav_ok);
fprintf('    ADAS_ADT_TOOLBOX_OK = %d\n', adt_ok);
fprintf('    ADAS_SF_TOOLBOX_OK  = %d\n', sf_ok);
fprintf('\n');

% ─────────────────────────────────────────────────────────────────────────────
% MATLAB version check
% ─────────────────────────────────────────────────────────────────────────────
ml_ver = ver('MATLAB');
if ~isempty(ml_ver)
    ml_year = str2double(ml_ver.Release(3:6));
    fprintf('  MATLAB %s (Release %s)\n', ml_ver.Version, ml_ver.Release);
    if ml_year >= 2024
        fprintf('  ✅  MATLAB version is compatible (R2024a+).\n');
    else
        fprintf('  ⚠️  MATLAB R2024a+ recommended for full plannerHybridAStar support.\n');
    end
end

fprintf('\n  Run next: main_adas  (interactive menu)\n\n');

if nargout == 0
    clear nav_ok adt_ok sf_ok vdb_ok;
end
end
