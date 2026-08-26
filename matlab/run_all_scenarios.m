% run_all_scenarios.m
% ADAS Vision — Automated Test Suite & Multi-Scenario Benchmarking
%
% Runs all 5 mandatory Indian road driving scenarios sequentially,
% evaluates safety and performance KPIs, generates a consolidated comparison
% table, exports metrics CSVs, and produces a summary comparison plot.
%
% Scenarios Tested:
%   1. Village Road (Unmarked winding road, oncoming motorcycle, VRUs)
%   2. Urban Intersection (Unsignalised 4-way junction, auto-rickshaw, pedestrians)
%   3. Highway Merge (Overloaded commercial truck cut-in, fast SUV)
%   4. Dense Market (Congested street, pushcarts, darting shoppers, scooter)
%   5. Cattle Crossing (Sudden cow entry, emergency brake)
%
% Usage:
%   >> run_all_scenarios

fprintf('=================================================================\n');
fprintf('       ADAS Vision — Automated Multi-Scenario Benchmark         \n');
fprintf('=================================================================\n\n');

scenarios = {
    'village_road',        @() scenario_village_road;
    'urban_intersection',  @() scenario_urban_intersection;
    'highway_merge',       @() scenario_highway_merge;
    'dense_market',        @() scenario_dense_market;
    'cattle_crossing',     @() scenario_cattle_crossing;
};

n_scen = size(scenarios, 1);
results = cell(n_scen, 1);

for i = 1:n_scen
    s_name = scenarios{i, 1};
    s_func = scenarios{i, 2};

    fprintf('\n>>> Running [%d/%d]: %s <<<\n', i, n_scen, s_name);
    try
        s_func();
        pause(1.0); % brief pause to let figure render and finish
        if evalin('base', 'exist(''scenario_result'',''var'')')
            res = evalin('base', 'scenario_result');
            m = collect_metrics(res, s_name, 'SaveCSV', true, 'PlotSummary', false);
            results{i} = m;
        end
    catch ME
        fprintf('[ERROR] Failed running scenario %s: %s\n', s_name, ME.message);
    end
end

% ---------------------------------------------------------------------------
% Consolidated Comparative Table
% ---------------------------------------------------------------------------
fprintf('\n\n========================================================================================================\n');
fprintf('                                     BENCHMARK SUMMARY RESULTS TABLE                                     \n');
fprintf('========================================================================================================\n');
fprintf('%-22s | %-10s | %-12s | %-12s | %-10s | %-10s | %-10s\n', ...
        'Scenario', 'Time (s)', 'Distance (m)', 'Smoothness', 'Collisions', 'Replan(ms)', 'Status');
fprintf('--------------------------------------------------------------------------------------------------------\n');

names       = {};
durations   = [];
distances   = [];
smoothnesses= [];
collisions  = [];
replans     = [];
completed   = {};

for i = 1:n_scen
    if ~isempty(results{i})
        m = results{i};
        stat_str = 'PASS';
        if m.collisions > 0
            stat_str = 'FAIL (Hit)';
        elseif ~m.scenario_completed
            stat_str = 'TIMEOUT';
        end

        fprintf('%-22s | %10.1f | %12.1f | %12.4f | %10d | %10.2f | %-10s\n', ...
                m.scenario, m.duration_s, m.distance_m, m.path_smoothness, ...
                m.collisions, m.mean_replan_ms, stat_str);

        names{end+1}        = m.scenario; %#ok<AGROW>
        durations(end+1)    = m.duration_s; %#ok<AGROW>
        distances(end+1)    = m.distance_m; %#ok<AGROW>
        smoothnesses(end+1) = m.path_smoothness; %#ok<AGROW>
        collisions(end+1)   = m.collisions; %#ok<AGROW>
        replans(end+1)      = m.mean_replan_ms; %#ok<AGROW>
        completed{end+1}    = stat_str; %#ok<AGROW>
    end
end
fprintf('========================================================================================================\n\n');

% ---------------------------------------------------------------------------
% Export Consolidated CSV
% ---------------------------------------------------------------------------
csv_name = 'metrics_all_scenarios_summary.csv';
try
    fid = fopen(csv_name, 'w');
    fprintf(fid, 'scenario,duration_s,distance_m,path_smoothness,collisions,mean_replan_ms,status\n');
    for i = 1:numel(names)
        fprintf(fid, '%s,%.2f,%.2f,%.4f,%d,%.2f,%s\n', ...
                names{i}, durations(i), distances(i), smoothnesses(i), collisions(i), replans(i), completed{i});
    end
    fclose(fid);
    fprintf('✅ Consolidated CSV saved: %s\n', fullfile(pwd, csv_name));
catch e
    fprintf('[WARN] CSV export failed: %s\n', e.message);
end

% ---------------------------------------------------------------------------
% Consolidated Summary Plot
% ---------------------------------------------------------------------------
if ~isempty(names)
    fig_sum = figure('Name', 'ADAS Multi-Scenario Comparative Benchmark', ...
                     'NumberTitle', 'off', 'Color', [0.08 0.08 0.12], ...
                     'Position', [150 150 900 480]);

    ax1 = subplot(2, 2, 1, 'Parent', fig_sum, 'Color', [0.13 0.13 0.18], 'XColor', 'w', 'YColor', 'w');
    bar(ax1, durations, 'FaceColor', [0.2 0.7 1.0], 'EdgeColor', 'none');
    set(ax1, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 15);
    title(ax1, 'Duration (s)', 'Color', 'w'); grid(ax1, 'on');

    ax2 = subplot(2, 2, 2, 'Parent', fig_sum, 'Color', [0.13 0.13 0.18], 'XColor', 'w', 'YColor', 'w');
    bar(ax2, distances, 'FaceColor', [0.3 0.85 0.4], 'EdgeColor', 'none');
    set(ax2, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 15);
    title(ax2, 'Distance Covered (m)', 'Color', 'w'); grid(ax2, 'on');

    ax3 = subplot(2, 2, 3, 'Parent', fig_sum, 'Color', [0.13 0.13 0.18], 'XColor', 'w', 'YColor', 'w');
    bar(ax3, smoothnesses, 'FaceColor', [1.0 0.7 0.2], 'EdgeColor', 'none');
    set(ax3, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 15);
    title(ax3, 'Path Smoothness (rad/m, lower=better)', 'Color', 'w'); grid(ax3, 'on');

    ax4 = subplot(2, 2, 4, 'Parent', fig_sum, 'Color', [0.13 0.13 0.18], 'XColor', 'w', 'YColor', 'w');
    bar(ax4, collisions, 'FaceColor', [0.9 0.3 0.3], 'EdgeColor', 'none');
    set(ax4, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 15);
    title(ax4, 'Collisions', 'Color', 'w'); grid(ax4, 'on');

    sgtitle(fig_sum, 'ADAS Vision — Performance Across 5 Indian Road Scenarios', 'Color', 'w', 'FontSize', 13);
    drawnow;
end

fprintf('\n🎉 Benchmark suite complete!\n');
