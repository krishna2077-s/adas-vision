% run_baseline_vs_adaptive.m
% ADAS Vision — Genuine Baseline vs. Adaptive Comparative Benchmark Suite
%
% Executes all 5 Indian road scenarios under two genuine closed-loop simulation passes:
%   1. ADAPTIVE (OURS): Dynamic occupancy grid, trajectory prediction cones,
%                       Hybrid A* replanning, R1-R7 temporal ratchet.
%   2. BASELINE: Fixed centerline trajectory, no Hybrid A* prediction/replanning,
%                purely reactive braking (traditional ADAS).
%
% Outputs:
%   - Side-by-side terminal comparison table
%   - Consolidated CSV: metrics_baseline_vs_adaptive.csv
%   - Comparative multi-panel chart
%
% Usage:
%   >> run_baseline_vs_adaptive

fprintf('========================================================================================\n');
fprintf('        ADAS Vision — Baseline vs. Adaptive Autonomous Benchmark Comparison             \n');
fprintf('========================================================================================\n\n');

scenarios = {
    'village_road',        @(m) scenario_village_road(m);
    'urban_intersection',  @(m) scenario_urban_intersection(m);
    'highway_merge',       @(m) scenario_highway_merge(m);
    'dense_market',        @(m) scenario_dense_market(m);
    'cattle_crossing',     @(m) scenario_cattle_crossing(m);
};

n_scen = size(scenarios, 1);
adaptive_results = cell(n_scen, 1);
baseline_results = cell(n_scen, 1);

% ---------------------------------------------------------------------------
% 1. Run Genuine Adaptive Tests
% ---------------------------------------------------------------------------
fprintf('>>> PHASE 1: Running All Scenarios with ADAPTIVE Hybrid A* Planner <<<\n');
for i = 1:n_scen
    s_name = scenarios{i, 1};
    s_func = scenarios{i, 2};
    fprintf('  [Adaptive %d/%d]: %s\n', i, n_scen, s_name);
    try
        res = s_func('adaptive');
        pause(0.5);
        if ~isempty(res)
            m = collect_metrics(res, [s_name, '_adaptive'], 'SaveCSV', false, 'PlotSummary', false);
            adaptive_results{i} = m;
        end
    catch ME
        fprintf('    [ERROR] %s: %s\n', s_name, ME.message);
    end
end

% ---------------------------------------------------------------------------
% 2. Run Genuine Baseline Tests (Centerline tracking, reactive braking only)
% ---------------------------------------------------------------------------
fprintf('\n>>> PHASE 2: Running All Scenarios in BASELINE Mode (Centerline, Reactive Braking) <<<\n');
for i = 1:n_scen
    s_name = scenarios{i, 1};
    s_func = scenarios{i, 2};
    fprintf('  [Baseline %d/%d]: %s\n', i, n_scen, s_name);
    try
        res = s_func('baseline');
        pause(0.5);
        if ~isempty(res)
            m = collect_metrics(res, [s_name, '_baseline'], 'SaveCSV', false, 'PlotSummary', false);
            baseline_results{i} = m;
        end
    catch ME
        fprintf('    [ERROR] %s: %s\n', s_name, ME.message);
    end
end

% ---------------------------------------------------------------------------
% Comparative Terminal Table
% ---------------------------------------------------------------------------
fprintf('\n\n========================================================================================================================\n');
fprintf('                                     BASELINE vs. ADAPTIVE PERFORMANCE COMPARISON TABLE                                 \n');
fprintf('========================================================================================================================\n');
fprintf('%-20s | %-10s | %-12s | %-12s | %-14s | %-10s | %-8s\n', ...
        'Scenario', 'Mode', 'Time (s)', 'Smoothness', 'Min Clearance', 'Collisions', 'Status');
fprintf('------------------------------------------------------------------------------------------------------------------------\n');

for i = 1:n_scen
    s_name = scenarios{i, 1};
    if ~isempty(baseline_results{i}) && ~isempty(adaptive_results{i})
        b = baseline_results{i};
        a = adaptive_results{i};
        
        stat_b = 'PASS'; if b.collisions > 0, stat_b = 'COLLISION'; elseif ~b.scenario_completed, stat_b = 'TIMEOUT'; end
        stat_a = 'PASS'; if a.collisions > 0, stat_a = 'COLLISION'; elseif ~a.scenario_completed, stat_a = 'TIMEOUT'; end

        fprintf('%-20s | %-10s | %12.1f | %12.4f | %12.2f m | %10d | %-8s\n', ...
                s_name, 'BASELINE', b.duration_s, b.path_smoothness, b.min_clearance_m, b.collisions, stat_b);
        fprintf('%-20s | %-10s | %12.1f | %12.4f | %12.2f m | %10d | %-8s\n', ...
                s_name, 'ADAPTIVE', a.duration_s, a.path_smoothness, a.min_clearance_m, a.collisions, stat_a);
        fprintf('------------------------------------------------------------------------------------------------------------------------\n');
    end
end
fprintf('========================================================================================================================\n\n');

% ---------------------------------------------------------------------------
% Export Consolidated Comparative CSV
% ---------------------------------------------------------------------------
csv_name = 'metrics_baseline_vs_adaptive.csv';
try
    fid = fopen(csv_name, 'w');
    fprintf(fid, 'scenario,mode,duration_s,distance_m,path_smoothness,min_clearance_m,collisions,mean_replan_ms,status\n');
    for i = 1:n_scen
        s_name = scenarios{i, 1};
        if ~isempty(baseline_results{i}) && ~isempty(adaptive_results{i})
            b = baseline_results{i};
            a = adaptive_results{i};
            stat_b = 'PASS'; if b.collisions > 0, stat_b = 'COLLISION'; elseif ~b.scenario_completed, stat_b = 'TIMEOUT'; end
            stat_a = 'PASS'; if a.collisions > 0, stat_a = 'COLLISION'; elseif ~a.scenario_completed, stat_a = 'TIMEOUT'; end
            
            fprintf(fid, '%s,BASELINE,%.2f,%.2f,%.4f,%.2f,%d,%.2f,%s\n', ...
                    s_name, b.duration_s, b.distance_m, b.path_smoothness, b.min_clearance_m, b.collisions, b.mean_replan_ms, stat_b);
            fprintf(fid, '%s,ADAPTIVE,%.2f,%.2f,%.4f,%.2f,%d,%.2f,%s\n', ...
                    s_name, a.duration_s, a.distance_m, a.path_smoothness, a.min_clearance_m, a.collisions, a.mean_replan_ms, stat_a);
        end
    end
    fclose(fid);
    fprintf('✅ Baseline vs. Adaptive CSV saved: %s\n', fullfile(pwd, csv_name));
catch e
    fprintf('[WARN] CSV export failed: %s\n', e.message);
end

% ---------------------------------------------------------------------------
% Summary Comparative Figure
% ---------------------------------------------------------------------------
try
    scen_labels = {'Village', 'Intersection', 'Highway', 'Market', 'Cattle'};
    clearance_base = cellfun(@(r) r.min_clearance_m, baseline_results);
    clearance_adapt = cellfun(@(r) r.min_clearance_m, adaptive_results);

    smooth_base = cellfun(@(r) r.path_smoothness*1000, baseline_results);
    smooth_adapt = cellfun(@(r) r.path_smoothness*1000, adaptive_results);

    fig_comp = figure('Name', 'Baseline vs. Adaptive Autonomous Planning Comparison', ...
                      'NumberTitle', 'off', 'Color', [0.08 0.08 0.12], ...
                      'Position', [100 120 950 480]);

    ax1 = subplot(1, 2, 1, 'Parent', fig_comp, 'Color', [0.13 0.13 0.18], 'XColor', 'w', 'YColor', 'w');
    b1 = bar(ax1, [clearance_base, clearance_adapt]);
    b1(1).FaceColor = [0.85 0.35 0.35]; b1(1).EdgeColor = 'none'; % Baseline red
    b1(2).FaceColor = [0.30 0.85 0.45]; b1(2).EdgeColor = 'none'; % Adaptive green
    set(ax1, 'XTick', 1:5, 'XTickLabel', scen_labels, 'XTickLabelRotation', 15);
    title(ax1, 'Minimum Obstacle Clearance (m, Higher=Safer)', 'Color', 'w');
    legend(ax1, {'Baseline (Centerline)', 'Adaptive (Hybrid A*)'}, 'TextColor', 'w', 'Location', 'northwest');
    grid(ax1, 'on');

    ax2 = subplot(1, 2, 2, 'Parent', fig_comp, 'Color', [0.13 0.13 0.18], 'XColor', 'w', 'YColor', 'w');
    b2 = bar(ax2, [smooth_base, smooth_adapt]);
    b2(1).FaceColor = [0.85 0.35 0.35]; b2(1).EdgeColor = 'none';
    b2(2).FaceColor = [0.30 0.85 0.45]; b2(2).EdgeColor = 'none';
    set(ax2, 'XTick', 1:5, 'XTickLabel', scen_labels, 'XTickLabelRotation', 15);
    title(ax2, 'Path Roughness (×1000 rad/m, Lower=Smoother)', 'Color', 'w');
    legend(ax2, {'Baseline (Centerline)', 'Adaptive (Hybrid A*)'}, 'TextColor', 'w', 'Location', 'northeast');
    grid(ax2, 'on');

    sgtitle(fig_comp, 'ADAS Vision: Quantitative Superiority of Adaptive Path Planning', 'Color', 'w', 'FontSize', 13);
    drawnow;
catch
end

fprintf('\n🎉 Comparative benchmark suite complete!\n');
