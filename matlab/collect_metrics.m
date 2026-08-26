function metrics = collect_metrics(log_data, scenario_name, varargin)
% collect_metrics — ADAS Vision Performance Metrics Collector
%
% Computes comprehensive performance metrics from a simulation log and
% saves them to a CSV file for the hackathon report.
%
% Usage:
%   metrics = collect_metrics(log_data, 'village_road')
%   metrics = collect_metrics(log_data, 'cattle_crossing', 'SaveCSV', false)
%
% Inputs:
%   log_data      — struct from run_cosimulation or scenario scripts
%   scenario_name — string label for this run (e.g. 'village_road')
%
% Optional name-value pairs:
%   'SaveCSV'     — true (default) to write metrics_<scenario>.csv
%   'OutDir'      — directory for CSV output (default: pwd)
%   'PlotSummary' — true (default) to show a bar-chart summary figure
%
% Returns:
%   metrics — struct with all computed metric values

% ---------------------------------------------------------------------------
% Parse arguments
% ---------------------------------------------------------------------------
p = inputParser();
p.addRequired ('log_data',      @isstruct);
p.addRequired ('scenario_name', @ischar);
p.addParameter('SaveCSV',       true,  @islogical);
p.addParameter('OutDir',        pwd,   @ischar);
p.addParameter('PlotSummary',   true,  @islogical);
p.parse(log_data, scenario_name, varargin{:});
opt = p.Results;

fprintf('=== collect_metrics: %s ===\n', scenario_name);

metrics = struct();
metrics.scenario = scenario_name;
metrics.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

% ---------------------------------------------------------------------------
% 1. Basic scenario stats
% ---------------------------------------------------------------------------
if isfield(log_data,'t') && ~isempty(log_data.t)
    metrics.duration_s    = log_data.t(end) - log_data.t(1);
    metrics.n_frames      = numel(log_data.t);
    metrics.control_hz    = metrics.n_frames / max(metrics.duration_s, 1e-6);
else
    metrics.duration_s = 0; metrics.n_frames = 0; metrics.control_hz = 0;
end

% ---------------------------------------------------------------------------
% 2. Distance travelled
% ---------------------------------------------------------------------------
if isfield(log_data,'x') && numel(log_data.x) > 1
    dx = diff(log_data.x);
    dy = diff(log_data.y);
    metrics.distance_m = sum(sqrt(dx.^2 + dy.^2));
else
    metrics.distance_m = 0;
end

% ---------------------------------------------------------------------------
% 3. Path smoothness (mean absolute curvature)
% ---------------------------------------------------------------------------
metrics.path_smoothness = compute_smoothness(log_data);

% ---------------------------------------------------------------------------
% 4. Speed statistics
% ---------------------------------------------------------------------------
if isfield(log_data,'speed_mps') && ~isempty(log_data.speed_mps)
    spd = log_data.speed_mps;
    metrics.mean_speed_kmh = mean(spd) * 3.6;
    metrics.max_speed_kmh  = max(spd)  * 3.6;
    metrics.min_speed_kmh  = min(spd)  * 3.6;
    % Time spent stationary (speed < 0.1 m/s) as fraction of total
    metrics.idle_fraction  = mean(spd < 0.1);
else
    metrics.mean_speed_kmh = 0; metrics.max_speed_kmh = 0;
    metrics.min_speed_kmh = 0; metrics.idle_fraction = 0;
end

% ---------------------------------------------------------------------------
% 5. Replanning latency
% ---------------------------------------------------------------------------
if isfield(log_data,'replan_latency_ms') && ~isempty(log_data.replan_latency_ms)
    rl = log_data.replan_latency_ms;
    metrics.n_replans          = numel(rl);
    metrics.mean_replan_ms     = mean(rl);
    metrics.max_replan_ms      = max(rl);
    metrics.p95_replan_ms      = prctile(rl, 95);
else
    metrics.n_replans = 0; metrics.mean_replan_ms = 0;
    metrics.max_replan_ms = 0; metrics.p95_replan_ms = 0;
end

% ---------------------------------------------------------------------------
% 6. Collisions
% ---------------------------------------------------------------------------
metrics.collisions = 0;
if isfield(log_data,'collisions')
    metrics.collisions = log_data.collisions;
end

% ---------------------------------------------------------------------------
% 7. Decision distribution
% ---------------------------------------------------------------------------
metrics.decision_dist = compute_decision_dist(log_data);
if isfield(metrics.decision_dist,'PROCEED')
    metrics.proceed_frac = metrics.decision_dist.PROCEED;
else
    metrics.proceed_frac = 0;
end

% ---------------------------------------------------------------------------
% 8. Minimum obstacle clearance (from tracks if available)
% ---------------------------------------------------------------------------
metrics.min_clearance_m = compute_min_clearance(log_data);

% ---------------------------------------------------------------------------
% 9. Lateral control error (RMS offset from centreline if available)
% ---------------------------------------------------------------------------
if isfield(log_data,'y') && numel(log_data.y) > 1
    % For a straight road scenario, y should stay near 0
    metrics.lateral_rms_m = sqrt(mean(log_data.y.^2));
else
    metrics.lateral_rms_m = 0;
end

% ---------------------------------------------------------------------------
% 10. Braking harshness (jerk in longitudinal speed)
% ---------------------------------------------------------------------------
metrics.max_decel_mps2 = 0;
metrics.mean_jerk = 0;
if isfield(log_data,'speed_mps') && numel(log_data.speed_mps) > 2
    dv = diff(log_data.speed_mps);
    dt_arr = diff(log_data.t);
    dt_arr(dt_arr < 1e-6) = 1e-6;
    accel = dv ./ dt_arr;
    decel = -accel(accel < 0);
    if ~isempty(decel)
        metrics.max_decel_mps2 = max(decel);
    end
    jerk = diff(accel) ./ dt_arr(1:end-1);
    metrics.mean_jerk = mean(abs(jerk));
end

% ---------------------------------------------------------------------------
% 11. Scenario completion flag
% ---------------------------------------------------------------------------
metrics.scenario_completed = 0;
if isfield(log_data,'scenario_completed')
    metrics.scenario_completed = double(log_data.scenario_completed);
end

% ---------------------------------------------------------------------------
% Print summary
% ---------------------------------------------------------------------------
fprintf('\n--- Metrics Summary ---\n');
fprintf('  Scenario          : %s\n',  metrics.scenario);
fprintf('  Duration          : %.1f s\n', metrics.duration_s);
fprintf('  Distance          : %.1f m\n', metrics.distance_m);
fprintf('  Mean speed        : %.1f km/h\n', metrics.mean_speed_kmh);
fprintf('  Path smoothness   : %.4f rad/m (lower = smoother)\n', metrics.path_smoothness);
fprintf('  Lateral RMS       : %.2f m\n', metrics.lateral_rms_m);
fprintf('  Min clearance     : %.1f m\n', metrics.min_clearance_m);
fprintf('  Collisions        : %d\n',    metrics.collisions);
fprintf('  Replans           : %d  (mean %.1f ms, P95 %.1f ms)\n',...
        metrics.n_replans, metrics.mean_replan_ms, metrics.p95_replan_ms);
fprintf('  Max decel         : %.2f m/s²\n', metrics.max_decel_mps2);
fprintf('  Idle fraction     : %.0f%%\n', metrics.idle_fraction * 100);
fprintf('  PROCEED fraction  : %.0f%%\n', metrics.proceed_frac * 100);
fprintf('  Completed         : %s\n', ternary(metrics.scenario_completed, 'YES', 'NO'));

% ---------------------------------------------------------------------------
% Save CSV
% ---------------------------------------------------------------------------
if opt.SaveCSV
    save_metrics_csv(metrics, opt.OutDir);
end

% ---------------------------------------------------------------------------
% Plot summary figure
% ---------------------------------------------------------------------------
if opt.PlotSummary
    plot_metrics_summary(metrics);
end

fprintf('\n  collect_metrics done.\n');
end


% ===========================================================================
% Internal helpers
% ===========================================================================

function s = compute_smoothness(ld)
    s = 0;
    if ~isfield(ld,'x') || numel(ld.x) < 3, return; end
    dx1 = diff(ld.x); dy1 = diff(ld.y);
    dx2 = diff(dx1);  dy2 = diff(dy1);
    ds  = sqrt(dx1(1:end-1).^2 + dy1(1:end-1).^2) + 1e-9;
    kap = abs(dx1(1:end-1).*dy2 - dy1(1:end-1).*dx2) ./ (ds.^3 + 1e-9);
    s   = mean(kap);
end

function dist = compute_decision_dist(ld)
    dist = struct('PROCEED',0,'CAUTION',0,'SLOW',0,'BRAKE',0,'EMERGENCY_STOP',0);
    if ~isfield(ld,'decision') || isempty(ld.decision), return; end
    n = numel(ld.decision);
    flds = fieldnames(dist);
    for k = 1:numel(flds)
        cnt = sum(cellfun(@(d) strcmpi(d, flds{k}), ld.decision));
        dist.(flds{k}) = cnt / n;
    end
end

function mc = compute_min_clearance(ld)
    mc = Inf;
    if ~isfield(ld,'n_tracks'), mc = 5.0; return; end
    % Approximate: if min distance info is in log use it, else use n_tracks heuristic
    if isfield(ld,'min_dist_m') && ~isempty(ld.min_dist_m)
        mc = min(ld.min_dist_m);
    else
        mc = 5.0;  % default when detailed track log not available
    end
    if isinf(mc), mc = 5.0; end
end

function save_metrics_csv(m, outdir)
    fname = fullfile(outdir, sprintf('metrics_%s.csv', strrep(m.scenario,' ','_')));
    try
        fid = fopen(fname, 'w');
        if fid < 0
            fprintf('[WARN] Could not write CSV to %s\n', fname);
            return;
        end
        fprintf(fid, 'metric,value,unit\n');
        fprintf(fid, 'scenario,%s,\n',        m.scenario);
        fprintf(fid, 'timestamp,%s,\n',        m.timestamp);
        fprintf(fid, 'duration_s,%.2f,s\n',    m.duration_s);
        fprintf(fid, 'distance_m,%.2f,m\n',    m.distance_m);
        fprintf(fid, 'mean_speed_kmh,%.2f,km/h\n', m.mean_speed_kmh);
        fprintf(fid, 'max_speed_kmh,%.2f,km/h\n',  m.max_speed_kmh);
        fprintf(fid, 'path_smoothness,%.5f,rad/m\n', m.path_smoothness);
        fprintf(fid, 'lateral_rms_m,%.3f,m\n',  m.lateral_rms_m);
        fprintf(fid, 'min_clearance_m,%.2f,m\n',m.min_clearance_m);
        fprintf(fid, 'collisions,%d,count\n',   m.collisions);
        fprintf(fid, 'n_replans,%d,count\n',    m.n_replans);
        fprintf(fid, 'mean_replan_ms,%.2f,ms\n',m.mean_replan_ms);
        fprintf(fid, 'p95_replan_ms,%.2f,ms\n', m.p95_replan_ms);
        fprintf(fid, 'max_decel_mps2,%.3f,m/s2\n', m.max_decel_mps2);
        fprintf(fid, 'idle_fraction,%.3f,frac\n',   m.idle_fraction);
        fprintf(fid, 'proceed_fraction,%.3f,frac\n',m.proceed_frac);
        fprintf(fid, 'scenario_completed,%d,bool\n',m.scenario_completed);
        fclose(fid);
        fprintf('  CSV saved: %s\n', fname);
    catch e
        fprintf('[WARN] CSV save failed: %s\n', e.message);
    end
end

function plot_metrics_summary(m)
    labels = {'Speed\n(km/h)', 'Smoothness\n(×100)', 'Clearance\n(m)',...
              'Replan\n(ms)', 'Idle\n(%%)', 'PROCEED\n(%%)'};
    vals   = [m.mean_speed_kmh, m.path_smoothness*100, m.min_clearance_m,...
              m.mean_replan_ms, m.idle_fraction*100, m.proceed_frac*100];

    fig2 = figure('Name',sprintf('Metrics — %s', m.scenario),...
                  'NumberTitle','off','Color',[0.08 0.08 0.10],...
                  'Position',[200 200 780 400]);
    ax2  = axes('Parent',fig2,'Color',[0.12 0.12 0.16],...
                'XColor','w','YColor','w','GridColor',[0.3 0.3 0.35],...
                'GridAlpha',0.5);
    hold(ax2,'on'); grid(ax2,'on');

    colors = [0.2 0.8 0.4; 0.2 0.6 1.0; 1.0 0.7 0.2;
              0.9 0.4 0.4; 0.8 0.5 1.0; 0.4 0.9 0.9];

    for k = 1:numel(vals)
        b = bar(ax2, k, vals(k), 0.6, 'FaceColor', colors(k,:), 'EdgeColor','none');
        text(ax2, k, vals(k)+max(vals)*0.02, sprintf('%.1f', vals(k)),...
             'Color','w','HorizontalAlignment','center','FontSize',9);
    end

    set(ax2,'XTick',1:numel(labels),'XTickLabel',...
        cellfun(@(s) strrep(s,'\n',newline), labels,'UniformOutput',false),...
        'XTickLabelRotation',0);
    title(ax2, sprintf('ADAS Metrics — %s  |  Collisions: %d  |  Replans: %d',...
          m.scenario, m.collisions, m.n_replans), 'Color','w','FontSize',12);
    ylabel(ax2, 'Value (units per label)', 'Color','w');

    % Collision indicator
    if m.collisions > 0
        text(ax2, 0.98, 0.95, sprintf('⚠ %d COLLISION(S)', m.collisions),...
             'Units','normalized','Color',[1 0.2 0.2],...
             'HorizontalAlignment','right','FontSize',12,'FontWeight','bold');
    else
        text(ax2, 0.98, 0.95, '✓ NO COLLISIONS',...
             'Units','normalized','Color',[0.2 0.9 0.3],...
             'HorizontalAlignment','right','FontSize',12,'FontWeight','bold');
    end

    drawnow;
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
