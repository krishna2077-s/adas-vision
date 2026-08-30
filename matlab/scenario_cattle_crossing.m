function scenario_result = scenario_cattle_crossing(mode)
% ADAS Vision — Driving Scenario 5: Sudden Cattle Crossing
%
% Tests emergency collision avoidance and adaptive resumption when cattle
% unpredictably enter the carriageway.
% Uses the unified closed-loop adaptive pipeline:
%   build_occupancy_grid -> predict_trajectories -> replan_trigger -> plan_path -> decision_with_ratchet
%
% Usage:
%   >> scenario_cattle_crossing
%   >> scenario_cattle_crossing('baseline')

if nargin < 1, mode = 'adaptive'; end

cfg = struct();
cfg.name  = 'cattle_crossing';
cfg.title = 'Scenario 5 — Cattle Crossing';
cfg.mode  = mode;
cfg.dt    = 0.033;
cfg.max_time = 50.0;
cfg.lane_half_width = 3.5;

cfg.ego_init.x       = 0.0;
cfg.ego_init.y       = -0.4;
cfg.ego_init.heading = 0.0;
cfg.ego_init.speed   = 4.17; % 15 km/h

cfg.goal = [160.0, 0.0, 0.0];
cfg.goal_tol = 4.0;  % Wider catch — prevents x-drift overshoot

% Actors
% Cow 1: Starts at x=75m, y=-3.5m (crosses to y=+3.5m)
cow1.id    = 'cow1';
cow1.class = 'cow';
cow1.x     = 75.0;
cow1.y     = -3.5;
cow1.vx    = 0.05;  % very slow x drift
cow1.vy    = 0.80;  % ~2.9 km/h lateral — realistic cow walk
cow1.width = 1.5;
cow1.length= 2.2;

% Cow 2: Following cow1 at x=79m, y=-4.0m
cow2.id    = 'cow2';
cow2.class = 'cow';
cow2.x     = 79.0;
cow2.y     = -4.2;
cow2.vx    = 0.05;
cow2.vy    = 0.70;  % ~2.5 km/h lateral — realistic cow walk
cow2.width = 1.5;
cow2.length= 2.2;

cfg.actors = {cow1, cow2};
cfg.update_actor = @(act, t, dt, ego) update_cattle_actors(act, t, dt);
cfg.draw_background = @(ax) draw_cattle_background(ax, cfg.lane_half_width);

scenario_result = adaptive_scenario_loop(cfg);
end

% ---------------------------------------------------------------------------
% Helper Functions
% ---------------------------------------------------------------------------
function act = update_cattle_actors(act, t, dt)
    % Cows start moving onto road at t=2.5s
    if t >= 2.5 && act.y < 3.8
        act.x = act.x + act.vx * dt;
        act.y = act.y + act.vy * dt;
    end
end

function draw_cattle_background(ax, lw)
    fill(ax, [0 180 180 0], [-lw -lw lw lw], [0.22 0.22 0.24], 'EdgeColor', 'none');
    % Grass shoulders
    fill(ax, [0 180 180 0], [-lw-5 -lw-5 -lw -lw], [0.15 0.28 0.12], 'EdgeColor', 'none');
    fill(ax, [0 180 180 0], [lw lw lw+5 lw+5], [0.15 0.28 0.12], 'EdgeColor', 'none');
    % Road lines
    plot(ax, [0 180], [-lw -lw], 'w-', 'LineWidth', 1.5);
    plot(ax, [0 180], [lw lw], 'w-', 'LineWidth', 1.5);
    plot(ax, [0 180], [0 0], 'w--', 'LineWidth', 0.8);
    % Cattle warning sign on shoulder
    plot(ax, 50, -lw-1.5, '^', 'MarkerSize', 14, 'MarkerFaceColor', [1 0.9 0], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    text(ax, 50, -lw-3.5, {'CATTLE','XING'}, 'Color', [1 0.9 0], 'FontSize', 8, 'HorizontalAlignment', 'center');
    ylim(ax, [-lw-6, lw+6]);
end
