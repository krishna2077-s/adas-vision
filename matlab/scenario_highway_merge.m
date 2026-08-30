function scenario_result = scenario_highway_merge(mode)
% ADAS Vision — Driving Scenario 3: Indian Highway Merge
%
% Tests vehicle navigation during highway merging with high speed differentials
% and unpredictable commercial vehicle cut-in behavior.
% Uses the unified closed-loop adaptive pipeline:
%   build_occupancy_grid -> predict_trajectories -> replan_trigger -> plan_path -> decision_with_ratchet
%
% Usage:
%   >> scenario_highway_merge
%   >> scenario_highway_merge('baseline')

if nargin < 1, mode = 'adaptive'; end

cfg = struct();
cfg.name  = 'highway_merge';
cfg.title = 'Scenario 3 — Highway Merge';
cfg.mode  = mode;
cfg.dt    = 0.033;
cfg.max_time = 90.0;  % Extended: 25 km/h needs more time to cover 200m
cfg.lane_half_width = 7.5; % 2-lane dual carriageway

cfg.ego_init.x       = 0.0;
cfg.ego_init.y       = -1.875; % Left cruising lane
cfg.ego_init.heading = 0.0;
cfg.ego_init.speed   = 13.89;  % 50 km/h — realistic highway speed

cfg.goal = [200.0, -1.875, 0.0];  % Shortened: reachable at 25 km/h in ~50s
cfg.goal_tol = 4.0;  % Wider catch — prevents x-drift overshoot

% Actors
% 1. Heavy Commercial Truck (Tata 1613 style, merges from on-ramp)
truck.id    = 'truck';
truck.class = 'truck';
truck.x     = 100.0;  % Starts well ahead — ego has free road for first 60m
truck.y     = -6.5;
truck.vx    = 3.33;   % 12 km/h
truck.vy    = 2.2;    % Fast merge — completes quickly, clears ego's path
truck.width = 2.5;
truck.length= 8.5;

% 2. Fast Overtaking SUV in right lane (y = +1.875)
suv.id    = 'suv';
suv.class = 'car';
suv.x     = -20.0;  % Closer start — overtake visible during sim
suv.y     = 1.875;
suv.vx    = 19.44; % 70 km/h fast overtaking SUV
suv.vy    = 0.0;
suv.width = 1.9;
suv.length= 4.8;

% 3. Wrong-way two-wheeler on left shoulder
bike.id    = 'bike';
bike.class = 'motorcycle';
bike.x     = 150.0;  % Within goal range
bike.y     = -8.5;
bike.vx    = -6.94; % Wrong-way ~25 km/h
bike.vy    = 0.0;
bike.width = 0.8;
bike.length= 1.8;

cfg.actors = {truck, suv, bike};
cfg.update_actor = @(act, t, dt, ego) update_highway_actors(act, t, dt, ego);
cfg.draw_background = @(ax) draw_highway_background(ax, cfg.lane_half_width);

scenario_result = adaptive_scenario_loop(cfg);
end

% ---------------------------------------------------------------------------
% Helper Functions
% ---------------------------------------------------------------------------
function act = update_highway_actors(act, t, dt, ego)
    if strcmp(act.id, 'truck')
        % Truck merges until reaching center of left lane (y = -1.875)
        if act.y < -1.875
            act.y = min(-1.875, act.y + act.vy * dt);
        end
        act.x = act.x + act.vx * dt;
    elseif strcmp(act.id, 'bike')
        % Wrong-way bike: slow down and swerve to shoulder if ego is approaching
        dist_to_ego = abs(act.x - ego.x);
        if dist_to_ego < 20.0 && abs(act.y - ego.y) < 2.5
            % Swerve hard to the far shoulder (y = -8.5, furthest from ego at y=-1.875)
            act.vx = max(-2.0, act.vx + 4.0 * dt); % Decelerate
            act.vy = -1.5; % Swerve to shoulder
        else
            act.vy = 0.0;
        end
        act.x = act.x + act.vx * dt;
        act.y = max(-8.5, act.y + act.vy * dt); % Clamp to shoulder
    else
        act.x = act.x + act.vx * dt;
        act.y = act.y + act.vy * dt;
    end
end

function draw_highway_background(ax, lw)
    fill(ax, [0 280 280 0], [-lw -lw lw lw], [0.18 0.18 0.22], 'EdgeColor', 'none');
    % Merge Ramp on-ramp taper
    fill(ax, [30 160 160 30], [-lw-4 -lw-4 -lw -lw], [0.18 0.18 0.22], 'EdgeColor', 'none');
    % Lane divider
    plot(ax, [0 280], [0 0], 'w--', 'LineWidth', 1.2);
    % Shoulders
    plot(ax, [0 280], [-lw -lw], 'w-', 'LineWidth', 2.0);
    plot(ax, [0 280], [lw lw], 'w-', 'LineWidth', 2.0);
    ylim(ax, [-lw-6, lw+4]);
end
