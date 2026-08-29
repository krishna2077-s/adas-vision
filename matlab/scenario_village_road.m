function scenario_result = scenario_village_road(mode)
% ADAS Vision — Driving Scenario 1: Unmarked Indian Village Road
%
% Tests adaptive navigation on a narrow, unmarked single-carriageway road with
% oncoming two-wheelers, crossing pedestrians, and parked pushcarts.
% Uses the unified closed-loop adaptive pipeline:
%   build_occupancy_grid -> predict_trajectories -> replan_trigger -> plan_path -> decision_with_ratchet
%
% Usage:
%   >> scenario_village_road
%   >> scenario_village_road('baseline')

if nargin < 1, mode = 'adaptive'; end

cfg = struct();
cfg.name  = 'village_road';
cfg.title = 'Scenario 1 — Unmarked Village Road';
cfg.mode  = mode;
cfg.dt    = 0.033;
cfg.max_time = 50.0;
cfg.lane_half_width = 3.2;

% Initial Ego Pose
cfg.ego_init.x       = 0.0;
cfg.ego_init.y       = -0.4; % Driving slightly left of center (LHT side)
cfg.ego_init.heading = 0.0;
cfg.ego_init.speed   = 8.33; % 30 km/h

% Goal
cfg.goal = [180.0, 0.0, 0.0];
cfg.goal_tol = 1.2;

% Scenario Actors
% 1. Oncoming motorcycle (x=160, moving on opposite side y=+1.2)
mcycle.id    = 'mcycle';
mcycle.class = 'motorcycle';
mcycle.x     = 160.0;
mcycle.y     = 1.2;
mcycle.vx    = -6.5; % Oncoming
mcycle.vy    = 0.0;
mcycle.width = 0.8;
mcycle.length= 2.0;

% 2. Pedestrian crossing road (x=85, y=-2.8 to +3.2)
ped.id    = 'ped';
ped.class = 'person';
ped.x     = 85.0;
ped.y     = -2.8;
ped.vx    = 0.05;
ped.vy    = 0.65; % crossing across road
ped.width = 0.5;
ped.length= 0.5;

% 3. Parked pushcart on left shoulder (x=130, y=1.8)
cart.id    = 'cart';
cart.class = 'pushcart';
cart.x     = 130.0;
cart.y     = 1.8;
cart.vx    = 0.0;
cart.vy    = 0.0;
cart.width = 1.3;
cart.length= 2.0;

cfg.actors = {mcycle, ped, cart};

% Actor dynamic motion callback
cfg.update_actor = @(act, t, dt, ego) update_village_actors(act, t, dt);

% Custom background graphics
cfg.draw_background = @(ax) draw_village_background(ax, cfg.lane_half_width);

% Run unified closed loop
scenario_result = adaptive_scenario_loop(cfg);
end

% ---------------------------------------------------------------------------
% Helper Functions
% ---------------------------------------------------------------------------
function act = update_village_actors(act, t, dt)
    if strcmp(act.id, 'ped')
        % Pedestrian crosses road once t >= 1.2s
        if t >= 1.2 && act.y < 3.2
            act.y = act.y + act.vy * dt;
            act.x = act.x + act.vx * dt;
        end
    else
        act.x = act.x + act.vx * dt;
        act.y = act.y + act.vy * dt;
    end
end

function draw_village_background(ax, lw)
    fill(ax, [0 200 200 0], [-lw -lw lw lw], [0.25 0.23 0.20], 'EdgeColor', 'none');
    % Dirt/grass shoulders (unmarked edges)
    fill(ax, [0 200 200 0], [-lw-5 -lw-5 -lw -lw], [0.18 0.28 0.12], 'EdgeColor', 'none');
    fill(ax, [0 200 200 0], [lw lw lw+5 lw+5], [0.18 0.28 0.12], 'EdgeColor', 'none');
    % Faint broken centerline
    plot(ax, [0 200], [0 0], '--', 'Color', [0.5 0.45 0.3], 'LineWidth', 0.8);
    ylim(ax, [-lw-6, lw+6]);
end
