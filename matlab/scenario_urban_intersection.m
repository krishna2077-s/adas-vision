function scenario_result = scenario_urban_intersection(mode)
% ADAS Vision — Driving Scenario 2: Unsignalised Urban Intersection
%
% Tests adaptive path planning and decision logic in a busy 4-way crossroad
% with auto-rickshaws, crossing pedestrians, cyclists, and cross-traffic.
% Uses the unified closed-loop adaptive pipeline:
%   build_occupancy_grid -> predict_trajectories -> replan_trigger -> plan_path -> decision_with_ratchet
%
% Usage:
%   >> scenario_urban_intersection
%   >> scenario_urban_intersection('baseline')

if nargin < 1, mode = 'adaptive'; end

cfg = struct();
cfg.name  = 'urban_intersection';
cfg.title = 'Scenario 2 — Urban Intersection';
cfg.mode  = mode;
cfg.dt    = 0.033;
cfg.max_time = 30.0;
cfg.lane_half_width = 4.0;

% Ego starting from South, heading North (+X axis in local coordinates)
cfg.ego_init.x       = 0.0;
cfg.ego_init.y       = 0.0;
cfg.ego_init.heading = 0.0;
cfg.ego_init.speed   = 6.94; % 25 km/h

cfg.goal = [120.0, 0.0, 0.0];
cfg.goal_tol = 4.0;

% Actors
% 1. Cross-traffic Car (from East to West at x=50m)
car1.id    = 'car1';
car1.class = 'car';
car1.x     = 50.0;
car1.y     = 25.0;
car1.vx    = 0.0;
car1.vy    = -5.55; % 20 km/h
car1.width = 1.8;
car1.length= 4.2;

% 2. Auto-rickshaw turning into intersection (x=55m)
rick.id    = 'rick';
rick.class = 'auto_rickshaw'; % Explicit auto-rickshaw actor
rick.x     = 55.0;
rick.y     = -20.0;
rick.vx    = 0.4;
rick.vy    = 3.88; % 14 km/h
rick.width = 1.4;
rick.length= 2.8;

% 3. Crossing Pedestrian (x=45m zebra crossing)
ped.id    = 'ped';
ped.class = 'person';
ped.x     = 45.0;
ped.y     = -6.0;
ped.vx    = 0.05;
ped.vy    = 1.1;
ped.width = 0.5;
ped.length= 0.5;

% 4. Cyclist proceeding through intersection (x=70m)
cyc.id    = 'cyc';
cyc.class = 'bicycle';
cyc.x     = 70.0;
cyc.y     = -1.5;
cyc.vx    = 3.88;
cyc.vy    = 0.1;
cyc.width = 0.6;
cyc.length= 1.8;

cfg.actors = {car1, rick, ped, cyc};
cfg.update_actor = @(act, t, dt, ego) update_intersection_actors(act, t, dt);
cfg.draw_background = @(ax) draw_intersection_background(ax, cfg.lane_half_width);

scenario_result = adaptive_scenario_loop(cfg);
end

% ---------------------------------------------------------------------------
% Helper Functions
% ---------------------------------------------------------------------------
function act = update_intersection_actors(act, t, dt)
    act.x = act.x + act.vx * dt;
    act.y = act.y + act.vy * dt;
end

function draw_intersection_background(ax, lw)
    fill(ax, [0 140 140 0], [-lw -lw lw lw], [0.20 0.20 0.22], 'EdgeColor', 'none');
    % Crossing Arm (y-axis crossing at x=50m)
    fill(ax, [42 58 58 42], [-40 -40 40 40], [0.20 0.20 0.22], 'EdgeColor', 'none');
    % Zebra Crossing Markings at x=45m
    for zy = -lw+0.4:0.8:lw-0.4
        plot(ax, [44 47], [zy zy], 'w-', 'LineWidth', 3);
    end
    ylim(ax, [-18, 18]);
end
