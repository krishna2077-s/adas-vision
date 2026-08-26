function scenario_result = adaptive_scenario_loop(scen_cfg)
% ADAS Vision — Unified Adaptive Autonomous Scenario Execution Engine
%
% Integrates the complete closed-loop pipeline:
%   Perception -> Dynamic Trajectory Prediction (predict_trajectories.m)
%              -> Binary Occupancy Grid (build_occupancy_grid.m)
%              -> Adaptive Replan Trigger (replan_trigger.m)
%              -> Hybrid A* Path Planner (plan_path.m)
%              -> R1-R7 Decision Engine with Temporal Ratchet (decision_with_ratchet.m)
%              -> Pure Pursuit Steering & PID Throttle/Brake
%              -> Non-linear Kinematic Bicycle Dynamics
%              -> Continuous Obstacle Clearance & Telemetry Logging
%
% Inputs:
%   scen_cfg - Struct containing scenario parameters, road geometry, actors, and goal.

fprintf('=== [Unified Loop] Scenario: %s ===\n', scen_cfg.name);

% ---------------------------------------------------------------------------
% 1. Vehicle and Simulation Parameters
% ---------------------------------------------------------------------------
dt       = 0.033; % ~30 Hz control loop
if isfield(scen_cfg, 'dt'), dt = scen_cfg.dt; end
max_time = 35.0;
if isfield(scen_cfg, 'max_time'), max_time = scen_cfg.max_time; end

veh.mass_kg       = 1450;
veh.wheelbase_m   = 2.70;
veh.lf_m          = 1.15;
veh.lr_m          = 1.55;
veh.width_m       = 1.85;
veh.length_m      = 4.40;
veh.max_steer_rad = deg2rad(35.0);
veh.min_turning_radius = 5.0;

% Ego State [x, y, heading, speed]
ego = scen_cfg.ego_init;
goal = scen_cfg.goal;
goal_tol = 4.0;
if isfield(scen_cfg, 'goal_tol'), goal_tol = scen_cfg.goal_tol; end

% Decision & Ratchet State
prev_committed  = 0; % PROCEED
raw_history     = [];
down_counter    = 0;
emergency_latch = 0;

% Path Planning & Replanning State
current_path.x   = [];
current_path.y   = [];
current_path.yaw = [];
last_replan_tic  = tic;
replan_time_ms_log = [];

% Log Structure
log_data = struct();
log_data.t = [];
log_data.x = [];
log_data.y = [];
log_data.heading = [];
log_data.speed_mps = [];
log_data.steer_rad = [];
log_data.throttle = [];
log_data.brake = [];
log_data.decision = {};
log_data.n_tracks = [];
log_data.min_dist_m = [];
log_data.collisions = 0;
log_data.replans = 0;
log_data.replan_latency_ms = [];

event_log = {};
last_logged_decision = '';

% ---------------------------------------------------------------------------
% 2. Setup Figure and Visualization
% ---------------------------------------------------------------------------
fig = figure('Name', sprintf('ADAS Scenario — %s (Closed-Loop Adaptive)', scen_cfg.title), ...
             'NumberTitle', 'off', 'Color', [0.06 0.06 0.10], ...
             'Position', [60 100 1020 540]);
ax  = axes('Parent', fig, 'Color', [0.12 0.12 0.16], ...
           'XColor', 'w', 'YColor', 'w', 'GridColor', [0.25 0.25 0.35], 'GridAlpha', 0.4);
hold(ax, 'on'); grid(ax, 'on');
axis(ax, 'equal');

% Call custom scenario background drawing (roads, stalls, markings)
if isfield(scen_cfg, 'draw_background')
    scen_cfg.draw_background(ax);
end

% Goal plot
plot(ax, goal(1), goal(2), 'p', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.85 0], 'MarkerEdgeColor', 'w');
text(ax, goal(1)+2, goal(2)+1, 'GOAL', 'Color', [1 0.85 0], 'FontSize', 9, 'FontWeight', 'bold');

% Graphic handles
h_path = plot(ax, NaN, NaN, 'c--', 'LineWidth', 2.0); % Planned Hybrid A* path
h_pred = plot(ax, NaN, NaN, 'r:', 'LineWidth', 1.5); % Forecast trajectories
h_traj = plot(ax, ego.x, ego.y, 'g-', 'LineWidth', 1.5); % Driven history
h_ego  = plot(ax, ego.x, ego.y, 'o', 'MarkerSize', 14, 'MarkerFaceColor', [0 0.85 0.3], 'MarkerEdgeColor', 'w', 'LineWidth', 2);

% Actor handles map
actor_handles = containers.Map();
for k = 1:numel(scen_cfg.actors)
    act = scen_cfg.actors{k};
    col = [0.85 0.45 0.1];
    marker = 's';
    msize = 14;
    if strcmpi(act.class, 'person'), col = [1.0 0.4 0.4]; marker = 'o'; msize = 10;
    elseif strcmpi(act.class, 'cow'), col = [0.6 0.4 0.2]; marker = 's'; msize = 18;
    elseif strcmpi(act.class, 'motorcycle'), col = [0.2 0.7 1.0]; marker = '^'; msize = 12;
    elseif strcmpi(act.class, 'truck'), col = [0.9 0.5 0.1]; marker = 's'; msize = 20;
    end
    h_act = plot(ax, act.x, act.y, marker, 'MarkerSize', msize, 'MarkerFaceColor', col, 'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
    actor_handles(act.id) = h_act;
end

h_status = text(ax, 0.02, 0.94, '', 'Units', 'normalized', 'Color', 'w', ...
                'FontSize', 10, 'VerticalAlignment', 'top', 'FontWeight', 'bold');

% ---------------------------------------------------------------------------
% 3. Master Simulation Loop
% ---------------------------------------------------------------------------
t = 0;
arrived = false;
traj_x = ego.x;
traj_y = ego.y;
actors = scen_cfg.actors;

while t < max_time && ishandle(fig)
    dist_to_goal = norm([ego.x, ego.y] - goal(1:2));
    if dist_to_goal < goal_tol
        arrived = true;
        break;
    end

    % -----------------------------------------------------------------------
    % A. Update Actor Motions (World State)
    % -----------------------------------------------------------------------
    tracks_struct = [];
    all_pred_x = [];
    all_pred_y = [];
    min_dist_this_frame = Inf;
    critical_label = 'None';
    in_path_hazard = false;
    closing_speed = 0.0;
    hazard_dist = Inf;
    hazard_ttc = Inf;

    for k = 1:numel(actors)
        % Dynamic actor motion update
        actors{k} = scen_cfg.update_actor(actors{k}, t, dt, ego);
        act = actors{k};
        
        % Update graphics
        if isKey(actor_handles, act.id)
            set(actor_handles(act.id), 'XData', act.x, 'YData', act.y);
        end

        % Build track struct for perception/prediction
        dx = act.x - ego.x;
        dy = act.y - ego.y;
        dist_act = norm([dx, dy]);
        
        if dist_act < min_dist_this_frame
            min_dist_this_frame = dist_act;
            critical_label = act.class;
        end

        % Only process actors within perceptual range (50m ahead)
        if dx > -5.0 && dx < 60.0 && abs(dy) < 25.0
            tr.track_id = k;
            tr.x = act.x;
            tr.y = act.y;
            tr.vx = act.vx;
            tr.vy = act.vy;
            tr.class = act.class;
            tracks_struct = [tracks_struct, tr]; %#ok<AGROW>

            % Check path obstruction & closing velocity
            lat_dist = abs(dy);
            if dx > 0 && lat_dist < (veh.width_m/2 + act.width/2 + 0.8)
                in_path_hazard = true;
                v_rel = (ego.speed - act.vx * cos(ego.heading));
                if v_rel > 0
                    ttc = dx / max(0.1, v_rel);
                else
                    ttc = 99.0;
                end
                if dist_act < hazard_dist
                    hazard_dist = dist_act;
                    hazard_ttc = ttc;
                    closing_speed = v_rel;
                end
            end
        end
    end

    % Check collision
    if min_dist_this_frame < 0.8
        log_data.collisions = log_data.collisions + 1;
    end

    % -----------------------------------------------------------------------
    % B. Predict Trajectories (predict_trajectories.m)
    % -----------------------------------------------------------------------
    predictions.x = {};
    predictions.y = {};
    if ~isempty(tracks_struct)
        [pred_x, pred_y, ~] = predict_trajectories(tracks_struct, 3.0, 15);
        predictions.x = pred_x;
        predictions.y = pred_y;
        for p = 1:numel(pred_x)
            all_pred_x = [all_pred_x; pred_x{p}(:)]; %#ok<AGROW>
            all_pred_y = [all_pred_y; pred_y{p}(:)]; %#ok<AGROW>
        end
    end
    set(h_pred, 'XData', all_pred_x, 'YData', all_pred_y);

    % -----------------------------------------------------------------------
    % C. Build Dynamic Occupancy Map (build_occupancy_grid.m)
    % -----------------------------------------------------------------------
    grid_len = 60; grid_width = 30; res = 4;
    map = binaryOccupancyMap(grid_len, grid_width, res);
    map.GridLocationInWorld = [ego.x - 5, ego.y - 15];

    % Stamp road limits
    if isfield(scen_cfg, 'lane_half_width')
        lw = scen_cfg.lane_half_width;
        x_span = (ego.x - 5):0.5:(ego.x + 55);
        y_top = lw * ones(size(x_span));
        y_bot = -lw * ones(size(x_span));
        setOccupancy(map, [x_span', y_top'], 1);
        setOccupancy(map, [x_span', y_bot'], 1);
    end

    % Stamp obstacles and predicted poses
    for k = 1:numel(tracks_struct)
        tr = tracks_struct(k);
        % Stamp current bounding footprint
        [ox, oy] = meshgrid((tr.x - 1.0):0.25:(tr.x + 1.0), (tr.y - 0.8):0.25:(tr.y + 0.8));
        valid_pts = [ox(:), oy(:)];
        setOccupancy(map, valid_pts, 1);
    end

    % -----------------------------------------------------------------------
    % D. Replan Trigger Evaluation (replan_trigger.m)
    % -----------------------------------------------------------------------
    time_since_replan_ms = toc(last_replan_tic) * 1000;
    [need_replan, replan_reason] = replan_trigger(current_path, predictions, time_since_replan_ms, map);

    % -----------------------------------------------------------------------
    % E. Hybrid A* Path Planning (plan_path.m)
    % -----------------------------------------------------------------------
    if need_replan
        ego_pose = [ego.x, ego.y, ego.heading];
        goal_pose = [min(goal(1), ego.x + 40), goal(2), 0];
        
        try
            % Try Navigation Toolbox Hybrid A*
            [px, py, pyaw, plan_time_ms] = plan_path(map, ego_pose, goal_pose, veh);
        catch
            % Robust fallback smooth spline if Navigation Toolbox is absent
            tic;
            target_y = goal(2);
            % Check if obstacles occupy current corridor
            if in_path_hazard && hazard_dist < 25.0
                % Maneuver laterally away from hazard
                target_y = ego.y + sign(ego.y - 0.1) * 2.0;
            end
            n_pts = 30;
            px = linspace(ego.x, goal_pose(1), n_pts)';
            py = linspace(ego.y, target_y, n_pts)';
            pyaw = zeros(size(px));
            plan_time_ms = toc * 1000;
        end

        if ~isempty(px)
            current_path.x = px;
            current_path.y = py;
            current_path.yaw = pyaw;
            set(h_path, 'XData', px, 'YData', py);
            last_replan_tic = tic;
            log_data.replans = log_data.replans + 1;
            replan_time_ms_log(end+1) = plan_time_ms; %#ok<AGROW>
        end
    end

    % -----------------------------------------------------------------------
    % F. Decision Engine with Temporal Ratchet (decision_with_ratchet.m)
    % -----------------------------------------------------------------------
    lane_offset = ego.y;
    degraded = false;
    
    [committed_level, target_speed_mps, lateral_cmd, rule_id, raw_history, down_counter, emergency_latch] = ...
        decision_with_ratchet(hazard_dist, hazard_ttc, critical_label, in_path_hazard, closing_speed, ...
                              degraded, lane_offset, prev_committed, raw_history, down_counter, emergency_latch);
    prev_committed = committed_level;

    % Map level to string
    level_names = {'PROCEED', 'CAUTION', 'SLOW', 'BRAKE', 'EMERGENCY_STOP'};
    decision_str = level_names{committed_level + 1};

    % Log decision changes
    if ~strcmp(decision_str, last_logged_decision)
        msg = sprintf('t=%.1fs: %s (%s) — %s at %.1fm', t, decision_str, rule_id, critical_label, hazard_dist);
        fprintf('  %s\n', msg);
        event_log{end+1} = msg; %#ok<AGROW>
        last_logged_decision = decision_str;
    end

    % -----------------------------------------------------------------------
    % G. Pure Pursuit Steering & PID Velocity Control
    % -----------------------------------------------------------------------
    % Pure pursuit lookahead
    la = 6.0;
    steer = 0.0;
    if ~isempty(current_path.x)
        path_pts = [current_path.x, current_path.y];
        dists_to_path = sqrt(sum((path_pts - [ego.x, ego.y]).^2, 2));
        [~, closest_idx] = min(dists_to_path);
        target_idx = min(size(path_pts, 1), closest_idx + round(la / 0.5));
        
        target_pt = path_pts(target_idx, :);
        dx_la = target_pt(1) - ego.x;
        dy_la = target_pt(2) - ego.y;
        alpha = atan2(dy_la, dx_la) - ego.heading;
        alpha = atan2(sin(alpha), cos(alpha)); % wrap to [-pi, pi]
        steer = atan2(2 * veh.wheelbase_m * sin(alpha), la);
        steer = max(-veh.max_steer_rad, min(veh.max_steer_rad, steer));
    end

    % Longitudinal acceleration (PID target speed tracking with emergency override)
    if committed_level == 4 % EMERGENCY_STOP
        accel = -7.0; % Max decel
        throttle = 0.0; brake = 1.0;
    elseif committed_level == 3 % BRAKE
        accel = -4.0;
        throttle = 0.0; brake = 0.8;
    else
        accel = 2.0 * (target_speed_mps - ego.speed);
        if accel >= 0
            throttle = min(1.0, accel / 2.5); brake = 0.0;
        else
            throttle = 0.0; brake = min(1.0, -accel / 7.0);
        end
    end

    % -----------------------------------------------------------------------
    % H. Vehicle Kinematics Update (Bicycle Model)
    % -----------------------------------------------------------------------
    ego.speed   = max(0.0, ego.speed + accel * dt);
    ego.heading = ego.heading + (ego.speed / veh.wheelbase_m) * tan(steer) * dt;
    ego.x       = ego.x + ego.speed * cos(ego.heading) * dt;
    ego.y       = ego.y + ego.speed * sin(ego.heading) * dt;

    traj_x(end+1) = ego.x; %#ok<AGROW>
    traj_y(end+1) = ego.y; %#ok<AGROW>

    % -----------------------------------------------------------------------
    % I. Telemetry Logging
    % -----------------------------------------------------------------------
    log_data.t(end+1)           = t;
    log_data.x(end+1)           = ego.x;
    log_data.y(end+1)           = ego.y;
    log_data.heading(end+1)     = ego.heading;
    log_data.speed_mps(end+1)   = ego.speed;
    log_data.steer_rad(end+1)   = steer;
    log_data.throttle(end+1)    = throttle;
    log_data.brake(end+1)       = brake;
    log_data.decision{end+1}    = decision_str;
    log_data.n_tracks(end+1)    = numel(tracks_struct);
    log_data.min_dist_m(end+1)  = min_dist_this_frame;

    % -----------------------------------------------------------------------
    % J. Graphics Update
    % -----------------------------------------------------------------------
    set(h_ego,  'XData', ego.x, 'YData', ego.y);
    set(h_traj, 'XData', traj_x, 'YData', traj_y);

    xlim(ax, [ego.x - 15, ego.x + 65]);

    col = scenario_color(decision_str);
    set(h_status, 'String', sprintf('t=%.1fs | Decision: %s [%s] | Speed: %.1f km/h | Min Clearance: %.1fm | Replans: %d', ...
        t, decision_str, rule_id, ego.speed*3.6, min_dist_this_frame, log_data.replans), 'Color', col);
    title(ax, sprintf('Scenario: %s  |  Status: %s', scen_cfg.title, decision_str), 'Color', col, 'FontSize', 12);

    drawnow limitrate;
    t = t + dt;
end

% ---------------------------------------------------------------------------
% 4. Summary and Result Packaging
% ---------------------------------------------------------------------------
fprintf('\n=== %s Complete ===\n', scen_cfg.name);
if arrived
    fprintf('  STATUS: GOAL REACHED in %.1f s\n', t);
else
    fprintf('  STATUS: TIMEOUT at %.1f s\n', t);
end
fprintf('  Events logged: %d\n', numel(event_log));
fprintf('  Total Replans: %d (Mean latency: %.2f ms)\n', log_data.replans, mean(replan_time_ms_log));
fprintf('  True Min Clearance: %.2f m\n', min(log_data.min_dist_m));
fprintf('  Collisions: %d\n\n', log_data.collisions);

log_data.replan_latency_ms = replan_time_ms_log;
log_data.scenario_completed = arrived;

scenario_result = log_data;
scenario_result.name = scen_cfg.name;
scenario_result.t_total = t;
scenario_result.arrived = arrived;
scenario_result.traj_x = traj_x;
scenario_result.traj_y = traj_y;

assignin('base', 'scenario_result', scenario_result);

function c = scenario_color(d)
    switch d
        case 'PROCEED',        c = [0.2 0.9 0.2];
        case 'CAUTION',        c = [1.0 0.8 0.0];
        case 'SLOW',           c = [1.0 0.5 0.0];
        case 'BRAKE',          c = [1.0 0.2 0.0];
        case 'EMERGENCY_STOP', c = [1.0 0.0 0.0];
        otherwise,             c = [0.8 0.8 0.8];
    end
end
end
