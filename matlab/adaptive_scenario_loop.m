function scenario_result = adaptive_scenario_loop(scen_cfg)
% ADAS Vision — Unified Adaptive Autonomous Scenario Execution Engine
%
% Supports two evaluation modes:
%   - 'adaptive' (default): Dynamic occupancy grid + prediction stamping + Frenet replan + R1-R7 ratchet
%   - 'baseline': Fixed road centerline + purely reactive threshold braking (no predictive replanning)
%
% ADAS Vision — SIH 2026 Autonomous Navigation Suite

mode = 'adaptive';
if isfield(scen_cfg, 'mode'), mode = lower(scen_cfg.mode); end
is_baseline = strcmp(mode, 'baseline');

fprintf('=== [%s Mode] Scenario: %s ===\n', upper(mode), scen_cfg.name);

% ---------------------------------------------------------------------------
% 1. Vehicle and Simulation Parameters
% ---------------------------------------------------------------------------
dt       = 0.033; % ~30 Hz control loop
if isfield(scen_cfg, 'dt'), dt = scen_cfg.dt; end
max_time = 50.0;
if isfield(scen_cfg, 'max_time'), max_time = scen_cfg.max_time; end

veh.mass_kg       = 1450;
veh.wheelbase_m   = 2.70;
veh.lf_m          = 1.15;
veh.lr_m          = 1.55;
veh.width_m       = 1.85;
veh.length_m      = 4.40;
veh.max_steer_rad = deg2rad(35.0);
veh.min_turning_radius = 4.5;

% Ego State [x, y, heading, speed]
ego = scen_cfg.ego_init;
goal = scen_cfg.goal;
goal_tol = 1.5; % Tolerance for arrival
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
fig_title = sprintf('ADAS Scenario — %s (%s)', scen_cfg.title, upper(mode));
fig = figure('Name', fig_title, 'NumberTitle', 'off', 'Color', [0.06 0.06 0.10], ...
             'Position', [60 100 1020 540]);
ax  = axes('Parent', fig, 'Color', [0.12 0.12 0.16], ...
           'XColor', 'w', 'YColor', 'w', 'GridColor', [0.25 0.25 0.35], 'GridAlpha', 0.4);
hold(ax, 'on'); grid(ax, 'on');
axis(ax, 'equal');

% Call custom scenario background drawing
if isfield(scen_cfg, 'draw_background')
    scen_cfg.draw_background(ax);
end

% Goal plot
plot(ax, goal(1), goal(2), 'p', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.85 0], 'MarkerEdgeColor', 'w');
text(ax, goal(1)+1.5, goal(2)+0.8, 'GOAL', 'Color', [1 0.85 0], 'FontSize', 9, 'FontWeight', 'bold');

% Graphic handles
path_col = 'c--';
if is_baseline, path_col = 'm--'; end
h_path  = plot(ax, NaN, NaN, path_col, 'LineWidth', 2.0); % Planned path
h_pred  = plot(ax, NaN, NaN, 'r:', 'LineWidth', 1.5); % Forecast trajectories
h_lidar = plot(ax, NaN, NaN, '.', 'Color', [0.2 0.85 1.0], 'MarkerSize', 5); % 3D LiDAR Obstacle Returns
h_traj  = plot(ax, ego.x, ego.y, 'g-', 'LineWidth', 1.5); % Driven history
h_ego   = plot(ax, ego.x, ego.y, 'o', 'MarkerSize', 14, 'MarkerFaceColor', [0 0.85 0.3], 'MarkerEdgeColor', 'w', 'LineWidth', 2);

% Actor handles map
actor_handles = containers.Map();
for k = 1:numel(scen_cfg.actors)
    act = scen_cfg.actors{k};
    col = [0.85 0.45 0.1];
    marker = 's';
    msize = 14;
    if strcmpi(act.class, 'person'), col = [1.0 0.4 0.4]; marker = 'o'; msize = 10;
    elseif strcmpi(act.class, 'cow'), col = [0.6 0.4 0.2]; marker = 's'; msize = 18;
    elseif strcmpi(act.class, 'auto_rickshaw') || strcmpi(act.class, 'rickshaw'), col = [0.9 0.8 0.1]; marker = 'd'; msize = 14;
    elseif strcmpi(act.class, 'pushcart') || strcmpi(act.class, 'thela'), col = [0.7 0.5 0.3]; marker = 's'; msize = 16;
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
    if dist_to_goal <= goal_tol
        % Final frame: vehicle overlaps directly ON TOP OF the goal star
        ego.x = goal(1);
        ego.y = goal(2);
        ego.speed = 0.0;
        traj_x(end+1) = ego.x; %#ok<AGROW>
        traj_y(end+1) = ego.y; %#ok<AGROW>
        set(h_ego,  'XData', ego.x, 'YData', ego.y);
        set(h_traj, 'XData', traj_x, 'YData', traj_y);
        set(h_path, 'XData', NaN, 'YData', NaN); % Clear dashed path line on arrival
        set(h_status, 'String', sprintf('t=%.1fs | [%s] GOAL REACHED | Speed: 0.0 km/h | Min Clearance: %.1fm | Replans: %d', ...
            t, upper(mode), min_dist_this_frame, log_data.replans), 'Color', [0.2 0.9 0.2]);
        title(ax, sprintf('Scenario: %s [%s Mode]  |  Status: GOAL REACHED', scen_cfg.title, upper(mode)), 'Color', [0.2 0.9 0.2], 'FontSize', 12);
        drawnow;
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
        actors{k} = scen_cfg.update_actor(actors{k}, t, dt, ego);
        act = actors{k};
        
        if isKey(actor_handles, act.id)
            set(actor_handles(act.id), 'XData', act.x, 'YData', act.y);
        end

        dx = act.x - ego.x;
        dy = act.y - ego.y;
        dist_act = norm([dx, dy]);
        
        if dist_act < min_dist_this_frame
            min_dist_this_frame = dist_act;
            critical_label = act.class;
        end

        if dx > -2.0 && dx < 50.0 && abs(dy) < 15.0
            tr.track_id = k;
            tr.x = act.x;
            tr.y = act.y;
            tr.vx = act.vx;
            tr.vy = act.vy;
            tr.class = act.class;
            tracks_struct = [tracks_struct, tr]; %#ok<AGROW>

            % In-path hazard check:
            is_ahead = (dx > 0.6 && dx < 40.0);
            lat_overlap = abs(dy) < (veh.width_m/2 + act.width/2 + 0.15);
            % If oncoming traffic is in opposite lane (y > 0.4), it does not block ego (y <= 0)
            is_opposite_lane = (act.y > 0.4 && ego.y < 0.1 && act.vx < -0.5);
            in_my_path = is_ahead && lat_overlap && ~is_opposite_lane;

            if in_my_path
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

    % Check collision: only count if vehicle is moving and contacts an obstacle
    if min_dist_this_frame < 0.6 && ego.speed > 0.4
        log_data.collisions = log_data.collisions + 1;
    end

    % -----------------------------------------------------------------------
    % B. Predict Trajectories (predict_trajectories.m)
    % -----------------------------------------------------------------------
    predictions.x = {};
    predictions.y = {};
    if ~isempty(tracks_struct)
        [pred_x, pred_y, ~] = predict_trajectories(tracks_struct, 2.0, 8);
        predictions.x = pred_x;
        predictions.y = pred_y;
        for p = 1:numel(pred_x)
            all_pred_x = [all_pred_x; pred_x{p}(:)]; %#ok<AGROW>
            all_pred_y = [all_pred_y; pred_y{p}(:)]; %#ok<AGROW>
        end
    end
    set(h_pred, 'XData', all_pred_x, 'YData', all_pred_y);

    % -----------------------------------------------------------------------
    % C. 3D LiDAR Simulation & Dynamic Occupancy Map Generation
    % -----------------------------------------------------------------------
    road_cfg_lidar = struct();
    if isfield(scen_cfg, 'lane_half_width'), road_cfg_lidar.lane_half_width = scen_cfg.lane_half_width; end
    
    try
        ptCloud = simulate_lidar(ego, actors, road_cfg_lidar);
        [lidar_dets, obs_pts, ~] = process_lidar_pointcloud(ptCloud);
        if ~isempty(obs_pts)
            set(h_lidar, 'XData', obs_pts(:, 1), 'YData', obs_pts(:, 2));
        else
            set(h_lidar, 'XData', NaN, 'YData', NaN);
        end
    catch
        obs_pts = [];
    end

    grid_len = 60; grid_width = 24; res = 4;
    map = binaryOccupancyMap(grid_len, grid_width, res);
    map.GridLocationInWorld = [ego.x - 5, ego.y - 12];

    % Stamp road limits
    if isfield(scen_cfg, 'lane_half_width')
        lw = scen_cfg.lane_half_width;
        x_span = (ego.x - 5):0.5:(ego.x + 55);
        y_top = lw * ones(size(x_span));
        y_bot = -lw * ones(size(x_span));
        setOccupancy(map, [x_span', y_top'], 1);
        setOccupancy(map, [x_span', y_bot'], 1);
    end

    % Stamp current obstacle footprints from tracks
    for k = 1:numel(tracks_struct)
        tr = tracks_struct(k);
        cls = lower(tr.class);
        hw = 0.75; hl = 0.9;
        if any(strcmp(cls, {'car','truck','bus'})),       hl = 2.2; hw = 1.0;
        elseif any(strcmp(cls, {'auto_rickshaw','rickshaw'})), hl = 1.3; hw = 0.65;
        elseif any(strcmp(cls, {'pushcart','thela'})),    hl = 1.0; hw = 0.55;
        elseif any(strcmp(cls, {'person','pedestrian'})), hl = 0.3; hw = 0.3;
        elseif any(strcmp(cls, {'cow','cattle','animal'})), hl = 1.1; hw = 0.6;
        elseif any(strcmp(cls, {'motorcycle','bike'})),   hl = 0.8; hw = 0.4;
        end
        [ox, oy] = meshgrid((tr.x - hl):0.25:(tr.x + hl), (tr.y - hw):0.25:(tr.y + hw));
        pts_stamp = [ox(:), oy(:)];
        in_map = pts_stamp(:,1) >= map.XWorldLimits(1) & pts_stamp(:,1) <= map.XWorldLimits(2) & ...
                 pts_stamp(:,2) >= map.YWorldLimits(1) & pts_stamp(:,2) <= map.YWorldLimits(2);
        if any(in_map)
            setOccupancy(map, pts_stamp(in_map, :), 1);
        end
    end

    % -----------------------------------------------------------------------
    % D. Path Planning & Replanning (plan_path.m)
    % -----------------------------------------------------------------------
    if is_baseline
        % BASELINE: Fixed centerline trajectory
        n_pts = 40;
        current_path.x = linspace(ego.x, min(goal(1), ego.x + 40), n_pts)';
        current_path.y = scen_cfg.ego_init.y * ones(size(current_path.x));
        current_path.yaw = zeros(size(current_path.x));
        set(h_path, 'XData', current_path.x, 'YData', current_path.y);
    else
        % ADAPTIVE: Real-time Replanning on Dynamic Occupancy Map
        time_since_replan_ms = toc(last_replan_tic) * 1000;
        [need_replan, ~] = replan_trigger(current_path, predictions, time_since_replan_ms, map);

        if need_replan || isempty(current_path.x)
            ego_pose = [ego.x, ego.y, ego.heading];
            goal_pose = [min(goal(1), ego.x + 40), goal(2), 0];
            
            [px, py, pyaw, plan_time_ms] = plan_path(map, ego_pose, goal_pose, veh);

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
    end

    % -----------------------------------------------------------------------
    % E. Decision Engine with Temporal Ratchet (decision_with_ratchet.m)
    % -----------------------------------------------------------------------
    lane_offset = ego.y;
    degraded = false;
    
    [committed_level, target_speed_mps, ~, rule_id, raw_history, down_counter, emergency_latch] = ...
        decision_with_ratchet(hazard_dist, hazard_ttc, critical_label, in_path_hazard, closing_speed, ...
                              degraded, lane_offset, prev_committed, raw_history, down_counter, emergency_latch);
    prev_committed = committed_level;

    level_names = {'PROCEED', 'CAUTION', 'SLOW', 'BRAKE', 'EMERGENCY_STOP'};
    decision_str = level_names{committed_level + 1};

    if ~strcmp(decision_str, last_logged_decision)
        msg = sprintf('t=%.1fs: %s (%s) — %s at %.1fm', t, decision_str, rule_id, critical_label, hazard_dist);
        fprintf('  %s\n', msg);
        event_log{end+1} = msg; %#ok<AGROW>
        last_logged_decision = decision_str;
    end

    % -----------------------------------------------------------------------
    % F. Speed-Adaptive Pure Pursuit Steering & PID Throttle/Brake
    % -----------------------------------------------------------------------
    la = max(3.0, min(12.0, 0.40 * ego.speed + 3.0));
    steer = 0.0;
    
    if ~isempty(current_path.x)
        path_pts = [current_path.x, current_path.y];
        dists_to_path = sqrt(sum((path_pts - [ego.x, ego.y]).^2, 2));
        [~, closest_idx] = min(dists_to_path);
        target_idx = min(size(path_pts, 1), closest_idx + max(1, round(la / 0.4)));
        
        target_pt = path_pts(target_idx, :);
        dx_la = target_pt(1) - ego.x;
        dy_la = target_pt(2) - ego.y;
        alpha = atan2(dy_la, dx_la) - ego.heading;
        alpha = atan2(sin(alpha), cos(alpha)); % wrap to [-pi, pi]
        steer = atan2(2 * veh.wheelbase_m * sin(alpha), la);
        steer = max(-veh.max_steer_rad, min(veh.max_steer_rad, steer));
    end

    % Longitudinal acceleration
    if committed_level == 4 % EMERGENCY_STOP
        accel = -6.0;
        throttle = 0.0; brake = 1.0;
    elseif committed_level == 3 % BRAKE
        accel = -3.5;
        throttle = 0.0; brake = 0.7;
    else
        accel = 2.2 * (target_speed_mps - ego.speed);
        if accel >= 0
            throttle = min(1.0, accel / 2.5); brake = 0.0;
        else
            throttle = 0.0; brake = min(1.0, -accel / 6.0);
        end
    end

    % -----------------------------------------------------------------------
    % G. Vehicle Kinematics Update (Bicycle Model)
    % -----------------------------------------------------------------------
    ego.speed   = max(0.0, ego.speed + accel * dt);
    ego.heading = ego.heading + (ego.speed / veh.wheelbase_m) * tan(steer) * dt;
    ego.x       = ego.x + ego.speed * cos(ego.heading) * dt;
    ego.y       = ego.y + ego.speed * sin(ego.heading) * dt;

    traj_x(end+1) = ego.x; %#ok<AGROW>
    traj_y(end+1) = ego.y; %#ok<AGROW>

    % -----------------------------------------------------------------------
    % H. Telemetry Logging
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
    % I. Graphics Update
    % -----------------------------------------------------------------------
    set(h_ego,  'XData', ego.x, 'YData', ego.y);
    set(h_traj, 'XData', traj_x, 'YData', traj_y);

    xlim(ax, [ego.x - 15, ego.x + 65]);

    col = scenario_color(decision_str);
    set(h_status, 'String', sprintf('t=%.1fs | [%s] %s [%s] | Speed: %.1f km/h | Min Clearance: %.1fm | Replans: %d', ...
        t, upper(mode), decision_str, rule_id, ego.speed*3.6, min_dist_this_frame, log_data.replans), 'Color', col);
    title(ax, sprintf('Scenario: %s [%s Mode]  |  Status: %s', scen_cfg.title, upper(mode), decision_str), 'Color', col, 'FontSize', 12);

    drawnow limitrate;
    t = t + dt;
end

% ---------------------------------------------------------------------------
% 4. Summary and Result Packaging
% ---------------------------------------------------------------------------
fprintf('\n=== %s [%s] Complete ===\n', scen_cfg.name, upper(mode));
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
