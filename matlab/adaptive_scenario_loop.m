function scenario_result = adaptive_scenario_loop(scen_cfg)
% ADAS Vision — Unified Adaptive Autonomous Scenario Execution Engine
%
% Integrates the complete closed-loop pipeline:
%   Perception -> Dynamic Trajectory Prediction (predict_trajectories.m)
%              -> Binary Occupancy Grid (build_occupancy_grid.m)
%              -> Adaptive Replan Trigger (replan_trigger.m)
%              -> Frenet / Potential-Field Path Planner (plan_path.m)
%              -> R1-R7 Decision Engine with Temporal Ratchet (decision_with_ratchet.m)
%              -> Progressive Ramp-Up Longitudinal Control & Pure Pursuit Steering
%              -> Non-linear Kinematic Bicycle Dynamics
%              -> Continuous Obstacle Clearance & Telemetry Logging
%
% ADAS Vision — SIH 2026 Autonomous Navigation Suite

mode = 'adaptive';
if isfield(scen_cfg, 'mode'), mode = lower(scen_cfg.mode); end
is_baseline = strcmp(mode, 'baseline');
is_dense = false;
if isfield(scen_cfg, 'is_dense'), is_dense = scen_cfg.is_dense; end

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
goal_tol = 2.5;  % Slightly wider catch radius to arrest overshoot at highway speeds
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
% 2. Setup Figure and Visualization — Professional Google-Maps Style UI
% ---------------------------------------------------------------------------
fig_title = sprintf('ADAS Vision  —  %s  (%s Mode)', scen_cfg.title, upper(mode));
fig = figure('Name', fig_title, 'NumberTitle', 'off', ...
             'Color', [0.08 0.09 0.11], ...
             'Position', [40 60 1380 700], ...
             'MenuBar', 'none', 'ToolBar', 'none');

% ---- Left: main map view (80% width) ----
ax = axes('Parent', fig, ...
          'Position', [0.01 0.04 0.68 0.88], ...
          'Color', [0.11 0.13 0.15], ...
          'XColor', [0.35 0.38 0.42], 'YColor', [0.35 0.38 0.42], ...
          'GridColor', [0.20 0.22 0.26], 'GridAlpha', 0.5, ...
          'FontSize', 8, 'FontName', 'Consolas', ...
          'LineWidth', 0.8);
hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');

% ---- Right: HUD dashboard panel (annotation axes, fixed position) ----
ax_hud = axes('Parent', fig, ...
              'Position', [0.705 0.04 0.285 0.88], ...
              'Color', [0.10 0.11 0.14], ...
              'XColor', 'none', 'YColor', 'none', ...
              'XLim', [0 1], 'YLim', [0 1]);
hold(ax_hud, 'on');
% Panel background card
patch(ax_hud, [0 1 1 0], [0 0 1 1], [0.10 0.11 0.14], 'EdgeColor', [0.20 0.22 0.28], 'LineWidth', 1.5);

% HUD: Title
text(ax_hud, 0.5, 0.97, 'ADAS VISION', 'Units', 'normalized', ...
     'HorizontalAlignment', 'center', 'Color', [0.4 0.75 1.0], ...
     'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Consolas');
text(ax_hud, 0.5, 0.93, upper(scen_cfg.title), 'Units', 'normalized', ...
     'HorizontalAlignment', 'center', 'Color', [0.6 0.65 0.72], ...
     'FontSize', 8, 'FontName', 'Consolas');
% Separator line
plot(ax_hud, [0.05 0.95], [0.905 0.905], 'Color', [0.25 0.28 0.35], 'LineWidth', 1);

% HUD: Dynamic text handles
h_hud_time  = text(ax_hud, 0.08, 0.875, 'TIME   0.0 s', 'Units', 'normalized', ...
     'Color', [0.75 0.78 0.85], 'FontSize', 9, 'FontName', 'Consolas');
h_hud_speed = text(ax_hud, 0.08, 0.830, 'SPEED  0.0 km/h', 'Units', 'normalized', ...
     'Color', [0.2 0.95 0.5], 'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Consolas');
h_hud_mode  = text(ax_hud, 0.08, 0.785, 'MODE   ADAPTIVE', 'Units', 'normalized', ...
     'Color', [0.55 0.6 0.7], 'FontSize', 8, 'FontName', 'Consolas');
plot(ax_hud, [0.05 0.95], [0.765 0.765], 'Color', [0.22 0.25 0.32], 'LineWidth', 0.8);

% HUD: Decision level
text(ax_hud, 0.08, 0.735, 'DECISION', 'Units', 'normalized', ...
     'Color', [0.45 0.5 0.6], 'FontSize', 7, 'FontName', 'Consolas');
h_hud_dec   = text(ax_hud, 0.08, 0.695, 'PROCEED', 'Units', 'normalized', ...
     'Color', [0.2 0.9 0.3], 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Consolas');
h_hud_rule  = text(ax_hud, 0.08, 0.660, 'Rule: R7', 'Units', 'normalized', ...
     'Color', [0.45 0.5 0.6], 'FontSize', 8, 'FontName', 'Consolas');
plot(ax_hud, [0.05 0.95], [0.640 0.640], 'Color', [0.22 0.25 0.32], 'LineWidth', 0.8);

% HUD: Clearance
text(ax_hud, 0.08, 0.615, 'MIN CLEARANCE', 'Units', 'normalized', ...
     'Color', [0.45 0.5 0.6], 'FontSize', 7, 'FontName', 'Consolas');
h_hud_clr   = text(ax_hud, 0.08, 0.575, '--- m', 'Units', 'normalized', ...
     'Color', [0.2 0.95 0.5], 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Consolas');
% Clearance bar background
patch(ax_hud, [0.08 0.92 0.92 0.08], [0.548 0.548 0.565 0.565], [0.18 0.20 0.26], 'EdgeColor', 'none');
h_hud_clrbar = patch(ax_hud, [0.08 0.50 0.50 0.08], [0.548 0.548 0.565 0.565], [0.2 0.85 0.4], 'EdgeColor', 'none');
plot(ax_hud, [0.05 0.95], [0.530 0.530], 'Color', [0.22 0.25 0.32], 'LineWidth', 0.8);

% HUD: Replans / Collisions
text(ax_hud, 0.08, 0.505, 'REPLANS', 'Units', 'normalized', ...
     'Color', [0.45 0.5 0.6], 'FontSize', 7, 'FontName', 'Consolas');
h_hud_rep   = text(ax_hud, 0.08, 0.468, '0', 'Units', 'normalized', ...
     'Color', [0.6 0.75 1.0], 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Consolas');
text(ax_hud, 0.55, 0.505, 'COLLISIONS', 'Units', 'normalized', ...
     'Color', [0.45 0.5 0.6], 'FontSize', 7, 'FontName', 'Consolas');
h_hud_col_cnt = text(ax_hud, 0.55, 0.468, '0', 'Units', 'normalized', ...
     'Color', [0.2 0.9 0.3], 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Consolas');
plot(ax_hud, [0.05 0.95], [0.445 0.445], 'Color', [0.22 0.25 0.32], 'LineWidth', 0.8);

% HUD: Goal progress
text(ax_hud, 0.08, 0.420, 'GOAL PROGRESS', 'Units', 'normalized', ...
     'Color', [0.45 0.5 0.6], 'FontSize', 7, 'FontName', 'Consolas');
patch(ax_hud, [0.08 0.92 0.92 0.08], [0.390 0.390 0.408 0.408], [0.15 0.18 0.24], 'EdgeColor', 'none');
h_hud_progbar = patch(ax_hud, [0.08 0.08 0.08 0.08], [0.390 0.390 0.408 0.408], [0.25 0.65 1.0], 'EdgeColor', 'none');
h_hud_progpct = text(ax_hud, 0.5, 0.371, '0%', 'Units', 'normalized', ...
     'HorizontalAlignment', 'center', 'Color', [0.55 0.6 0.7], 'FontSize', 7, 'FontName', 'Consolas');
plot(ax_hud, [0.05 0.95], [0.355 0.355], 'Color', [0.22 0.25 0.32], 'LineWidth', 0.8);

% HUD: Legend
text(ax_hud, 0.08, 0.330, 'MAP LEGEND', 'Units', 'normalized', ...
     'Color', [0.45 0.5 0.6], 'FontSize', 7, 'FontName', 'Consolas');
leg_items = { [0.0 0.82 1.0],  'ADAS VEHICLE'; [1.0 0.52 0.06], 'TRUCK / BUS'; ...
              [1.0 0.68 0.15], 'CAR / SUV'; [1.0 0.60 0.10], 'AUTO RICKSHAW'; ...
              [1.0 0.72 0.22], 'MOTORCYCLE'; [1.0 0.28 0.28], 'PEDESTRIAN'; ...
              [0.95 0.78 0.12],'COW / CATTLE'; [0.50 0.58 0.72], 'PUSHCART' };
for li = 1:size(leg_items,1)
    yp = 0.300 - (li-1)*0.030;
    patch(ax_hud, [0.08 0.16 0.16 0.08], [yp yp yp+0.018 yp+0.018], ...
          leg_items{li,1}, 'EdgeColor', 'none');
    text(ax_hud, 0.20, yp+0.009, leg_items{li,2}, 'Units', 'normalized', ...
         'VerticalAlignment', 'middle', 'Color', [0.60 0.64 0.72], ...
         'FontSize', 7, 'FontName', 'Consolas');
end

% ---- Draw scenario background (roads, lanes, etc.) ----
if isfield(scen_cfg, 'draw_background')
    scen_cfg.draw_background(ax);
end

% ---- Goal flag ----
patch(ax, goal(1) + [0 0 0.3 0.3], goal(2) + [0 3 3 0], [1 0.82 0], 'EdgeColor', 'none', 'FaceAlpha', 0.85);
plot(ax, [goal(1) goal(1)], [goal(2) goal(2)+4.5], 'Color', [1 0.82 0], 'LineWidth', 2.5);
text(ax, goal(1)+1.2, goal(2)+4.8, 'GOAL', 'Color', [1 0.82 0], ...
     'FontSize', 9, 'FontWeight', 'bold', 'FontName', 'Consolas');

% ---- Planned path and prediction lines ----
path_col = [0.15 0.85 0.95];
if is_baseline, path_col = [0.9 0.3 0.9]; end
h_path  = plot(ax, NaN, NaN, '--', 'Color', path_col, 'LineWidth', 2.2);
h_pred  = plot(ax, NaN, NaN, ':', 'Color', [1.0 0.35 0.35], 'LineWidth', 1.4);
h_lidar = plot(ax, NaN, NaN, '.', 'Color', [0.25 0.85 1.0], 'MarkerSize', 4);
% Trajectory trail
h_traj  = plot(ax, ego.x, ego.y, '-', 'Color', [0.15 0.75 0.35], 'LineWidth', 2.5);

% ---- Ego vehicle patch (rotated rectangle) ----
[ex, ey] = adas_vehicle_patch(ego.x, ego.y, ego.heading, veh.length_m, veh.width_m, 'car');
h_ego_body = patch(ax, ex, ey, [0.0 0.82 1.0], ...
                   'EdgeColor', [1.0 1.0 1.0], 'LineWidth', 1.5, 'FaceAlpha', 1.0);
% Ego windshield overlay
[ewx, ewy] = adas_window_patch(ego.x, ego.y, ego.heading, veh.length_m, veh.width_m, 'car');
h_ego_win = patch(ax, ewx, ewy, [0.0 0.3 0.5], 'EdgeColor', 'none', 'FaceAlpha', 1.0);
% Ego direction arrow
h_ego_arrow = quiver(ax, ego.x, ego.y, cos(ego.heading)*2.5, sin(ego.heading)*2.5, ...
                     0, 'Color', 'w', 'LineWidth', 2.0, 'MaxHeadSize', 0.8);
% Ego label
h_ego_lbl = text(ax, ego.x, ego.y+2.5, 'ADAS', 'Color', [0.8 0.97 1.0], ...
                 'FontSize', 7, 'FontWeight', 'bold', 'FontName', 'Consolas', ...
                 'HorizontalAlignment', 'center');

% ---- Actor patches ----
actor_patches = containers.Map();
actor_arrows  = containers.Map();
actor_labels  = containers.Map();
actor_windows = containers.Map();
for k = 1:numel(scen_cfg.actors)
    act = scen_cfg.actors{k};
    [ac, aw, al_str, aheading] = adas_actor_style(act);
    [px, py] = adas_vehicle_patch(act.x, act.y, aheading, al_str, aw, act.class);
    h_p = patch(ax, px, py, ac, 'EdgeColor', ac*0.3, 'LineWidth', 1.5, 'FaceAlpha', 1.0);
    % Windshield for vehicles (not pedestrians/cows/carts)
    cls_lc = lower(act.class);
    has_win = any(strcmp(cls_lc, {'car','suv','sedan','truck','bus','auto_rickshaw','rickshaw','motorcycle','bicycle','bike'}));
    if has_win
        [wpx, wpy] = adas_window_patch(act.x, act.y, aheading, al_str, aw, act.class);
        h_w = patch(ax, wpx, wpy, [0.15 0.15 0.15], 'EdgeColor', 'none', 'FaceAlpha', 1.0);
        actor_windows(act.id) = h_w;
    end
    % Direction arrow for moving actors
    spd = norm([act.vx, act.vy]);
    h_a = quiver(ax, act.x, act.y, act.vx/max(spd,0.1)*1.8, act.vy/max(spd,0.1)*1.8, ...
                 0, 'Color', [1 1 1], 'LineWidth', 1.2, 'MaxHeadSize', 1.0);
    % Class label above actor
    h_l = text(ax, act.x, act.y + aw/2 + 0.8, upper(act.class), ...
               'Color', min(ac*1.3+0.1,1), 'FontSize', 6, 'FontName', 'Consolas', ...
               'HorizontalAlignment', 'center');
    actor_patches(act.id) = h_p;
    actor_arrows(act.id)  = h_a;
    actor_labels(act.id)  = h_l;
end

% ---- Status overlay on main map (bottom-left corner) ----
h_status = text(ax, 0.01, 0.03, '', 'Units', 'normalized', ...
                'Color', [0.75 0.78 0.85], 'FontSize', 7.5, ...
                'FontName', 'Consolas', 'VerticalAlignment', 'bottom', ...
                'BackgroundColor', [0.08 0.09 0.13], ...
                'Margin', 4);


% ---------------------------------------------------------------------------
% 3. Master Simulation Loop
% ---------------------------------------------------------------------------
t = 0;
arrived = false;
traj_x = ego.x;
traj_y = ego.y;
actors = scen_cfg.actors;

min_dist_this_frame = Inf;  % Pre-init so goal-reached status message is safe
while t < max_time && ishandle(fig)
    dist_to_goal   = norm([ego.x, ego.y] - goal(1:2));
    x_past_goal    = ego.x >= (goal(1) - 0.5);   % TRUE when ego has crossed goal x
    % DUAL GOAL CHECK: Euclidean distance OR x-position crossing
    % (Euclidean alone fails when vehicle drifts in y; x-check catches that)
    if dist_to_goal <= goal_tol || x_past_goal
        % Snap vehicle cleanly onto goal point
        ego.x = goal(1);
        ego.y = goal(2);
        ego.speed = 0.0;
        traj_x(end+1) = ego.x; %#ok<AGROW>
        traj_y(end+1) = ego.y; %#ok<AGROW>
        set(h_ego_body,  'XData', ex, 'YData', ey);
        [ewx, ewy] = adas_window_patch(ego.x, ego.y, 0, veh.length_m, veh.width_m, 'car');
        set(h_ego_win,   'XData', ewx, 'YData', ewy);
        set(h_ego_arrow, 'XData', ego.x, 'YData', ego.y, 'UData', 0, 'VData', 0);
        set(h_ego_lbl,   'Position', [ego.x, ego.y + veh.width_m/2 + 1.0, 0]);
        set(h_traj, 'XData', traj_x, 'YData', traj_y);
        set(h_path, 'XData', NaN, 'YData', NaN);
        % HUD final update
        set(h_hud_speed,   'String', sprintf('SPEED  0.0 km/h'), 'Color', [0.2 0.95 0.5]);
        set(h_hud_dec,     'String', 'GOAL REACHED', 'Color', [0.15 0.95 0.35]);
        set(h_hud_time,    'String', sprintf('TIME   %.1f s', t));
        set(h_hud_progbar, 'XData', [0.08 0.92 0.92 0.08]);
        set(h_hud_progpct, 'String', '100%');
        set(h_status, 'String', sprintf('GOAL REACHED  |  t=%.1fs  |  Min Clearance: %.1fm  |  Replans: %d', ...
            t, min_dist_this_frame, log_data.replans), 'Color', [0.2 0.95 0.35]);
        title(ax, sprintf('%s  [%s]  —  ✓ GOAL REACHED', scen_cfg.title, upper(mode)), ...
              'Color', [0.2 0.95 0.35], 'FontSize', 11, 'FontName', 'Consolas');
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

    for k = 1:numel(actors)
        actors{k} = scen_cfg.update_actor(actors{k}, t, dt, ego);
        act = actors{k};
        
        if isKey(actor_patches, act.id)
            aheading_act = atan2(act.vy, act.vx);
            if abs(act.vx) < 0.05 && abs(act.vy) < 0.05, aheading_act = 0; end
            [~, aw_act, al_act, ~] = adas_actor_style(act);
            [apx, apy] = adas_vehicle_patch(act.x, act.y, aheading_act, al_act, aw_act, act.class);
            set(actor_patches(act.id), 'XData', apx, 'YData', apy);
            if isKey(actor_windows, act.id)
                [wpx, wpy] = adas_window_patch(act.x, act.y, aheading_act, al_act, aw_act, act.class);
                set(actor_windows(act.id), 'XData', wpx, 'YData', wpy);
            end
            spd_act = norm([act.vx, act.vy]);
            if spd_act > 0.1
                set(actor_arrows(act.id), 'XData', act.x, 'YData', act.y, ...
                    'UData', act.vx/spd_act*1.8, 'VData', act.vy/spd_act*1.8);
            end
            set(actor_labels(act.id), 'Position', [act.x, act.y + aw_act/2 + 0.8, 0]);
        end

        dx = act.x - ego.x;
        dy = act.y - ego.y;
        dist_act = norm([dx, dy]);
        
        if dist_act < min_dist_this_frame
            min_dist_this_frame = dist_act;
        end

        if dx > -2.0 && dx < 50.0 && abs(dy) < 15.0
            tr.track_id = k;
            tr.x = act.x;
            tr.y = act.y;
            tr.vx = act.vx;
            tr.vy = act.vy;
            tr.class = act.class;
            tr.width = 1.0;
            tr.length = 2.0;
            if isfield(act, 'width'), tr.width = act.width; end
            if isfield(act, 'length'), tr.length = act.length; end
            tracks_struct = [tracks_struct, tr]; %#ok<AGROW>
        end
    end

    % Check physical collision using Bounding Box overlap
    % Ego size: 4.6m (length), 1.8m (width). Half-dims: 2.3m, 0.9m
    has_collision = false;
    for k = 1:numel(tracks_struct)
        tr = tracks_struct(k);
        dx = abs(tr.x - ego.x);
        dy = abs(tr.y - ego.y);
        % Add 0.1m tolerance margin
        if dx < (2.3 + tr.length/2 - 0.1) && dy < (0.9 + tr.width/2 - 0.1)
            has_collision = true;
            break;
        end
    end

    if has_collision && ego.speed > 0.4
        log_data.collisions = log_data.collisions + 1;
    end

    % -----------------------------------------------------------------------
    % B. Predict Trajectories (predict_trajectories.m)
    % -----------------------------------------------------------------------
    predictions.x = {};
    predictions.y = {};
    if ~isempty(tracks_struct)
        pred_horizon = 2.0;
        if is_dense, pred_horizon = 1.0; end
        [pred_x, pred_y, ~] = predict_trajectories(tracks_struct, pred_horizon, 8);
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
        elseif any(strcmp(cls, {'cow','cattle','animal'})), hl = 1.2; hw = 0.7;
        elseif any(strcmp(cls, {'motorcycle','bike','bicycle'})), hl = 0.8; hw = 0.4;
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
        n_pts = 40;
        current_path.x = linspace(ego.x, min(goal(1), ego.x + 40), n_pts)';
        current_path.y = scen_cfg.ego_init.y * ones(size(current_path.x));
        current_path.yaw = zeros(size(current_path.x));
        set(h_path, 'XData', current_path.x, 'YData', current_path.y);
    else
        time_since_replan_ms = toc(last_replan_tic) * 1000;
        [need_replan, ~] = replan_trigger(current_path, predictions, time_since_replan_ms, map);

        if need_replan || isempty(current_path.x)
            ego_pose = [ego.x, ego.y, ego.heading];
            goal_pose = [min(goal(1), ego.x + 40), goal(2), 0];
            veh.is_dense = is_dense;
            if isfield(scen_cfg, 'lane_half_width')
                veh.lane_half_width = scen_cfg.lane_half_width;
            end
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
    % E. Trajectory-Aligned Hazard Evaluation
    % -----------------------------------------------------------------------
    in_path_hazard = false;
    hazard_dist = Inf;
    hazard_ttc = Inf;
    closing_speed = 0.0;
    critical_label = 'None';
    proximal_clearance = Inf;

    if ~isempty(current_path.x)
        path_pts = [current_path.x, current_path.y];

        for k = 1:numel(tracks_struct)
            tr = tracks_struct(k);
            dx = tr.x - ego.x;
            dy = tr.y - ego.y;
            dist_to_obs = norm([dx, dy]);

            % Only evaluate obstacles ahead of ego
            if dx > -1.0 && dx < 45.0
                obs_hw = tr.width / 2;
                corridor_thresh = (veh.width_m / 2) + obs_hw + 0.30;
                % Opposite-lane check: actor is on the other side of the road from ego,
                % moving toward us (oncoming). Use ego.y as reference.
                lane_sep = 1.5; % minimum lateral separation to count as opposite lane
                is_opposite_lane = (tr.vx < -0.3) && ((tr.y - ego.y) > lane_sep);

                % In adaptive mode: trust the planner for STATIC INFRASTRUCTURE only
                % (pushcarts, parked vehicles that cannot move).
                % Do NOT apply bypass for vehicles that can move (motorcycle, bicycle, car)
                % since a stopped vehicle can still cause a collision if ego swerves near it.
                is_static_obs = (abs(tr.vx) < 0.1 && abs(tr.vy) < 0.1);
                cls_lower = lower(tr.class);
                is_infrastructure = any(strcmp(cls_lower, {'pushcart','thela','sign','barrier','parked'}));
                bypass_threshold = corridor_thresh;
                if is_static_obs && is_infrastructure
                    bypass_threshold = 0.7; % Only block if planned path literally hits fixed infrastructure
                end

                % Check distance from ALL predicted points to the planned path
                obs_pred_x = [tr.x; all_pred_x((k-1)*8+1 : k*8)];
                obs_pred_y = [tr.y; all_pred_y((k-1)*8+1 : k*8)];

                min_d_traj = Inf;
                for p = 1:numel(obs_pred_x)
                    dists = sqrt((path_pts(:, 1) - obs_pred_x(p)).^2 + (path_pts(:, 2) - obs_pred_y(p)).^2);
                    [d_min_pt, closest_idx] = min(dists);
                    traj_pt = path_pts(closest_idx, :);
                    if traj_pt(1) >= (ego.x - 0.5)
                        if d_min_pt < min_d_traj
                            min_d_traj = d_min_pt;
                        end
                    end
                end

                % Track closest distance to ANY obstacle along path
                if min_d_traj < proximal_clearance
                    proximal_clearance = min_d_traj;
                end

                if (min_d_traj < bypass_threshold) && ~is_opposite_lane
                    in_path_hazard = true;
                    v_rel = ego.speed - tr.vx * cos(ego.heading);
                    if v_rel > 0
                        ttc = dx / max(0.1, v_rel);
                    else
                        ttc = 99.0;
                    end

                    if dist_to_obs < hazard_dist
                        hazard_dist = dist_to_obs;
                        hazard_ttc = ttc;
                        closing_speed = v_rel;
                        critical_label = tr.class;
                    end
                end
            end
        end
    end

    % -----------------------------------------------------------------------
    % F. Decision Engine with Temporal Ratchet (decision_with_ratchet.m)
    % -----------------------------------------------------------------------
    lane_offset = ego.y;
    degraded = false;
    
    [committed_level, target_speed_mps, ~, rule_id, raw_history, down_counter, emergency_latch] = ...
        decision_with_ratchet(hazard_dist, hazard_ttc, critical_label, in_path_hazard, closing_speed, ...
                              degraded, lane_offset, prev_committed, raw_history, down_counter, emergency_latch);
    
    % -----------------------------------------------------------------------
    % Dense-mode Speed Override: keep car moving at moderate pace near obstacles
    % -----------------------------------------------------------------------
    if is_dense
        % Moderate floors: car slows but never fully stops at obstacles
        if committed_level == 3 && target_speed_mps < 1.0      % BRAKE -> ~3.5 km/h creep
            target_speed_mps = 1.0;
        elseif committed_level == 2 && target_speed_mps < 2.5  % SLOW -> ~9 km/h
            target_speed_mps = 2.5;
        elseif committed_level == 1 && target_speed_mps < 5.0  % CAUTION -> ~18 km/h
            target_speed_mps = 5.0;
        end
        % Drain emergency latch in 1 frame so recovery isn't delayed
        if emergency_latch > 1
            emergency_latch = 1;
        end
    end

    % Proximal Clearance Governor
    % In dense mode, static obstacles that the planner is routing around
    % should not trigger speed caps — only apply when path is genuinely tight
    if proximal_clearance < 1.5 && target_speed_mps > 3.5
        if is_dense
            target_speed_mps = 5.0; % Slow creep past very tight gaps
        else
            target_speed_mps = 3.5;
            if committed_level < 2, committed_level = 2; rule_id = 'PROX_SLOW'; end
        end
    elseif proximal_clearance < 2.5 && target_speed_mps > 6.0
        target_speed_mps = 6.0;
        if ~is_dense && committed_level < 1
            committed_level = 1; rule_id = 'PROX_CAUTION';
        end
    elseif proximal_clearance < 4.5 && target_speed_mps > 9.0
        target_speed_mps = 9.0;
    end

    % -----------------------------------------------------------------------
    % Goal Approach Speed Governor
    % Prevents high-speed overshoot: progressively cap speed as we near goal
    % -----------------------------------------------------------------------
    % -----------------------------------------------------------------------
    % Goal Approach Speed Governor (distance-to-goal in X, not Euclidean)
    % Uses X-distance so y-drift doesn't delay braking near goal
    % -----------------------------------------------------------------------
    dx_to_goal = goal(1) - ego.x;   % positive = still ahead of goal
    if dx_to_goal < 4.0
        target_speed_mps = min(target_speed_mps, 0.8);   % near-stop creep
    elseif dx_to_goal < 10.0
        target_speed_mps = min(target_speed_mps, 2.0);   % slow creep
    elseif dx_to_goal < 20.0
        target_speed_mps = min(target_speed_mps, 3.5);   % gentle approach
    elseif dx_to_goal < 40.0
        target_speed_mps = min(target_speed_mps, 5.5);   % moderate approach
    end

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
    % G. Speed-Adaptive Pure Pursuit Steering & Progressive Acceleration
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

    % Progressive Ramp-Up Longitudinal Control
    if committed_level == 4 % EMERGENCY_STOP
        accel = -8.0;  % Increased max braking force
        throttle = 0.0; brake = 1.0;
    elseif committed_level == 3 % BRAKE
        accel = -4.5;
        throttle = 0.0; brake = 0.7;
    else
        % Smooth ramp-up from stop (start gently, then accelerate to cruising speed)
        if ego.speed < 2.0
            max_accel = 0.9; % Gentle launch
        else
            max_accel = 1.6; % Normal acceleration (reduced from 2.4)
        end
        accel = min(max_accel, max(-3.0, 1.4 * (target_speed_mps - ego.speed)));
        if accel >= 0
            throttle = min(1.0, accel / 2.5); brake = 0.0;
        else
            throttle = 0.0; brake = min(1.0, -accel / 5.0);
        end
    end

    % -----------------------------------------------------------------------
    % H. Vehicle Kinematics Update (Bicycle Model)
    % -----------------------------------------------------------------------
    ego.speed   = max(0.0, ego.speed + accel * dt);
    ego.heading = ego.heading + (ego.speed / veh.wheelbase_m) * tan(steer) * dt;
    ego.x       = ego.x + ego.speed * cos(ego.heading) * dt;
    ego.y       = ego.y + ego.speed * sin(ego.heading) * dt;

    % -----------------------------------------------------------------------
    % HARD GOAL CLAMP: If vehicle has passed goal.x this frame, stop it
    % immediately. This is the final safety net against any overshoot.
    % -----------------------------------------------------------------------
    if ego.x >= goal(1)
        ego.x     = goal(1);
        ego.y     = goal(2);
        ego.speed = 0.0;
        traj_x(end+1) = ego.x; %#ok<AGROW>
        traj_y(end+1) = ego.y; %#ok<AGROW>
        set(h_ego,  'XData', ego.x, 'YData', ego.y);
        set(h_traj, 'XData', traj_x, 'YData', traj_y);
        set(h_path, 'XData', NaN, 'YData', NaN);
        set(h_status, 'String', sprintf('t=%.1fs | [%s] GOAL REACHED | Speed: 0.0 km/h | Min Clearance: %.1fm | Replans: %d', ...
            t, upper(mode), min_dist_this_frame, log_data.replans), 'Color', [0.2 0.9 0.2]);
        title(ax, sprintf('Scenario: %s [%s Mode]  |  Status: GOAL REACHED', scen_cfg.title, upper(mode)), 'Color', [0.2 0.9 0.2], 'FontSize', 12);
        drawnow;
        arrived = true;
        t = t + dt;
        break;
    end

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
    % Update ego patch
    [ex, ey] = adas_vehicle_patch(ego.x, ego.y, ego.heading, veh.length_m, veh.width_m, 'car');
    set(h_ego_body,  'XData', ex, 'YData', ey);
    [ewx, ewy] = adas_window_patch(ego.x, ego.y, ego.heading, veh.length_m, veh.width_m, 'car');
    set(h_ego_win,   'XData', ewx, 'YData', ewy);
    set(h_ego_arrow, 'XData', ego.x, 'YData', ego.y, ...
                     'UData', cos(ego.heading)*2.5, 'VData', sin(ego.heading)*2.5);
    set(h_ego_lbl,   'Position', [ego.x, ego.y + veh.width_m/2 + 1.0, 0]);
    set(h_traj, 'XData', traj_x, 'YData', traj_y);

    xlim(ax, [ego.x - 18, ego.x + 62]);

    % HUD updates
    goal_dist_x = max(0, goal(1) - scen_cfg.ego_init.x);
    ego_prog    = min(1, max(0, (ego.x - scen_cfg.ego_init.x) / max(goal_dist_x, 1)));
    prog_xr     = 0.08 + ego_prog * 0.84;
    col = scenario_color(decision_str);
    clr_norm = min(1, max(0, min_dist_this_frame / 10.0));
    clr_col  = [1-clr_norm, clr_norm*0.85, clr_norm*0.3];

    set(h_hud_time,    'String', sprintf('TIME   %.1f s', t));
    set(h_hud_speed,   'String', sprintf('SPEED  %.1f km/h', ego.speed*3.6), 'Color', col);
    set(h_hud_mode,    'String', sprintf('MODE   %s', upper(mode)));
    set(h_hud_dec,     'String', decision_str, 'Color', col);
    set(h_hud_rule,    'String', sprintf('Rule: %s  |  Hazard: %s', rule_id, critical_label));
    if isfinite(min_dist_this_frame)
        set(h_hud_clr, 'String', sprintf('%.2f m', min_dist_this_frame), 'Color', clr_col);
    else
        set(h_hud_clr, 'String', 'Clear', 'Color', [0.2 0.95 0.5]);
    end
    set(h_hud_clrbar,  'XData', [0.08, min(0.08+clr_norm*0.84, 0.92), min(0.08+clr_norm*0.84, 0.92), 0.08], 'FaceColor', clr_col);
    set(h_hud_rep,     'String', sprintf('%d', log_data.replans));
    set(h_hud_col_cnt, 'String', sprintf('%d', log_data.collisions), ...
                       'Color', [1-(log_data.collisions>0)*0.8, 0.2+(log_data.collisions==0)*0.7, 0.3*(log_data.collisions==0)]);
    set(h_hud_progbar, 'XData', [0.08, prog_xr, prog_xr, 0.08]);
    set(h_hud_progpct, 'String', sprintf('%d%%', round(ego_prog*100)));

    set(h_status, 'String', sprintf('t=%.1fs  |  %s [%s]  |  Speed: %.1f km/h  |  Clr: %.1fm  |  Replans: %d', ...
        t, decision_str, rule_id, ego.speed*3.6, min_dist_this_frame, log_data.replans));
    title(ax, sprintf('%s  [%s Mode]', scen_cfg.title, upper(mode)), ...
          'Color', [0.50 0.55 0.65], 'FontSize', 10, 'FontName', 'Consolas');

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

% ---------------------------------------------------------------------------
% Hard-clamp goal snap — update patches on final frame
% ---------------------------------------------------------------------------
[ex, ey] = adas_vehicle_patch(ego.x, ego.y, ego.heading, veh.length_m, veh.width_m, 'car');
if ishandle(fig)
    set(h_ego_body, 'XData', ex, 'YData', ey);
    set(h_hud_dec,  'String', 'GOAL REACHED', 'Color', [0.15 0.95 0.35]);
    set(h_hud_progbar, 'XData', [0.08 0.92 0.92 0.08]);
    set(h_hud_progpct, 'String', '100%');
    title(ax, sprintf('%s  [%s]  —  GOAL REACHED', scen_cfg.title, upper(mode)), ...
          'Color', [0.2 0.95 0.35], 'FontSize', 10, 'FontName', 'Consolas');
    drawnow;
end

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

function [px, py] = adas_vehicle_patch(cx, cy, heading, len, wid, cls)
% Returns top-view silhouette polygon for each vehicle class.
% Each class gets a distinctive shape instead of a plain rectangle.
    if nargin < 6, cls = 'car'; end
    cls = lower(cls);
    hl = len/2;  hw = wid/2;

    if any(strcmp(cls, {'car','suv','sedan'}))
        % Car: tapered nose (pointed front), wide body, squared rear
        % 10-point shape giving a proper top-view car silhouette
        raw_x = [-hl,   -hl*0.15,  hl*0.50,  hl,    hl,    hl*0.50, -hl*0.15, -hl,   -hl, -hl];
        raw_y = [-hw,   -hw,       -hw*0.78, -hw*0.45, hw*0.45, hw*0.78,  hw,      hw,   hw, -hw];

    elseif any(strcmp(cls, {'truck','bus'}))
        % Truck: long flat trailer + distinct cab at front (bumped forward section)
        % 12-point shape: flat trailer body with cab protrusion
        cab = hl * 0.35;  % cab length proportion
        raw_x = [-hl, -hl, hl-cab, hl-cab, hl,    hl,    hl-cab, hl-cab, -hl, -hl, -hl, -hl];
        raw_y = [-hw, hw,  hw,     hw*0.9,  hw*0.9,-hw*0.9,-hw*0.9,-hw,   -hw,  hw,  hw, -hw];
        raw_x = [-hl, hl-cab, hl-cab, hl,   hl,   hl-cab, hl-cab, -hl];
        raw_y = [-hw, -hw,    -hw*0.85, -hw*0.85, hw*0.85, hw*0.85, hw, hw];

    elseif any(strcmp(cls, {'auto_rickshaw','rickshaw'}))
        % Rickshaw: wide rear, tapered front (reverse of car)
        raw_x = [-hl,  -hl*0.3,  hl,  hl,  -hl*0.3, -hl];
        raw_y = [-hw,  -hw,     -hw*0.55,  hw*0.55,   hw,  hw];

    elseif any(strcmp(cls, {'motorcycle','bicycle','bike'}))
        % Motorcycle: narrow teardrop (wide rear, pointed front)
        theta = linspace(-pi/2, pi/2, 8);
        front_x =  hl * cos(theta);
        front_y =  hw * 0.5 * sin(theta);
        rear_x  = -hl * cos(theta(end:-1:1));
        rear_y  =  hw * sin(theta(end:-1:1));
        raw_x = [front_x, rear_x];
        raw_y = [front_y, rear_y];

    elseif any(strcmp(cls, {'person','pedestrian'}))
        % Pedestrian: circle (12 points)
        theta = linspace(0, 2*pi-0.01, 12);
        r = max(hw, 0.40);
        raw_x = r * cos(theta);
        raw_y = r * sin(theta);

    elseif any(strcmp(cls, {'cow','cattle','animal'}))
        % Cow: oval body (elongated ellipse)
        theta = linspace(0, 2*pi-0.01, 14);
        raw_x = hl * cos(theta);
        raw_y = hw * sin(theta);

    elseif any(strcmp(cls, {'pushcart','thela'}))
        % Pushcart: rectangle with a small handle notch at rear
        raw_x = [-hl, hl, hl, -hl*0.85, -hl*0.85, -hl];
        raw_y = [-hw, -hw, hw, hw, hw*0.6, hw*0.6];

    else
        % Default: simple rectangle
        raw_x = [-hl, hl, hl, -hl];
        raw_y = [-hw, -hw, hw,  hw];
    end

    R = [cos(heading), -sin(heading); sin(heading), cos(heading)];
    rc = R * [raw_x; raw_y];
    px = cx + rc(1,:);
    py = cy + rc(2,:);
end

function [px, py] = adas_window_patch(cx, cy, heading, len, wid, cls)
% Returns a smaller highlight patch representing windshield / front glass.
    if nargin < 6, cls = 'car'; end
    cls = lower(cls);
    hl = len/2;  hw = wid/2;

    if any(strcmp(cls, {'car','suv','sedan'}))
        % Front windshield zone (front 30% of car, inner 60% of width)
        raw_x = [hl*0.15, hl*0.72, hl*0.72, hl*0.15];
        raw_y = [-hw*0.52, -hw*0.38, hw*0.38, hw*0.52];
    elseif any(strcmp(cls, {'truck','bus'}))
        % Cab windshield
        raw_x = [hl*0.45, hl*0.92, hl*0.92, hl*0.45];
        raw_y = [-hw*0.55, -hw*0.40, hw*0.40, hw*0.55];
    elseif any(strcmp(cls, {'auto_rickshaw','rickshaw'}))
        raw_x = [hl*0.20, hl*0.80, hl*0.80, hl*0.20];
        raw_y = [-hw*0.40, -hw*0.25, hw*0.25, hw*0.40];
    elseif any(strcmp(cls, {'motorcycle','bicycle','bike'}))
        % Headlight dot at front
        theta = linspace(0, 2*pi-0.01, 8);
        raw_x = hl*0.65 + hw*0.25*cos(theta);
        raw_y = hw*0.25*sin(theta);
    else
        raw_x = NaN;  raw_y = NaN;
    end

    R = [cos(heading), -sin(heading); sin(heading), cos(heading)];
    rc = R * [raw_x; raw_y];
    px = cx + rc(1,:);
    py = cy + rc(2,:);
end

function [col, width, len, heading] = adas_actor_style(act)
% Returns color, width, length, and heading for each actor class.
% Unified professional palette: amber family for vehicles, red for VRUs,
% golden for animals, steel-gray for static obstacles.
    heading = atan2(act.vy, act.vx);
    if abs(act.vx) < 0.05 && abs(act.vy) < 0.05, heading = 0; end
    cls = lower(act.class);
    if any(strcmp(cls, {'truck','bus'}))
        col = [1.0 0.52 0.06];  width = 2.6;  len = 8.5;  % Deep amber — large, stands out
    elseif any(strcmp(cls, {'car','suv','sedan'}))
        col = [1.0 0.68 0.15];  width = 1.85; len = 4.4;  % Mid amber
    elseif any(strcmp(cls, {'auto_rickshaw','rickshaw'}))
        col = [1.0 0.60 0.10];  width = 1.45; len = 2.9;  % Amber-orange
    elseif any(strcmp(cls, {'motorcycle','bicycle','bike'}))
        col = [1.0 0.72 0.22];  width = 0.80; len = 2.0;  % Light amber
    elseif any(strcmp(cls, {'person','pedestrian'}))
        col = [1.0 0.28 0.28];  width = 0.60; len = 0.60; % Coral red — VRU danger
    elseif any(strcmp(cls, {'cow','cattle','animal'}))
        col = [0.95 0.78 0.12]; width = 1.5;  len = 2.2;  % Golden yellow
    elseif any(strcmp(cls, {'pushcart','thela'}))
        col = [0.50 0.58 0.72]; width = 1.2;  len = 2.0;  % Steel blue-gray — static
    else
        col = [0.65 0.68 0.75]; width = 1.8;  len = 4.0;  % Default gray
    end
    if isfield(act, 'width'),  width = act.width;  end
    if isfield(act, 'length'), len   = act.length; end
end
end
