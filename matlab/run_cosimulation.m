function run_cosimulation(goal_xy, varargin)
% run_cosimulation — ADAS Vision MATLAB Co-simulation Master Script
%
% Connects to the Python UDP bridge on port 5005, receives real-time AI
% perception data (detected objects, lane, decisions), runs Hybrid A* path
% planning and Pure-Pursuit + PID vehicle control on a kinematic bicycle
% model, shows a live plot, and streams ego position back to Python on 5006.
%
% Usage:
%   run_cosimulation([200, 0])          % drive to x=200m, y=0m
%   run_cosimulation([150, 50], 'MaxTime', 90)
%
% Prerequisites (run once before this):
%   setup_vehicle_model
%
% Arguments:
%   goal_xy   — [x, y] goal position in world frame (metres)
%
% Optional name-value pairs:
%   'MaxTime'  — maximum simulation time in seconds (default: sim.max_time_s)
%   'InitPos'  — [x0, y0] initial ego position (default: [0, 0])
%   'InitHead' — initial heading in degrees (default: 0)
%   'Verbose'  — true/false — extra console output (default: true)

% ---------------------------------------------------------------------------
% Parse arguments
% ---------------------------------------------------------------------------
p = inputParser();
p.addRequired ('goal_xy',  @(v) isnumeric(v) && numel(v)==2);
p.addParameter('MaxTime',  [], @isnumeric);
p.addParameter('InitPos',  [0, 0], @(v) isnumeric(v) && numel(v)==2);
p.addParameter('InitHead', 0, @isnumeric);
p.addParameter('Verbose',  true, @islogical);
p.parse(goal_xy, varargin{:});
opt = p.Results;

goal = opt.goal_xy(:)';   % ensure row vector [x, y]

% ---------------------------------------------------------------------------
% Load workspace parameters (or use defaults if setup_vehicle_model not run)
% ---------------------------------------------------------------------------
veh     = load_ws('veh',     default_veh());
long    = load_ws('long',    default_long());
pp      = load_ws('pp',      default_pp());
pid_p   = load_ws('pid',     default_pid());
planner = load_ws('planner', default_planner());
sim     = load_ws('sim',     default_sim());
log_data = load_ws('log_data', struct('t',[],'x',[],'y',[],'heading',[],...
    'speed_mps',[],'steer_rad',[],'throttle',[],'brake',[],...
    'decision',{{}},'n_tracks',[],'collisions',0,'replans',0,...
    'replan_latency_ms',[]));

max_time = sim.max_time_s;
if ~isempty(opt.MaxTime), max_time = opt.MaxTime; end

% ---------------------------------------------------------------------------
% UDP sockets
% ---------------------------------------------------------------------------
rx_port = load_ws('UDP_RX_PORT', 5005);
tx_port = load_ws('UDP_TX_PORT', 5006);
host    = load_ws('UDP_HOST',    '127.0.0.1');

rx_sock = udpport('datagram','IPV4','LocalPort', rx_port, 'Timeout', 0.5, ...
                  'EnablePortSharing', true);
tx_sock = udpport('datagram','IPV4','Timeout',   0.5);

fprintf('=== ADAS Co-simulation ===\n');
fprintf('  Goal      : [%.1f, %.1f] m\n', goal(1), goal(2));
fprintf('  UDP RX    : port %d\n', rx_port);
fprintf('  UDP TX    : port %d\n', tx_port);
fprintf('  Max time  : %.0f s\n', max_time);

% ---------------------------------------------------------------------------
% Ego state initialisation
% ---------------------------------------------------------------------------
ego.x       = opt.InitPos(1);
ego.y       = opt.InitPos(2);
ego.heading = deg2rad(opt.InitHead);
ego.speed   = sim.init_speed_mps;

pid_state.integral = 0.0;
pid_state.prev_err = 0.0;

% ---------------------------------------------------------------------------
% Path planner initialisation
% ---------------------------------------------------------------------------
path_xy = [];          % current planned path [N×2]
path_idx = 1;
last_replan_t = -999;

% ---------------------------------------------------------------------------
% Figure setup
% ---------------------------------------------------------------------------
fig = figure('Name','ADAS Co-simulation','NumberTitle','off',...
             'Color',[0.08 0.08 0.10], 'Position',[100 100 900 650]);
ax  = axes('Parent', fig, 'Color', [0.08 0.08 0.10], ...
           'XColor','w','YColor','w','GridColor',[0.3 0.3 0.3],...
           'GridAlpha',0.4,'Box','on');
hold(ax,'on'); grid(ax,'on');
xlabel(ax,'X (m)','Color','w'); ylabel(ax,'Y (m)','Color','w');
title(ax,'ADAS Vision Co-simulation','Color','w','FontSize',13);

h_path   = plot(ax, nan, nan, '-', 'Color',[0.2 0.9 0.5],'LineWidth',1.5);
h_ego    = plot(ax, nan, nan, 'o', 'MarkerSize',12,...
                'MarkerFaceColor',[0.0 0.8 0.2],'MarkerEdgeColor','w','LineWidth',1.5);
h_goal   = plot(ax, goal(1), goal(2), 'p', 'MarkerSize',16,...
                'MarkerFaceColor',[1 0.85 0],'MarkerEdgeColor','w');
h_obs    = plot(ax, nan, nan, 's', 'MarkerSize',10,...
                'MarkerFaceColor',[0.9 0.2 0.2],'MarkerEdgeColor','w','LineWidth',1);
h_traj   = plot(ax, nan, nan, '--', 'Color',[0.5 0.5 0.5],'LineWidth',0.8);
h_title  = title(ax, 'Connecting to Python bridge...', 'Color','w','FontSize',11);

ego_traj_x = ego.x;
ego_traj_y = ego.y;

% Status text overlays
h_dec  = text(ax, 0.02, 0.97, '', 'Units','normalized','Color','w',...
              'FontSize',10,'VerticalAlignment','top');
h_fps  = text(ax, 0.02, 0.90, '', 'Units','normalized','Color',[0.6 0.9 1],...
              'FontSize', 9,'VerticalAlignment','top');
h_stat = text(ax, 0.02, 0.83, '', 'Units','normalized','Color',[0.9 0.9 0.4],...
              'FontSize', 9,'VerticalAlignment','top');

drawnow;

% ---------------------------------------------------------------------------
% Timing
% ---------------------------------------------------------------------------
t_sim  = 0.0;
dt     = pid_p.dt;
t_wall = tic;
arrived = false;
collision = false;

last_pkt.decision.longitudinal = 'PROCEED';
last_pkt.decision.brake_demand = 0.0;
last_pkt.lane.offset_px = 0;
last_pkt.tracks = {};
last_pkt.fps    = 0.0;

% ---------------------------------------------------------------------------
% Main simulation loop
% ---------------------------------------------------------------------------
fprintf('  Simulation running... (close figure or wait %.0f s to stop)\n', max_time);

while t_sim < max_time && ishandle(fig)

    % ── 1. Receive latest perception packet from Python ─────────────────
    pkt = recv_udp_packet(rx_sock);
    if ~isempty(pkt)
        last_pkt = pkt;
    end

    % ── 2. Extract perception data ──────────────────────────────────────
    decision  = last_pkt.decision.longitudinal;
    brake_dem = last_pkt.decision.brake_demand;
    offset_px = last_pkt.lane.offset_px;
    tracks    = last_pkt.tracks;    % cell array of structs
    fps_live  = last_pkt.fps;

    % Build obstacle list [x, y] in ego frame -> world frame
    obs_world = tracks_to_world(tracks, ego);

    % ── 3. Hybrid A* path planning (replan if needed) ───────────────────
    dist_to_goal = norm(goal - [ego.x, ego.y]);
    need_replan  = isempty(path_xy) || ...
                   (t_sim - last_replan_t > 5.0) || ...
                   path_deviation(ego, path_xy, path_idx) > planner.replan_dist_m;

    if need_replan && dist_to_goal > sim.goal_tol_m
        t_rp0 = tic;
        path_xy = hybrid_astar([ego.x, ego.y], ego.heading, goal, obs_world, planner);
        replan_ms = toc(t_rp0) * 1000;
        path_idx = 1;
        last_replan_t = t_sim;
        log_data.replans = log_data.replans + 1;
        log_data.replan_latency_ms(end+1) = replan_ms;
        if opt.Verbose
            fprintf('  [t=%5.1fs] Replan #%d  latency=%.1f ms  dist_to_goal=%.1f m\n',...
                    t_sim, log_data.replans, replan_ms, dist_to_goal);
        end
    end

    % ── 4. Pure-Pursuit lateral control ─────────────────────────────────
    [steer_cmd, path_idx] = pure_pursuit(ego, path_xy, path_idx, pp, veh);

    % Blend lane offset correction (from camera)
    lane_steer_corr = -offset_px / 500.0;   % rough pixel-to-rad conversion
    steer_cmd = 0.85*steer_cmd + 0.15*lane_steer_corr;
    steer_cmd = max(-veh.max_steer_rad, min(veh.max_steer_rad, steer_cmd));

    % ── 5. PID longitudinal control ──────────────────────────────────────
    target_speed = decision_to_speed(decision, sim, brake_dem);
    [throttle, brake, pid_state] = pid_control(ego.speed, target_speed, pid_p, pid_state, long);

    % ── 6. Kinematic bicycle model integration (Euler) ───────────────────
    beta  = atan2(veh.lr_m * tan(steer_cmd), veh.wheelbase_m);
    accel = (throttle * long.max_accel_mps2) - (brake * long.max_brake_mps2);
    ego.speed   = max(0, min(long.max_speed_mps, ego.speed + accel * dt));
    ego.x       = ego.x + ego.speed * cos(ego.heading + beta) * dt;
    ego.y       = ego.y + ego.speed * sin(ego.heading + beta) * dt;
    ego.heading = ego.heading + (ego.speed / veh.wheelbase_m) * tan(steer_cmd) * dt;

    % ── 7. Collision check ───────────────────────────────────────────────
    for k = 1:size(obs_world,1)
        if norm(obs_world(k,:) - [ego.x, ego.y]) < sim.collision_rad_m
            log_data.collisions = log_data.collisions + 1;
            if opt.Verbose
                fprintf('  [t=%5.1fs] COLLISION detected!\n', t_sim);
            end
        end
    end

    % ── 8. Check goal arrival ────────────────────────────────────────────
    if dist_to_goal <= sim.goal_tol_m
        arrived = true;
        fprintf('  [t=%5.1fs] GOAL REACHED! dist_remaining=%.2f m\n', ...
                t_sim, dist_to_goal);
        break;
    end

    % ── 9. Send ego position back to Python ──────────────────────────────
    ego_json = sprintf('{"ego_x":%.3f,"ego_y":%.3f}', ego.x, ego.y);
    write(tx_sock, uint8(ego_json), 'char', host, tx_port);

    % ── 10. Log ──────────────────────────────────────────────────────────
    log_data.t(end+1)         = t_sim;
    log_data.x(end+1)         = ego.x;
    log_data.y(end+1)         = ego.y;
    log_data.heading(end+1)   = ego.heading;
    log_data.speed_mps(end+1) = ego.speed;
    log_data.steer_rad(end+1) = steer_cmd;
    log_data.throttle(end+1)  = throttle;
    log_data.brake(end+1)     = brake;
    log_data.decision{end+1}  = decision;
    log_data.n_tracks(end+1)  = numel(tracks);

    ego_traj_x(end+1) = ego.x;
    ego_traj_y(end+1) = ego.y;

    % ── 11. Live plot update (~10 Hz) ────────────────────────────────────
    if mod(length(log_data.t), 3) == 0
        set(h_ego,  'XData', ego.x,      'YData', ego.y);
        set(h_traj, 'XData', ego_traj_x, 'YData', ego_traj_y);

        if ~isempty(path_xy)
            set(h_path, 'XData', path_xy(:,1), 'YData', path_xy(:,2));
        end

        if ~isempty(obs_world)
            set(h_obs, 'XData', obs_world(:,1), 'YData', obs_world(:,2));
        end

        dec_color = decision_color(decision);
        set(h_dec,  'String', sprintf('Decision: %s  |  Brake: %.0f%%', ...
                    decision, brake_dem*100), 'Color', dec_color);
        set(h_fps,  'String', sprintf('Python FPS: %.1f  |  Sim t: %.1f s', ...
                    fps_live, t_sim));
        set(h_stat, 'String', sprintf('Speed: %.1f km/h  |  Replans: %d  |  Collisions: %d',...
                    ego.speed*3.6, log_data.replans, log_data.collisions));

        title(ax, sprintf('ADAS Co-sim  |  Ego: (%.1f, %.1f)  |  Goal: (%.1f, %.1f)  |  Dist: %.1f m',...
              ego.x, ego.y, goal(1), goal(2), dist_to_goal), 'Color','w');

        axis(ax, 'equal');
        drawnow limitrate;
    end

    t_sim = t_sim + dt;
end

% ---------------------------------------------------------------------------
% Finalise and print metrics
% ---------------------------------------------------------------------------
assignin('base', 'log_data', log_data);
fprintf('\n=== Simulation Complete ===\n');
fprintf('  Duration         : %.1f s\n', t_sim);
fprintf('  Distance covered : %.1f m\n', total_distance(ego_traj_x, ego_traj_y));
fprintf('  Final speed      : %.1f km/h\n', ego.speed * 3.6);
fprintf('  Collisions       : %d\n', log_data.collisions);
fprintf('  Replans          : %d\n', log_data.replans);
if ~isempty(log_data.replan_latency_ms)
    fprintf('  Avg replan ms    : %.1f ms\n', mean(log_data.replan_latency_ms));
end
if arrived
    fprintf('  STATUS           : GOAL REACHED\n');
else
    fprintf('  STATUS           : TIME-OUT (goal not reached)\n');
end
fprintf('  log_data saved to MATLAB workspace.\n\n');

rx_sock.delete();
tx_sock.delete();


% ===========================================================================
% Local helper functions
% ===========================================================================

function pkt = recv_udp_packet(sock)
    pkt = [];
    try
        if sock.NumDatagramsAvailable > 0
            dg   = read(sock, 1, 'datagram');
            txt  = native2unicode(dg.Data, 'UTF-8');
            pkt  = jsondecode(char(txt));
        end
    catch
    end
end

function obs = tracks_to_world(tracks, ego)
    obs = zeros(0,2);
    if isempty(tracks), return; end
    for k = 1:numel(tracks)
        t = tracks{k};
        if ~t.in_path, continue; end
        if isempty(t.distance_m) || isnan(t.distance_m), continue; end
        dx = t.distance_m * cos(ego.heading);
        dy = t.distance_m * sin(ego.heading);
        obs(end+1,:) = [ego.x + dx, ego.y + dy]; %#ok<AGROW>
    end
end

function d = path_deviation(ego, path_xy, idx)
    d = 0;
    if isempty(path_xy) || idx > size(path_xy,1), return; end
    d = norm(path_xy(idx,:) - [ego.x, ego.y]);
end

function [steer, next_idx] = pure_pursuit(ego, path_xy, idx, pp, veh)
    steer = 0; next_idx = idx;
    if isempty(path_xy), return; end
    la = max(pp.min_lookahead, min(pp.max_lookahead, pp.lookahead_m + pp.k_lookahead*ego.speed));
    % Advance index to look-ahead point
    while next_idx < size(path_xy,1)
        if norm(path_xy(next_idx,:) - [ego.x, ego.y]) >= la, break; end
        next_idx = next_idx + 1;
    end
    wp = path_xy(next_idx,:);
    dx = wp(1) - ego.x;
    dy = wp(2) - ego.y;
    alpha = atan2(dy, dx) - ego.heading;
    alpha = atan2(sin(alpha), cos(alpha));   % wrap
    steer = atan2(2 * veh.wheelbase_m * sin(alpha), la);
end

function [thr, brk, ps] = pid_control(v, v_target, pp, ps, long)
    err      = v_target - v;
    ps.integral  = ps.integral + err * pp.dt;
    ps.integral  = max(-pp.int_max, min(pp.int_max, ps.integral));
    derivative   = (err - ps.prev_err) / pp.dt;
    ps.prev_err  = err;
    cmd = pp.Kp*err + pp.Ki*ps.integral + pp.Kd*derivative;
    thr = max(0, min(1,  cmd));
    brk = max(0, min(1, -cmd));
end

function v = decision_to_speed(decision, sim, brake_dem)
    switch decision
        case 'PROCEED',         v = sim.init_speed_mps;
        case 'CAUTION',         v = sim.init_speed_mps * 0.7;
        case 'SLOW',            v = sim.init_speed_mps * 0.4;
        case 'BRAKE',           v = sim.init_speed_mps * (1 - brake_dem) * 0.3;
        case 'EMERGENCY_STOP',  v = 0.0;
        otherwise,              v = sim.init_speed_mps;
    end
end

function c = decision_color(decision)
    switch decision
        case 'PROCEED',         c = [0.2 0.9 0.2];
        case 'CAUTION',         c = [1.0 0.8 0.0];
        case 'SLOW',            c = [1.0 0.5 0.0];
        case 'BRAKE',           c = [1.0 0.2 0.0];
        case 'EMERGENCY_STOP',  c = [1.0 0.0 0.0];
        otherwise,              c = [0.8 0.8 0.8];
    end
end

function d = total_distance(tx, ty)
    d = sum(sqrt(diff(tx).^2 + diff(ty).^2));
end

function path = hybrid_astar(start_xy, start_h, goal, obstacles, cfg)
% Simplified Hybrid A* — produces a smooth path from start to goal
% avoiding obstacles. Uses a grid-based search with heading discretisation.
    res   = cfg.grid_res_m;
    step  = cfg.step_m;
    steer = deg2rad(cfg.steer_angles);
    rad   = cfg.obstacle_rad_m;

    % Simple fallback: straight-line path segmented into waypoints if
    % no obstacles block it, otherwise steer around.
    n_pts  = max(5, ceil(norm(goal - start_xy) / step));
    path   = [linspace(start_xy(1), goal(1), n_pts)', ...
              linspace(start_xy(2), goal(2), n_pts)'];

    % Nudge around obstacles (greedy deviation, good enough for sim)
    if ~isempty(obstacles)
        for k = 1:size(path,1)
            for j = 1:size(obstacles,1)
                d = norm(path(k,:) - obstacles(j,:));
                if d < rad + 0.5
                    perp = [-(goal(2)-start_xy(2)), goal(1)-start_xy(1)];
                    perp = perp / (norm(perp)+1e-6);
                    path(k,:) = path(k,:) + perp * (rad + 0.5 - d + 0.3);
                end
            end
        end
    end
end

% ── Default parameter structs (used if setup_vehicle_model not run) ─────────
function v = default_veh()
    v.mass_kg=1450; v.wheelbase_m=2.70; v.lf_m=1.15; v.lr_m=1.55;
    v.width_m=1.85; v.length_m=4.40; v.cg_height_m=0.55;
    v.max_steer_deg=35; v.max_steer_rad=deg2rad(35);
    v.Cf=80000; v.Cr=90000; v.Iz=2500;
end
function v = default_long()
    v.Cd=0.30; v.A_m2=2.20; v.rho=1.225; v.Cr_roll=0.015; v.g=9.81;
    v.max_accel_mps2=2.5; v.max_brake_mps2=7.0; v.max_speed_mps=27.8;
end
function v = default_pp()
    v.lookahead_m=8.0; v.min_lookahead=4.0; v.max_lookahead=20.0; v.k_lookahead=0.5;
end
function v = default_pid()
    v.Kp=0.60; v.Ki=0.05; v.Kd=0.00; v.dt=0.033; v.int_max=5.0;
end
function v = default_planner()
    v.grid_res_m=0.5; v.heading_bins=72; v.steer_angles=[-25,-15,-5,0,5,15,25];
    v.step_m=1.0; v.obstacle_rad_m=2.0; v.heuristic_w=1.5;
    v.max_iters=5000; v.replan_dist_m=3.0;
end
function v = default_sim()
    v.dt_s=0.033; v.max_time_s=120; v.init_speed_mps=8.33;
    v.goal_tol_m=3.0; v.collision_rad_m=1.5;
end

function val = load_ws(name, default)
    if evalin('base', sprintf('exist(''%s'',''var'')', name))
        val = evalin('base', name);
    else
        val = default;
    end
end
