% setup_vehicle_model.m
% ADAS Vision Co-simulation — Vehicle Model & Controller Parameter Setup
%
% Run this script ONCE before calling run_cosimulation() or any scenario.
% Populates the MATLAB base workspace with all vehicle physics constants,
% controller gains, and UDP network parameters.
%
% Usage:
%   >> setup_vehicle_model

fprintf('=== ADAS Vision — Vehicle Model Setup ===\n');

% ---------------------------------------------------------------------------
% UDP Network parameters (must match udp_bridge.py)
% ---------------------------------------------------------------------------
UDP_RX_PORT  = 5005;   % port to receive perception JSON from Python
UDP_TX_PORT  = 5006;   % port to send ego position back to Python
UDP_HOST     = '127.0.0.1';
UDP_TIMEOUT  = 0.5;    % seconds to wait for a UDP packet before using last value
UDP_BUFSIZE  = 65535;

assignin('base', 'UDP_RX_PORT',  UDP_RX_PORT);
assignin('base', 'UDP_TX_PORT',  UDP_TX_PORT);
assignin('base', 'UDP_HOST',     UDP_HOST);
assignin('base', 'UDP_TIMEOUT',  UDP_TIMEOUT);
assignin('base', 'UDP_BUFSIZE',  UDP_BUFSIZE);

% ---------------------------------------------------------------------------
% Vehicle geometry (kinematic bicycle model)
% ---------------------------------------------------------------------------
veh.mass_kg       = 1450;        % kerb mass (kg)
veh.wheelbase_m   = 2.70;        % front-to-rear axle (m)
veh.lf_m          = 1.15;        % CG-to-front axle (m)
veh.lr_m          = 1.55;        % CG-to-rear  axle (m)
veh.width_m       = 1.85;        % vehicle width (m)
veh.length_m      = 4.40;        % vehicle length (m)
veh.cg_height_m   = 0.55;        % CG height (m)
veh.max_steer_deg = 35.0;        % maximum steering angle (degrees)
veh.max_steer_rad = deg2rad(35.0);

% Tyre (simplified linear — sufficient for bicycle model)
veh.Cf = 80000;    % front cornering stiffness (N/rad)
veh.Cr = 90000;    % rear  cornering stiffness (N/rad)
veh.Iz = 2500;     % yaw moment of inertia (kg·m²)

assignin('base', 'veh', veh);

% ---------------------------------------------------------------------------
% Longitudinal dynamics (simplified)
% ---------------------------------------------------------------------------
long.Cd            = 0.30;       % aerodynamic drag coefficient
long.A_m2          = 2.20;       % frontal area (m²)
long.rho           = 1.225;      % air density (kg/m³)
long.Cr_roll       = 0.015;      % rolling resistance coefficient
long.g             = 9.81;       % gravity (m/s²)
long.max_accel_mps2 = 2.5;
long.max_brake_mps2 = 7.0;       % maximum deceleration (m/s²)
long.max_speed_mps  = 27.8;      % ~100 km/h speed cap

assignin('base', 'long', long);

% ---------------------------------------------------------------------------
% Pure-Pursuit lateral controller
% ---------------------------------------------------------------------------
pp.lookahead_m    = 8.0;         % look-ahead distance (m) — fixed
pp.min_lookahead  = 4.0;         % minimum (low-speed)
pp.max_lookahead  = 20.0;        % maximum (high-speed)
pp.k_lookahead    = 0.5;         % velocity scaling factor for adaptive lookahead

assignin('base', 'pp', pp);

% ---------------------------------------------------------------------------
% Longitudinal PID controller
% ---------------------------------------------------------------------------
pid.Kp  = 0.60;
pid.Ki  = 0.05;
pid.Kd  = 0.00;
pid.dt  = 0.033;                 % ~30 Hz control rate (s)
pid.int_max = 5.0;               % integrator anti-windup clamp

assignin('base', 'pid', pid);

% ---------------------------------------------------------------------------
% Hybrid A* path planner
% ---------------------------------------------------------------------------
planner.grid_res_m      = 0.5;   % grid cell resolution (m)
planner.heading_bins    = 72;    % heading discretisation (360/72 = 5 deg/bin)
planner.steer_angles    = [-25, -15, -5, 0, 5, 15, 25]; % degrees
planner.step_m          = 1.0;   % simulation step per node expansion (m)
planner.obstacle_rad_m  = 2.0;   % obstacle inflation radius (m)
planner.heuristic_w     = 1.5;   % A* heuristic weight (>1 = faster, suboptimal)
planner.max_iters       = 5000;  % maximum A* iterations per plan
planner.replan_dist_m   = 3.0;   % replan when ego deviates this far from path (m)

assignin('base', 'planner', planner);

% ---------------------------------------------------------------------------
% Simulation parameters
% ---------------------------------------------------------------------------
sim.dt_s           = 0.033;      % simulation timestep (~30 Hz)
sim.max_time_s     = 120.0;      % maximum scenario duration (s)
sim.init_speed_mps = 8.33;       % initial ego speed (~30 km/h)
sim.goal_tol_m     = 3.0;        % distance to goal to declare "arrived"
sim.collision_rad_m = 1.5;       % ego collision radius (m)

assignin('base', 'sim', sim);

% ---------------------------------------------------------------------------
% Map / world origin
% ---------------------------------------------------------------------------
map.origin_lat = 12.9716;        % reference latitude  (Bangalore, India)
map.origin_lon = 77.5946;        % reference longitude
map.scale_mpp  = 1.0;            % metres per pixel (for plotting)

assignin('base', 'map', map);

% ---------------------------------------------------------------------------
% Logging structure (pre-allocated, grown dynamically)
% ---------------------------------------------------------------------------
log_data = struct();
log_data.t          = [];        % simulation time (s)
log_data.x          = [];        % ego X position (m)
log_data.y          = [];        % ego Y position (m)
log_data.heading    = [];        % ego heading (rad)
log_data.speed_mps  = [];        % ego speed (m/s)
log_data.steer_rad  = [];        % commanded steering angle (rad)
log_data.throttle   = [];        % throttle command [0,1]
log_data.brake      = [];        % brake command [0,1]
log_data.decision   = {};        % longitudinal decision string
log_data.n_tracks   = [];        % number of tracked objects
log_data.collisions = 0;
log_data.replans    = 0;
log_data.replan_latency_ms = [];

assignin('base', 'log_data', log_data);

fprintf('  Vehicle model  : %.0f kg, %.2f m wheelbase\n', veh.mass_kg, veh.wheelbase_m);
fprintf('  Controller     : Pure-Pursuit (%.1f m lookahead) + PID (Kp=%.2f)\n', ...
        pp.lookahead_m, pid.Kp);
fprintf('  Planner        : Hybrid A* (%.1f m grid, %d steer angles)\n', ...
        planner.grid_res_m, numel(planner.steer_angles));
fprintf('  UDP RX         : port %d   TX: port %d\n', UDP_RX_PORT, UDP_TX_PORT);
fprintf('  All parameters loaded into MATLAB workspace.\n');
fprintf('  Next step: run_cosimulation([200, 0])\n\n');
