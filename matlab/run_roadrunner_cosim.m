function run_roadrunner_cosim(scenarioName)
% RUN_ROADRUNNER_COSIM Master script to co-simulate RoadRunner with ADAS Vision
%
% Connects MATLAB to RoadRunner using the roadrunner API, streams dynamic
% ground-truth actor poses, builds local dynamic occupancy maps, runs Hybrid A*
% path planning, evaluates R1-R7 decision logic, and commands the Ego vehicle
% in a closed feedback loop.
%
% Usage:
%   run_roadrunner_cosim
%   run_roadrunner_cosim('scenario_village_road.rrscenario')

if nargin < 1
    scenarioName = 'scenario_village_road.rrscenario';
end

projectPath = fullfile(pwd, '..', 'roadrunner', 'Project');

fprintf('=====================================================\n');
fprintf('   ADAS Vision — RoadRunner 3D Co-Simulation Engine  \n');
fprintf('=====================================================\n');
fprintf('  Project Path : %s\n', projectPath);
fprintf('  Scenario File: %s\n', scenarioName);

% 1. Connect to RoadRunner API
try
    rrApp = roadrunner(projectPath);
    openScenario(rrApp, scenarioName);
    fprintf('  ✅ Connected to RoadRunner app instance.\n');
catch ME
    fprintf('  [NOTE] RoadRunner application not currently open/installed.\n');
    fprintf('  Error details: %s\n', ME.message);
    fprintf('  Falling back to simulated RoadRunner Ground-Truth API loop...\n');
    run_simulated_rr_loop(scenarioName);
    return;
end

% 2. Get Scenario Simulation Object
rrSim = createSimulation(rrApp);
set(rrSim, 'StepSize', 0.033); % ~30 Hz control loop
set(rrSim, 'MaxSimulationTime', 45.0);

% 3. Initialize Ego Vehicle & Controller State
setup_vehicle_model;
veh     = evalin('base', 'veh');
pp      = evalin('base', 'pp');
pid_p   = evalin('base', 'pid');
sim_cfg = evalin('base', 'sim');

ego_handle = getActor(rrSim, 'EgoVehicle');

% Initialize Local Occupancy Grid and Planner
current_path = struct('x', [], 'y', [], 'yaw', []);
last_replan_t = -999;
pid_state = struct('integral', 0.0, 'prev_err', 0.0);

% 4. Start RoadRunner Simulation
start(rrSim);
fprintf('  Simulation status: RUNNING...\n');

while strcmp(get(rrSim, 'SimulationStatus'), 'Running')
    simTime = get(rrSim, 'SimulationTime');
    
    % -------------------------------------------------------------
    % A. Read Ground Truth & Actor States from RoadRunner
    % -------------------------------------------------------------
    actorPoses = getActorPoses(rrSim);
    egoPose    = getActorPose(rrSim, ego_handle);
    
    % Extract ego [x, y, heading, speed]
    ego.x       = egoPose.Position(1);
    ego.y       = egoPose.Position(2);
    ego.heading = egoPose.Orientation(3); % Yaw in radians
    ego.speed   = norm(egoPose.Velocity(1:2));
    
    % Filter obstacle tracks (excluding Ego)
    tracks = extract_rr_obstacles(actorPoses, egoPose);
    
    % -------------------------------------------------------------
    % B. Trajectory Prediction & Risk Occupancy Grid
    % -------------------------------------------------------------
    [pred_x, pred_y, ~] = predict_trajectories(tracks, 3.0, 15);
    
    % Build Dynamic Binary Occupancy Map
    occ_map = build_occupancy_grid_from_rr(tracks, ego);
    
    % -------------------------------------------------------------
    % C. Adaptive Path Planning (Hybrid A*)
    % -------------------------------------------------------------
    dist_to_goal = norm([200.0, 0.0] - [ego.x, ego.y]);
    time_since_replan = (simTime - last_replan_t) * 1000;
    
    [need_replan, ~] = replan_trigger(current_path, struct('x',{pred_x},'y',{pred_y}), ...
                                           time_since_replan, occ_map);
    
    if need_replan && dist_to_goal > 4.0
        goal_pose = [min(200.0, ego.x + 40), 0.0, 0.0];
        [px, py, pyaw, ~] = plan_path(occ_map, [ego.x, ego.y, ego.heading], goal_pose, veh);
        if ~isempty(px)
            current_path.x = px;
            current_path.y = py;
            current_path.yaw = pyaw;
        end
        last_replan_t = simTime;
    end
    
    % -------------------------------------------------------------
    % D. Decision Engine with Ratchet
    % -------------------------------------------------------------
    nearest_hazard = find_nearest_hazard(tracks);
    decision = decision_with_ratchet(nearest_hazard, false, simTime);
    
    % -------------------------------------------------------------
    % E. Vehicle Control (Pure Pursuit + PID)
    % -------------------------------------------------------------
    steer_cmd = pure_pursuit_controller(ego, current_path, pp, veh);
    target_speed = decision_to_target_speed(decision.level, sim_cfg);
    [throttle_cmd, brake_cmd, pid_state] = pid_controller(ego.speed, target_speed, pid_p, pid_state);
    
    % -------------------------------------------------------------
    % F. Write Action Back to RoadRunner Ego Vehicle
    % -------------------------------------------------------------
    egoCmd = struct();
    egoCmd.SteeringAngle = steer_cmd;
    egoCmd.Throttle      = throttle_cmd;
    egoCmd.Brake         = brake_cmd;
    setActorControl(rrSim, ego_handle, egoCmd);
    
    % Step RoadRunner simulation
    step(rrSim);
end

stop(rrSim);
fprintf('✅ RoadRunner Co-simulation complete!\n');
end

% ---------------------------------------------------------------------------
% Fallback function for running scenario loop when RR app is offline
% ---------------------------------------------------------------------------
function run_simulated_rr_loop(scenarioName)
    fprintf('  Running scenario via closed-loop adaptive engine...\n');
    clean_name = strrep(strrep(scenarioName, '.rrscenario', ''), 'scenario_', '');
    switch lower(clean_name)
        case 'village_road', scenario_village_road('adaptive');
        case 'urban_intersection', scenario_urban_intersection('adaptive');
        case 'highway_merge', scenario_highway_merge('adaptive');
        case 'dense_market', scenario_dense_market('adaptive');
        case 'cattle_crossing', scenario_cattle_crossing('adaptive');
        otherwise, scenario_village_road('adaptive');
    end
end

function hazard = find_nearest_hazard(tracks)
    hazard = [];
    if isempty(tracks), return; end
    min_d = inf;
    for k = 1:numel(tracks)
        t = tracks(k);
        if isfield(t, 'distance_m') && ~isempty(t.distance_m) && t.distance_m < min_d
            min_d = t.distance_m;
            hazard = t;
        end
    end
end

function target_v = decision_to_target_speed(level, sim_cfg)
    base_v = sim_cfg.init_speed_mps;
    switch level
        case 0, target_v = base_v;         % PROCEED
        case 1, target_v = base_v * 0.7;   % CAUTION
        case 2, target_v = base_v * 0.4;   % SLOW
        case 3, target_v = base_v * 0.15;  % BRAKE
        case 4, target_v = 0.0;            % EMERGENCY_STOP
        otherwise, target_v = base_v;
    end
end

function steer = pure_pursuit_controller(ego, current_path, pp, veh)
    steer = 0.0;
    if isempty(current_path) || isempty(current_path.x), return; end
    la = max(pp.min_lookahead, min(pp.max_lookahead, pp.lookahead_m + pp.k_lookahead * ego.speed));
    
    pts = [current_path.x, current_path.y];
    dists = sqrt((pts(:,1) - ego.x).^2 + (pts(:,2) - ego.y).^2);
    [~, target_idx] = min(abs(dists - la));
    
    wp = pts(target_idx, :);
    dx = wp(1) - ego.x;
    dy = wp(2) - ego.y;
    alpha = atan2(dy, dx) - ego.heading;
    alpha = atan2(sin(alpha), cos(alpha)); % Wrap angle
    steer = atan2(2 * veh.wheelbase_m * sin(alpha), la);
    steer = max(-veh.max_steer_rad, min(veh.max_steer_rad, steer));
end

function [thr, brk, pid_s] = pid_controller(v, v_target, pid_p, pid_s)
    err = v_target - v;
    pid_s.integral = pid_s.integral + err * pid_p.dt;
    pid_s.integral = max(-pid_p.int_max, min(pid_p.int_max, pid_s.integral));
    deriv = (err - pid_s.prev_err) / pid_p.dt;
    pid_s.prev_err = err;
    cmd = pid_p.Kp * err + pid_p.Ki * pid_s.integral + pid_p.Kd * deriv;
    thr = max(0.0, min(1.0, cmd));
    brk = max(0.0, min(1.0, -cmd));
end
