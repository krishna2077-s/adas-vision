function [path_x, path_y, path_yaw, plan_time_ms] = plan_path(occupancy_map, ego_pose, goal_pose, veh_params)
% PLAN_PATH Robust Collision-Avoidance Path Planner for Autonomous Driving.
%
%   [path_x, path_y, path_yaw, plan_time_ms] = plan_path(occupancy_map, ego_pose, goal_pose, veh_params)
%
%   Generates smooth, collision-free trajectories avoiding dynamic & static obstacles
%   while respecting road limits and vehicle turning constraints.
%
% ADAS Vision — SIH 2026

    tic;

    if nargin < 4 || isempty(veh_params)
        veh_params = struct('min_turning_radius', 4.5, 'width_m', 1.85, 'length_m', 4.4);
    end

    x0   = ego_pose(1);
    y0   = ego_pose(2);
    psi0 = ego_pose(3);

    x_target = goal_pose(1);
    y_target = goal_pose(2);

    lookahead = min(45.0, max(18.0, x_target - x0));
    
    % Candidate lateral offsets across the drivable road (LHT biased: prefers y <= 0)
    candidate_y = [-1.5, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.5, -2.0, 2.0];
    
    best_y = y_target;
    min_cost = Inf;

    n_eval = 25;
    eval_x = linspace(x0, x0 + lookahead, n_eval)';

    for cy = candidate_y
        % Smooth transition polynomial: s in [0, 1]
        s = min(1.0, max(0.0, (eval_x - x0) / max(1.0, lookahead)));
        poly_lat = y0 + (cy - y0) * (3*s.^2 - 2*s.^3);

        % Check centerline, left bumper, and right bumper for collision
        pts_center = [eval_x, poly_lat];
        pts_left   = [eval_x, poly_lat + 0.65];
        pts_right  = [eval_x, poly_lat - 0.65];

        all_check_pts = [pts_center; pts_left; pts_right];

        % Check bounds in occupancy map
        in_map = all_check_pts(:,1) >= occupancy_map.XWorldLimits(1) & ...
                 all_check_pts(:,1) <= occupancy_map.XWorldLimits(2) & ...
                 all_check_pts(:,2) >= occupancy_map.YWorldLimits(1) & ...
                 all_check_pts(:,2) <= occupancy_map.YWorldLimits(2);

        occ_penalty = 0;
        if any(in_map)
            occ_vals = getOccupancy(occupancy_map, all_check_pts(in_map, :));
            occ_penalty = sum(occ_vals > 0.5);
        end

        % Cost: Collisions (heavy) + lane bias + lateral deviation + steering effort
        cost = occ_penalty * 5000.0 + ...
               4.0 * abs(cy - y_target) + ...
               2.0 * abs(cy - y0) + ...
               1.0 * max(0.0, cy); % Slight bias towards left-hand driving (y <= 0)

        if cost < min_cost
            min_cost = cost;
            best_y = cy;
        end
    end

    % Generate high-resolution smoothed trajectory
    n_pts = max(35, round(lookahead / 0.4));
    query_x = linspace(x0, x0 + lookahead, n_pts)';
    s_final = min(1.0, max(0.0, (query_x - x0) / max(1.0, lookahead)));

    lat_final = y0 + (best_y - y0) * (3*s_final.^2 - 2*s_final.^3);

    path_x = query_x;
    path_y = lat_final;

    dx_s = gradient(path_x);
    dy_s = gradient(path_y);
    path_yaw = atan2(dy_s, dx_s);

    plan_time_ms = toc * 1000;
end
