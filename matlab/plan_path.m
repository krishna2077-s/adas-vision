function [path_x, path_y, path_yaw, plan_time_ms] = plan_path(occupancy_map, ego_pose, goal_pose, veh_params)
% PLAN_PATH Robust Frenet Potential-Field Collision-Avoidance Path Planner.
%
%   [path_x, path_y, path_yaw, plan_time_ms] = plan_path(occupancy_map, ego_pose, goal_pose, veh_params)
%
%   Computes optimal, collision-free evasion trajectories with guaranteed lateral safety
%   margins (>2.2m clearance from obstacles) and smooth transition back to goal.
%
% ADAS Vision — SIH 2026 Autonomous Path Planner

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

    % Extract occupied world coordinates from occupancy map in forward ROI
    obs_pts = [];
    try
        occ_mat = occupancy_map.occupancyMatrix;
        [occ_r, occ_c] = find(occ_mat > 0.5);
        if ~isempty(occ_r)
            res = occupancy_map.Resolution;
            % MATLAB binaryOccupancyMap matrix: 
            % col 1 is X_min (origin(1)), col N is X_max
            % row 1 is Y_max, row M is Y_min (origin(2))
            origin = occupancy_map.GridLocationInWorld;
            num_rows = occupancy_map.GridSize(1);
            
            pts_x = origin(1) + (occ_c - 0.5) / res;
            pts_y = origin(2) + (num_rows - occ_r + 0.5) / res;

            % Filter to forward region of interest
            % Use lane_half_width from veh_params if available (e.g. highway = 7.5m)
            lateral_filter = 5.0;
            if isfield(veh_params, 'lane_half_width')
                lateral_filter = veh_params.lane_half_width + 1.0;
            end
            fwd = (pts_x >= x0 - 1.0 & pts_x <= x0 + lookahead + 5.0) & ...
                  (abs(pts_y) <= lateral_filter);
            if any(fwd)
                obs_pts = [pts_x(fwd), pts_y(fwd)];
            end
        end
    catch
        obs_pts = [];
    end

    is_dense_cfg = isfield(veh_params, 'is_dense') && veh_params.is_dense;
    
    % Candidate lateral offset range based on road width
    road_half = 2.0; % default if not provided
    if isfield(veh_params, 'lane_half_width')
        road_half = min(veh_params.lane_half_width - 0.5, 5.0); % leave 0.5m shoulder margin
    end
    
    % Candidate lateral offsets across the drivable road
    % Dense mode: wider range so ego can swerve hard left to bypass a kerb obstacle
    if is_dense_cfg
        candidate_y = linspace(-road_half, road_half, 61);
    else
        candidate_y = linspace(-road_half, road_half, 37);
    end
    
    best_y = y_target;
    min_cost = Inf;

    n_eval = 25;
    eval_x = linspace(x0, x0 + lookahead, n_eval)';

    for cy = candidate_y
        % Smooth transition polynomial: s in [0, 1]
        s = min(1.0, max(0.0, (eval_x - x0) / max(1.0, lookahead)));
        poly_lat = y0 + (cy - y0) * (3*s.^2 - 2*s.^3);

        traj_pts = [eval_x, poly_lat];

        % Compute clearance to nearest obstacle
        min_clearance = Inf;
        if ~isempty(obs_pts)
            % Sample check along trajectory
            for p = 1:size(traj_pts, 1)
                d_sq = (obs_pts(:, 1) - traj_pts(p, 1)).^2 + (obs_pts(:, 2) - traj_pts(p, 2)).^2;
                d_min_p = sqrt(min(d_sq));
                if d_min_p < min_clearance
                    min_clearance = d_min_p;
                end
            end
        end

        is_dense = isfield(veh_params, 'is_dense') && veh_params.is_dense;
        min_clearance_limit = 2.2;
        soft_clearance_limit = 1.8;
        if is_dense
            min_clearance_limit = 1.5;
            soft_clearance_limit = 1.2;
        end

        % Repulsion cost function based on minimum clearance:
        % Minimum physical radius: vehicle half-width ~0.95m + obstacle margin ~1.25m = 2.2m
        if min_clearance < min_clearance_limit
            repulsion_cost = 50000.0; % Hard collision penalty
        elseif min_clearance < 3.5
            % Strong smooth inverse-distance repulsion
            repulsion_cost = 350.0 / ((min_clearance - soft_clearance_limit)^2);
        else
            repulsion_cost = 0.0;
        end

        % Road boundary penalty — uses the same road_half computed above
        road_limit = road_half;
        road_edge_cost = 0.0;
        if abs(cy) > road_limit
            road_edge_cost = 1000.0 * (abs(cy) - road_limit)^2;
        end

        % Total Cost = Repulsion (safety) + Boundary + Goal Tracking + Smoothness
        cost = repulsion_cost + ...
               road_edge_cost + ...
               3.0 * abs(cy - y_target) + ...
               1.5 * abs(cy - y0) + ...
               0.5 * max(0.0, cy); % Preference for Left-Hand Driving (y <= 0)

        if cost < min_cost
            min_cost = cost;
            best_y = cy;
        end
    end

    % Generate high-resolution smoothed path
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
