function [path_x, path_y, path_yaw, plan_time_ms] = plan_path(occupancy_map, ego_pose, goal_pose, veh_params)
% PLAN_PATH Hybrid A* and Adaptive Frenet-Frame Collision-Avoidance Path Planner.
%
%   [path_x, path_y, path_yaw, plan_time_ms] = plan_path(occupancy_map, ego_pose, goal_pose, veh_params)
%
%   Inputs:
%       occupancy_map: a binaryOccupancyMap object
%       ego_pose:      [x, y, yaw] current ego pose
%       goal_pose:     [x, y, yaw] target position
%       veh_params:    struct with vehicle dimensions & turning limits
%
%   Outputs:
%       path_x:        Nx1 array of smoothed path X coordinates
%       path_y:        Nx1 array of smoothed path Y coordinates
%       path_yaw:      Nx1 array of smoothed path headings
%       plan_time_ms:  Planning execution time in milliseconds
%
% ADAS Vision — Robust Autonomous Path Planner

    tic;

    if nargin < 4 || isempty(veh_params)
        veh_params = struct('min_turning_radius', 4.5, 'width_m', 1.85, 'length_m', 4.4);
    end
    min_turning_radius = 4.5;
    if isfield(veh_params, 'min_turning_radius'), min_turning_radius = veh_params.min_turning_radius; end

    path_x = [];
    path_y = [];
    path_yaw = [];

    % 1. Try Navigation Toolbox Hybrid A* if available and start/goal are clear
    try
        % Check if start pose is free in occupancy map
        start_free = true;
        if checkOccupancy(occupancy_map, ego_pose(1:2)) > 0
            start_free = false;
        end

        if start_free
            planner = plannerHybridAStar(occupancy_map, ...
                'MinTurningRadius', min_turning_radius, ...
                'MotionPrimitiveLength', 1.8, ...
                'NumMotionPrimitives', 7);
            
            refPath = plan(planner, ego_pose, goal_pose);
            
            if ~isempty(refPath) && ~isempty(refPath.States)
                raw_x = refPath.States(:, 1);
                raw_y = refPath.States(:, 2);
                
                dx = diff(raw_x); dy = diff(raw_y);
                dists = [0; cumsum(sqrt(dx.^2 + dy.^2))];
                total_dist = dists(end);
                
                if total_dist >= 1.0 && length(raw_x) >= 3
                    query_dists = (0:0.4:total_dist)';
                    path_x = spline(dists, raw_x, query_dists);
                    path_y = spline(dists, raw_y, query_dists);
                    dx_s = gradient(path_x); dy_s = gradient(path_y);
                    path_yaw = atan2(dy_s, dx_s);
                end
            end
        end
    catch
        % Fall through to adaptive trajectory generator
    end

    % 2. Adaptive Collision-Avoidance Trajectory Generator (Guaranteed Fallback)
    if isempty(path_x)
        x0 = ego_pose(1);
        y0 = ego_pose(2);
        psi0 = ego_pose(3);
        
        x_target = goal_pose(1);
        y_target = goal_pose(2);
        
        lookahead = min(40.0, max(15.0, x_target - x0));
        
        % Sample potential lateral corridor offsets y in [-2.5, +2.5]
        candidate_y = linspace(-2.4, 2.4, 25);
        best_y = y_target;
        min_cost = Inf;
        
        % Sample collision checks along candidate trajectories
        eval_x = linspace(x0, x0 + lookahead, 20)';
        
        for cy = candidate_y
            % Quintic-like smooth lateral transition
            s = min(1.0, max(0.0, (eval_x - x0) / max(1.0, lookahead)));
            % Smoothstep polynomial: 3*s^2 - 2*s^3
            poly_lat = y0 + (cy - y0) * (3*s.^2 - 2*s.^3);
            
            eval_pts = [eval_x, poly_lat];
            
            % Check occupancy collision cost
            in_lims = eval_pts(:,1) >= occupancy_map.XWorldLimits(1) & ...
                      eval_pts(:,1) <= occupancy_map.XWorldLimits(2) & ...
                      eval_pts(:,2) >= occupancy_map.YWorldLimits(1) & ...
                      eval_pts(:,2) <= occupancy_map.YWorldLimits(2);
            
            occ_count = 0;
            if any(in_lims)
                occ_vals = getOccupancy(occupancy_map, eval_pts(in_lims, :));
                occ_count = sum(occ_vals > 0.5);
            end
            
            % Cost function: Collision penalty + Deviation from goal + Lateral jerk penalty
            cost = occ_count * 1000.0 + 3.0 * abs(cy - y_target) + 1.5 * abs(cy - y0);
            
            if cost < min_cost
                min_cost = cost;
                best_y = cy;
            end
        end
        
        % Generate final smoothed query path
        n_pts = max(30, round(lookahead / 0.4));
        query_x = linspace(x0, x0 + lookahead, n_pts)';
        s_final = min(1.0, max(0.0, (query_x - x0) / max(1.0, lookahead)));
        
        % Blend smoothly with initial heading
        lat_blend = y0 + (best_y - y0) * (3*s_final.^2 - 2*s_final.^3);
        
        path_x = query_x;
        path_y = lat_blend;
        
        dx_s = gradient(path_x); dy_s = gradient(path_y);
        path_yaw = atan2(dy_s, dx_s);
    end

    plan_time_ms = toc * 1000;
end
