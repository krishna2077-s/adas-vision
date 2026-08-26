function [path_x, path_y, path_yaw, plan_time_ms] = plan_path(occupancy_map, ego_pose, goal_pose, veh_params)
% PLAN_PATH Implements path planning using Hybrid A*
%   [path_x, path_y, path_yaw, plan_time_ms] = plan_path(occupancy_map, ego_pose, goal_pose, veh_params)
%
%   Inputs:
%       occupancy_map: a binaryOccupancyMap object
%       ego_pose: [x, y, yaw] current ego position
%       goal_pose: [x, y, yaw] target position
%       veh_params: struct with min_turning_radius (default 5.0m)
%
%   Outputs:
%       path_x: Nx1 array of smoothed path x coordinates
%       path_y: Nx1 array of smoothed path y coordinates
%       path_yaw: Nx1 array of smoothed path headings
%       plan_time_ms: Planning time in milliseconds

    % Start timing
    tic;

    % Set default turning radius if not provided
    if nargin < 4 || ~isfield(veh_params, 'min_turning_radius')
        min_turning_radius = 5.0;
    else
        min_turning_radius = veh_params.min_turning_radius;
    end

    % Create planner
    % We use Hybrid A* with forward and reverse motions
    planner = plannerHybridAStar(occupancy_map, ...
        'MinTurningRadius', min_turning_radius, ...
        'MotionPrimitiveLength', 2.0, ...
        'NumMotionPrimitives', 7);
    
    % The planner allows both forward and reverse motions by default for vehicles.

    % Try planning the path
    try
        refPath = plan(planner, ego_pose, goal_pose);
        
        % If plan is successful, extract the poses
        if isempty(refPath.States)
            path_x = [];
            path_y = [];
            path_yaw = [];
            plan_time_ms = toc * 1000;
            return;
        end
        
        raw_x = refPath.States(:, 1);
        raw_y = refPath.States(:, 2);
        
        % Calculate cumulative distance along path for interpolation
        dx = diff(raw_x);
        dy = diff(raw_y);
        dists = [0; cumsum(sqrt(dx.^2 + dy.^2))];
        total_dist = dists(end);
        
        % Check if path is too short for spline interpolation
        if total_dist < 0.5 || length(raw_x) < 2
            path_x = raw_x;
            path_y = raw_y;
            path_yaw = refPath.States(:, 3);
            plan_time_ms = toc * 1000;
            return;
        end

        % Create query points at 0.5m spacing
        query_dists = (0:0.5:total_dist)';
        
        % Smooth the path using cubic spline interpolation
        path_x = spline(dists, raw_x, query_dists);
        path_y = spline(dists, raw_y, query_dists);
        
        % Compute yaw from spline tangents
        % Using central differences for the interior and forward/backward for ends
        dx_spline = gradient(path_x);
        dy_spline = gradient(path_y);
        path_yaw = atan2(dy_spline, dx_spline);
        
    catch ME
        % If no path found or other error, return empty
        warning('Path planning failed: %s', ME.message);
        path_x = [];
        path_y = [];
        path_yaw = [];
    end

    % Record planning time
    plan_time_ms = toc * 1000;
end
