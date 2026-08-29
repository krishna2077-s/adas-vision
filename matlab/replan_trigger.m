function [should_replan, reason] = replan_trigger(current_path, predictions, time_since_last_replan_ms, occupancy_map)
% REPLAN_TRIGGER Decides whether replanning is needed
%   [should_replan, reason] = replan_trigger(current_path, predictions, time_since_last_replan_ms, occupancy_map)
%
%   Hackathon Requirement: Real-time replanning check
%   Triggers replanning if:
%       - time_since_last_replan_ms >= 200 (periodic replan every 200ms)
%       - Any predicted trajectory intersects the current planned path within 2m clearance
%       - Current path passes through newly occupied cells in the occupancy map
%       - No current path exists (first frame)

    % Default to no replan
    should_replan = false;
    reason = 'No replanning needed';
    
    % Condition 1: No current path exists
    if isempty(current_path) || ~isfield(current_path, 'x') || isempty(current_path.x)
        should_replan = true;
        reason = 'No current path exists (first frame)';
        return;
    end
    
    % Condition 2: Periodic replan every 1000ms (reduces computation lag)
    if time_since_last_replan_ms >= 1000
        should_replan = true;
        reason = 'Periodic replanning (time >= 1000ms)';
        return;
    end
    
    path_points = [current_path.x, current_path.y];
    
    % Condition 3: Path-prediction intersection check
    % If min distance < 1.2m (tighter threshold for dense markets), trigger replan.
    if ~isempty(predictions) && isfield(predictions, 'x') && ~isempty(predictions.x)
        num_preds = length(predictions.x);
        for p = 1:num_preds
            pred_pts = [predictions.x{p}, predictions.y{p}];
            if ~isempty(pred_pts)
                dists = pdist2(path_points, pred_pts);
                if min(dists(:)) < 1.2
                    should_replan = true;
                    reason = 'Predicted obstacle trajectory intersects path within 1.2m clearance';
                    return;
                end
            end
        end
    end
    
    % Condition 4: Current path passes through newly occupied cells in the occupancy map
    if ~isempty(occupancy_map)
        % Ensure points are within map bounds
        valid_idx = path_points(:,1) >= occupancy_map.XWorldLimits(1) & ...
                    path_points(:,1) <= occupancy_map.XWorldLimits(2) & ...
                    path_points(:,2) >= occupancy_map.YWorldLimits(1) & ...
                    path_points(:,2) <= occupancy_map.YWorldLimits(2);
        
        valid_points = path_points(valid_idx, :);
        
        if ~isempty(valid_points)
            % getOccupancy checks if the given xy locations are occupied
            is_occ = getOccupancy(occupancy_map, valid_points);
            if any(is_occ)
                should_replan = true;
                reason = 'Current path passes through newly occupied cells';
                return;
            end
        end
    end

end
