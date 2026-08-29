function [should_replan, reason] = replan_trigger(current_path, predictions, time_since_last_replan_ms, occupancy_map)
% REPLAN_TRIGGER Decides whether replanning is needed
%   [should_replan, reason] = replan_trigger(current_path, predictions, time_since_last_replan_ms, occupancy_map)
%
%   Triggers replanning if:
%       - time_since_last_replan_ms >= 1500 (periodic replan every 1.5s — NOT 200ms)
%       - Any predicted trajectory intersects the current planned path within 1.0m clearance
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
    
    % Condition 2: Periodic replan every 1500ms (was 200ms — caused pathological thrashing)
    if time_since_last_replan_ms >= 1500
        should_replan = true;
        reason = 'Periodic replanning (time >= 1500ms)';
        return;
    end
    
    path_points = [current_path.x, current_path.y];
    
    % Condition 3: Path-prediction intersection check
    % Only check the NEXT 8 path points ahead (not the full 40-point path behind ego)
    check_pts = path_points(1:min(8, size(path_points, 1)), :);
    
    if ~isempty(predictions) && isfield(predictions, 'x') && ~isempty(predictions.x)
        num_preds = length(predictions.x);
        for p = 1:num_preds
            pred_pts = [predictions.x{p}, predictions.y{p}];
            if ~isempty(pred_pts) && ~isempty(check_pts)
                dists = pdist2(check_pts, pred_pts);
                if min(dists(:)) < 1.0
                    should_replan = true;
                    reason = 'Predicted obstacle trajectory intersects path within 1.0m clearance';
                    return;
                end
            end
        end
    end
    
    % Condition 4: Current path passes through newly occupied cells
    if ~isempty(occupancy_map)
        valid_idx = check_pts(:,1) >= occupancy_map.XWorldLimits(1) & ...
                    check_pts(:,1) <= occupancy_map.XWorldLimits(2) & ...
                    check_pts(:,2) >= occupancy_map.YWorldLimits(1) & ...
                    check_pts(:,2) <= occupancy_map.YWorldLimits(2);
        
        valid_points = check_pts(valid_idx, :);
        
        if ~isempty(valid_points)
            is_occ = getOccupancy(occupancy_map, valid_points);
            if any(is_occ)
                should_replan = true;
                reason = 'Current path passes through newly occupied cells';
                return;
            end
        end
    end

end

