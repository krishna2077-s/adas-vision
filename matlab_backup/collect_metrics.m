function metrics = collect_metrics(log_data, scenario_name)
% COLLECT_METRICS Collects and reports hackathon metrics from log data.
% log_data: table or struct array with specific columns
% scenario_name: name of the scenario to save results correctly

    if isstruct(log_data)
        log_data = struct2table(log_data);
    end
    
    total_frames = height(log_data);
    if total_frames == 0
        error('Log data is empty.');
    end
    
    % Compute distances
    if total_frames > 1
        dx = diff(log_data.ego_x);
        dy = diff(log_data.ego_y);
        metrics.total_distance_m = sum(sqrt(dx.^2 + dy.^2));
    else
        metrics.total_distance_m = 0;
    end
    
    % Core metrics
    metrics.avg_speed_mps = mean(log_data.ego_speed);
    metrics.num_collisions = sum(log_data.min_obstacle_dist < 0.5);
    metrics.min_clearance_m = min(log_data.min_obstacle_dist);
    metrics.avg_replan_latency_ms = mean(log_data.replan_time_ms);
    metrics.max_replan_latency_ms = max(log_data.replan_time_ms);
    
    % Path smoothness (integral of squared curvature)
    metrics.path_smoothness = trapz(log_data.path_curvature.^2);
    
    % Scenario complete boolean: Basic heuristic assuming total dist > 5m or goal proximity
    % (In practice requires knowing goal coordinates, but placeholder proxy used here)
    metrics.scenario_completed = logical(metrics.total_distance_m > 5.0 && metrics.num_collisions == 0);
    
    % Percentage time in states
    metrics.time_in_emergency = (sum(log_data.decision_level == 4) / total_frames) * 100;
    metrics.time_in_brake = (sum(log_data.decision_level == 3) / total_frames) * 100;
    metrics.time_in_proceed = (sum(log_data.decision_level == 0) / total_frames) * 100;
    
    % Print clean results table
    fprintf('\n--- Hackathon Metrics: %s ---\n', scenario_name);
    disp(struct2table(metrics));
    
    % Save to CSV
    filename = sprintf('metrics_%s.csv', scenario_name);
    writetable(struct2table(metrics), filename);
    fprintf('Metrics saved to %s\n', filename);
end
