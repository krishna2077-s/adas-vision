function export_result_json(scenario_result, output_path)
% EXPORT_RESULT_JSON — Serialises scenario_result struct to JSON file.
% Called by generated scenario scripts. Does NOT modify any existing file.
% This is the ONLY new MATLAB file added to the project.

    s = struct();
    
    % Core outcomes
    s.arrived         = scenario_result.arrived;
    s.t_total_s       = scenario_result.t_total;
    s.collisions      = scenario_result.collisions;
    s.replans         = scenario_result.replans;
    
    if isfield(scenario_result, 'min_dist_m') && ~isempty(scenario_result.min_dist_m)
        s.min_clearance_m = min(scenario_result.min_dist_m);
    else
        s.min_clearance_m = 99.0;
    end
    
    % Replan stats
    if isfield(scenario_result, 'replan_latency_ms') && ~isempty(scenario_result.replan_latency_ms)
        s.mean_replan_ms = mean(scenario_result.replan_latency_ms);
        s.max_replan_ms  = max(scenario_result.replan_latency_ms);
    else
        s.mean_replan_ms = 0; 
        s.max_replan_ms = 0;
    end
    
    % Trajectory (sampled to max 500 points for lighter JSON)
    if isfield(scenario_result, 'traj_x') && ~isempty(scenario_result.traj_x)
        n    = numel(scenario_result.traj_x);
        step = max(1, floor(n/500));
        idx  = 1:step:n;
        s.traj_x = scenario_result.traj_x(idx);
        s.traj_y = scenario_result.traj_y(idx);
    else
        s.traj_x = [];
        s.traj_y = [];
    end
    
    % Speed metrics
    if isfield(scenario_result,'speed_mps') && ~isempty(scenario_result.speed_mps)
        spd = scenario_result.speed_mps;
        s.mean_speed_kmh = mean(spd)*3.6;
        s.max_speed_kmh  = max(spd)*3.6;
    else
        s.mean_speed_kmh = 0;
        s.max_speed_kmh  = 0;
    end
    
    % Decision distribution
    if isfield(scenario_result,'decision') && ~isempty(scenario_result.decision)
        d   = scenario_result.decision;
        n_d = numel(d);
        for lv = {'PROCEED','CAUTION','SLOW','BRAKE','EMERGENCY_STOP'}
            key = ['decision_' lower(lv{1})];
            s.(key) = sum(cellfun(@(x) strcmpi(x,lv{1}), d)) / max(n_d, 1);
        end
    else
        s.decision_proceed = 1.0;
        s.decision_caution = 0.0;
        s.decision_slow = 0.0;
        s.decision_brake = 0.0;
        s.decision_emergency_stop = 0.0;
    end
    
    % Path smoothness
    if isfield(scenario_result, 'traj_x') && numel(scenario_result.traj_x) > 3
        dx1 = diff(scenario_result.traj_x); dy1 = diff(scenario_result.traj_y);
        dx2 = diff(dx1); dy2 = diff(dy1);
        ds  = sqrt(dx1(1:end-1).^2 + dy1(1:end-1).^2) + 1e-9;
        kap = abs(dx1(1:end-1).*dy2 - dy1(1:end-1).*dx2)./(ds.^3+1e-9);
        s.path_smoothness = mean(kap);
    else
        s.path_smoothness = 0;
    end

    % Write JSON file
    fid = fopen(output_path, 'w');
    if fid < 0
        warning('export_result_json: cannot write to %s', output_path);
        return;
    end
    fprintf(fid, '%s', jsonencode(s));
    fclose(fid);
    fprintf('[Scenario Lab] Result exported: %s\n', output_path);
end
