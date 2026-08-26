function [longitudinal_cmd, target_speed_mps, lateral_cmd] = stateflow_reference(nearest_distance_m, ttc_s, object_class, in_path, closing_speed, degraded, lane_offset_px)
% STATEFLOW_REFERENCE Implements R1-R7 decision logic for Stateflow porting.
%
%   This function serves as a direct reference for porting the Python
%   decision_engine.py logic to a Simulink Stateflow chart.
%
%   Inputs:
%       nearest_distance_m - Distance to nearest obstacle in meters.
%       ttc_s              - Time-to-collision in seconds.
%       object_class       - String representing the class of the obstacle.
%       in_path            - Boolean flag indicating if obstacle is in ego path.
%       closing_speed      - Relative closing speed (positive if closing in).
%       degraded           - Boolean flag for degraded system mode (e.g. low visibility).
%       lane_offset_px     - Lateral deviation from lane center in pixels.
%
%   Outputs:
%       longitudinal_cmd   - Command string: 'EMERGENCY_STOP', 'BRAKE', 'SLOW', 'CAUTION', 'PROCEED'
%       target_speed_mps   - Target speed in meters per second.
%       lateral_cmd        - Command string: 'CORRECT', 'KEEP_LANE', 'HOLD'

    % Helper classifications
    vru_classes = {'person', 'bicycle', 'cow', 'dog', 'cat'};
    is_vru = ismember(lower(string(object_class)), vru_classes);
    is_static_sign = ismember(lower(string(object_class)), {'stop sign', 'traffic light'});
    
    % Derived risk assessment (Simplified for reference mapping)
    if ttc_s <= 3.0 || nearest_distance_m <= 10.0
        risk = 'HIGH';
    elseif ttc_s <= 6.0 || nearest_distance_m <= 25.0
        risk = 'MEDIUM';
    else
        risk = 'LOW';
    end

    % --- Rule Evaluation ---
    
    % Initialize default
    longitudinal_cmd = 'PROCEED';
    
    % Dynamic margins based on Python engine logic
    brake_ttc_threshold = 2.5;
    if is_vru
        brake_ttc_threshold = brake_ttc_threshold + 0.8; % VRU margin
    end
    
    slow_distance_threshold = 20.0;
    if degraded
        slow_distance_threshold = slow_distance_threshold + 5.0; % Degraded margin
    end

    % Evaluated in order of priority:
    
    % R1: EMERGENCY_STOP if distance <= 5.0 OR (TTC <= 1.2 AND closing_speed > 0)
    if nearest_distance_m <= 5.0 || (ttc_s <= 1.2 && closing_speed > 0)
        longitudinal_cmd = 'EMERGENCY_STOP';
        
    % R2: BRAKE if TTC <= 2.5 AND closing_speed > 0 (margin added for VRU above)
    elseif ttc_s <= brake_ttc_threshold && closing_speed > 0
        longitudinal_cmd = 'BRAKE';
        
    % R3: BRAKE if risk is HIGH AND in_path AND distance <= 8.0 AND closing_speed <= 0
    elseif strcmp(risk, 'HIGH') && in_path && nearest_distance_m <= 8.0 && closing_speed <= 0
        longitudinal_cmd = 'BRAKE';
        
    % R4: SLOW if class is 'stop sign' or 'traffic light' AND distance <= 25.0
    elseif is_static_sign && nearest_distance_m <= 25.0
        longitudinal_cmd = 'SLOW';
        
    % R5: SLOW if risk is MEDIUM AND in_path AND distance <= 20.0 (margin added for degraded above)
    elseif strcmp(risk, 'MEDIUM') && in_path && nearest_distance_m <= slow_distance_threshold
        longitudinal_cmd = 'SLOW';
        
    % R6: CAUTION if TTC <= 4.0 OR (risk >= MEDIUM AND in_path)
    elseif ttc_s <= 4.0 || (ismember(risk, {'HIGH', 'MEDIUM'}) && in_path)
        longitudinal_cmd = 'CAUTION';
        
    % R7: PROCEED (default, clear path)
    else
        longitudinal_cmd = 'PROCEED';
    end
    
    % --- VRU Safety Floor ---
    % if class in {person, bicycle, cow, dog, cat} AND in_path AND distance <= 15m → force minimum CAUTION
    if is_vru && in_path && nearest_distance_m <= 15.0
        if strcmp(longitudinal_cmd, 'PROCEED')
            longitudinal_cmd = 'CAUTION';
        end
    end
    
    % --- Degraded Mode Floor ---
    % if degraded AND any hazard exists → force minimum CAUTION
    has_hazard = (nearest_distance_m < 100); 
    if degraded && has_hazard
        if strcmp(longitudinal_cmd, 'PROCEED')
            longitudinal_cmd = 'CAUTION';
        end
    end

    % --- Target Speeds ---
    % PROCEED=13.9 (50km/h), CAUTION=8.3 (30km/h), SLOW=5.6 (20km/h), BRAKE=0, E-STOP=0
    switch longitudinal_cmd
        case 'EMERGENCY_STOP'
            target_speed_mps = 0;
        case 'BRAKE'
            target_speed_mps = 0;
        case 'SLOW'
            target_speed_mps = 5.6;
        case 'CAUTION'
            target_speed_mps = 8.3;
        case 'PROCEED'
            target_speed_mps = 13.9;
        otherwise
            target_speed_mps = 13.9;
    end
    
    % --- Lateral Control ---
    % if abs(lane_offset_px) > 50 → CORRECT toward center, else KEEP_LANE.
    % If longitudinal is EMERGENCY_STOP → lateral = HOLD.
    if strcmp(longitudinal_cmd, 'EMERGENCY_STOP')
        lateral_cmd = 'HOLD';
    elseif abs(lane_offset_px) > 50
        lateral_cmd = 'CORRECT';
    else
        lateral_cmd = 'KEEP_LANE';
    end
    
end
