function [committed_level, target_speed, lateral_cmd, rule_id, raw_history, down_counter, emergency_latch] = decision_with_ratchet(nearest_dist, ttc, obj_class, in_path, closing_speed, degraded, lane_offset, prev_committed, raw_history, down_counter, emergency_latch)
% DECISION_WITH_RATCHET 
% Enhanced decision logic with temporal ratchet (N-of-M voting). 
% Serves as reference for Stateflow porting. Matches decision_engine.py.
%
% Levels: PROCEED=0, CAUTION=1, SLOW=2, BRAKE=3, EMERGENCY_STOP=4
PROCEED = 0; CAUTION = 1; SLOW = 2; BRAKE = 3; EMERGENCY_STOP = 4;

% 1. Compute raw_level (R1-R7 rules)
raw_level = PROCEED;
raw_rule_id = 'R7';

if in_path
    if nearest_dist <= 1.2
        raw_level = EMERGENCY_STOP; raw_rule_id = 'R1';
    elseif ttc <= 0.9
        raw_level = EMERGENCY_STOP; raw_rule_id = 'R2/R3';
    elseif ttc <= 1.8
        raw_level = BRAKE; raw_rule_id = 'R4/R5';
    elseif nearest_dist <= 6.0
        raw_level = SLOW; raw_rule_id = 'R6';
    elseif nearest_dist <= 15.0
        raw_level = CAUTION; raw_rule_id = 'R7';
    end
else
    % Clear path -> reset latch immediately
    emergency_latch = 0;
end

% Update raw_history array (size 5 history maintained)
if length(raw_history) >= 5
    raw_history = [raw_history(2:end), raw_level];
else
    raw_history = [raw_history, raw_level];
end

% 2. Temporal ratchet escalation (N-of-M voting)
voted_level = prev_committed;
hist_len = length(raw_history);

if in_path
    if hist_len >= 3 && sum(raw_history(end-2:end) == EMERGENCY_STOP) >= 2
        voted_level = EMERGENCY_STOP;
    elseif hist_len >= 4 && sum(raw_history(end-3:end) >= BRAKE) >= 2
        voted_level = BRAKE;
    elseif hist_len >= 3 && sum(raw_history(end-2:end) >= SLOW) >= 2
        voted_level = SLOW;
    elseif raw_level > prev_committed
        voted_level = raw_level; 
    end
else
    % Path is clear: direct recovery towards PROCEED
    voted_level = PROCEED;
end

% 3. Temporal ratchet de-escalation (FASTER recovery)
if voted_level < prev_committed
    if emergency_latch > 0 && in_path
        voted_level = prev_committed; % Block de-escalation due to latch while hazard remains
    else
        req_down_frames = 2; % Rapid de-escalation once hazard clears
        if degraded
            req_down_frames = 4;
        end
        
        if down_counter >= req_down_frames
            voted_level = max(0, prev_committed - 1); % Go DOWN 1 level at a time
            down_counter = 0;
        else
            voted_level = prev_committed;
            down_counter = down_counter + 1;
        end
    end
else
    down_counter = 0;
end

% 4. VRU proximity floor (only when in path)
is_vru = contains(lower(obj_class), {'person','bicycle','cow','dog','cat'});
if is_vru && in_path && nearest_dist <= 6
    voted_level = max(voted_level, CAUTION);
end

% 5. Degraded mode floor
if degraded && in_path
    voted_level = max(voted_level, CAUTION);
end

committed_level = voted_level;

% Manage emergency latch
if committed_level == EMERGENCY_STOP && in_path
    emergency_latch = 3;
elseif emergency_latch > 0
    emergency_latch = emergency_latch - 1;
end

% 6. Lateral arbitration
offset_thresh = 50;
if committed_level >= EMERGENCY_STOP
    lateral_cmd = 'HOLD';
else
    if committed_level >= BRAKE
        offset_thresh = 100;
    end
    
    if lane_offset > offset_thresh
        lateral_cmd = 'CORRECT_LEFT';
    elseif lane_offset < -offset_thresh
        lateral_cmd = 'CORRECT_RIGHT';
    else
        lateral_cmd = 'HOLD';
    end
end

% 7. Map committed_level to target_speed
speed_map = [13.9, 11.1, 7.0, 2.5, 0.0]; % PROCEED=50, CAUTION=40, SLOW=25, BRAKE=9, ESTOP=0 km/h
target_speed = speed_map(committed_level + 1);

% 8. Map committed_level to rule_id
rule_map = {'R7', 'R6', 'R5/R4', 'R2/R3', 'R1'};
rule_id = rule_map{committed_level + 1};

end
