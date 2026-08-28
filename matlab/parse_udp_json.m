function [num_tracks, track_ids, track_x, track_y, track_distance, track_ttc, track_in_path, track_closing_speed, road_center_x, road_confidence, road_offset_px, decision_level, decision_brake, decision_throttle, degraded, fps] = parse_udp_json(udp_bytes)
% PARSE_UDP_JSON Parses JSON string received via UDP for Simulink ADAS
%
% This function is designed to be used inside a Simulink MATLAB Function block.
% It takes a raw UDP byte array, decodes the JSON, and outputs structured
% signals with fixed dimensions (required by Simulink).

    % Simulink MATLAB Function Block Requirements:
    % - Outputs must have fixed size and type.
    % - We pad array outputs to a maximum of 20 elements.
    MAX_TRACKS = 20;

    % Initialize outputs with default values
    num_tracks = 0;
    track_ids = zeros(1, MAX_TRACKS);
    track_x = zeros(1, MAX_TRACKS);
    track_y = zeros(1, MAX_TRACKS);
    track_distance = zeros(1, MAX_TRACKS);
    track_ttc = zeros(1, MAX_TRACKS);
    track_in_path = false(1, MAX_TRACKS);
    track_closing_speed = zeros(1, MAX_TRACKS);
    
    road_center_x = 0.0;
    road_confidence = 0.0;
    road_offset_px = 0.0;
    
    decision_level = 0.0;
    decision_brake = 0.0;
    decision_throttle = 0.0;
    
    degraded = false;
    fps = 0.0;

    % Convert byte array to character vector
    % UDP bytes might be zero-padded, strip null characters
    json_str = char(udp_bytes(udp_bytes > 0)');

    if isempty(json_str)
        return;
    end

    try
        % Decode JSON
        data = jsondecode(json_str);

        % Extract generic info
        if isfield(data, 'fps')
            fps = double(data.fps);
        end
        if isfield(data, 'degraded')
            degraded = logical(data.degraded);
        end

        % Extract road info
        if isfield(data, 'road')
            if isfield(data.road, 'center_x')
                road_center_x = double(data.road.center_x);
            end
            if isfield(data.road, 'confidence')
                road_confidence = double(data.road.confidence);
            end
            if isfield(data.road, 'offset_px')
                road_offset_px = double(data.road.offset_px);
            end
        end

        % Extract decision info
        if isfield(data, 'decision')
            if isfield(data.decision, 'level')
                decision_level = double(data.decision.level);
            end
            if isfield(data.decision, 'brake')
                decision_brake = double(data.decision.brake);
            end
            if isfield(data.decision, 'throttle')
                decision_throttle = double(data.decision.throttle);
            end
        end

        % Extract track info
        if isfield(data, 'tracks') && ~isempty(data.tracks)
            num_t = length(data.tracks);
            num_tracks = min(num_t, MAX_TRACKS);
            
            for i = 1:num_tracks
                t = data.tracks(i);
                if isfield(t, 'id')
                    track_ids(i) = double(t.id);
                end
                if isfield(t, 'x')
                    track_x(i) = double(t.x);
                end
                if isfield(t, 'y')
                    track_y(i) = double(t.y);
                end
                if isfield(t, 'distance')
                    track_distance(i) = double(t.distance);
                end
                if isfield(t, 'ttc')
                    track_ttc(i) = double(t.ttc);
                end
                if isfield(t, 'in_path')
                    track_in_path(i) = logical(t.in_path);
                end
                if isfield(t, 'closing_speed')
                    track_closing_speed(i) = double(t.closing_speed);
                end
            end
        end
    catch
        % If parsing fails, output defaults
    end
end
