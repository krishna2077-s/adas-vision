% udp_receiver_test.m
% 
% A standalone MATLAB script to receive and parse JSON packets from a Python UDP bridge.
% It listens on port 5005, reads the packets, parses them using jsondecode(), 
% and prints a summary of the ADAS perception data.
%
% Usage: Run the script. Press Ctrl+C to terminate.

disp('Starting UDP Receiver on port 5005...');
disp('Press Ctrl+C to stop.');

% Open a UDP socket on port 5005
% We use the modern udpport() function introduced in MATLAB R2021a.
u = udpport('Datagram', 'ipv4', 'LocalPort', 5005);
% Set timeout (e.g. wait for 1 second for packets)
u.Timeout = 1;

try
    while true
        % Check if data is available
        if u.NumBytesAvailable > 0
            % Read data from the port (returns a datagram packet array)
            data = read(u, u.NumBytesAvailable, 'string');
            
            % Process the latest datagram packet
            if ~isempty(data)
                % Extract the JSON string from the last datagram
                jsonStr = data(end).Data;
                
                try
                    % Parse the JSON string into a MATLAB struct
                    packet = jsondecode(jsonStr);
                    
                    % Extract frame info
                    frame_id = packet.frame_id;
                    fps = packet.fps;
                    
                    % Print header
                    fprintf('\n--- Frame ID: %d | FPS: %.1f ---\n', frame_id, fps);
                    
                    % Print tracked objects
                    num_objects = length(packet.objects);
                    fprintf('Tracked Objects: %d\n', num_objects);
                    
                    for i = 1:num_objects
                        obj = packet.objects(i);
                        % Handle potential missing fields gracefully
                        obj_id = obj.track_id;
                        obj_class = obj.class;
                        dist = obj.distance;
                        ttc = obj.ttc;
                        in_path = obj.in_path;
                        risk = obj.risk;
                        
                        fprintf('  Obj [%d] %s: Dist=%.1fm, TTC=%.1fs, InPath=%d, Risk=%s\n', ...
                                obj_id, obj_class, dist, ttc, in_path, risk);
                    end
                    
                    % Print road boundary info
                    if isfield(packet, 'road') && ~isempty(packet.road)
                        r = packet.road;
                        fprintf('Road Boundary: Center_X=%.1fpx, Conf=%.2f, Source=%s\n', ...
                                r.center_x, r.confidence, r.source);
                    end
                    
                    % Print decision info
                    if isfield(packet, 'decision') && ~isempty(packet.decision)
                        d = packet.decision;
                        fprintf('Decision: %s | Rule: %s | Reason: %s\n', ...
                                d.longitudinal, d.rule_id, d.reason);
                    end
                    
                    % Print traffic light info
                    if isfield(packet, 'traffic_light') && ~isempty(packet.traffic_light)
                        tl = packet.traffic_light;
                        fprintf('Traffic Light State: %s\n', tl.state);
                    end
                    
                catch ME
                    fprintf('Failed to parse JSON: %s\n', ME.message);
                end
            end
        end
        % Pause briefly to prevent freezing
        pause(0.01);
    end
catch ME
    % Expected to exit via Ctrl+C
    disp('Stopping UDP Receiver...');
    clear u;
end
