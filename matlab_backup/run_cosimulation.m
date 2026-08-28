function run_cosimulation(goal_x, goal_y)
% RUN_COSIMULATION Orchestrates the full co-simulation pipeline
% Tests the Python perception integration without Simulink blocks.

    if nargin < 2
        goal_x = 100;
        goal_y = 0;
    end

    % 1. Load parameters
    setup_vehicle_model;
    
    % Prepare UDP objects
    % Port 5005 to receive from Python
    udp_rx = udpport("LocalPort", 5005, "Timeout", sim.dt);
    
    % Port 5006 to send to Python
    udp_tx = udpport("LocalPort", 5006);
    python_ip = "127.0.0.1";
    python_port = 5007; % Assuming Python listens on 5007
    
    % Initialize states
    ego_x = ego.x0;
    ego_y = ego.y0;
    ego_yaw = ego.yaw0;
    ego_v = ego.v0;
    
    % Metrics storage
    log_time = [];
    log_x = [];
    log_y = [];
    log_v = [];
    log_decision = [];
    log_min_dist = [];
    
    figure('Name', 'ADAS Co-Simulation', 'NumberTitle', 'off');
    
    disp('Starting Co-Simulation... Press Ctrl+C to stop.');
    
    try
        % 4. Run simulation loop at 20 Hz
        for t = 0 : sim.dt : sim.duration
            tic;
            
            % a. Receive latest UDP packet
            if udp_rx.NumBytesAvailable > 0
                udp_bytes = read(udp_rx, udp_rx.NumBytesAvailable, "uint8");
            else
                udp_bytes = zeros(1,0,'uint8');
            end
            
            % b. Parse it
            [num_tracks, track_ids, track_x, track_y, track_dist, ...
             track_ttc, track_in_path, track_cs, road_cx, road_conf, ...
             road_off, dec_level, dec_brake, dec_throttle, degraded, fps] = parse_udp_json(udp_bytes);
            
            % c. Build occupancy grid (mockup)
            % build_occupancy_grid(track_x, track_y, num_tracks);
            
            % d. Run decision logic (mockup)
            decision = dec_level;
            
            % e. Compute Pure Pursuit steering angle
            dx = goal_x - ego_x;
            dy = goal_y - ego_y;
            target_yaw = atan2(dy, dx);
            yaw_error = target_yaw - ego_yaw;
            yaw_error = atan2(sin(yaw_error), cos(yaw_error)); % wrap to pi
            steer = max(-veh.max_steer_rad, min(veh.max_steer_rad, yaw_error));
            
            % f. Update bicycle model kinematics
            accel = dec_throttle * veh.max_accel - dec_brake * abs(veh.max_decel);
            if accel == 0 && decision == 0
                accel = 1.0; % Default cruise acceleration if no inputs
            end
            
            ego_x = ego_x + ego_v * cos(ego_yaw) * sim.dt;
            ego_y = ego_y + ego_v * sin(ego_yaw) * sim.dt;
            ego_yaw = ego_yaw + (ego_v / veh.wheelbase) * tan(steer) * sim.dt;
            ego_v = ego_v + accel * sim.dt;
            ego_v = max(0, min(veh.max_speed_mps, ego_v)); % clamp speed
            
            % g. Send ego pose back via UDP
            pose_struct = struct('x', ego_x, 'y', ego_y, 'yaw', ego_yaw, 'v', ego_v);
            pose_json = jsonencode(pose_struct);
            write(udp_tx, uint8(pose_json), python_ip, python_port);
            
            % h. Log metrics
            min_d = inf;
            if num_tracks > 0
                min_d = min(track_dist(1:num_tracks));
            end
            
            log_time = [log_time; t];
            log_x = [log_x; ego_x];
            log_y = [log_y; ego_y];
            log_v = [log_v; ego_v];
            log_decision = [log_decision; decision];
            log_min_dist = [log_min_dist; min_d];
            
            % i. Plot live
            clf;
            hold on;
            plot(log_x, log_y, 'b-', 'LineWidth', 2);
            plot(ego_x, ego_y, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
            plot(goal_x, goal_y, 'g*', 'MarkerSize', 10);
            
            if num_tracks > 0
                for i = 1:num_tracks
                    tx_world = ego_x + track_x(i)*cos(ego_yaw) - track_y(i)*sin(ego_yaw);
                    ty_world = ego_y + track_x(i)*sin(ego_yaw) + track_y(i)*cos(ego_yaw);
                    plot(tx_world, ty_world, 'ro', 'MarkerFaceColor', 'r');
                end
            end
            
            axis equal;
            grid on;
            title(sprintf('Time: %.1fs | Speed: %.1f m/s | Dist to Goal: %.1fm', t, ego_v, norm([dx, dy])));
            drawnow;
            
            % Wait to maintain 20 Hz
            elapsed = toc;
            if elapsed < sim.dt
                pause(sim.dt - elapsed);
            end
            
            if norm([dx, dy]) < 2.0
                disp('Goal reached!');
                break;
            end
        end
        
    catch ME
        disp('Simulation stopped or errored.');
        disp(ME.message);
    end
    
    % Cleanup UDP
    clear udp_rx udp_tx;
    
    % 5. Compute summary metrics
    total_dist = sum(sqrt(diff(log_x).^2 + diff(log_y).^2));
    num_collisions = sum(log_min_dist < 0.5);
    avg_latency = NaN; % Requires Python timestamps
    goal_reached = norm([goal_x - ego_x, goal_y - ego_y]) < 2.0;
    
    fprintf('\n=== Co-Simulation Summary ===\n');
    fprintf('Total Distance Traveled: %.2f m\n', total_dist);
    fprintf('Number of Collisions (min clearance < 0.5m): %d\n', num_collisions);
    fprintf('Average Replanning Latency: N/A\n');
    fprintf('Scenario Completion (Reached Goal): %s\n', mat2str(goal_reached));

end
