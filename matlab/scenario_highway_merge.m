% scenario_highway_merge.m
% ADAS Vision — Driving Scenario 3: Indian Highway Merge
%
% Tests vehicle navigation during highway merging with high speed differentials
% and unpredictable commercial vehicle behavior common on Indian highways.
%
% Elements:
%   - 2-lane dual carriageway (70 km/h design speed) + 1 on-ramp merging lane
%   - Ego vehicle: entering from the on-ramp or cruising in left lane (speed: 50 km/h)
%   - Slow overloaded commercial truck (Tata 1613 style): merging at 25 km/h without indicator
%   - Fast SUV in overtaking lane (85 km/h)
%   - Two-wheeler riding on the shoulder against traffic
%
% Usage:
%   >> scenario_highway_merge

fprintf('=== Scenario 3: Highway Merge ===\n');

% ---------------------------------------------------------------------------
% Road geometry
% ---------------------------------------------------------------------------
road_len_m = 300.0;
lw         = 3.75;  % standard Indian highway lane width (m)
merge_start = 50.0;
merge_end   = 180.0;

% ---------------------------------------------------------------------------
% Actors
% ---------------------------------------------------------------------------
% Ego vehicle (starts on main highway left lane, accelerating to cruising speed)
ego.x       = 0.0;
ego.y       = lw / 2;
ego.heading = 0.0;
ego.speed   = 13.88; % ~50 km/h
ego.target_speed = 16.66; % ~60 km/h

% Slow overloaded commercial truck in merge lane (lane y = -lw/2 - 1.5)
truck.x       = 40.0;
truck.y       = -lw - 1.0;
truck.heading = 0.0;
truck.speed   = 6.94; % 25 km/h
truck.length  = 9.0;
truck.width   = 2.5;
truck.merging = false;

% Fast overtaking SUV in right lane (y = 3*lw/2)
suv.x       = -40.0;
suv.y       = lw * 1.5;
suv.heading = 0.0;
suv.speed   = 23.6; % 85 km/h
suv.length  = 4.8;
suv.width   = 1.9;

% Errant two-wheeler on left shoulder
bike.x       = 140.0;
bike.y       = -lw - 2.5;
bike.heading = pi; % driving against traffic
bike.speed   = 4.0;

% ---------------------------------------------------------------------------
% Simulation settings
% ---------------------------------------------------------------------------
dt     = 0.033;
t_end  = 22.0;
goal   = [road_len_m - 20, lw/2];
tol    = 5.0;
wb     = 2.70;
max_st = deg2rad(35);

% ---------------------------------------------------------------------------
% Visualization Setup
% ---------------------------------------------------------------------------
fig = figure('Name','Scenario 3 — Indian Highway Merge','NumberTitle','off',...
             'Color',[0.06 0.06 0.10],'Position',[60 120 1000 480]);
ax  = axes('Parent',fig,'Color',[0.12 0.12 0.16],...
           'XColor','w','YColor','w','GridColor',[0.25 0.25 0.35],'GridAlpha',0.4);
hold(ax,'on'); grid(ax,'on');
axis(ax,'equal');
xlim(ax, [-10, road_len_m]);
ylim(ax, [-lw*3, lw*3]);
xlabel(ax, 'Longitudinal Position (m)', 'Color','w');
ylabel(ax, 'Lateral Position (m)', 'Color','w');

% Main Highway Lanes (Dark asphalt)
fill(ax, [0 road_len_m road_len_m 0], [0 0 2*lw 2*lw], [0.18 0.18 0.22], 'EdgeColor','none');

% Merge Ramp
fill(ax, [merge_start merge_end merge_end merge_start], ...
     [-lw*2 -lw*2 0 0], [0.18 0.18 0.22], 'EdgeColor','none');

% Road markings
plot(ax, [0 road_len_m], [2*lw 2*lw], 'w-', 'LineWidth', 2); % Outer right
plot(ax, [0 road_len_m], [0 0], 'w-', 'LineWidth', 1.5);    % Edge line
plot(ax, [0 road_len_m], [lw lw], 'w--', 'LineWidth', 1.0); % Lane divider

% Merge entry taper & solid line
plot(ax, [merge_start merge_end], [-lw*2 -lw*2], 'w-', 'LineWidth', 1.5);
plot(ax, [merge_start merge_end-30], [0 0], 'w--', 'LineWidth', 1.2);

% Shoulder grass
fill(ax, [0 road_len_m road_len_m 0], [-lw*3 -lw*3 -lw*2 -lw*2], [0.12 0.24 0.10], 'EdgeColor','none');
fill(ax, [0 road_len_m road_len_m 0], [2*lw 2*lw 2*lw+5 2*lw+5], [0.12 0.24 0.10], 'EdgeColor','none');

% Goal Marker
plot(ax, goal(1), goal(2), 'p', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.85 0], 'MarkerEdgeColor', 'w');
text(ax, goal(1)+2, goal(2)+1, 'GOAL', 'Color', [1 0.85 0], 'FontSize', 9);

% Actor Graphics Handles
h_ego   = plot(ax, ego.x, ego.y, 'o', 'MarkerSize', 14, 'MarkerFaceColor', [0 0.85 0.3], 'MarkerEdgeColor','w', 'LineWidth', 2);
h_truck = plot(ax, truck.x, truck.y, 's', 'MarkerSize', 20, 'MarkerFaceColor', [0.85 0.45 0.1], 'MarkerEdgeColor','w', 'LineWidth', 2);
h_suv   = plot(ax, suv.x, suv.y, 'd', 'MarkerSize', 14, 'MarkerFaceColor', [0.3 0.6 1.0], 'MarkerEdgeColor','w');
h_bike  = plot(ax, bike.x, bike.y, '^', 'MarkerSize', 10, 'MarkerFaceColor', [1.0 0.2 0.2], 'MarkerEdgeColor','w');
h_traj  = plot(ax, ego.x, ego.y, '--', 'Color', [0.4 0.9 0.4], 'LineWidth', 1.2);

text(ax, truck.x-5, truck.y-2, 'TRUCK (Heavy)', 'Color', [1 0.7 0.2], 'FontSize', 8);
text(ax, suv.x-5, suv.y+2, 'SUV (Fast)', 'Color', [0.5 0.8 1.0], 'FontSize', 8);

h_status = text(ax, 0.02, 0.94, '', 'Units', 'normalized', 'Color', 'w', ...
                'FontSize', 10, 'VerticalAlignment', 'top', 'FontWeight', 'bold');

% ---------------------------------------------------------------------------
% Simulation Loop
% ---------------------------------------------------------------------------
t = 0;
arrived = false;
traj_x = ego.x;
traj_y = ego.y;
event_log = {};

while t < t_end && ishandle(fig)
    dist_to_goal = norm([ego.x, ego.y] - goal);
    if dist_to_goal < tol
        arrived = true;
        break;
    end

    % 1. Update Actors
    % Truck drifts into main left lane abruptly around t=3.5s
    if t > 3.5 && truck.x < 170
        truck.merging = true;
        truck.y = min(lw/2, truck.y + 0.45 * dt); % lateral cut-in
    end
    truck.x = truck.x + truck.speed * dt;

    suv.x  = suv.x + suv.speed * dt;
    bike.x = bike.x - bike.speed * dt;

    % 2. Perception & Decision
    d_truck = norm([truck.x - ego.x, truck.y - ego.y]);
    in_path_truck = (truck.x > ego.x) && (abs(truck.y - ego.y) < 1.8);

    decision = 'PROCEED';
    v_cmd = ego.target_speed;
    steer = 0;

    if in_path_truck && d_truck < 18.0
        decision = 'BRAKE';
        v_cmd = truck.speed * 0.7;
        event_log{end+1} = sprintf('t=%.1fs: BRAKE — Truck cut-in gap: %.1fm', t, d_truck); %#ok<AGROW>
        % If overtaking lane is clear of SUV, initiate courteous lane change
        if (ego.x - suv.x > 25.0) || (suv.x - ego.x > 35.0)
            steer = deg2rad(6.0); % change to right lane to pass
        end
    elseif in_path_truck && d_truck < 35.0
        decision = 'CAUTION';
        v_cmd = truck.speed * 1.1;
        event_log{end+1} = sprintf('t=%.1fs: CAUTION — Merging truck ahead: %.1fm', t, d_truck); %#ok<AGROW>
    elseif truck.merging && d_truck < 45.0
        decision = 'SLOW';
        v_cmd = ego.target_speed * 0.85;
    end

    % 3. Ego Dynamics
    accel = 1.8 * (v_cmd - ego.speed);
    ego.speed = max(0, ego.speed + accel * dt);
    ego.heading = ego.heading + (ego.speed / wb) * tan(steer) * dt;
    ego.x = ego.x + ego.speed * cos(ego.heading) * dt;
    ego.y = ego.y + ego.speed * sin(ego.heading) * dt;

    % Keep on road boundaries
    ego.y = max(0.5, min(2*lw - 0.5, ego.y));

    traj_x(end+1) = ego.x; %#ok<AGROW>
    traj_y(end+1) = ego.y; %#ok<AGROW>

    % 4. Graphics Update
    set(h_ego,   'XData', ego.x,   'YData', ego.y);
    set(h_truck, 'XData', truck.x, 'YData', truck.y);
    set(h_suv,   'XData', suv.x,   'YData', suv.y);
    set(h_bike,  'XData', bike.x,  'YData', bike.y);
    set(h_traj,  'XData', traj_x,  'YData', traj_y);

    % Camera follow ego
    xlim(ax, [ego.x - 30, ego.x + 120]);

    col = scenario_color(decision);
    set(h_status, 'String', sprintf('t=%.1fs | Decision: %s | Ego Speed: %.1f km/h | Truck Gap: %.1fm', ...
        t, decision, ego.speed*3.6, d_truck), 'Color', col);
    title(ax, sprintf('Scenario 3 — Highway Merge | %s', decision), 'Color', col, 'FontSize', 12);

    drawnow limitrate;
    t = t + dt;
end

fprintf('\n=== Scenario 3 Complete ===\n');
if arrived
    fprintf('  Goal reached in %.1f s\n', t);
else
    fprintf('  Simulation ended at t=%.1f s\n', t);
end
fprintf('  Events logged: %d\n', numel(event_log));

scenario_result.name    = 'highway_merge';
scenario_result.t_total = t;
scenario_result.arrived = arrived;
scenario_result.traj_x  = traj_x;
scenario_result.traj_y  = traj_y;
assignin('base', 'scenario_result', scenario_result);

function c = scenario_color(d)
    switch d
        case 'PROCEED',  c = [0.2 0.9 0.2];
        case 'CAUTION',  c = [1.0 0.8 0.0];
        case 'SLOW',     c = [1.0 0.5 0.0];
        case 'BRAKE',    c = [1.0 0.2 0.0];
        otherwise,       c = [0.8 0.8 0.8];
    end
end
