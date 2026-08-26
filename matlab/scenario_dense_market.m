% scenario_dense_market.m
% ADAS Vision — Driving Scenario 4: Congested Indian Market Street
%
% Tests autonomous navigation in an ultra-dense, unstructured urban market
% with high density of Vulnerable Road Users (VRUs), street vendors, and narrow navigable space.
%
% Elements:
%   - Narrow street (4.0m drivable space) bounded by market stalls & parked pushcarts
%   - Multiple pedestrians crossing without formal crosswalks
%   - Street vendor pushcart (thela) partially occupying road
%   - Oncoming scooter filtering between pedestrians
%   - Stray animal crossing slowly
%
% Usage:
%   >> scenario_dense_market

fprintf('=== Scenario 4: Dense Market Street ===\n');

% ---------------------------------------------------------------------------
% Geometry & Setup
% ---------------------------------------------------------------------------
street_len = 120.0;
sw         = 3.5;   % half-width of street (m)

% Actors
ego.x       = 0.0;
ego.y       = 0.0;
ego.heading = 0.0;
ego.speed   = 3.33;  % 12 km/h (creeping speed for market)
ego.target_speed = 3.88; % ~14 km/h

% Static Pushcart (Thela) parked at x=35m, y=-1.2m
thela.x     = 35.0;
thela.y     = -1.4;
thela.l     = 2.2;
thela.w     = 1.2;

% Pedestrian 1: wandering across from left to right at x=20m
ped1.x = 22.0; ped1.y = 2.5; ped1.vx = 0.2; ped1.vy = -0.7;

% Pedestrian 2: child/shopper darting from right behind pushcart at x=40m
ped2.x = 42.0; ped2.y = -2.8; ped2.vx = -0.3; ped2.vy = 0.9;

% Pedestrian 3: walking along road edge at x=65m
ped3.x = 65.0; ped3.y = 1.6; ped3.vx = 0.8; ped3.vy = 0.05;

% Oncoming Scooter: weaving through at x=90m
scooter.x = 90.0; scooter.y = 0.6; scooter.speed = 4.5; scooter.heading = pi;

% Simulation params
dt     = 0.033;
t_end  = 30.0;
goal   = [street_len - 10, 0];
tol    = 4.0;
wb     = 2.70;

% ---------------------------------------------------------------------------
% Graphics Setup
% ---------------------------------------------------------------------------
fig = figure('Name','Scenario 4 — Dense Indian Market Street','NumberTitle','off',...
             'Color',[0.06 0.06 0.08],'Position',[80 100 950 500]);
ax  = axes('Parent',fig,'Color',[0.14 0.13 0.15],...
           'XColor','w','YColor','w','GridColor',[0.3 0.28 0.35],'GridAlpha',0.4);
hold(ax,'on'); grid(ax,'on');
axis(ax,'equal');
xlim(ax, [-5, street_len]);
ylim(ax, [-sw-3, sw+3]);
xlabel(ax, 'Street Distance (m)', 'Color','w');
ylabel(ax, 'Lateral Offset (m)', 'Color','w');

% Street surface
fill(ax, [0 street_len street_len 0], [-sw -sw sw sw], [0.22 0.20 0.21], 'EdgeColor','none');

% Market Stalls (left side colorful awnings)
for sx = 0:15:street_len-10
    col_aw = [0.8 + 0.15*rand(), 0.3 + 0.4*rand(), 0.2];
    fill(ax, [sx sx+12 sx+12 sx], [sw sw sw+2.5 sw+2.5], col_aw, 'EdgeColor','none');
    text(ax, sx+2, sw+1.2, 'STALL', 'Color','k','FontSize',7,'FontWeight','bold');
end

% Market Stalls (right side)
for sx = 5:16:street_len-10
    col_aw = [0.2, 0.5 + 0.3*rand(), 0.7 + 0.2*rand()];
    fill(ax, [sx sx+13 sx+13 sx], [-sw-2.5 -sw-2.5 -sw -sw], col_aw, 'EdgeColor','none');
    text(ax, sx+2, -sw-1.2, 'SHOP', 'Color','w','FontSize',7);
end

% Pushcart (Thela) icon
fill(ax, [thela.x-1 thela.x+1 thela.x+1 thela.x-1], ...
         [thela.y-0.6 thela.y-0.6 thela.y+0.6 thela.y+0.6], [0.6 0.4 0.2], 'EdgeColor','w');
text(ax, thela.x-2, thela.y, '🛒 PUSHCART', 'Color',[1 0.8 0.4],'FontSize',8);

% Goal Marker
plot(ax, goal(1), goal(2), 'p', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.85 0], 'MarkerEdgeColor', 'w');
text(ax, goal(1)+2, goal(2)+1, 'EXIT', 'Color', [1 0.85 0], 'FontSize', 9);

% Actor Handles
h_ego  = plot(ax, ego.x, ego.y, 'o', 'MarkerSize', 15, 'MarkerFaceColor', [0 0.85 0.3], 'MarkerEdgeColor','w', 'LineWidth',2);
h_ped1 = plot(ax, ped1.x, ped1.y, 'o', 'MarkerSize', 9, 'MarkerFaceColor', [1 0.4 0.4], 'MarkerEdgeColor','w');
h_ped2 = plot(ax, ped2.x, ped2.y, 'o', 'MarkerSize', 9, 'MarkerFaceColor', [1 0.8 0.2], 'MarkerEdgeColor','w');
h_ped3 = plot(ax, ped3.x, ped3.y, 'o', 'MarkerSize', 9, 'MarkerFaceColor', [0.8 0.4 1.0], 'MarkerEdgeColor','w');
h_scoot= plot(ax, scooter.x, scooter.y, '^', 'MarkerSize', 12, 'MarkerFaceColor', [0.2 0.7 1.0], 'MarkerEdgeColor','w');
h_traj = plot(ax, ego.x, ego.y, '--', 'Color', [0.4 0.9 0.4], 'LineWidth', 1.2);

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

    % 1. Update VRUs & obstacles
    ped1.x = ped1.x + ped1.vx * dt;
    ped1.y = ped1.y + ped1.vy * dt;

    % Darting pedestrian starts moving at t=3.0s
    if t > 3.0 && ped2.y < 1.0
        ped2.x = ped2.x + ped2.vx * dt;
        ped2.y = ped2.y + ped2.vy * dt;
    end

    ped3.x = ped3.x + ped3.vx * dt;
    ped3.y = ped3.y + ped3.vy * dt;

    scooter.x = scooter.x - scooter.speed * dt;

    % 2. Perception & Closest Obstacle Analysis
    peds = [ped1.x, ped1.y; ped2.x, ped2.y; ped3.x, ped3.y; thela.x, thela.y; scooter.x, scooter.y];
    labels = {'Pedestrian (crossing)', 'Child (darting)', 'Shopper (edge)', 'Parked Pushcart', 'Oncoming Scooter'};

    min_d = Inf;
    crit_label = '';
    steer = 0;

    for k = 1:size(peds, 1)
        dx = peds(k, 1) - ego.x;
        dy = peds(k, 2) - ego.y;
        if dx > -1.0 && dx < 25.0
            dist = norm([dx, dy]);
            if dist < min_d
                min_d = dist;
                crit_label = labels{k};
                % Nudge steering away from lateral obstacles
                if abs(dy) < 1.6 && dx > 0
                    steer = -sign(dy) * deg2rad(7.0); % steer opposite side
                end
            end
        end
    end

    % 3. Decision Rule Hierarchy with VRU Safety Floor
    decision = 'PROCEED';
    v_cmd = ego.target_speed;

    if min_d < 4.5
        decision = 'EMERGENCY_STOP';
        v_cmd = 0.0;
        event_log{end+1} = sprintf('t=%.1fs: E-STOP — %s at %.1fm', t, crit_label, min_d); %#ok<AGROW>
    elseif min_d < 9.0
        decision = 'BRAKE';
        v_cmd = 0.8;
        event_log{end+1} = sprintf('t=%.1fs: BRAKE — %s in path (%.1fm)', t, crit_label, min_d); %#ok<AGROW>
    elseif min_d < 16.0
        decision = 'CAUTION';
        v_cmd = ego.target_speed * 0.6;
    end

    % 4. Ego Kinematics
    accel = 2.0 * (v_cmd - ego.speed);
    ego.speed = max(0, ego.speed + accel * dt);
    ego.heading = ego.heading + (ego.speed / wb) * tan(steer) * dt;
    ego.x = ego.x + ego.speed * cos(ego.heading) * dt;
    ego.y = ego.y + ego.speed * sin(ego.heading) * dt;

    % Clamp within drivable space between stalls
    ego.y = max(-sw + 0.8, min(sw - 0.8, ego.y));

    traj_x(end+1) = ego.x; %#ok<AGROW>
    traj_y(end+1) = ego.y; %#ok<AGROW>

    % 5. Graphics Refresh
    set(h_ego,   'XData', ego.x,     'YData', ego.y);
    set(h_ped1,  'XData', ped1.x,    'YData', ped1.y);
    set(h_ped2,  'XData', ped2.x,    'YData', ped2.y);
    set(h_ped3,  'XData', ped3.x,    'YData', ped3.y);
    set(h_scoot, 'XData', scooter.x, 'YData', scooter.y);
    set(h_traj,  'XData', traj_x,    'YData', traj_y);

    xlim(ax, [ego.x - 10, ego.x + 50]);

    col = scenario_color(decision);
    set(h_status, 'String', sprintf('t=%.1fs | %s | Speed: %.1f km/h | Hazard: %s (%.1fm)', ...
        t, decision, ego.speed*3.6, crit_label, min_d), 'Color', col);
    title(ax, sprintf('Scenario 4 — Dense Indian Market | %s', decision), 'Color', col, 'FontSize', 12);

    drawnow limitrate;
    t = t + dt;
end

fprintf('\n=== Scenario 4 Complete ===\n');
if arrived
    fprintf('  Market corridor cleared in %.1f s\n', t);
else
    fprintf('  Simulation finished at t=%.1f s\n', t);
end
fprintf('  Events logged: %d\n', numel(event_log));

scenario_result.name    = 'dense_market';
scenario_result.t_total = t;
scenario_result.arrived = arrived;
scenario_result.traj_x  = traj_x;
scenario_result.traj_y  = traj_y;
assignin('base', 'scenario_result', scenario_result);

function c = scenario_color(d)
    switch d
        case 'PROCEED',        c = [0.2 0.9 0.2];
        case 'CAUTION',        c = [1.0 0.8 0.0];
        case 'SLOW',           c = [1.0 0.5 0.0];
        case 'BRAKE',          c = [1.0 0.2 0.0];
        case 'EMERGENCY_STOP', c = [1.0 0.0 0.0];
        otherwise,             c = [0.8 0.8 0.8];
    end
end
