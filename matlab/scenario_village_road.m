% scenario_village_road.m
% ADAS Vision — Driving Scenario 1: Narrow Village Road
%
% Creates a narrow winding village road with:
%   - A pedestrian crossing mid-road ahead
%   - An oncoming motorcycle
%   - A parked pushcart on the left shoulder
%
% Runs a standalone simulation (no UDP bridge required) and shows a
% live animated plot of the ego vehicle and all road actors.
%
% Usage:
%   >> scenario_village_road

fprintf('=== Scenario 1: Village Road ===\n');

% ---------------------------------------------------------------------------
% Road geometry — narrow winding road (3.5 m lane width)
% ---------------------------------------------------------------------------
road_width_m = 7.0;       % two lanes, 3.5 m each
road_len_m   = 200.0;

% Centreline waypoints (winding village road)
cx = [0, 20, 45, 70, 100, 130, 160, 190, 200];
cy = [0,  4,  2, -3,   0,   5,   2,  -1,   0];

% Interpolate for smooth road
t_road = linspace(0, 1, 500);
road_cx = interp1(linspace(0,1,numel(cx)), cx, t_road, 'pchip');
road_cy = interp1(linspace(0,1,numel(cy)), cy, t_road, 'pchip');

% Road edges
dx = gradient(road_cx);
dy = gradient(road_cy);
norms = sqrt(dx.^2 + dy.^2);
nx = -dy ./ (norms + 1e-9);
ny =  dx ./ (norms + 1e-9);

left_x  = road_cx + nx * (road_width_m/2);
left_y  = road_cy + ny * (road_width_m/2);
right_x = road_cx - nx * (road_width_m/2);
right_y = road_cy - ny * (road_width_m/2);

% ---------------------------------------------------------------------------
% Actors
% ---------------------------------------------------------------------------
% Ego vehicle (starts at origin, heading east)
ego.x = 0; ego.y = 0; ego.heading = 0; ego.speed = 8.33;  % ~30 km/h

% Actor 1: Pedestrian crossing at x=80 m
ped.x = 80; ped.y = 0; ped.speed = 1.2;  % walking across
ped.heading = pi/2;   % crossing perpendicular

% Actor 2: Oncoming motorcycle at x=160 m, heading west (-x)
moto.x = 160; moto.y = 1.5; moto.speed = 11.1; moto.heading = pi;

% Actor 3: Parked pushcart at x=55 m, left shoulder
cart.x = 55; cart.y = 3.0;  % stationary

% ---------------------------------------------------------------------------
% Simulation parameters
% ---------------------------------------------------------------------------
dt     = 0.033;
t_end  = 25.0;
goal   = [200, 0];
tol    = 4.0;

% Pure-pursuit (simplified)
pp_la  = 8.0;
wb     = 2.70;
max_st = deg2rad(35);

% ---------------------------------------------------------------------------
% Figure setup
% ---------------------------------------------------------------------------
fig = figure('Name','Scenario 1 — Village Road','NumberTitle','off',...
             'Color',[0.07 0.08 0.07],'Position',[80 80 1000 600]);
ax  = axes('Parent',fig,'Color',[0.12 0.14 0.10],...
           'XColor','w','YColor','w','GridColor',[0.25 0.3 0.2],...
           'GridAlpha',0.5,'Box','on');
hold(ax,'on'); grid(ax,'on');
xlabel(ax,'X (m)','Color','w'); ylabel(ax,'Y (m)','Color','w');
title(ax,'Scenario 1 — Village Road  |  Initialising...','Color',[0.6 1 0.6],'FontSize',12);

% Road fill
fill(ax,[left_x, fliplr(right_x)],[left_y, fliplr(right_y)],...
     [0.25 0.28 0.22],'EdgeColor','none','FaceAlpha',0.6);

% Road edges (white lines)
plot(ax, left_x,  left_y,  '-', 'Color',[1 1 1],   'LineWidth',1.5);
plot(ax, right_x, right_y, '-', 'Color',[1 1 1],   'LineWidth',1.5);
plot(ax, road_cx, road_cy, '--','Color',[0.8 0.7 0],'LineWidth',0.8,'LineStyle','--');

% Goal marker
plot(ax, goal(1), goal(2), 'p','MarkerSize',18,...
     'MarkerFaceColor',[1 0.85 0],'MarkerEdgeColor','w','LineWidth',1.5);
text(ax, goal(1)+2, goal(2)+3, 'GOAL','Color',[1 0.85 0],'FontSize',10);

% Static actors
plot(ax, cart.x, cart.y, 's','MarkerSize',14,...
     'MarkerFaceColor',[0.6 0.4 0.1],'MarkerEdgeColor','w','LineWidth',1.5);
text(ax, cart.x+1, cart.y+2, 'Pushcart','Color',[0.8 0.6 0.2],'FontSize',9);

% Pedestrian crossing zone
xp = [ped.x-1, ped.x+1, ped.x+1, ped.x-1];
yp = [-road_width_m/2, -road_width_m/2, road_width_m/2, road_width_m/2];
fill(ax, xp, yp, [0.9 0.9 0.9],'FaceAlpha',0.3,'EdgeColor','none');
text(ax, ped.x-1, road_width_m/2+1, 'PEDESTRIAN CROSSING','Color',[0.9 0.9 0.7],'FontSize',8);

% Dynamic actor handles
h_ego  = plot(ax, ego.x,  ego.y,  'o','MarkerSize',14,'MarkerFaceColor',[0 0.85 0.3],...
              'MarkerEdgeColor','w','LineWidth',2);
h_ped  = plot(ax, ped.x,  ped.y,  '^','MarkerSize',12,'MarkerFaceColor',[0.9 0.3 0.9],...
              'MarkerEdgeColor','w','LineWidth',1.5);
h_moto = plot(ax, moto.x, moto.y, 'd','MarkerSize',12,'MarkerFaceColor',[0.9 0.5 0.1],...
              'MarkerEdgeColor','w','LineWidth',1.5);
h_traj = plot(ax, ego.x, ego.y, '--','Color',[0.4 0.8 0.4],'LineWidth',1);

traj_x = ego.x; traj_y = ego.y;

% Status text
h_status = text(ax, 0.02, 0.97, '','Units','normalized','Color','w',...
                'FontSize',10,'VerticalAlignment','top','FontWeight','bold');

drawnow;

% ---------------------------------------------------------------------------
% Simulation loop
% ---------------------------------------------------------------------------
t = 0;
arrived = false;
event_log = {};

while t < t_end && ishandle(fig)

    dist_to_goal = norm(goal - [ego.x, ego.y]);
    if dist_to_goal < tol
        arrived = true; break;
    end

    % ── Pure-pursuit steering ────────────────────────────────────────────
    % Find lookahead point on road centreline
    dists = sqrt((road_cx - ego.x).^2 + (road_cy - ego.y).^2);
    [~, nearest] = min(dists);
    la_idx = nearest;
    la_acc = 0;
    while la_acc < pp_la && la_idx < numel(road_cx)-1
        la_acc = la_acc + norm([road_cx(la_idx+1)-road_cx(la_idx),...
                                road_cy(la_idx+1)-road_cy(la_idx)]);
        la_idx = la_idx + 1;
    end
    wp = [road_cx(la_idx), road_cy(la_idx)];
    alpha = atan2(wp(2)-ego.y, wp(1)-ego.x) - ego.heading;
    alpha = atan2(sin(alpha), cos(alpha));
    steer = atan2(2*wb*sin(alpha), pp_la);
    steer = max(-max_st, min(max_st, steer));

    % ── Decision (simple rule-based for scenario) ────────────────────────
    dist_ped  = norm([ped.x - ego.x, ped.y - ego.y]);
    dist_moto = norm([moto.x - ego.x, moto.y - ego.y]);
    dist_cart = norm([cart.x - ego.x, cart.y - ego.y]);

    if dist_ped < 15 && abs(ped.y - ego.y) < 3.5
        decision = 'SLOW';  speed_target = 2.5;
        event = sprintf('t=%.1fs: SLOW — pedestrian %.1fm ahead', t, dist_ped);
    elseif dist_moto < 25 && abs(moto.y - ego.y) < 5
        decision = 'CAUTION'; speed_target = 5.0;
        event = sprintf('t=%.1fs: CAUTION — oncoming motorcycle %.1fm', t, dist_moto);
    elseif dist_cart < 10 && abs(cart.y - ego.y) < 4
        decision = 'CAUTION'; speed_target = 4.0;
        event = '';
    else
        decision = 'PROCEED'; speed_target = 8.33;
        event = '';
    end
    if ~isempty(event), event_log{end+1} = event; fprintf('  %s\n', event); end

    % ── Speed control (simple P) ─────────────────────────────────────────
    err = speed_target - ego.speed;
    ego.speed = max(0, min(15, ego.speed + 0.8*err*dt));

    % ── Ego integration ──────────────────────────────────────────────────
    ego.x = ego.x + ego.speed * cos(ego.heading) * dt;
    ego.y = ego.y + ego.speed * sin(ego.heading) * dt;
    ego.heading = ego.heading + (ego.speed / wb) * tan(steer) * dt;

    % ── Actor updates ────────────────────────────────────────────────────
    ped.x  = ped.x  + ped.speed  * cos(ped.heading)  * dt;
    ped.y  = ped.y  + ped.speed  * sin(ped.heading)  * dt;
    moto.x = moto.x + moto.speed * cos(moto.heading) * dt;
    moto.y = moto.y + moto.speed * sin(moto.heading) * dt;

    traj_x(end+1) = ego.x;
    traj_y(end+1) = ego.y;

    % ── Plot update ──────────────────────────────────────────────────────
    set(h_ego,  'XData', ego.x,  'YData', ego.y);
    set(h_ped,  'XData', ped.x,  'YData', ped.y);
    set(h_moto, 'XData', moto.x, 'YData', moto.y);
    set(h_traj, 'XData', traj_x, 'YData', traj_y);

    col = scenario_color(decision);
    set(h_status, 'String', sprintf('t=%.1f s  |  Decision: %s  |  Speed: %.1f km/h  |  Dist: %.1f m',...
                  t, decision, ego.speed*3.6, dist_to_goal), 'Color', col);
    title(ax, sprintf('Scenario 1 — Village Road  |  %s', decision),...
          'Color', col, 'FontSize', 12);

    drawnow limitrate;
    t = t + dt;
end

% ---------------------------------------------------------------------------
% Summary
% ---------------------------------------------------------------------------
fprintf('\n=== Scenario 1 Complete ===\n');
if arrived
    fprintf('  EGO reached goal in %.1f s\n', t);
else
    fprintf('  Scenario ended at t=%.1f s (goal not reached)\n', t);
end
fprintf('  Events logged: %d\n', numel(event_log));
for k = 1:numel(event_log)
    fprintf('    %s\n', event_log{k});
end

% Stash data for collect_metrics
scenario_result.name    = 'village_road';
scenario_result.t_total = t;
scenario_result.arrived = arrived;
scenario_result.traj_x  = traj_x;
scenario_result.traj_y  = traj_y;
assignin('base','scenario_result', scenario_result);


% ---------------------------------------------------------------------------
function c = scenario_color(decision)
    switch decision
        case 'PROCEED',  c = [0.2 0.9 0.2];
        case 'CAUTION',  c = [1.0 0.8 0.0];
        case 'SLOW',     c = [1.0 0.5 0.0];
        case 'BRAKE',    c = [1.0 0.2 0.0];
        otherwise,       c = [0.8 0.8 0.8];
    end
end
