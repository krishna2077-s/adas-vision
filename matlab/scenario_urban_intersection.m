% scenario_urban_intersection.m
% ADAS Vision — Driving Scenario 2: Urban Intersection
%
% Creates a 4-way junction with:
%   - Cross-traffic cars from the left and right
%   - A rickshaw entering from the right
%   - Pedestrians crossing at the zebra crossing
%   - A cyclist on the far side
%
% Usage:
%   >> scenario_urban_intersection

fprintf('=== Scenario 2: Urban Intersection ===\n');

% ---------------------------------------------------------------------------
% Road geometry — 4-way intersection (6 m lane width each direction)
% ---------------------------------------------------------------------------
lw = 3.0;   % half lane width (m)
rl = 80.0;  % road arm length (m)

% ---------------------------------------------------------------------------
% Actors
% ---------------------------------------------------------------------------
% Ego: approaching from south, heading north (+y)
ego.x = 0; ego.y = -rl; ego.heading = pi/2; ego.speed = 8.33;

% Cross-traffic car 1: from west, heading east (+x)
car1.x = -rl; car1.y = lw/2; car1.speed = 11.1; car1.heading = 0;

% Cross-traffic car 2: from east, heading west (-x)
car2.x = rl; car2.y = -lw/2; car2.speed = 9.7; car2.heading = pi;

% Rickshaw: from right (east), entering intersection
rick.x = rl*0.6; rick.y = lw/2+1.5; rick.speed = 5.5; rick.heading = pi;

% Pedestrian 1: crossing east-west (zebra crossing at y=0)
ped1.x = -lw; ped1.y = 0; ped1.speed = 1.4; ped1.heading = 0;

% Pedestrian 2: also crossing
ped2.x = lw*2; ped2.y = 0; ped2.speed = 1.1; ped2.heading = pi;

% Cyclist: on far (north) side
cyc.x = lw/2; cyc.y = rl*0.3; cyc.speed = 4.2; cyc.heading = pi/2;

% ---------------------------------------------------------------------------
% Simulation parameters
% ---------------------------------------------------------------------------
dt     = 0.033;
t_end  = 30.0;
goal   = [0, rl];      % drive through intersection to north
tol    = 5.0;
wb     = 2.70;
max_st = deg2rad(35);

% ---------------------------------------------------------------------------
% Figure setup
% ---------------------------------------------------------------------------
fig = figure('Name','Scenario 2 — Urban Intersection','NumberTitle','off',...
             'Color',[0.06 0.06 0.10],'Position',[90 90 850 820]);
ax  = axes('Parent',fig,'Color',[0.14 0.14 0.18],...
           'XColor','w','YColor','w','GridColor',[0.25 0.25 0.35],...
           'GridAlpha',0.5,'Box','on');
hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
xlabel(ax,'X (m)','Color','w'); ylabel(ax,'Y (m)','Color','w');
title(ax,'Scenario 2 — Urban Intersection','Color',[0.6 0.8 1],'FontSize',12);
xlim(ax,[-rl-10, rl+10]); ylim(ax,[-rl-10, rl+10]);

% Draw roads
road_color = [0.22 0.22 0.28];
% N-S road
fill(ax,[-lw*2, lw*2, lw*2, -lw*2],[-rl-5, -rl-5, rl+5, rl+5],...
     road_color,'EdgeColor','none');
% E-W road
fill(ax,[-rl-5, rl+5, rl+5, -rl-5],[-lw*2, -lw*2, lw*2, lw*2],...
     road_color,'EdgeColor','none');

% Lane markings
line_color = [0.85 0.85 0.85];
plot(ax, [0 0], [-rl-5 -lw*2], '--','Color',line_color,'LineWidth',1);
plot(ax, [0 0], [ lw*2  rl+5], '--','Color',line_color,'LineWidth',1);
plot(ax, [-rl-5 -lw*2], [0 0], '--','Color',line_color,'LineWidth',1);
plot(ax, [ lw*2  rl+5], [0 0], '--','Color',line_color,'LineWidth',1);

% Zebra crossings (N and S of intersection)
zc_y = [-lw*2-2, lw*2+2];
for zy = zc_y
    for xi = -lw*1.8 : 1.2 : lw*1.8
        fill(ax,[xi, xi+0.8, xi+0.8, xi],[zy, zy, zy+0.8, zy+0.8],...
             [0.9 0.9 0.9],'EdgeColor','none','FaceAlpha',0.5);
    end
end

% Goal
plot(ax, goal(1), goal(2), 'p','MarkerSize',18,...
     'MarkerFaceColor',[1 0.85 0],'MarkerEdgeColor','w');
text(ax, goal(1)+2, goal(2)+3,'GOAL','Color',[1 0.85 0],'FontSize',10);

% Actor plot handles
h_ego  = plot(ax, ego.x, ego.y, 'o','MarkerSize',14,'MarkerFaceColor',[0 0.85 0.3],...
              'MarkerEdgeColor','w','LineWidth',2);
h_car1 = plot(ax, car1.x, car1.y, 's','MarkerSize',13,'MarkerFaceColor',[0.9 0.2 0.2],...
              'MarkerEdgeColor','w','LineWidth',1.5);
h_car2 = plot(ax, car2.x, car2.y, 's','MarkerSize',13,'MarkerFaceColor',[0.9 0.2 0.2],...
              'MarkerEdgeColor','w','LineWidth',1.5);
h_rick = plot(ax, rick.x, rick.y, 'd','MarkerSize',12,'MarkerFaceColor',[0.8 0.5 0.1],...
              'MarkerEdgeColor','w','LineWidth',1.5);
h_ped1 = plot(ax, ped1.x, ped1.y, '^','MarkerSize',11,'MarkerFaceColor',[0.9 0.3 0.9],...
              'MarkerEdgeColor','w','LineWidth',1.5);
h_ped2 = plot(ax, ped2.x, ped2.y, '^','MarkerSize',11,'MarkerFaceColor',[0.9 0.3 0.9],...
              'MarkerEdgeColor','w','LineWidth',1.5);
h_cyc  = plot(ax, cyc.x, cyc.y, 'v','MarkerSize',11,'MarkerFaceColor',[0.3 0.7 1.0],...
              'MarkerEdgeColor','w','LineWidth',1.5);
h_traj = plot(ax, ego.x, ego.y, '--','Color',[0.4 0.9 0.4],'LineWidth',1);

% Labels
text(ax, car1.x, car1.y+3, 'Car (W→E)','Color',[1 0.5 0.5],'FontSize',8);
text(ax, car2.x, car2.y+3, 'Car (E→W)','Color',[1 0.5 0.5],'FontSize',8);
text(ax, rick.x, rick.y+3, 'Rickshaw','Color',[1 0.7 0.3],'FontSize',8);

h_status = text(ax, 0.02, 0.98,'','Units','normalized','Color','w',...
                'FontSize',10,'VerticalAlignment','top','FontWeight','bold');

drawnow;

% ---------------------------------------------------------------------------
% Simulation loop
% ---------------------------------------------------------------------------
t = 0;
arrived = false;
traj_x = ego.x; traj_y = ego.y;
event_log = {};
in_intersection = false;

while t < t_end && ishandle(fig)

    dist_to_goal = norm(goal - [ego.x, ego.y]);
    if dist_to_goal < tol, arrived = true; break; end

    in_intersection = abs(ego.x) < lw*2 && abs(ego.y) < lw*2;

    % ── Decision ────────────────────────────────────────────────────────
    dist_c1  = norm([car1.x - ego.x, car1.y - ego.y]);
    dist_c2  = norm([car2.x - ego.x, car2.y - ego.y]);
    dist_rk  = norm([rick.x - ego.x, rick.y - ego.y]);
    dist_p1  = norm([ped1.x - ego.x, ped1.y - ego.y]);
    dist_p2  = norm([ped2.x - ego.x, ped2.y - ego.y]);
    dist_cyc = norm([cyc.x  - ego.x, cyc.y  - ego.y]);

    in_int_zone = norm([ego.x, ego.y]) < lw*3 + 5;

    if (dist_p1 < 10 || dist_p2 < 10) && in_int_zone
        decision = 'BRAKE'; speed_target = 1.0;
        ev = sprintf('t=%.1fs: BRAKE — pedestrian in crossing (%.1fm)', t, min(dist_p1,dist_p2));
    elseif (dist_c1 < 20 || dist_c2 < 20) && in_int_zone
        decision = 'SLOW'; speed_target = 3.0;
        ev = sprintf('t=%.1fs: SLOW — cross-traffic %.1fm', t, min(dist_c1,dist_c2));
    elseif dist_rk < 18 && in_int_zone
        decision = 'CAUTION'; speed_target = 5.0;
        ev = sprintf('t=%.1fs: CAUTION — rickshaw %.1fm', t, dist_rk);
    elseif in_int_zone
        decision = 'CAUTION'; speed_target = 5.0;
        ev = '';
    else
        decision = 'PROCEED'; speed_target = 8.33;
        ev = '';
    end
    if ~isempty(ev), event_log{end+1} = ev; fprintf('  %s\n', ev); end

    % ── Pure-pursuit: track goal directly ────────────────────────────────
    alpha = atan2(goal(2)-ego.y, goal(1)-ego.x) - ego.heading;
    alpha = atan2(sin(alpha), cos(alpha));
    pp_la = 8.0;
    steer = atan2(2*wb*sin(alpha), pp_la);
    steer = max(-max_st, min(max_st, steer));

    % ── Speed ────────────────────────────────────────────────────────────
    ego.speed = max(0, min(15, ego.speed + 0.8*(speed_target-ego.speed)*dt));

    % ── Integrate ───────────────────────────────────────────────────────
    ego.x = ego.x + ego.speed * cos(ego.heading) * dt;
    ego.y = ego.y + ego.speed * sin(ego.heading) * dt;
    ego.heading = ego.heading + (ego.speed/wb)*tan(steer)*dt;

    % ── Actors ──────────────────────────────────────────────────────────
    car1.x = car1.x + car1.speed * cos(car1.heading) * dt;
    car2.x = car2.x + car2.speed * cos(car2.heading) * dt;
    rick.x = rick.x + rick.speed * cos(rick.heading) * dt;
    ped1.x = ped1.x + ped1.speed * cos(ped1.heading) * dt;
    ped2.x = ped2.x + ped2.speed * cos(ped2.heading) * dt;
    cyc.y  = cyc.y  + cyc.speed  * sin(cyc.heading)  * dt;

    traj_x(end+1) = ego.x; traj_y(end+1) = ego.y;

    % ── Plot ────────────────────────────────────────────────────────────
    set(h_ego,  'XData', ego.x,  'YData', ego.y);
    set(h_car1, 'XData', car1.x, 'YData', car1.y);
    set(h_car2, 'XData', car2.x, 'YData', car2.y);
    set(h_rick, 'XData', rick.x, 'YData', rick.y);
    set(h_ped1, 'XData', ped1.x, 'YData', ped1.y);
    set(h_ped2, 'XData', ped2.x, 'YData', ped2.y);
    set(h_cyc,  'XData', cyc.x,  'YData', cyc.y);
    set(h_traj, 'XData', traj_x, 'YData', traj_y);

    col = scenario_color(decision);
    set(h_status,'String',...
        sprintf('t=%.1f s  |  %s  |  Speed: %.1f km/h  |  Dist: %.1f m',...
        t, decision, ego.speed*3.6, dist_to_goal),'Color',col);
    title(ax, sprintf('Scenario 2 — Urban Intersection  |  %s', decision),...
          'Color', col, 'FontSize', 12);

    drawnow limitrate;
    t = t + dt;
end

fprintf('\n=== Scenario 2 Complete ===\n');
if arrived
    fprintf('  Goal reached in %.1f s\n', t);
else
    fprintf('  Ended at t=%.1f s\n', t);
end
fprintf('  Events: %d\n', numel(event_log));

scenario_result.name = 'urban_intersection';
scenario_result.t_total = t; scenario_result.arrived = arrived;
scenario_result.traj_x = traj_x; scenario_result.traj_y = traj_y;
assignin('base','scenario_result', scenario_result);

function c = scenario_color(d)
    switch d
        case 'PROCEED',  c=[0.2 0.9 0.2];
        case 'CAUTION',  c=[1.0 0.8 0.0];
        case 'SLOW',     c=[1.0 0.5 0.0];
        case 'BRAKE',    c=[1.0 0.2 0.0];
        otherwise,       c=[0.8 0.8 0.8];
    end
end
