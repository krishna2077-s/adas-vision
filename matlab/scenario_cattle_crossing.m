% scenario_cattle_crossing.m
% ADAS Vision — Driving Scenario 5: Cattle Crossing
%
% Creates a village road where TWO COWS suddenly walk onto the road
% from the left shoulder after 3 seconds. The ego vehicle must detect
% the obstacle, brake, and wait for the cows to clear before proceeding.
%
% Usage:
%   >> scenario_cattle_crossing

fprintf('=== Scenario 5: Cattle Crossing ===\n');

% ---------------------------------------------------------------------------
% Road geometry — straight rural road (6 m wide)
% ---------------------------------------------------------------------------
road_width_m = 6.0;
road_len_m   = 180.0;
lw           = road_width_m / 2;

% Road edges
rx = [0, road_len_m];

% ---------------------------------------------------------------------------
% Actors
% ---------------------------------------------------------------------------
% Ego: starts 20 m from beginning, heading east
ego.x = 5; ego.y = 0; ego.heading = 0; ego.speed = 11.1;  % ~40 km/h

% Cow 1: starts on left shoulder, enters road at t=3 s
cow1.x = 80; cow1.y = -lw - 2;   % starts off-road (left)
cow1.speed = 0.8; cow1.heading = pi/2;   % walks straight across (+y)
cow1.active = false;  % not moving yet

% Cow 2: slightly behind, enters road at t=3.5 s
cow2.x = 77; cow2.y = -lw - 3;
cow2.speed = 0.7; cow2.heading = pi/2;
cow2.active = false;

goal    = [160, 0];
goal_tol = 4.0;

% ---------------------------------------------------------------------------
% Simulation parameters
% ---------------------------------------------------------------------------
dt     = 0.033;
t_end  = 35.0;
wb     = 2.70;
max_st = deg2rad(35);
pp_la  = 10.0;

% ---------------------------------------------------------------------------
% Figure
% ---------------------------------------------------------------------------
fig = figure('Name','Scenario 5 — Cattle Crossing','NumberTitle','off',...
             'Color',[0.06 0.07 0.05],'Position',[100 150 1000 550]);
ax  = axes('Parent',fig,'Color',[0.13 0.15 0.10],...
           'XColor','w','YColor','w','GridColor',[0.25 0.3 0.20],...
           'GridAlpha',0.5,'Box','on');
hold(ax,'on'); grid(ax,'on');
xlabel(ax,'X (m)','Color','w'); ylabel(ax,'Y (m)','Color','w');
title(ax,'Scenario 5 — Cattle Crossing  |  Waiting for event...','Color',[0.7 1 0.5],'FontSize',12);
xlim(ax,[0, road_len_m+10]); ylim(ax,[-lw-6, lw+6]);

% Road fill
fill(ax,[0, road_len_m, road_len_m, 0],[-lw, -lw, lw, lw],...
     [0.22 0.26 0.18],'EdgeColor','none','FaceAlpha',0.7);

% Road markings
plot(ax,[0 road_len_m], [ lw  lw],'w-','LineWidth',1.5);
plot(ax,[0 road_len_m], [-lw -lw],'w-','LineWidth',1.5);
plot(ax,[0 road_len_m], [0 0],'--','Color',[0.8 0.75 0.0],'LineWidth',0.8);

% Grass shoulders
fill(ax,[0, road_len_m, road_len_m, 0],[-lw-6,-lw-6,-lw,-lw],...
     [0.15 0.30 0.10],'EdgeColor','none','FaceAlpha',0.5);
fill(ax,[0, road_len_m, road_len_m, 0],[lw, lw, lw+6, lw+6],...
     [0.15 0.30 0.10],'EdgeColor','none','FaceAlpha',0.5);

% Cattle warning sign (on left shoulder near x=65)
plot(ax, 65, -lw-1.5, '^','MarkerSize',14,...
     'MarkerFaceColor',[1 0.9 0],'MarkerEdgeColor','k','LineWidth',1.5);
text(ax, 65, -lw-3.5,'CATTLE\nXING','Color',[1 0.9 0],'FontSize',8,...
     'HorizontalAlignment','center');

% Goal marker
plot(ax, goal(1), goal(2),'p','MarkerSize',18,...
     'MarkerFaceColor',[1 0.85 0],'MarkerEdgeColor','w');
text(ax, goal(1)+2, goal(2)+2,'GOAL','Color',[1 0.85 0],'FontSize',10);

% Actor handles
h_ego  = plot(ax, ego.x,  ego.y,  'o','MarkerSize',16,'MarkerFaceColor',[0 0.85 0.3],...
              'MarkerEdgeColor','w','LineWidth',2);
h_cow1 = plot(ax, cow1.x, cow1.y, 's','MarkerSize',18,'MarkerFaceColor',[0.55 0.40 0.20],...
              'MarkerEdgeColor','w','LineWidth',2);
h_cow2 = plot(ax, cow2.x, cow2.y, 's','MarkerSize',18,'MarkerFaceColor',[0.55 0.40 0.20],...
              'MarkerEdgeColor','w','LineWidth',2);
h_traj = plot(ax, ego.x, ego.y,'--','Color',[0.4 0.9 0.4],'LineWidth',1);

% Labels
text(ax, cow1.x+1, cow1.y+1, '🐄','FontSize',16,'Color',[0.8 0.6 0.2]);
text(ax, cow2.x+1, cow2.y+1, '🐄','FontSize',16,'Color',[0.8 0.6 0.2]);

% Countdown label
h_count = text(ax, 80, lw+2,'Cows enter in 3.0 s','Color',[1 0.7 0.2],...
               'FontSize',10,'HorizontalAlignment','center','FontWeight','bold');
h_status = text(ax, 0.02, 0.97,'','Units','normalized','Color','w',...
                'FontSize',10,'VerticalAlignment','top','FontWeight','bold');

drawnow;

% ---------------------------------------------------------------------------
% Simulation
% ---------------------------------------------------------------------------
t = 0;
arrived = false;
traj_x = ego.x; traj_y = ego.y;
event_log = {};
waiting   = false;

while t < t_end && ishandle(fig)

    dist_to_goal = norm(goal - [ego.x, ego.y]);
    if dist_to_goal < goal_tol, arrived = true; break; end

    % ── Cattle event trigger ─────────────────────────────────────────────
    if t >= 3.0,   cow1.active = true; end
    if t >= 3.5,   cow2.active = true; end

    % ── Update cows ─────────────────────────────────────────────────────
    if cow1.active && cow1.y < lw + 3
        cow1.y = cow1.y + cow1.speed * dt;
        % Stop once across
        if cow1.y > lw + 2, cow1.active = false; end
    end
    if cow2.active && cow2.y < lw + 3
        cow2.y = cow2.y + cow2.speed * dt;
        if cow2.y > lw + 2, cow2.active = false; end
    end

    % ── Decision ────────────────────────────────────────────────────────
    cow_in_road1 = cow1.y > -lw && cow1.y < lw;
    cow_in_road2 = cow2.y > -lw && cow2.y < lw;
    cow_in_road  = cow_in_road1 || cow_in_road2;

    dist_c1 = norm([cow1.x - ego.x, cow1.y - ego.y]);
    dist_c2 = norm([cow2.x - ego.x, cow2.y - ego.y]);
    dist_cow = min(dist_c1, dist_c2);

    if cow_in_road && dist_cow < 8
        decision = 'EMERGENCY_STOP'; speed_target = 0;
        if ~waiting
            ev = sprintf('t=%.1fs: EMERGENCY_STOP — cow %.1fm in-path', t, dist_cow);
            event_log{end+1} = ev; fprintf('  %s\n', ev);
            waiting = true;
        end
    elseif cow_in_road && dist_cow < 20
        decision = 'BRAKE'; speed_target = 0.5;
        if t > 3.5
            ev = sprintf('t=%.1fs: BRAKE — cow %.1fm ahead (in road)', t, dist_cow);
            if isempty(event_log) || ~strcmp(event_log{end}, ev)
                event_log{end+1} = ev; fprintf('  %s\n', ev);
            end
        end
    elseif dist_cow < 35 && cow_in_road
        decision = 'SLOW'; speed_target = 3.0;
    elseif dist_cow < 50 && t > 2.5
        decision = 'CAUTION'; speed_target = 6.0;
        if t < 4.5
            ev = sprintf('t=%.1fs: CAUTION — cattle detected %.1fm ahead', t, dist_cow);
            if isempty(event_log) || ~strcmp(event_log{end}, ev)
                event_log{end+1} = ev; fprintf('  %s\n', ev);
            end
        end
    else
        decision = 'PROCEED'; speed_target = 11.1;
        if waiting && ~cow_in_road
            ev = sprintf('t=%.1fs: PROCEED — road clear, resuming', t);
            event_log{end+1} = ev; fprintf('  %s\n', ev);
            waiting = false;
        end
    end

    % ── Steering: straight to goal ───────────────────────────────────────
    alpha = atan2(goal(2)-ego.y, goal(1)-ego.x) - ego.heading;
    alpha = atan2(sin(alpha), cos(alpha));
    steer = atan2(2*wb*sin(alpha), pp_la);
    steer = max(-max_st, min(max_st, steer));

    % ── Speed control ────────────────────────────────────────────────────
    accel = 1.2 * (speed_target - ego.speed);
    ego.speed = max(0, min(15, ego.speed + accel*dt));

    % ── Integrate ego ────────────────────────────────────────────────────
    ego.x = ego.x + ego.speed * cos(ego.heading) * dt;
    ego.y = ego.y + ego.speed * sin(ego.heading) * dt;
    ego.heading = ego.heading + (ego.speed/wb)*tan(steer)*dt;

    traj_x(end+1) = ego.x; traj_y(end+1) = ego.y;

    % ── Plot ────────────────────────────────────────────────────────────
    set(h_ego,  'XData', ego.x,  'YData', ego.y);
    set(h_cow1, 'XData', cow1.x, 'YData', cow1.y);
    set(h_cow2, 'XData', cow2.x, 'YData', cow2.y);
    set(h_traj, 'XData', traj_x, 'YData', traj_y);

    % Countdown
    if t < 3.0
        set(h_count,'String',sprintf('⚠ Cows enter in %.1f s', 3.0 - t),'Visible','on');
    elseif t < 5.0
        set(h_count,'String','⚠ CATTLE ON ROAD','Color',[1 0.2 0.2],...
            'FontSize',12,'Visible','on');
    else
        set(h_count,'Visible','off');
    end

    col = scenario_color(decision);
    set(h_status,'String',...
        sprintf('t=%.1f s  |  %s  |  Speed: %.1f km/h  |  Cow dist: %.1f m',...
        t, decision, ego.speed*3.6, dist_cow),'Color',col);
    title(ax, sprintf('Scenario 5 — Cattle Crossing  |  %s', decision),...
          'Color', col, 'FontSize', 12);

    drawnow limitrate;
    t = t + dt;
end

fprintf('\n=== Scenario 5 Complete ===\n');
if arrived
    fprintf('  Goal reached in %.1f s\n', t);
else
    fprintf('  Ended at t=%.1f s\n', t);
end
fprintf('  Events: %d\n', numel(event_log));
for k = 1:numel(event_log)
    fprintf('    %s\n', event_log{k});
end

scenario_result.name = 'cattle_crossing';
scenario_result.t_total = t; scenario_result.arrived = arrived;
scenario_result.traj_x = traj_x; scenario_result.traj_y = traj_y;
assignin('base','scenario_result', scenario_result);

function c = scenario_color(d)
    switch d
        case 'PROCEED',         c=[0.2 0.9 0.2];
        case 'CAUTION',         c=[1.0 0.8 0.0];
        case 'SLOW',            c=[1.0 0.5 0.0];
        case 'BRAKE',           c=[1.0 0.2 0.0];
        case 'EMERGENCY_STOP',  c=[1.0 0.0 0.0];
        otherwise,              c=[0.8 0.8 0.8];
    end
end
