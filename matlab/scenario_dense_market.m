function scenario_result = scenario_dense_market(mode)
% ADAS Vision — Driving Scenario 4: Congested Indian Market Street
%
% Tests autonomous navigation in an ultra-dense, unstructured urban market
% with high density of Vulnerable Road Users (VRUs), street vendors, and narrow navigable space.
% Uses the unified closed-loop adaptive pipeline:
%   build_occupancy_grid -> predict_trajectories -> replan_trigger -> plan_path -> decision_with_ratchet
%
% Usage:
%   >> scenario_dense_market
%   >> scenario_dense_market('baseline')

if nargin < 1, mode = 'adaptive'; end

cfg = struct();
cfg.name  = 'dense_market';
cfg.title = 'Scenario 4 — Dense Indian Market';
cfg.mode  = mode;
cfg.dt    = 0.033;
cfg.max_time = 55.0;  % Extended to allow navigation around dense obstacles
cfg.is_dense = true;
cfg.lane_half_width = 3.5; % 3.5m half width (7m total street)

cfg.ego_init.x       = 0.0;
cfg.ego_init.y       = 0.0;
cfg.ego_init.heading = 0.0;
cfg.ego_init.speed   = 2.78; % 10 km/h — congested market crawl

cfg.goal = [85.0, 0.0, 0.0];
cfg.goal_tol = 5.0;

% Actors
% 1. Static Pushcart (Thela) parked at x=32m — moved to y=-2.5 (right kerb)
%    so there is a clear left-lane corridor for the ego to bypass
thela.id    = 'thela';
thela.class = 'pushcart';
thela.x     = 32.0;
thela.y     = -2.5;   % Hugging right kerb — leaves >2.5m clear on the left
thela.vx    = 0.0;
thela.vy    = 0.0;
thela.width = 1.2;    % Slightly narrower for realistic market pushcart
thela.length= 2.0;

% 2. Crossing Shopper (x=20m, crossing from left stall y=+2.8 to right y=-2.5)
ped1.id    = 'ped1';
ped1.class = 'person';
ped1.x     = 22.0;
ped1.y     = 2.8;
ped1.vx    = 0.02;
ped1.vy    = -0.40; % ~1.4 km/h casual shopper crossing
ped1.width = 0.5;
ped1.length= 0.5;

% 3. Darting child/shopper from behind pushcart (x=45m)
ped2.id    = 'ped2';
ped2.class = 'person';
ped2.x     = 45.0;
ped2.y     = -2.5;
ped2.vx    = -0.05;
ped2.vy    = 0.50; % ~1.8 km/h darting child jog
ped2.width = 0.5;
ped2.length= 0.5;

% 4. Oncoming Scooter — starts behind pushcart, swerves to center to overtake
scoot.id    = 'scoot';
scoot.class = 'motorcycle';
scoot.x     = 78.0;   
scoot.y     = -2.5;   % Starts in the same lane as the pushcart
scoot.vx    = -4.17;  % ~15 km/h oncoming scooter
scoot.vy    =  0.0;
scoot.width = 0.8;
scoot.length= 1.8;

cfg.actors = {thela, ped1, ped2, scoot};
cfg.update_actor = @(act, t, dt, ego) update_market_actors(act, t, dt);
cfg.draw_background = @(ax) draw_market_background(ax, cfg.lane_half_width);

scenario_result = adaptive_scenario_loop(cfg);
end

% ---------------------------------------------------------------------------
% Helper Functions
% ---------------------------------------------------------------------------
function act = update_market_actors(act, t, dt)
    if strcmp(act.id, 'ped1')
        % Pedestrian 1 crosses from left stall to right — walk fully off the road
        if t >= 0.5 && act.y > -8.0
            act.x = act.x + act.vx * dt;
            act.y = act.y + act.vy * dt;
        end
    elseif strcmp(act.id, 'ped2')
        % Pedestrian 2 darts out from behind pushcart at t=4s — walk fully off the road
        if t >= 4.0 && act.y < 8.0
            act.x = act.x + act.vx * dt;
            act.y = act.y + act.vy * dt;
        end
    elseif strcmp(act.id, 'scoot')
        % Scooter: swerve to the positive-y side of the road (y = +1.8) to avoid ego at y=0
        overtake_start_x = 48.0;

        if act.x > overtake_start_x
            % Approaching straight
            act.x = act.x + act.vx * dt;
        else
            % Swerve to y = +1.8 (left side — away from ego near y=0 and pushcart at -2.5)
            if act.y < 1.8
                act.vy = 0.8; % Swerve towards positive-y side
            else
                act.vy = 0.0;
            end
            act.x = act.x + act.vx * dt;
            act.y = act.y + act.vy * dt;
        end
    else
        act.x = act.x + act.vx * dt;
        act.y = act.y + act.vy * dt;
    end
end

function draw_market_background(ax, lw)
    fill(ax, [0 110 110 0], [-lw -lw lw lw], [0.22 0.20 0.21], 'EdgeColor', 'none');
    % Colorful market stalls on left and right
    for sx = 0:14:100
        fill(ax, [sx sx+11 sx+11 sx], [lw lw lw+2.5 lw+2.5], [0.85 0.35 0.2], 'EdgeColor', 'none');
        text(ax, sx+2, lw+1.2, 'STALL', 'Color', 'k', 'FontSize', 7, 'FontWeight', 'bold');
        fill(ax, [sx sx+11 sx+11 sx], [-lw-2.5 -lw-2.5 -lw -lw], [0.2 0.55 0.75], 'EdgeColor', 'none');
        text(ax, sx+2, -lw-1.2, 'SHOP', 'Color', 'w', 'FontSize', 7);
    end
    ylim(ax, [-lw-4, lw+4]);
end
