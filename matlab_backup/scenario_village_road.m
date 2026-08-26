% SCENARIO_VILLAGE_ROAD
% This script creates a driving scenario testing the ego vehicle's ability 
% to navigate an unmarked winding village road while handling an oncoming 
% motorcycle, a crossing pedestrian, and a static pushcart.

scenario = drivingScenario('SampleTime', 0.05);

% 2. Build narrow winding road (6m wide, ~500m long) with gentle S-curves
waypoints = [0 0 0; 100 20 0; 200 -20 0; 300 10 0; 400 -10 0; 500 0 0];
road(scenario, waypoints, 'Width', 6);

% 3 & 4. Add actors and trajectories
% Ego vehicle: speed 30 km/h (8.33 m/s)
ego = vehicle(scenario, 'ClassID', 1, 'Position', [0 0 0]);
trajectory(ego, waypoints, 8.33);

% Oncoming motorcycle: 200m ahead, speed 30 km/h (8.33 m/s)
moto = actor(scenario, 'ClassID', 3, 'Length', 2.2, 'Width', 0.8, 'Position', [200 -20 0]);
moto_waypoints = flipud(waypoints(1:3, :)); % Moving towards ego
trajectory(moto, moto_waypoints, 8.33);

% Pedestrian crossing: 150m ahead, speed 4 km/h (1.1 m/s) moving left
ped = actor(scenario, 'ClassID', 4, 'Length', 0.5, 'Width', 0.5, 'Position', [150 15 0]);
trajectory(ped, [150 15 0; 150 -15 0], 1.1);

% Static pushcart: parked on left shoulder at 300m ahead
cart = actor(scenario, 'ClassID', 2, 'Length', 2.0, 'Width', 1.0, 'Position', [300 13 0]);

% 5. Plot the scenario
plot(scenario);
title('Scenario 1: Unmarked Village Road');
