% SCENARIO_CATTLE_CROSSING
% Scenario 5: Sudden Cattle Crossing
% Pass criteria: Ego vehicle must stop with > 2m clearance from the cows.

% 1. Create drivingScenario
scenario = drivingScenario('SampleTime', 0.05);

% 2. Build a straight village road (6m wide, 300m)
road(scenario, [0 0 0; 300 0 0], 'Width', 6);

% 3. Add actors
% Ego vehicle: starting position, speed 30 km/h (8.33 m/s)
ego = vehicle(scenario, 'ClassID', 1, 'Position', [0 0 0]);
trajectory(ego, [0 0 0; 300 0 0], 8.33);

% 2 cows behind an obstacle at 120m, walking onto road at 3 km/h (0.83 m/s)
cow1 = actor(scenario, 'ClassID', 5, 'Length', 2.0, 'Width', 1.5, 'Position', [120 6 0]);
cow2 = actor(scenario, 'ClassID', 5, 'Length', 2.0, 'Width', 1.5, 'Position', [122 7 0]);

% 4. Set cow trajectories: stationary for first 3 seconds, then cross
time_pts = [0, 3, 15]; % 0 to 3s stationary, then moving until 15s
traj1 = [120 6 0; 120 6 0; 120 -4 0];
trajectory(cow1, traj1, time_pts);

traj2 = [122 7 0; 122 7 0; 122 -3 0];
trajectory(cow2, traj2, time_pts);

% 5. Plot the scenario
plot(scenario);
title('Scenario 5: Sudden Cattle Crossing');
