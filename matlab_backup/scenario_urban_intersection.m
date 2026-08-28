% SCENARIO_URBAN_INTERSECTION
% Scenario 2: Busy Urban Intersection Without Signals

% 1. Create drivingScenario
scenario = drivingScenario('SampleTime', 0.05);

% 2. Build 4-way intersection (two roads crossing at right angles, 8m wide)
road(scenario, [0 -100 0; 0 100 0], 'Width', 8); % North-South
road(scenario, [-100 0 0; 100 0 0], 'Width', 8); % East-West

% 3 & 4. Add actors and trajectories
% Ego vehicle approaching from south, speed 20 km/h (5.56 m/s)
ego = vehicle(scenario, 'ClassID', 1, 'Position', [2 -100 0]);
trajectory(ego, [2 -100 0; 2 100 0], 5.56);

% 2 cars crossing from the left (west to east), speed 20 km/h, staggered
car1 = vehicle(scenario, 'ClassID', 1, 'Position', [-100 -2 0]);
trajectory(car1, [-100 -2 0; 100 -2 0], 5.56);

car2 = vehicle(scenario, 'ClassID', 1, 'Position', [-130 -2 0]);
trajectory(car2, [-130 -2 0; 100 -2 0], 5.56);

% 1 auto-rickshaw (small car) turning right from the east
rickshaw = actor(scenario, 'ClassID', 1, 'Length', 3.5, 'Width', 1.5, 'Position', [100 2 0]);
trajectory(rickshaw, [100 2 0; 15 2 0; 2 15 0; 2 100 0], 5.56);

% 3 pedestrians crossing diagonally through intersection at 4 km/h (1.11 m/s)
ped1 = actor(scenario, 'ClassID', 4, 'Length', 0.5, 'Width', 0.5);
trajectory(ped1, [-8 -8 0; 8 8 0], 1.11);
ped2 = actor(scenario, 'ClassID', 4, 'Length', 0.5, 'Width', 0.5);
trajectory(ped2, [-10 -8 0; 6 8 0], 1.11);
ped3 = actor(scenario, 'ClassID', 4, 'Length', 0.5, 'Width', 0.5);
trajectory(ped3, [8 -8 0; -8 8 0], 1.11);

% 1 cyclist going straight from north, speed 15 km/h (4.17 m/s)
cyclist = actor(scenario, 'ClassID', 3, 'Length', 2.0, 'Width', 0.6);
trajectory(cyclist, [-2 100 0; -2 -100 0], 4.17);

% 5. Plot the scenario
plot(scenario);
title('Scenario 2: Busy Urban Intersection');
