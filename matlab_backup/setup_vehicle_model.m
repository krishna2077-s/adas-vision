% SETUP_VEHICLE_MODEL Script to setup bicycle vehicle model parameters
% This script defines parameters in the MATLAB workspace for Simulink.

% 1. Define vehicle parameters
veh.mass = 1500; % Vehicle mass (kg)
veh.wheelbase = 2.7; % Wheelbase (m)
veh.lf = 1.2; % Distance from front axle to CG (m)
veh.lr = 1.5; % Distance from rear axle to CG (m)
veh.Cf = 80000; % Front cornering stiffness (N/rad)
veh.Cr = 80000; % Rear cornering stiffness (N/rad)
veh.Iz = 3000; % Yaw inertia (kg*m^2)
veh.max_steer_rad = deg2rad(35); % Maximum steering angle (rad)
veh.max_speed_mps = 16.7; % Maximum speed (m/s) -> 60 km/h
veh.max_accel = 3.0; % Maximum acceleration (m/s^2)
veh.max_decel = -8.0; % Maximum deceleration (m/s^2)

% 2. Define Pure Pursuit controller parameters
pp.lookahead_dist = 8.0; % Nominal lookahead distance (m)
pp.min_lookahead = 4.0; % Minimum lookahead distance (m)
pp.max_lookahead = 15.0; % Maximum lookahead distance (m)

% 3. Define PID longitudinal controller
pid.Kp = 0.6; % Proportional gain
pid.Ki = 0.05; % Integral gain
pid.Kd = 0.1; % Derivative gain
pid.dt = 0.05; % Control loop time step (20 Hz)

% 4. Define initial ego state
ego.x0 = 0; % Initial X position
ego.y0 = 0; % Initial Y position
ego.yaw0 = 0; % Initial yaw angle (rad)
ego.v0 = 0; % Initial velocity (m/s)

% 5. Define simulation parameters
sim.dt = 0.05; % Simulation time step (20 Hz)
sim.duration = 60; % Simulation duration (seconds)

% 6. Print a summary of all parameters
disp('=== Vehicle Model Parameters Setup Complete ===');
disp('Vehicle Properties:');
disp(veh);
disp('Pure Pursuit Controller:');
disp(pp);
disp('PID Controller:');
disp(pid);
disp('Initial Ego State:');
disp(ego);
disp('Simulation Parameters:');
disp(sim);
