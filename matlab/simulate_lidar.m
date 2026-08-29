function ptCloud = simulate_lidar(ego, actors, road_cfg, lidar_params)
% SIMULATE_LIDAR Generates a synthetic 3D LiDAR point cloud from scene geometry.
%
%   ptCloud = SIMULATE_LIDAR(ego, actors, road_cfg, lidar_params)
%
%   Inputs:
%       ego          - Struct with fields: x, y, heading (rad), speed (m/s)
%       actors       - Cell array of actor structs (x, y, vx, vy, class, width, length, height)
%       road_cfg     - Struct with road boundaries and static geometry (e.g. lane_half_width, x_limits)
%       lidar_params - (Optional) Struct with sensor configuration:
%                      .mount_pos      - [x, y, z] offset relative to ego (default: [1.5, 0, 1.8] m)
%                      .num_channels   - Vertical laser beams (default: 32)
%                      .v_fov          - [min_deg, max_deg] (default: [-15, 15])
%                      .h_fov          - [min_deg, max_deg] (default: [-75, 75])
%                      .h_res_deg      - Horizontal angular resolution (default: 0.6 deg)
%                      .max_range      - Maximum detection range in meters (default: 65)
%                      .range_noise_std- Gaussian range noise in meters (default: 0.02)
%
%   Outputs:
%       ptCloud - Struct containing:
%                 .Location     - [N x 3] matrix of (X, Y, Z) point coordinates in world frame
%                 .Intensity    - [N x 1] vector of reflection intensity (0..1)
%                 .Count        - Total point count N
%                 .EgoFrame     - [N x 3] matrix in ego-centric frame (X: forward, Y: left, Z: up)
%                 .SensorPos    - [1 x 3] sensor position in world frame
%                 .Range        - [N x 1] range measurements in meters
%
% ADAS Vision — SIH 2026 Perception Pipeline

if nargin < 4 || isempty(lidar_params)
    lidar_params = struct();
end

% Default LiDAR sensor specs (Automotive 32-beam solid-state / spinning LiDAR)
mount_pos   = [1.5, 0, 1.8]; % [dx, dy, dz] from vehicle rear-axle center
num_ch      = 32;
v_fov       = [-16.0, 14.0];  % Vertical FOV (degrees)
h_fov       = [-75.0, 75.0];  % Forward sector FOV (degrees)
h_res       = 0.6;           % Horizontal resolution in deg (~250 rays per ring)
max_range   = 65.0;          % Range in meters
noise_std   = 0.02;          % 2cm range accuracy

if isfield(lidar_params, 'mount_pos'), mount_pos = lidar_params.mount_pos; end
if isfield(lidar_params, 'num_channels'), num_ch = lidar_params.num_channels; end
if isfield(lidar_params, 'v_fov'), v_fov = lidar_params.v_fov; end
if isfield(lidar_params, 'h_fov'), h_fov = lidar_params.h_fov; end
if isfield(lidar_params, 'h_res_deg'), h_res = lidar_params.h_res_deg; end
if isfield(lidar_params, 'max_range'), max_range = lidar_params.max_range; end
if isfield(lidar_params, 'range_noise_std'), noise_std = lidar_params.range_noise_std; end

% Ego world position & sensor world position
psi = ego.heading;
R_body_to_world = [cos(psi), -sin(psi); sin(psi), cos(psi)];
sensor_world_xy = [ego.x, ego.y]' + R_body_to_world * mount_pos(1:2)';
sensor_world_z  = mount_pos(3);

v_angles = linspace(v_fov(1), v_fov(2), num_ch);
h_angles = h_fov(1):h_res:h_fov(2);

[V_grid, H_grid] = ndgrid(v_angles, h_angles);
V_rad = deg2rad(V_grid(:));
H_rad = deg2rad(H_grid(:));

% Ray unit vectors in ego sensor coordinates (X: forward, Y: left, Z: up)
ray_dir_sensor = [cos(V_rad) .* cos(H_rad), ...
                  cos(V_rad) .* sin(H_rad), ...
                  sin(V_rad)];

% Ray unit vectors in world frame
ray_dir_world_xy = (R_body_to_world * ray_dir_sensor(:, 1:2)')';
ray_dir_world = [ray_dir_world_xy, ray_dir_sensor(:, 3)];

num_rays = size(ray_dir_world, 1);
ray_ranges = Inf(num_rays, 1);
ray_intensity = zeros(num_rays, 1);

% ---------------------------------------------------------------------------
% 1. Raycast Ground Plane (Z = 0)
% ---------------------------------------------------------------------------
dz = ray_dir_world(:, 3);
down_mask = dz < -1e-4;
ground_t = -sensor_world_z ./ dz(down_mask);
valid_ground = ground_t > 0 & ground_t <= max_range;

down_indices = find(down_mask);
matched_indices = down_indices(valid_ground);
t_vals = ground_t(valid_ground);

hit_x = sensor_world_xy(1) + t_vals .* ray_dir_world(matched_indices, 1);
hit_y = sensor_world_xy(2) + t_vals .* ray_dir_world(matched_indices, 2);

% Check road limits if road_cfg is provided
if nargin >= 3 && ~isempty(road_cfg) && isfield(road_cfg, 'lane_half_width')
    lw = road_cfg.lane_half_width;
    on_road = abs(hit_y) <= (lw + 2.0);
    road_mask = valid_ground;
    road_mask(valid_ground) = on_road;
    
    matched_indices = down_indices(road_mask);
    t_vals = ground_t(road_mask);
end

ray_ranges(matched_indices) = t_vals;
ray_intensity(matched_indices) = 0.25 + 0.1 * rand(numel(matched_indices), 1); % Asphalt reflectivity

% ---------------------------------------------------------------------------
% 2. Raycast Actors (3D Oriented Bounding Boxes)
% ---------------------------------------------------------------------------
if nargin >= 2 && ~isempty(actors)
    for k = 1:numel(actors)
        act = actors{k};
        
        cls = 'car';
        if isfield(act, 'class'), cls = lower(act.class); end
        
        L = 4.4; W = 1.8; H = 1.5; % Default car
        reflectivity = 0.7;
        
        switch cls
            case {'person', 'pedestrian'}
                L = 0.5; W = 0.5; H = 1.7; reflectivity = 0.35;
            case {'cow', 'cattle', 'animal'}
                L = 2.2; W = 0.9; H = 1.4; reflectivity = 0.45;
            case {'auto_rickshaw', 'rickshaw', 'auto'}
                L = 2.8; W = 1.3; H = 1.8; reflectivity = 0.80;
            case {'pushcart', 'thela'}
                L = 2.0; W = 1.1; H = 1.2; reflectivity = 0.40;
            case {'motorcycle', 'bike'}
                L = 2.1; W = 0.8; H = 1.3; reflectivity = 0.65;
            case {'truck', 'bus'}
                L = 8.5; W = 2.5; H = 3.0; reflectivity = 0.85;
        end
        
        if isfield(act, 'length'), L = act.length; end
        if isfield(act, 'width'),  W = act.width; end
        if isfield(act, 'height'), H = act.height; end
        
        yaw = 0;
        if isfield(act, 'heading'), yaw = act.heading; end
        if isfield(act, 'vx') && isfield(act, 'vy') && norm([act.vx, act.vy]) > 0.1
            yaw = atan2(act.vy, act.vx);
        end
        
        % Actor center in world
        box_center = [act.x, act.y, H/2];
        
        % Transform sensor rays into Actor-Local Coordinate Frame
        R_actor = [cos(yaw), -sin(yaw), 0; ...
                   sin(yaw),  cos(yaw), 0; ...
                   0,         0,        1];
        
        % Sensor origin in actor frame
        sensor_to_actor = ([sensor_world_xy; sensor_world_z] - box_center');
        sensor_in_actor = (R_actor' * sensor_to_actor)';
        
        % Ray directions in actor frame
        rays_in_actor = (R_actor' * ray_dir_world')';
        
        % Ray-AABB intersection (Slab method)
        box_min = [-L/2, -W/2, -H/2];
        box_max = [ L/2,  W/2,  H/2];
        
        inv_dir = 1.0 ./ (rays_in_actor + 1e-12 * (rays_in_actor == 0));
        t0 = (box_min - sensor_in_actor) .* inv_dir;
        t1 = (box_max - sensor_in_actor) .* inv_dir;
        
        tmin_xyz = min(t0, t1);
        tmax_xyz = max(t0, t1);
        
        t_near = max(tmin_xyz, [], 2);
        t_far  = min(tmax_xyz, [], 2);
        
        hit_mask = (t_near <= t_far) & (t_far > 0) & (t_near < ray_ranges) & (t_near < max_range);
        
        if any(hit_mask)
            hit_idx = find(hit_mask);
            t_hit = max(0.1, t_near(hit_mask));
            ray_ranges(hit_idx) = t_hit;
            ray_intensity(hit_idx) = reflectivity + 0.1 * randn(numel(hit_idx), 1);
        end
    end
end

% ---------------------------------------------------------------------------
% 3. Extract Valid Returns & Add Sensor Measurement Noise
% ---------------------------------------------------------------------------
valid_hits = (ray_ranges > 0.2) & (ray_ranges < max_range);
valid_ranges = ray_ranges(valid_hits);
valid_intensity = max(0.0, min(1.0, ray_intensity(valid_hits)));

% Add Gaussian range measurement noise
valid_ranges = valid_ranges + noise_std * randn(size(valid_ranges));

valid_dirs_world = ray_dir_world(valid_hits, :);
pts_world = [sensor_world_xy(1) + valid_ranges .* valid_dirs_world(:, 1), ...
             sensor_world_xy(2) + valid_ranges .* valid_dirs_world(:, 2), ...
             sensor_world_z     + valid_ranges .* valid_dirs_world(:, 3)];

% Compute points in vehicle body frame (Ego-centric: X-forward, Y-left, Z-up)
dx_w = pts_world(:, 1) - ego.x;
dy_w = pts_world(:, 2) - ego.y;
dz_w = pts_world(:, 3);

pts_ego_x =  cos(psi) * dx_w + sin(psi) * dy_w;
pts_ego_y = -sin(psi) * dx_w + cos(psi) * dy_w;
pts_ego_z = dz_w;
pts_ego = [pts_ego_x, pts_ego_y, pts_ego_z];

% Construct output struct
ptCloud = struct();
ptCloud.Location  = pts_world;
ptCloud.EgoFrame  = pts_ego;
ptCloud.Intensity = valid_intensity;
ptCloud.Count     = size(pts_world, 1);
ptCloud.SensorPos = [sensor_world_xy; sensor_world_z]';
ptCloud.Range     = valid_ranges;

end
