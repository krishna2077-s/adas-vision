function map = build_occupancy_grid_from_lidar(obstacle_pts, ego, grid_cfg)
% BUILD_OCCUPANCY_GRID_FROM_LIDAR Creates a 2D occupancy map directly from 3D LiDAR point clouds.
%
%   map = BUILD_OCCUPANCY_GRID_FROM_LIDAR(obstacle_pts, ego, grid_cfg)
%
%   Inputs:
%       obstacle_pts - [K x 3] Non-ground 3D points in world coordinates
%       ego          - Struct with ego state (x, y, heading)
%       grid_cfg     - (Optional) Struct with map dimensions:
%                      .length_m    (default: 60)
%                      .width_m     (default: 30)
%                      .resolution  (default: 4 cells/meter -> 0.25m cell size)
%                      .inflation_r (default: 0.35m inflation radius)
%
%   Outputs:
%       map - binaryOccupancyMap object populated with LiDAR obstacle hits
%
% ADAS Vision — SIH 2026 Perception Pipeline

if nargin < 3 || isempty(grid_cfg)
    grid_cfg = struct();
end

length_m   = 60.0;
width_m    = 30.0;
resolution = 4.0;
inflation_r= 0.35;

if isfield(grid_cfg, 'length_m'),   length_m   = grid_cfg.length_m; end
if isfield(grid_cfg, 'width_m'),    width_m    = grid_cfg.width_m; end
if isfield(grid_cfg, 'resolution'), resolution = grid_cfg.resolution; end
if isfield(grid_cfg, 'inflation_r'),inflation_r= grid_cfg.inflation_r; end

% Create binaryOccupancyMap centered dynamically around the ego vehicle
map = binaryOccupancyMap(length_m, width_m, resolution);
map.GridLocationInWorld = [ego.x - 5.0, ego.y - width_m/2];

if isempty(obstacle_pts)
    return;
end

% Filter obstacle points falling within the map's world boundaries
x_min = map.GridLocationInWorld(1);
x_max = x_min + length_m;
y_min = map.GridLocationInWorld(2);
y_max = y_min + width_m;

in_bounds = (obstacle_pts(:, 1) >= x_min & obstacle_pts(:, 1) <= x_max) & ...
            (obstacle_pts(:, 2) >= y_min & obstacle_pts(:, 2) <= y_max);

valid_pts = obstacle_pts(in_bounds, 1:2);

if isempty(valid_pts)
    return;
end

% Set occupied cells from raw obstacle points
setOccupancy(map, valid_pts, 1);

% Optional: Inflate occupied cells by vehicle safety radius
if inflation_r > 0
    inflate(map, inflation_r);
end

end
