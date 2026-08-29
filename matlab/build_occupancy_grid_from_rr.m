function map = build_occupancy_grid_from_rr(tracks, ego)
% BUILD_OCCUPANCY_GRID_FROM_RR Generates a binaryOccupancyMap from RoadRunner tracks
%
% Inputs:
%   tracks - Struct array of obstacle tracks in Ego local frame
%   ego    - Current ego vehicle pose struct
%
% Output:
%   map    - MATLAB binaryOccupancyMap object

grid_len = 50.0; % meters
grid_width = 30.0; % meters
resolution = 4; % 0.25m per cell

map = binaryOccupancyMap(grid_len, grid_width, resolution);

% Ego position in grid coordinates (shift y to [0, width])
y_shift = grid_width / 2;

if isempty(tracks), return; end

for k = 1:numel(tracks)
    t = tracks(k);
    
    % Track position in local occupancy grid
    grid_x = t.x;
    grid_y = t.y + y_shift;
    
    if grid_x < 0 || grid_x > grid_len || grid_y < 0 || grid_y > grid_width
        continue;
    end
    
    % Dimension scaling based on class
    cls = lower(t.class);
    switch cls
        case {'car', 'vehicle'}, dim_x = 4.5; dim_y = 2.0;
        case {'truck', 'bus'},   dim_x = 7.0; dim_y = 2.5;
        case {'person', 'pedestrian'}, dim_x = 1.0; dim_y = 1.0;
        case {'cow', 'animal', 'livestock'}, dim_x = 2.5; dim_y = 1.8;
        case {'rickshaw', 'auto_rickshaw', 'autorickshaw'}, dim_x = 2.8; dim_y = 1.5;
        case {'pushcart', 'cart', 'thela'}, dim_x = 2.2; dim_y = 1.5;
        otherwise, dim_x = 2.0; dim_y = 1.5;
    end
    
    x_min = max(0, grid_x - dim_x/2);
    x_max = min(grid_len, grid_x + dim_x/2);
    y_min = max(0, grid_y - dim_y/2);
    y_max = min(grid_width, grid_y + dim_y/2);
    
    [X_mesh, Y_mesh] = meshgrid(x_min:0.25:x_max, y_min:0.25:y_max);
    pts = [X_mesh(:), Y_mesh(:)];
    
    if ~isempty(pts)
        setOccupancy(map, pts, 1);
    end
end
end
