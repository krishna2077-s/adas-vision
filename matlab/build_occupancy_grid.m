function map = build_occupancy_grid(packet)
% BUILD_OCCUPANCY_GRID Creates a binary occupancy map from ADAS perception & LiDAR data.
%
%   MAP = build_occupancy_grid(PACKET) takes a parsed JSON struct or perception
%   packet and builds a 50m x 30m binaryOccupancyMap (resolution 0.25m).
%   It stamps road limits, multi-sensor fused obstacle bounding boxes, and raw LiDAR hits.
%
%   Inputs:
%       packet - A MATLAB struct containing parsed ADAS perception data.
%                Fields: 'objects', 'road', and optionally 'lidar_points' or 'lidar_clusters'.
%
%   Outputs:
%       map - A binaryOccupancyMap object.
%
% ADAS Vision — SIH 2026 Perception Pipeline

    % Define map parameters
    width = 30;     % meters (lateral width, 30m total)
    length_m = 50;  % meters (longitudinal distance, 50m total)
    resolution = 4; % cells per meter (0.25m resolution)
    
    % Initialize map
    map = binaryOccupancyMap(length_m, width, resolution);
    
    % Ego vehicle is typically placed at (X=0, Y=width/2) in the grid so Y ranges 
    % effectively from -15 to +15. Shift by +15m.
    y_offset = width / 2;
    
    % 1. Mark road boundaries
    lane_width = 3.5;
    if isfield(packet, 'road') && ~isempty(packet.road)
        x_pts = 0:0.5:length_m;
        y_left = (lane_width/2 + y_offset) * ones(size(x_pts));
        y_right = (-lane_width/2 + y_offset) * ones(size(x_pts));
        
        setOccupancy(map, [x_pts', y_left'], 1);
        setOccupancy(map, [x_pts', y_right'], 1);
    end
    
    % 2. Stamp raw LiDAR obstacle points if provided
    if isfield(packet, 'lidar_points') && ~isempty(packet.lidar_points)
        l_pts = packet.lidar_points;
        % Transform to grid coords
        gx = l_pts(:, 1);
        gy = l_pts(:, 2) + y_offset;
        in_grid = (gx >= 0 & gx <= length_m) & (gy >= 0 & gy <= width);
        if any(in_grid)
            setOccupancy(map, [gx(in_grid), gy(in_grid)], 1);
        end
    end
    
    % 3. Stamp tracked & classified objects with Indian road dimensions & safety buffers
    if isfield(packet, 'objects')
        for i = 1:numel(packet.objects)
            obj = packet.objects(i);
            
            if isfield(obj, 'distance')
                obj_x = obj.distance;
            elseif isfield(obj, 'x')
                obj_x = obj.x;
            else
                continue;
            end
            
            if isfield(obj, 'lateral_offset')
                obj_y = obj.lateral_offset;
            elseif isfield(obj, 'y')
                obj_y = obj.y;
            else
                obj_y = 0;
            end
            
            grid_x = obj_x;
            grid_y = obj_y + y_offset;
            
            if grid_x < 0 || grid_x > length_m || grid_y < 0 || grid_y > width
                continue;
            end
            
            cls = lower(string(obj.class));
            switch cls
                case {'car'}
                    dim_x = 4.6; dim_y = 2.0;
                case {'truck', 'bus'}
                    dim_x = 8.5; dim_y = 2.6;
                case {'auto_rickshaw', 'rickshaw', 'auto'}
                    dim_x = 3.0; dim_y = 1.4;
                case {'pushcart', 'thela'}
                    dim_x = 2.2; dim_y = 1.2;
                case {'motorcycle', 'bicycle', 'bike'}
                    dim_x = 2.0; dim_y = 0.8;
                case {'person', 'pedestrian'}
                    dim_x = 0.8; dim_y = 0.8;
                case {'cow', 'cattle', 'animal', 'dog'}
                    dim_x = 2.6; dim_y = 1.8; % Wider uncertainty margin for animals
                otherwise
                    dim_x = 1.4; dim_y = 1.4;
            end
            
            x_min = max(0, grid_x - dim_x/2);
            x_max = min(length_m, grid_x + dim_x/2);
            y_min = max(0, grid_y - dim_y/2);
            y_max = min(width, grid_y + dim_y/2);
            
            [X_mesh, Y_mesh] = meshgrid(x_min:0.25:x_max, y_min:0.25:y_max);
            pts = [X_mesh(:), Y_mesh(:)];
            
            if ~isempty(pts)
                setOccupancy(map, pts, 1);
            end
        end
    end
end
