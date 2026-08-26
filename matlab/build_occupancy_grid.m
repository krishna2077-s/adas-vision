function map = build_occupancy_grid(packet)
% BUILD_OCCUPANCY_GRID Creates a binary occupancy map from ADAS perception data.
%
%   MAP = build_occupancy_grid(PACKET) takes a parsed JSON struct (PACKET) from 
%   the UDP receiver and builds a 50m x 30m binaryOccupancyMap (resolution 0.25m).
%   It marks road boundaries and tracked objects as occupied cells.
%
%   Inputs:
%       packet - A MATLAB struct containing parsed ADAS perception data.
%                Expected to have 'objects' and optionally 'road' fields.
%                For objects, it should contain 'class', 'distance', and 'lateral_offset'.
%
%   Outputs:
%       map - A binaryOccupancyMap object.

    % Define map parameters
    width = 30;  % meters (lateral width, 30m total)
    length_m = 50; % meters (longitudinal distance, 50m total)
    resolution = 4; % cells per meter (0.25m resolution)
    
    % Initialize map
    map = binaryOccupancyMap(length_m, width, resolution);
    
    % Ego vehicle is typically placed at (X=0, Y=width/2) in the grid so Y ranges 
    % effectively from -15 to +15. Shift by +15m.
    y_offset = width / 2;
    
    % 1. Mark road boundaries
    % We assume roughly 3.5m lane width for demonstration. Marking sides as non-drivable.
    lane_width = 3.5;
    if isfield(packet, 'road') && ~isempty(packet.road)
        x_pts = 0:0.5:length_m;
        y_left = (lane_width/2 + y_offset) * ones(size(x_pts));
        y_right = (-lane_width/2 + y_offset) * ones(size(x_pts));
        
        % Set occupied cells for boundaries
        setOccupancy(map, [x_pts', y_left'], 1);
        setOccupancy(map, [x_pts', y_right'], 1);
    end
    
    % 2. Stamp tracked objects
    if isfield(packet, 'objects')
        for i = 1:numel(packet.objects)
            obj = packet.objects(i);
            
            % Monocular distance (longitudinal X) + lateral offset (Y)
            if isfield(obj, 'distance')
                obj_x = obj.distance;
            else
                continue; % skip if no distance
            end
            
            if isfield(obj, 'lateral_offset')
                obj_y = obj.lateral_offset;
            else
                obj_y = 0; % default straight ahead
            end
            
            % Map to occupancy grid coordinates
            grid_x = obj_x;
            grid_y = obj_y + y_offset;
            
            % Skip if out of map bounds
            if grid_x < 0 || grid_x > length_m || grid_y < 0 || grid_y > width
                continue;
            end
            
            % Determine dimensions based on class
            cls = lower(string(obj.class));
            switch cls
                case {'car', 'truck', 'bus'}
                    dim_x = 4.5;
                    dim_y = 2.0;
                case {'motorcycle', 'bicycle'}
                    dim_x = 2.0;
                    dim_y = 0.8;
                case 'person'
                    dim_x = 0.5;
                    dim_y = 0.5;
                case 'cow'
                    dim_x = 2.0;
                    dim_y = 1.5;
                otherwise
                    dim_x = 1.0;
                    dim_y = 1.0;
            end
            
            % Calculate corners of the occupied rectangle
            x_min = grid_x - dim_x/2;
            x_max = grid_x + dim_x/2;
            y_min = grid_y - dim_y/2;
            y_max = grid_y + dim_y/2;
            
            % Clamp to grid limits
            x_min = max(0, x_min);
            x_max = min(length_m, x_max);
            y_min = max(0, y_min);
            y_max = min(width, y_max);
            
            % Set grid points within bounding box to 1 (occupied)
            [X_mesh, Y_mesh] = meshgrid(x_min:0.25:x_max, y_min:0.25:y_max);
            pts = [X_mesh(:), Y_mesh(:)];
            
            if ~isempty(pts)
                setOccupancy(map, pts, 1);
            end
        end
    end
end
