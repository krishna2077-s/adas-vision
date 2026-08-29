function [lidar_detections, obstacle_pts, ground_pts] = process_lidar_pointcloud(ptCloud, proc_params)
% PROCESS_LIDAR_POINTCLOUD Ground segmentation, ROI filtering & 3D clustering.
%
%   [lidar_detections, obstacle_pts, ground_pts] = PROCESS_LIDAR_POINTCLOUD(ptCloud, proc_params)
%
%   Inputs:
%       ptCloud     - Struct from simulate_lidar or pointCloud object
%       proc_params - (Optional) Struct with parameters:
%                     .ground_height_thresh - Height threshold for ground in ego frame (default: 0.15m)
%                     .cluster_dist_thresh  - Euclidean clustering threshold in m (default: 0.85m)
%                     .min_cluster_pts      - Minimum points to form valid cluster (default: 6)
%                     .roi_x                - Forward ROI in ego frame [min_x, max_x] (default: [-5, 55])
%                     .roi_y                - Lateral ROI in ego frame [min_y, max_y] (default: [-12, 12])
%                     .roi_z                - Vertical ROI in ego frame [min_z, max_z] (default: [-0.4, 3.5])
%
%   Outputs:
%       lidar_detections - Struct array of detected obstacle clusters with fields:
%                          .id           - Cluster ID
%                          .x, .y, .z    - Centroid coordinates in world frame
%                          .ego_x, .ego_y- Centroid in ego-centric frame
%                          .length       - Estimated bounding box length (m)
%                          .width        - Estimated bounding box width (m)
%                          .height       - Estimated bounding box height (m)
%                          .num_points   - Number of LiDAR points in cluster
%                          .points_world - [M x 3] points in this cluster (world)
%                          .points_ego   - [M x 3] points in this cluster (ego)
%                          .class_hint   - Inferred class ('pedestrian', 'vehicle', 'animal', etc.)
%       obstacle_pts     - [K x 3] Non-ground obstacle points in world frame
%       ground_pts       - [G x 3] Segmented ground points in world frame
%
% ADAS Vision — SIH 2026 Perception Pipeline

if nargin < 2 || isempty(proc_params)
    proc_params = struct();
end

% Default parameters
ground_thresh = 0.18; % Ground height in world/ego (Z < 0.18m is road surface)
cluster_dist  = 0.90; % Spatial cluster distance
min_pts       = 5;    % Minimum points per obstacle
roi_x         = [-4.0, 55.0];
roi_y         = [-15.0, 15.0];
roi_z         = [-0.2, 3.5];

if isfield(proc_params, 'ground_height_thresh'), ground_thresh = proc_params.ground_height_thresh; end
if isfield(proc_params, 'cluster_dist_thresh'),  cluster_dist  = proc_params.cluster_dist_thresh; end
if isfield(proc_params, 'min_cluster_pts'),      min_pts       = proc_params.min_cluster_pts; end
if isfield(proc_params, 'roi_x'),                roi_x         = proc_params.roi_x; end
if isfield(proc_params, 'roi_y'),                roi_y         = proc_params.roi_y; end
if isfield(proc_params, 'roi_z'),                roi_z         = proc_params.roi_z; end

if isempty(ptCloud) || ptCloud.Count == 0
    lidar_detections = [];
    obstacle_pts = zeros(0, 3);
    ground_pts = zeros(0, 3);
    return;
end

pts_world = ptCloud.Location;
pts_ego   = ptCloud.EgoFrame;

% ---------------------------------------------------------------------------
% 1. Region of Interest (ROI) Filtering in Ego Frame
% ---------------------------------------------------------------------------
in_roi = (pts_ego(:, 1) >= roi_x(1) & pts_ego(:, 1) <= roi_x(2)) & ...
         (pts_ego(:, 2) >= roi_y(1) & pts_ego(:, 2) <= roi_y(2)) & ...
         (pts_ego(:, 3) >= roi_z(1) & pts_ego(:, 3) <= roi_z(2));

pts_world = pts_world(in_roi, :);
pts_ego   = pts_ego(in_roi, :);

if isempty(pts_world)
    lidar_detections = [];
    obstacle_pts = zeros(0, 3);
    ground_pts = zeros(0, 3);
    return;
end

% ---------------------------------------------------------------------------
% 2. Ground Plane Segmentation
% ---------------------------------------------------------------------------
% Ground returns have Z near 0 (in world) or Z <= -1.6m (in sensor/body)
is_ground = pts_world(:, 3) <= ground_thresh;

ground_pts   = pts_world(is_ground, :);
obstacle_pts = pts_world(~is_ground, :);
obs_ego      = pts_ego(~is_ground, :);

if size(obstacle_pts, 1) < min_pts
    lidar_detections = [];
    return;
end

% ---------------------------------------------------------------------------
% 3. Euclidean Spatial Clustering (Vectorized & Fast)
% ---------------------------------------------------------------------------
N_obs = size(obstacle_pts, 1);
visited = false(N_obs, 1);
cluster_ids = zeros(N_obs, 1);
cur_cluster = 0;

% Use 2D (X, Y) projection for rapid spatial clustering
obs_xy = obstacle_pts(:, 1:2);

for i = 1:N_obs
    if visited(i), continue; end
    
    % Start new cluster
    cur_cluster = cur_cluster + 1;
    visited(i) = true;
    cluster_ids(i) = cur_cluster;
    
    % Breadth-first search / queue for neighbors
    queue = i;
    head = 1;
    
    while head <= numel(queue)
        curr = queue(head);
        head = head + 1;
        
        % Distances to unvisited points
        dists_sq = sum((obs_xy(~visited, :) - obs_xy(curr, :)).^2, 2);
        neighbor_unvisited_idx = find(~visited);
        new_neighbors = neighbor_unvisited_idx(dists_sq < (cluster_dist^2));
        
        if ~isempty(new_neighbors)
            visited(new_neighbors) = true;
            cluster_ids(new_neighbors) = cur_cluster;
            queue = [queue; new_neighbors]; %#ok<AGROW>
        end
    end
end

% ---------------------------------------------------------------------------
% 4. Extract 3D Bounding Boxes and Detections
% ---------------------------------------------------------------------------
detections = [];
det_count = 0;

for c = 1:cur_cluster
    idx = find(cluster_ids == c);
    if numel(idx) < min_pts, continue; end
    
    c_pts_world = obstacle_pts(idx, :);
    c_pts_ego   = obs_ego(idx, :);
    
    det_count = det_count + 1;
    
    % Centroid
    centroid_w = mean(c_pts_world, 1);
    centroid_e = mean(c_pts_ego, 1);
    
    % Bounding box extents
    min_w = min(c_pts_world, [], 1);
    max_w = max(c_pts_world, [], 1);
    
    dim_x = max(0.4, max_w(1) - min_w(1));
    dim_y = max(0.4, max_w(2) - min_w(2));
    dim_z = max(0.4, max_w(3) - min_w(3));
    
    % Geometric class classification heuristic from 3D size & aspect ratio
    footprint_area = dim_x * dim_y;
    vol = footprint_area * dim_z;
    
    if vol < 0.8 && dim_z > 1.1 && dim_x < 1.0 && dim_y < 1.0
        class_hint = 'pedestrian';
    elseif vol < 2.5 && dim_x > 1.2 && dim_y < 1.2
        class_hint = 'cattle';
    elseif dim_x > 2.0 && dim_x < 3.2 && dim_y > 1.0 && dim_y < 1.6
        class_hint = 'auto_rickshaw';
    elseif dim_x >= 3.5 || dim_y >= 1.8
        class_hint = 'vehicle';
    elseif dim_x < 1.5 && dim_y > 0.8
        class_hint = 'pushcart';
    else
        class_hint = 'obstacle';
    end
    
    det = struct();
    det.id           = det_count;
    det.x            = centroid_w(1);
    det.y            = centroid_w(2);
    det.z            = centroid_w(3);
    det.ego_x        = centroid_e(1);
    det.ego_y        = centroid_e(2);
    det.length       = dim_x;
    det.width        = dim_y;
    det.height       = dim_z;
    det.num_points   = numel(idx);
    det.points_world = c_pts_world;
    det.points_ego   = c_pts_ego;
    det.class_hint   = class_hint;
    
    detections = [detections, det]; %#ok<AGROW>
end

lidar_detections = detections;

end
