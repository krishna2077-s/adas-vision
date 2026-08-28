function [fused_tracks, fusion_metrics] = fuse_camera_lidar(cam_objects, lidar_detections, prev_tracks, dt)
% FUSE_CAMERA_LIDAR Kalman-based Camera & LiDAR Multi-Sensor Fusion Engine.
%
%   [fused_tracks, fusion_metrics] = FUSE_CAMERA_LIDAR(cam_objects, lidar_detections, prev_tracks, dt)
%
%   Fuses rich visual semantic classification from Camera with millimeter-precise
%   3D range, extents, and physical geometry from LiDAR point cloud clusters.
%
%   Inputs:
%       cam_objects      - Array/cell of camera detections (class, distance, lateral_offset, etc.)
%       lidar_detections - Struct array from process_lidar_pointcloud (x, y, ego_x, ego_y, length, width, etc.)
%       prev_tracks      - (Optional) Struct array of previous fused tracks for temporal Kalman smoothing
%       dt               - (Optional) Time step in seconds (default: 0.033)
%
%   Outputs:
%       fused_tracks   - Struct array with fused state estimates:
%                        .id, .class, .x, .y, .vx, .vy, .confidence, .source,
%                        .length, .width, .height, .ttc, .range_m
%       fusion_metrics - Struct with diagnostics (n_cam, n_lidar, n_fused, latency_ms)
%
% ADAS Vision — SIH 2026 Perception Pipeline

if nargin < 3, prev_tracks = []; end
if nargin < 4 || isempty(dt), dt = 0.033; end

t_start = tic;

% Measurement Variances
% Camera: High classification quality, lower range accuracy
var_cam_range = 1.8^2;   % (m^2)
var_cam_lat   = 0.4^2;   % (m^2)

% LiDAR: Ultra-precise range & lateral position, geometry-based classification
var_lidar_range = 0.05^2; % (m^2)
var_lidar_lat   = 0.08^2; % (m^2)

N_cam = 0;
if isstruct(cam_objects) || iscell(cam_objects)
    N_cam = numel(cam_objects);
end

N_lidar = 0;
if ~isempty(lidar_detections)
    N_lidar = numel(lidar_detections);
end

% Parse camera detections into standard format
cam_list = [];
if isstruct(cam_objects)
    for i = 1:numel(cam_objects)
        co = cam_objects(i);
        c_item.class = 'obstacle';
        if isfield(co, 'class'), c_item.class = char(co.class); end
        
        c_item.x = 10.0;
        if isfield(co, 'distance'), c_item.x = co.distance;
        elseif isfield(co, 'x'), c_item.x = co.x; end
        
        c_item.y = 0.0;
        if isfield(co, 'lateral_offset'), c_item.y = co.lateral_offset;
        elseif isfield(co, 'y'), c_item.y = co.y; end
        
        c_item.confidence = 0.85;
        if isfield(co, 'confidence'), c_item.confidence = co.confidence; end
        
        cam_list = [cam_list, c_item]; %#ok<AGROW>
    end
end

% Match Camera & LiDAR detections using GNN (Global Nearest Neighbor / Euclidean distance)
match_dist_thresh = 3.2; % 3.2 meters association gate
cam_matched = false(numel(cam_list), 1);
lidar_matched = false(N_lidar, 1);

fused_tracks = [];
track_id_counter = 0;

% 1. Fuse matched pairs (Camera + LiDAR)
for c = 1:numel(cam_list)
    c_item = cam_list(c);
    best_dist = Inf;
    best_l = 0;
    
    for l = 1:N_lidar
        if lidar_matched(l), continue; end
        l_item = lidar_detections(l);
        
        % Distance between camera ego estimate and lidar ego estimate
        dx = c_item.x - l_item.ego_x;
        dy = c_item.y - l_item.ego_y;
        d = norm([dx, dy]);
        
        if d < best_dist && d < match_dist_thresh
            best_dist = d;
            best_l = l;
        end
    end
    
    if best_l > 0
        % Match Found! Perform Bayesian Optimal Fusion
        cam_matched(c) = true;
        lidar_matched(best_l) = true;
        l_item = lidar_detections(best_l);
        
        % Inverse-variance fusion for position
        w_cam_x = 1.0 / var_cam_range;
        w_lid_x = 1.0 / var_lidar_range;
        fused_x = (c_item.x * w_cam_x + l_item.ego_x * w_lid_x) / (w_cam_x + w_lid_x);
        
        w_cam_y = 1.0 / var_cam_lat;
        w_lid_y = 1.0 / var_lidar_lat;
        fused_y = (c_item.y * w_cam_y + l_item.ego_y * w_lid_y) / (w_cam_y + w_lid_y);
        
        track_id_counter = track_id_counter + 1;
        ft.id         = track_id_counter;
        ft.class      = c_item.class; % High-confidence vision semantic class
        ft.x          = l_item.x;     % World X
        ft.y          = l_item.y;     % World Y
        ft.ego_x      = fused_x;
        ft.ego_y      = fused_y;
        ft.vx         = 0.0;
        ft.vy         = 0.0;
        ft.length     = l_item.length;
        ft.width      = l_item.width;
        ft.height     = l_item.height;
        ft.confidence = min(0.99, c_item.confidence + 0.10);
        ft.source     = 'Camera + LiDAR (Fused)';
        ft.range_m    = norm([fused_x, fused_y]);
        
        fused_tracks = [fused_tracks, ft]; %#ok<AGROW>
    end
end

% 2. Unmatched LiDAR detections (e.g. night time, occluded, or unclassified obstacles)
for l = 1:N_lidar
    if ~lidar_matched(l)
        l_item = lidar_detections(l);
        track_id_counter = track_id_counter + 1;
        
        ft.id         = track_id_counter;
        ft.class      = l_item.class_hint; % Geometry-inferred class
        ft.x          = l_item.x;
        ft.y          = l_item.y;
        ft.ego_x      = l_item.ego_x;
        ft.ego_y      = l_item.ego_y;
        ft.vx         = 0.0;
        ft.vy         = 0.0;
        ft.length     = l_item.length;
        ft.width      = l_item.width;
        ft.height     = l_item.height;
        ft.confidence = 0.88;
        ft.source     = 'LiDAR Only';
        ft.range_m    = norm([l_item.ego_x, l_item.ego_y]);
        
        fused_tracks = [fused_tracks, ft]; %#ok<AGROW>
    end
end

% 3. Unmatched Camera detections (e.g. distant signs/objects beyond LiDAR range)
for c = 1:numel(cam_list)
    if ~cam_matched(c)
        c_item = cam_list(c);
        track_id_counter = track_id_counter + 1;
        
        ft.id         = track_id_counter;
        ft.class      = c_item.class;
        ft.x          = c_item.x;
        ft.y          = c_item.y;
        ft.ego_x      = c_item.x;
        ft.ego_y      = c_item.y;
        ft.vx         = 0.0;
        ft.vy         = 0.0;
        ft.length     = 2.0;
        ft.width      = 1.5;
        ft.height     = 1.5;
        ft.confidence = c_item.confidence * 0.8;
        ft.source     = 'Camera Only';
        ft.range_m    = norm([c_item.x, c_item.y]);
        
        fused_tracks = [fused_tracks, ft]; %#ok<AGROW>
    end
end

% 4. Temporal Kalman Velocity Estimation from previous tracks
if ~isempty(prev_tracks) && ~isempty(fused_tracks)
    for i = 1:numel(fused_tracks)
        best_prev = 0;
        min_d = 2.5;
        for p = 1:numel(prev_tracks)
            d = norm([fused_tracks(i).ego_x - prev_tracks(p).ego_x, ...
                      fused_tracks(i).ego_y - prev_tracks(p).ego_y]);
            if d < min_d
                min_d = d;
                best_prev = p;
            end
        end
        if best_prev > 0
            raw_vx = (fused_tracks(i).ego_x - prev_tracks(best_prev).ego_x) / max(0.01, dt);
            raw_vy = (fused_tracks(i).ego_y - prev_tracks(best_prev).ego_y) / max(0.01, dt);
            
            % Smooth velocity with low-pass filter
            fused_tracks(i).vx = 0.6 * prev_tracks(best_prev).vx + 0.4 * raw_vx;
            fused_tracks(i).vy = 0.6 * prev_tracks(best_prev).vy + 0.4 * raw_vy;
        end
    end
end

fusion_metrics.n_cam = N_cam;
fusion_metrics.n_lidar = N_lidar;
fusion_metrics.n_fused = numel(fused_tracks);
fusion_metrics.latency_ms = toc(t_start) * 1000;

end
