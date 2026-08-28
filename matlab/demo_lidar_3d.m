function demo_lidar_3d()
% DEMO_LIDAR_3D Interactive 3D LiDAR Simulation & Multi-Sensor Fusion Demo
%
%   Run this script in MATLAB to demonstrate:
%     1. 32-Beam 3D LiDAR Point Cloud Raycasting (Velodyne / Ouster style)
%     2. Real-Time Ground Plane Segmentation (Asphalt vs Obstacles)
%     3. 3D Spatial Clustering & Oriented Bounding Box Fitting
%     4. True Camera + LiDAR Multi-Sensor Bayesian Fusion
%     5. Live Occupancy Map Generation for Hybrid A* Planning
%
% ADAS Vision — SIH Grand Jury Demonstration Suite

fprintf('\n=========================================================\n');
fprintf('  ADAS Vision — 3D LiDAR Simulation & Fusion Demo\n');
fprintf('=========================================================\n');

% 1. Setup Ego Vehicle & Indian Road Environment
ego.x       = 0.0;
ego.y       = 0.0;
ego.heading = 0.0; % Facing East (+X)
ego.speed   = 8.33; % 30 km/h

road_cfg.lane_half_width = 3.5;
road_cfg.x_limits        = [-10, 70];

% 2. Setup Typical Indian Road Traffic Actors
actors = { ...
    struct('id', 'act_1', 'class', 'auto_rickshaw', 'x', 16.0, 'y',  0.8, 'vx', 4.0, 'vy', 0.0, 'length', 2.8, 'width', 1.3, 'height', 1.8), ...
    struct('id', 'act_2', 'class', 'cow',           'x', 26.0, 'y', -1.2, 'vx', 0.2, 'vy', 0.6, 'length', 2.2, 'width', 0.9, 'height', 1.4), ...
    struct('id', 'act_3', 'class', 'person',        'x', 12.0, 'y',  2.4, 'vx', 0.0, 'vy', -0.8, 'length', 0.5, 'width', 0.5, 'height', 1.7), ...
    struct('id', 'act_4', 'class', 'pushcart',      'x', 38.0, 'y',  2.2, 'vx', 1.2, 'vy', 0.0, 'length', 2.0, 'width', 1.1, 'height', 1.2), ...
    struct('id', 'act_5', 'class', 'truck',         'x', 48.0, 'y', -0.5, 'vx', 6.0, 'vy', 0.0, 'length', 8.5, 'width', 2.5, 'height', 3.0)  ...
};

% 3. Simulate 3D LiDAR Point Cloud
fprintf('[1/4] Simulating 32-Beam 3D LiDAR point cloud...\n');
lidar_params.num_channels = 32;
lidar_params.v_fov        = [-16, 14];
lidar_params.h_fov        = [-75, 75];
lidar_params.h_res_deg    = 0.5;
lidar_params.max_range    = 60.0;

t_start = tic;
ptCloud = simulate_lidar(ego, actors, road_cfg, lidar_params);
sim_ms = toc(t_start) * 1000;
fprintf('      Generated %d points in %.1f ms.\n', ptCloud.Count, sim_ms);

% 4. Process Point Cloud: Ground Segmentation & 3D Clustering
fprintf('[2/4] Segmenting ground plane & clustering 3D obstacles...\n');
t_proc = tic;
[lidar_dets, obs_pts, ground_pts] = process_lidar_pointcloud(ptCloud);
proc_ms = toc(t_proc) * 1000;
fprintf('      Ground Points: %d | Obstacle Points: %d | Detected Clusters: %d (in %.1f ms)\n', ...
        size(ground_pts, 1), size(obs_pts, 1), numel(lidar_dets), proc_ms);

% 5. Mock Camera Detections for Fusion
cam_objects = [ ...
    struct('class', 'auto_rickshaw', 'distance', 15.6, 'lateral_offset',  0.9, 'confidence', 0.92), ...
    struct('class', 'cow',           'distance', 26.5, 'lateral_offset', -1.1, 'confidence', 0.88), ...
    struct('class', 'person',        'distance', 11.8, 'lateral_offset',  2.5, 'confidence', 0.94)  ...
];

fprintf('[3/4] Fusing Camera Semantics with LiDAR 3D Spatial Geometry...\n');
[fused_tracks, fusion_metrics] = fuse_camera_lidar(cam_objects, lidar_dets);
fprintf('      Fused Tracks: %d (Fusion latency: %.2f ms)\n', ...
        numel(fused_tracks), fusion_metrics.latency_ms);

% 6. Build Occupancy Grid from LiDAR Non-Ground Points
fprintf('[4/4] Building Binary Occupancy Map from LiDAR returns...\n');
occ_map = build_occupancy_grid_from_lidar(obs_pts, ego);

% ---------------------------------------------------------------------------
% 7. Interactive Multi-Panel 3D Visualization
% ---------------------------------------------------------------------------
fig = figure('Name', 'ADAS Vision — 3D LiDAR & Sensor Fusion Dashboard', ...
             'NumberTitle', 'off', 'Color', [0.07 0.07 0.11], 'Position', [40 60 1200 680]);

% Panel 1: 3D Point Cloud View
ax1 = subplot(2, 2, [1 3], 'Parent', fig);
set(ax1, 'Color', [0.10 0.10 0.15], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', ...
    'GridColor', [0.3 0.3 0.4], 'GridAlpha', 0.5);
hold(ax1, 'on'); grid(ax1, 'on'); view(ax1, [-45, 28]);
axis(ax1, 'equal');
xlim(ax1, [-5, 60]); ylim(ax1, [-12, 12]); zlim(ax1, [-1, 4.5]);
xlabel(ax1, 'X: Forward (m)', 'Color', 'w');
ylabel(ax1, 'Y: Lateral (m)', 'Color', 'w');
zlabel(ax1, 'Z: Height (m)', 'Color', 'w');
title(ax1, '3D LiDAR Point Cloud — Ground Segmentation & 3D Bounding Boxes', ...
      'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% Draw Ground Points (Subsampled for speed & aesthetic blue-gray)
if ~isempty(ground_pts)
    g_step = max(1, round(size(ground_pts, 1) / 3000));
    scatter3(ax1, ground_pts(1:g_step:end, 1), ground_pts(1:g_step:end, 2), ground_pts(1:g_step:end, 3), ...
             3, [0.3 0.45 0.65], 'filled', 'MarkerFaceAlpha', 0.4);
end

% Draw Obstacle Points (Vivid Cyan/Orange by height)
if ~isempty(obs_pts)
    scatter3(ax1, obs_pts(:, 1), obs_pts(:, 2), obs_pts(:, 3), ...
             14, obs_pts(:, 3), 'filled');
    colormap(ax1, 'parula');
end

% Draw Ego Vehicle Box
draw_3d_box(ax1, ego.x + 1.5, ego.y, 0.75, 4.4, 1.8, 1.5, [0 1 0.4], 2.0);
text(ax1, ego.x - 2, ego.y, 2.2, 'EGO VEHICLE', 'Color', [0 1 0.4], 'FontWeight', 'bold', 'FontSize', 9);

% Draw 3D Bounding Boxes for Detected LiDAR Clusters
colors = lines(numel(lidar_dets));
for d = 1:numel(lidar_dets)
    det = lidar_dets(d);
    c_col = colors(d, :);
    draw_3d_box(ax1, det.x, det.y, det.z, det.length, det.width, det.height, c_col, 1.8);
    lbl = sprintf('#%d %s (%.1fm)', det.id, upper(det.class_hint), norm([det.ego_x, det.ego_y]));
    text(ax1, det.x, det.y, det.z + det.height/2 + 0.4, lbl, ...
         'Color', 'w', 'FontSize', 8, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.6]);
end

% Panel 2: Bird's Eye View (BEV) Sensor Fusion
ax2 = subplot(2, 2, 2, 'Parent', fig);
set(ax2, 'Color', [0.10 0.10 0.15], 'XColor', 'w', 'YColor', 'w', 'GridColor', [0.3 0.3 0.4]);
hold(ax2, 'on'); grid(ax2, 'on'); axis(ax2, 'equal');
xlim(ax2, [-5, 60]); ylim(ax2, [-10, 10]);
title(ax2, 'BEV Multi-Sensor Fusion (Camera Semantics + LiDAR Depth)', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax2, 'Longitudinal X (m)', 'Color', 'w'); ylabel(ax2, 'Lateral Y (m)', 'Color', 'w');

% Road edges
y_top = road_cfg.lane_half_width * ones(1, 100);
x_r = linspace(-5, 60, 100);
plot(ax2, x_r,  y_top, 'w--', 'LineWidth', 1.5);
plot(ax2, x_r, -y_top, 'w--', 'LineWidth', 1.5);

% Camera FOV Triangle (Green translucent)
cam_fov_x = [0, 50, 50, 0];
cam_fov_y = [0, 50*tan(deg2rad(30)), -50*tan(deg2rad(30)), 0];
patch(ax2, cam_fov_x, cam_fov_y, [0 0.8 0.4], 'FaceAlpha', 0.08, 'EdgeColor', [0 0.8 0.4], 'LineStyle', ':');

% Fused Track Markers
for f = 1:numel(fused_tracks)
    ft = fused_tracks(f);
    plot(ax2, ft.x, ft.y, 's', 'MarkerSize', 12, 'MarkerFaceColor', [1 0.3 0.3], 'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
    text(ax2, ft.x + 1, ft.y, sprintf('%s [%s]', ft.class, ft.source), ...
         'Color', [1 0.9 0.2], 'FontSize', 8, 'FontWeight', 'bold');
end
plot(ax2, ego.x, ego.y, 'o', 'MarkerSize', 12, 'MarkerFaceColor', [0 1 0.4], 'MarkerEdgeColor', 'w');

% Panel 3: Binary Occupancy Grid
ax3 = subplot(2, 2, 4, 'Parent', fig);
show(occ_map, 'Parent', ax3);
title(ax3, 'Real-Time Occupancy Map (Derived from LiDAR Point Cloud)', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');
set(ax3, 'XColor', 'w', 'YColor', 'w');

fprintf('\n✅ Demo rendering complete. Close figure window when finished.\n\n');

end

% ---------------------------------------------------------------------------
% Helper Function: Draw 3D Wireframe Bounding Box
% ---------------------------------------------------------------------------
function draw_3d_box(ax, cx, cy, cz, L, W, H, col, lw)
    dx = L / 2; dy = W / 2; dz = H / 2;
    % 8 vertices
    V = [
        cx - dx, cy - dy, cz - dz;
        cx + dx, cy - dy, cz - dz;
        cx + dx, cy + dy, cz - dz;
        cx - dx, cy + dy, cz - dz;
        cx - dx, cy - dy, cz + dz;
        cx + dx, cy - dy, cz + dz;
        cx + dx, cy + dy, cz + dz;
        cx - dx, cy + dy, cz + dz;
    ];
    % 12 edges
    edges = [
        1 2; 2 3; 3 4; 4 1; % Bottom
        5 6; 6 7; 7 8; 8 5; % Top
        1 5; 2 6; 3 7; 4 8  % Pillars
    ];
    for e = 1:size(edges, 1)
        plot3(ax, [V(edges(e, 1), 1), V(edges(e, 2), 1)], ...
                  [V(edges(e, 1), 2), V(edges(e, 2), 2)], ...
                  [V(edges(e, 1), 3), V(edges(e, 2), 3)], ...
                  '-', 'Color', col, 'LineWidth', lw);
    end
end
