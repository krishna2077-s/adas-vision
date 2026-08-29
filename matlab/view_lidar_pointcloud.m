% view_lidar_pointcloud.m
% ADAS Vision — Real 3D Automotive LiDAR Point Cloud Visualizer
%
% Visualizes real automotive 3D LiDAR point clouds (Velodyne HDL-64 & PandaSet)
% demonstrating true multi-sensor 3D perception.
%
% Usage:
%   >> view_lidar_pointcloud

lidar_dir = fullfile(fileparts(pwd), 'data', 'lidar');
if ~exist(lidar_dir, 'dir')
    lidar_dir = pwd;
end

pcd_files = {'HDL64LidarData.pcd', 'highwayScene.pcd', 'PandasetLidarData.pcd'};

fprintf('=================================================================\n');
fprintf('       ADAS Vision — Real 3D Automotive LiDAR Point Cloud       \n');
fprintf('=================================================================\n\n');

% Load Velodyne HDL-64 LiDAR Point Cloud
pcd_path = fullfile(lidar_dir, pcd_files{1});
if ~exist(pcd_path, 'file')
    pcd_path = which(pcd_files{1});
end

if isempty(pcd_path) || ~exist(pcd_path, 'file')
    fprintf('[ERROR] PCD file not found in %s\n', lidar_dir);
    return;
end

fprintf('Loading LiDAR point cloud: %s ...\n', pcd_files{1});
ptCloud = pcread(pcd_path);

fprintf('  Total Laser Points: %d\n', ptCloud.Count);
fprintf('  X Bounds (Longitudinal): [%.1f, %.1f] m\n', ptCloud.XLimits(1), ptCloud.XLimits(2));
fprintf('  Y Bounds (Lateral):      [%.1f, %.1f] m\n', ptCloud.YLimits(1), ptCloud.YLimits(2));
fprintf('  Z Bounds (Elevation):    [%.1f, %.1f] m\n\n', ptCloud.ZLimits(1), ptCloud.ZLimits(2));

% Create 3D Interactive Viewer
fig = figure('Name', 'ADAS Vision — 3D Velodyne LiDAR Point Cloud Scan', ...
             'Color', [0.05 0.05 0.08], 'Position', [100 100 960 620]);
ax = axes('Parent', fig, 'Color', [0.08 0.08 0.12], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

% Color points by height (Z-elevation) or intensity
pcshow(ptCloud, 'VerticalAxis', 'Z', 'VerticalAxisDir', 'Up', 'Parent', ax);
colormap(ax, 'jet');
colorbar(ax, 'Color', 'w');

title(ax, sprintf('Velodyne HDL-64 Automotive 3D LiDAR Scan (%d Laser Returns)', ptCloud.Count), ...
      'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');
xlabel(ax, 'X (Forward, meters)', 'Color', 'w');
ylabel(ax, 'Y (Lateral, meters)', 'Color', 'w');
zlabel(ax, 'Z (Elevation, meters)', 'Color', 'w');
grid(ax, 'on');

% Set viewing perspective (Bird''s-Eye 3D Angle)
view(ax, [-45 35]);

fprintf('✅ 3D LiDAR Visualizer launched! You can click and drag the mouse in the window to rotate in 3D.\n');
