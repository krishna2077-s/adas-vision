function tracks = extract_rr_obstacles(actorPoses, egoPose)
% EXTRACT_RR_OBSTACLES Transforms RoadRunner ground-truth actor poses to ego frame tracks
%
% Inputs:
%   actorPoses - Array of pose structs from getActorPoses(rrSim)
%   egoPose    - Pose struct of Ego vehicle
%
% Output:
%   tracks     - Array of structs compatible with predict_trajectories and decision engine

tracks = struct('id', {}, 'class', {}, 'x', {}, 'y', {}, 'vx', {}, 'vy', {}, ...
                'distance_m', {}, 'closing_speed_mps', {}, 'ttc_s', {}, 'w', {}, 'h', {});

if isempty(actorPoses), return; end

ego_x = egoPose.Position(1);
ego_y = egoPose.Position(2);
ego_yaw = egoPose.Orientation(3);

ego_v = norm(egoPose.Velocity(1:2));

for k = 1:numel(actorPoses)
    pose = actorPoses(k);
    
    % Skip Ego itself
    if isfield(pose, 'ActorID') && isfield(egoPose, 'ActorID') && pose.ActorID == egoPose.ActorID
        continue;
    end
    
    dx = pose.Position(1) - ego_x;
    dy = pose.Position(2) - ego_y;
    
    % Transform position to Ego local frame
    long_dist =  dx * cos(ego_yaw) + dy * sin(ego_yaw);
    lat_offset = -dx * sin(ego_yaw) + dy * cos(ego_yaw);
    
    % Ignore objects behind ego or too far away (>60m)
    if long_dist < 0.5 || long_dist > 60.0, continue; end
    
    % Velocity transformation
    vx_w = pose.Velocity(1);
    vy_w = pose.Velocity(2);
    vx_ego =  vx_w * cos(ego_yaw) + vy_w * sin(ego_yaw);
    vy_ego = -vx_w * sin(ego_yaw) + vy_w * cos(ego_yaw);
    
    closing_speed = ego_v - vx_ego;
    ttc = inf;
    if closing_speed > 0.5
        ttc = long_dist / closing_speed;
    end
    
    cls_name = 'car';
    if isfield(pose, 'Class') && ~isempty(pose.Class)
        cls_name = lower(char(pose.Class));
    end
    
    t = struct();
    t.id = k;
    t.class = cls_name;
    t.x = long_dist;
    t.y = lat_offset;
    t.vx = vx_ego;
    t.vy = vy_ego;
    t.distance_m = sqrt(long_dist^2 + lat_offset^2);
    t.closing_speed_mps = closing_speed;
    t.ttc_s = ttc;
    t.w = 1.8;
    t.h = 4.0;
    
    tracks(end+1) = t; %#ok<AGROW>
end
end
