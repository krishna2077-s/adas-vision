function [pred_x, pred_y, pred_sizes] = predict_trajectories(tracks_struct, dt_horizon, num_steps)
% PREDICT_TRAJECTORIES Predicts future trajectories of tracked objects
%   [pred_x, pred_y, pred_sizes] = predict_trajectories(tracks_struct, dt_horizon, num_steps)
%
%   Inputs:
%       tracks_struct: array of structs with fields: track_id, x, y, vx, vy, class
%       dt_horizon: prediction horizon in seconds (default 3.0)
%       num_steps: number of prediction steps (default 15)
%
%   Outputs:
%       pred_x: cell array of Nx1 x-coordinates for predictions
%       pred_y: cell array of Nx1 y-coordinates for predictions
%       pred_sizes: cell array of [width, height] sizes considering uncertainty

    if nargin < 2
        dt_horizon = 3.0;
    end
    if nargin < 3
        num_steps = 15;
    end

    dt = dt_horizon / num_steps;
    time_steps = (1:num_steps)' * dt;
    
    num_tracks = length(tracks_struct);
    
    pred_x = cell(num_tracks, 1);
    pred_y = cell(num_tracks, 1);
    pred_sizes = cell(num_tracks, 1);
    
    % Social Force model inspiration:
    % Pedestrians and animals have different lateral uncertainties due to their
    % unpredictable nature compared to constrained vehicles.
    
    for i = 1:num_tracks
        track = tracks_struct(i);
        
        % Roll forward with constant velocity
        x_pred = track.x + track.vx * time_steps;
        y_pred = track.y + track.vy * time_steps;
        
        pred_x{i} = x_pred;
        pred_y{i} = y_pred;
        
        % Apply class-dependent lateral uncertainty inflation
        switch lower(track.class)
            case 'person'
                lateral_inflation = 1.5;
            case {'cow', 'dog', 'cat'}
                lateral_inflation = 2.0; % Animals are unpredictable
            case {'auto_rickshaw', 'rickshaw', 'autorickshaw'}
                lateral_inflation = 1.2; % Agile Indian 3-wheelers
            case {'pushcart', 'thela'}
                lateral_inflation = 0.8; % Street vendors/pushcarts
            case {'bicycle', 'motorcycle'}
                lateral_inflation = 1.0;
            case {'car', 'truck', 'bus'}
                lateral_inflation = 0.5;
            otherwise
                lateral_inflation = 1.0;
        end
        
        % Base size assumptions if not provided, just using inflation for now
        % Assume base width/height of 1m if track doesn't specify size
        base_w = 1.0; 
        base_h = 1.0;
        if isfield(track, 'w')
            base_w = track.w;
        end
        if isfield(track, 'h')
            base_h = track.h;
        end
        
        % Inflated sizes (treating uncertainty as a bounding box inflation)
        inflated_w = base_w + 2 * lateral_inflation;
        inflated_h = base_h + 2 * lateral_inflation;
        
        pred_sizes{i} = [inflated_w, inflated_h];
    end
end
