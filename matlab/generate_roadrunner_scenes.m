% generate_roadrunner_scenes.m
% ADAS Vision — RoadRunner Scene Generator
%
% Creates two detailed driving scenarios using MATLAB's drivingScenario object
% (compatible with Automated Driving Toolbox) and exports them as:
%   1. MATLAB .mat scenario files
%   2. OpenDRIVE (.xodr) XML files — importable into RoadRunner
%
% Scenarios:
%   1. Indian Village Road — narrow, winding, unmarked carriageway
%   2. Urban Unsignalised Intersection — busy 4-way junction, Chandni Chowk style
%
% Usage:
%   >> generate_roadrunner_scenes
%   >> generate_roadrunner_scenes('village')
%   >> generate_roadrunner_scenes('intersection')

function generate_roadrunner_scenes(which_scene)
if nargin < 1, which_scene = 'both'; end

fprintf('=== ADAS Vision — RoadRunner Scene Generator ===\n\n');

out_dir = fullfile(fileparts(mfilename('fullpath')), 'roadrunner');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

if strcmpi(which_scene, 'both') || strcmpi(which_scene, 'village')
    build_village_road_scene(out_dir);
end
if strcmpi(which_scene, 'both') || strcmpi(which_scene, 'intersection')
    build_urban_intersection_scene(out_dir);
end

fprintf('\n✅ Scene files saved to: %s\n', out_dir);
fprintf('   To open in RoadRunner: File → Import → OpenDRIVE...\n');
fprintf('   To use in MATLAB:      load the .mat files directly.\n\n');
end


% =============================================================================
% Scene 1: Indian Village Road
% =============================================================================
function build_village_road_scene(out_dir)
fprintf('[1/2] Building Village Road scene...\n');

% ── Try using Automated Driving Toolbox drivingScenario ──────────────────
adt_ok = ~isempty(ver('Automated Driving Toolbox'));

if adt_ok
    scenario = drivingScenario('SampleTime', 0.033, 'StopTime', 35);

    % --- Road geometry: narrow winding village road ---
    % Main road: 200m long, 6.5m wide (one lane each direction, no markings)
    roadCenters = [
         0,   0,  0.0;
        30,   0.5, 0.0;
        60,   1.2, 0.0;
        90,   0.4, 0.0;
       120,  -0.8, 0.0;
       150,  -0.3, 0.0;
       180,   0.2, 0.0;
       210,   0.0, 0.0;
    ];
    laneWidth = 3.0;  % 3m each direction = 6m total
    road(scenario, roadCenters, laneWidth * 2, ...
        lanespec([1, 1], 'Width', [laneWidth, laneWidth]));

    % --- Actors ---
    % Ego vehicle: Maruti Suzuki Swift Dzire
    ego = vehicle(scenario, ...
        'ClassID',      1, ...
        'Length',       4.40, ...
        'Width',        1.73, ...
        'Height',       1.50, ...
        'PlotColor',    [0, 0.8, 0.3], ...
        'Name',         'EgoVehicle');
    trajectory(ego, [0 0 0], 0);

    % Motorcycle: oncoming at 30 km/h
    mcycle = vehicle(scenario, ...
        'ClassID',      2, ...
        'Length',       2.0, ...
        'Width',        0.8, ...
        'Height',       1.2, ...
        'PlotColor',    [0.2, 0.7, 1.0], ...
        'Name',         'Motorcycle_Oncoming');
    mcy_wp = [ 200,  1.0, 0;
               150,  1.2, 0;
               100,  0.8, 0;
                50,  0.5, 0 ];
    trajectory(mcycle, mcy_wp, [8.33, 8.33, 8.33, 8.33]);

    % Pedestrian crossing
    ped = vehicle(scenario, ...
        'ClassID',      4, ...
        'Length',       0.5, ...
        'Width',        0.5, ...
        'Height',       1.7, ...
        'PlotColor',    [1.0, 0.4, 0.4], ...
        'Name',         'Pedestrian1');
    ped_wp = [85, -3.5, 0; 85, 3.5, 0];
    trajectory(ped, ped_wp, [0.7, 0.7]);

    % Pushcart (stationary)
    pushcart = vehicle(scenario, ...
        'ClassID',      3, ...
        'Length',       2.2, ...
        'Width',        1.4, ...
        'Height',       1.0, ...
        'PlotColor',    [0.7, 0.5, 0.3], ...
        'Name',         'Pushcart');
    trajectory(pushcart, [130, 2.0, 0], 0);

    % Save .mat scenario
    mat_file = fullfile(out_dir, 'village_road_scene.mat');
    save(mat_file, 'scenario');
    fprintf('  ✓ Saved: %s\n', mat_file);

    % Export OpenDRIVE
    xodr_file = fullfile(out_dir, 'village_road.xodr');
    export_village_xodr(xodr_file, roadCenters, laneWidth);
    fprintf('  ✓ Saved: %s\n', xodr_file);

    % Launch visualisation
    plot(scenario, 'Waypoints', 'on');
    title('ADAS Scene 1 — Indian Village Road (RoadRunner Preview)', ...
          'Color', 'w');
    set(gcf, 'Color', [0.08 0.08 0.12]);
else
    % Fallback: no toolbox — generate OpenDRIVE XML only
    xodr_file = fullfile(out_dir, 'village_road.xodr');
    export_village_xodr(xodr_file, ...
        [0,0,0; 60,1.2,0; 120,-0.8,0; 210,0,0], 3.0);
    fprintf('  ✓ OpenDRIVE saved: %s\n', xodr_file);
    fprintf('  [INFO] drivingScenario skipped (Automated Driving Toolbox not found)\n');
end
end


% =============================================================================
% Scene 2: Urban Unsignalised Intersection
% =============================================================================
function build_urban_intersection_scene(out_dir)
fprintf('[2/2] Building Urban Intersection scene...\n');

adt_ok = ~isempty(ver('Automated Driving Toolbox'));

if adt_ok
    scenario = drivingScenario('SampleTime', 0.033, 'StopTime', 40);

    % Main road (East-West, 80m)
    road(scenario, [-40 0 0; 40 0 0], 8.0, ...
        lanespec([2, 2], 'Width', [2, 2, 2, 2]));

    % Cross road (North-South, 60m)
    road(scenario, [0 -30 0; 0 30 0], 7.0, ...
        lanespec([1, 1], 'Width', [3.5, 3.5]));

    % Ego vehicle
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoVehicle', ...
        'PlotColor', [0 0.8 0.3]);
    trajectory(ego, [-35 -1 0; -5 -1 0; 30 2 0], [8.33, 6.0, 8.33]);

    % Auto-rickshaw crossing at intersection
    auto_rick = vehicle(scenario, ...
        'ClassID',   3, ...
        'Length',    3.5, ...
        'Width',     1.5, ...
        'Height',    1.7, ...
        'PlotColor', [1.0, 0.85, 0.1], ...
        'Name',      'AutoRickshaw');
    trajectory(auto_rick, [0 25 0; 0 -5 0], [4.16, 4.16]);

    % Two pedestrians crossing
    for p_idx = 1:2
        ped_p = vehicle(scenario, 'ClassID', 4, ...
            'Length', 0.5, 'Width', 0.5, 'Height', 1.7, ...
            'PlotColor', [1.0, 0.4, 0.4], ...
            'Name', sprintf('Pedestrian%d', p_idx));
        if p_idx == 1
            trajectory(ped_p, [-3 + p_idx*1.2, -8, 0; -3 + p_idx*1.2, 8, 0], [0.8, 0.8]);
        else
            trajectory(ped_p, [-3 + p_idx*1.2, 8, 0; -3 + p_idx*1.2, -8, 0], [0.9, 0.9]);
        end
    end

    % Scooter from East at speed
    scooter = vehicle(scenario, ...
        'ClassID',   2, 'Length', 1.8, 'Width', 0.8, ...
        'PlotColor', [0.2, 0.7, 1.0], 'Name', 'Scooter_EastBound');
    trajectory(scooter, [35 1 0; -20 1 0], [8.33, 8.33]);

    % Save
    mat_file = fullfile(out_dir, 'urban_intersection_scene.mat');
    save(mat_file, 'scenario');
    fprintf('  ✓ Saved: %s\n', mat_file);

    xodr_file = fullfile(out_dir, 'urban_intersection.xodr');
    export_intersection_xodr(xodr_file);
    fprintf('  ✓ Saved: %s\n', xodr_file);

    figure;
    plot(scenario, 'Waypoints', 'on');
    title('ADAS Scene 2 — Urban Unsignalised Intersection (RoadRunner Preview)', ...
          'Color', 'w');
    set(gcf, 'Color', [0.08 0.08 0.12]);
else
    xodr_file = fullfile(out_dir, 'urban_intersection.xodr');
    export_intersection_xodr(xodr_file);
    fprintf('  ✓ OpenDRIVE saved: %s\n', xodr_file);
    fprintf('  [INFO] drivingScenario skipped (Automated Driving Toolbox not found)\n');
end
end


% =============================================================================
% OpenDRIVE XML Exporters
% =============================================================================
function export_village_xodr(fname, road_centers, lane_width)
% Generate minimal OpenDRIVE 1.7 file for village road

fid = fopen(fname, 'w');
if fid < 0
    warning('Cannot create file: %s', fname);
    return;
end

% Compute total road length approximation
total_len = sum(sqrt(sum(diff(road_centers(:,1:2)).^2, 2)));

fprintf(fid, '<?xml version="1.0" encoding="UTF-8"?>\n');
fprintf(fid, '<OpenDRIVE xmlns="http://www.opendrive.org" rev_major="1" rev_minor="7">\n');
fprintf(fid, '  <header revMajor="1" revMinor="7" name="IndianVillageRoad" version="1.0"\n');
fprintf(fid, '          date="2026-08-29" north="0" south="0" east="0" west="0"\n');
fprintf(fid, '          vendor="ADASVision"/>\n\n');

fprintf(fid, '  <road name="VillageRoad_Unmarked" length="%.2f" id="1" junction="-1">\n', total_len);
fprintf(fid, '    <link>\n');
fprintf(fid, '      <predecessor elementType="road" elementId="-1"/>\n');
fprintf(fid, '      <successor elementType="road" elementId="-1"/>\n');
fprintf(fid, '    </link>\n');
fprintf(fid, '    <type s="0.0" type="rural"/>\n');
fprintf(fid, '    <planView>\n');
fprintf(fid, '      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="%.2f">\n', total_len);
fprintf(fid, '        <line/>\n');
fprintf(fid, '      </geometry>\n');
fprintf(fid, '    </planView>\n');
fprintf(fid, '    <elevationProfile>\n');
fprintf(fid, '      <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>\n');
fprintf(fid, '    </elevationProfile>\n');
fprintf(fid, '    <lateralProfile/>\n');
fprintf(fid, '    <lanes>\n');
fprintf(fid, '      <laneSection s="0.0">\n');
fprintf(fid, '        <left>\n');
fprintf(fid, '          <lane id="1" type="driving" level="false">\n');
fprintf(fid, '            <width sOffset="0.0" a="%.2f" b="0" c="0" d="0"/>\n', lane_width);
fprintf(fid, '            <roadMark sOffset="0.0" type="none" weight="standard" color="white"/>\n');
fprintf(fid, '          </lane>\n');
fprintf(fid, '        </left>\n');
fprintf(fid, '        <center>\n');
fprintf(fid, '          <lane id="0" type="none" level="false">\n');
fprintf(fid, '            <roadMark sOffset="0.0" type="none" weight="standard" color="white"/>\n');
fprintf(fid, '          </lane>\n');
fprintf(fid, '        </center>\n');
fprintf(fid, '        <right>\n');
fprintf(fid, '          <lane id="-1" type="driving" level="false">\n');
fprintf(fid, '            <width sOffset="0.0" a="%.2f" b="0" c="0" d="0"/>\n', lane_width);
fprintf(fid, '            <roadMark sOffset="0.0" type="none" weight="standard" color="white"/>\n');
fprintf(fid, '          </lane>\n');
fprintf(fid, '        </right>\n');
fprintf(fid, '      </laneSection>\n');
fprintf(fid, '    </lanes>\n');
fprintf(fid, '    <objects/>\n');
fprintf(fid, '    <signals/>\n');
fprintf(fid, '  </road>\n\n');
fprintf(fid, '</OpenDRIVE>\n');
fclose(fid);
end


function export_intersection_xodr(fname)
fid = fopen(fname, 'w');
if fid < 0
    warning('Cannot create file: %s', fname);
    return;
end

fprintf(fid, '<?xml version="1.0" encoding="UTF-8"?>\n');
fprintf(fid, '<OpenDRIVE xmlns="http://www.opendrive.org" rev_major="1" rev_minor="7">\n');
fprintf(fid, '  <header revMajor="1" revMinor="7" name="IndianUrbanIntersection" version="1.0"\n');
fprintf(fid, '          date="2026-08-29" north="0" south="0" east="0" west="0"\n');
fprintf(fid, '          vendor="ADASVision"/>\n\n');

% East approach (road 1)
fprintf(fid, '  <road name="EastApproach" length="35.0" id="1" junction="10">\n');
fprintf(fid, '    <link><successor elementType="junction" elementId="10"/></link>\n');
fprintf(fid, '    <planView>\n');
fprintf(fid, '      <geometry s="0.0" x="-35.0" y="0.0" hdg="0.0" length="35.0">\n');
fprintf(fid, '        <line/></geometry>\n');
fprintf(fid, '    </planView>\n');
fprintf(fid, '    <elevationProfile><elevation s="0.0" a="0" b="0" c="0" d="0"/></elevationProfile>\n');
fprintf(fid, '    <lateralProfile/>\n');
fprintf(fid, '    <lanes>\n');
fprintf(fid, '      <laneSection s="0.0">\n');
fprintf(fid, '        <left>\n');
fprintf(fid, '          <lane id="1" type="driving" level="false">\n');
fprintf(fid, '            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>\n');
fprintf(fid, '            <roadMark sOffset="0.0" type="broken" weight="standard" color="white" laneChange="both"/>\n');
fprintf(fid, '          </lane>\n');
fprintf(fid, '        </left>\n');
fprintf(fid, '        <center><lane id="0" type="none" level="false"><roadMark sOffset="0.0" type="solid" weight="standard" color="white"/></lane></center>\n');
fprintf(fid, '        <right>\n');
fprintf(fid, '          <lane id="-1" type="driving" level="false">\n');
fprintf(fid, '            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>\n');
fprintf(fid, '            <roadMark sOffset="0.0" type="broken" weight="standard" color="white" laneChange="both"/>\n');
fprintf(fid, '          </lane>\n');
fprintf(fid, '        </right>\n');
fprintf(fid, '      </laneSection>\n');
fprintf(fid, '    </lanes>\n');
fprintf(fid, '  </road>\n\n');

% North approach (road 2)
fprintf(fid, '  <road name="NorthApproach" length="25.0" id="2" junction="10">\n');
fprintf(fid, '    <link><successor elementType="junction" elementId="10"/></link>\n');
fprintf(fid, '    <planView>\n');
fprintf(fid, '      <geometry s="0.0" x="0.0" y="-25.0" hdg="1.5708" length="25.0">\n');
fprintf(fid, '        <line/></geometry>\n');
fprintf(fid, '    </planView>\n');
fprintf(fid, '    <elevationProfile><elevation s="0.0" a="0" b="0" c="0" d="0"/></elevationProfile>\n');
fprintf(fid, '    <lateralProfile/>\n');
fprintf(fid, '    <lanes>\n');
fprintf(fid, '      <laneSection s="0.0">\n');
fprintf(fid, '        <left><lane id="1" type="driving" level="false"><width sOffset="0.0" a="3.0" b="0" c="0" d="0"/></lane></left>\n');
fprintf(fid, '        <center><lane id="0" type="none" level="false"><roadMark sOffset="0.0" type="solid" weight="standard" color="white"/></lane></center>\n');
fprintf(fid, '        <right><lane id="-1" type="driving" level="false"><width sOffset="0.0" a="3.0" b="0" c="0" d="0"/></lane></right>\n');
fprintf(fid, '      </laneSection>\n');
fprintf(fid, '    </lanes>\n');
fprintf(fid, '  </road>\n\n');

% Junction box
fprintf(fid, '  <junction name="UrbanIntersectionJunction" id="10">\n');
fprintf(fid, '    <connection id="1" incomingRoad="1" connectingRoad="2" contactPoint="start">\n');
fprintf(fid, '      <laneLink from="-1" to="1"/>\n');
fprintf(fid, '    </connection>\n');
fprintf(fid, '    <connection id="2" incomingRoad="2" connectingRoad="1" contactPoint="start">\n');
fprintf(fid, '      <laneLink from="-1" to="1"/>\n');
fprintf(fid, '    </connection>\n');
fprintf(fid, '  </junction>\n\n');

fprintf(fid, '</OpenDRIVE>\n');
fclose(fid);
end
