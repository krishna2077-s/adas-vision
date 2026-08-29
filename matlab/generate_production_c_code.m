% generate_production_c_code.m
% ADAS Vision — MATLAB Coder C/C++ Generation Utility
%
% This script demonstrates an Enterprise-Only feature: converting the 
% custom ADAS Decision Engine (decision_with_ratchet.m) into raw, optimized
% C/C++ code suitable for deployment onto embedded automotive ECUs.
%
% This requires the MATLAB Coder toolbox, which is heavily restricted 
% in standard trials.

function generate_production_c_code()
    fprintf('=== ADAS Vision — Production C/C++ Code Generator ===\n\n');
    
    if isempty(ver('coder'))
        fprintf('[ERROR] MATLAB Coder toolbox not found. This is an Enterprise feature.\n');
        return;
    end
    
    fprintf('  [1/4] Configuring Code Generation for decision_with_ratchet.m...\n');
    
    % Define the precise types and sizes of the inputs for the C compiler
    % Inputs: (hazard_dist, hazard_ttc, critical_label, in_path_hazard, closing_speed, ...
    %          degraded, lane_offset, prev_committed, raw_history, down_counter, emergency_latch)
    
    % Use basic types for static compilation
    ARGS = cell(1, 11);
    ARGS{1} = coder.typeof(0.0); % hazard_dist (double)
    ARGS{2} = coder.typeof(0.0); % hazard_ttc (double)
    
    % critical_label is a string/char array. We'll specify it as a variable size char up to 20
    ARGS{3} = coder.typeof('A', [1 20], [false true]); 
    
    ARGS{4} = coder.typeof(false); % in_path_hazard (logical)
    ARGS{5} = coder.typeof(0.0); % closing_speed (double)
    ARGS{6} = coder.typeof(false); % degraded (logical)
    ARGS{7} = coder.typeof(0.0); % lane_offset (double)
    ARGS{8} = coder.typeof(0); % prev_committed (double/int)
    
    % raw_history is an array, let's say up to 10 elements
    ARGS{9} = coder.typeof(0, [1 10], [false true]);
    
    ARGS{10} = coder.typeof(0); % down_counter
    ARGS{11} = coder.typeof(0); % emergency_latch
    
    cfg = coder.config('lib'); % Generate a static C/C++ library
    cfg.TargetLang = 'C++';
    cfg.GenerateReport = true;
    cfg.ReportPotentialDifferences = false;
    cfg.HardwareImplementation.ProdHWDeviceType = 'ARM Compatible->ARM Cortex-A'; % Automotive typical
    
    fprintf('  [2/4] Target Language: C++ (ARM Cortex-A Profile)\n');
    fprintf('  [3/4] Running Code Generation (this may take a minute)...\n');
    
    out_dir = fullfile(pwd, 'codegen', 'lib', 'decision_engine_c');
    
    try
        % Execute the Coder
        codegen -config cfg -args ARGS -d out_dir decision_with_ratchet
        
        fprintf('\n  [4/4] SUCCESS! C++ Code Generated Successfully.\n\n');
        fprintf('  Location: %s\n', out_dir);
        fprintf('  Key files generated:\n');
        fprintf('    - decision_with_ratchet.cpp (Main Logic)\n');
        fprintf('    - decision_with_ratchet.h   (Header)\n');
        fprintf('    - decision_with_ratchet_types.h\n\n');
        
        fprintf('  You can now show the judges the "codegen" folder to prove\n');
        fprintf('  that your logic is hardware-ready for Automotive ECUs!\n');
    catch ME
        fprintf('\n[ERROR] Code generation failed: %s\n', ME.message);
    end
end
