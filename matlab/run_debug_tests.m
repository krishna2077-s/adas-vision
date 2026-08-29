cd('c:\Users\laksh\OneDrive\Desktop\adas-vision\adas-vision\matlab');
startup;

fprintf('\n==========================================\n');
fprintf('TESTING SCENARIO 2: urban_intersection\n');
fprintf('==========================================\n');
res2 = scenario_urban_intersection();
fprintf('S2 RESULT: arrived=%d, collisions=%d, t=%.2f\n\n', res2.arrived, res2.collisions, res2.t_total);

fprintf('\n==========================================\n');
fprintf('TESTING SCENARIO 4: dense_market\n');
fprintf('==========================================\n');
res4 = scenario_dense_market();
fprintf('S4 RESULT: arrived=%d, collisions=%d, t=%.2f\n\n', res4.arrived, res4.collisions, res4.t_total);

fprintf('\n==========================================\n');
fprintf('TESTING SCENARIO 5: cattle_crossing\n');
fprintf('==========================================\n');
res5 = scenario_cattle_crossing();
fprintf('S5 RESULT: arrived=%d, collisions=%d, t=%.2f\n\n', res5.arrived, res5.collisions, res5.t_total);

exit;
