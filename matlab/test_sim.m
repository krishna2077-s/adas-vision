diary('sim_log.txt');
cd('c:\Users\laksh\OneDrive\Desktop\adas-vision\adas-vision\matlab');
startup;
res = scenario_village_road();
fprintf('FINAL STATUS: Arrived=%d, Collisions=%d, TotalTime=%.2f\n', res.arrived, res.collisions, res.t_total);
diary off;
exit;
