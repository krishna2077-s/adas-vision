function run_rr_scenario(scenarioName)
% RUN_RR_SCENARIO Runner script for RoadRunner scenarios
if nargin < 1, scenarioName = 'scenario_village_road.rrscenario'; end
run_roadrunner_cosim(scenarioName);
end
