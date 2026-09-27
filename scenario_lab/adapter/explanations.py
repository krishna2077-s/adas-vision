import datetime
from scenario_lab.schema.scenario_schema import SimulationResult, ResultExplanation, Scenario

def build_explanation(result_data: dict, scenario_id: str, mode: str) -> SimulationResult:
    """Pure deterministic explanation generation from simulation data."""
    
    arrived = result_data.get('arrived', False)
    collisions = result_data.get('collisions', 0)
    t_total_s = result_data.get('t_total_s', 0.0)
    min_clearance = result_data.get('min_clearance_m', 99.0)
    em_stop = result_data.get('decision_emergency_stop', 0.0)
    
    if collisions > 0:
        outcome = f"COLLISION — {collisions} contact event(s) detected"
        status = "COLLISION"
        failure_reason = f"Collision detected at minimum clearance {min_clearance:.2f}m"
    elif arrived:
        outcome = f"Goal reached in {t_total_s:.1f}s with 0 collisions"
        status = "GOAL_REACHED"
        failure_reason = None
    else:
        outcome = f"Timeout at {t_total_s:.1f}s — goal not reached"
        status = "TIMEOUT"
        failure_reason = "Goal not reached within time limit — path blocked or scenario too long"
        
    critical_events = []
    if em_stop > 0.01:
        critical_events.append(f"EMERGENCY_STOP engaged ({em_stop*100:.0f}% of frames)")
    if min_clearance < 2.0:
        critical_events.append(f"Critical clearance: {min_clearance:.2f}m")
        
    explanation = ResultExplanation(
        outcome_summary=outcome,
        critical_events=critical_events,
        avoidance_successful=(collisions == 0),
        failure_reason=failure_reason
    )
    
    import uuid
    result_id = str(uuid.uuid4())
    
    return SimulationResult(
        result_id=result_id,
        scenario_id=scenario_id,
        executed_at=datetime.datetime.utcnow().isoformat() + "Z",
        mode=mode,
        arrived=arrived,
        t_total_s=t_total_s,
        status=status,
        collisions=collisions,
        min_clearance_m=min_clearance,
        replans=result_data.get('replans', 0),
        mean_replan_ms=result_data.get('mean_replan_ms', 0.0),
        max_replan_ms=result_data.get('max_replan_ms', 0.0),
        path_smoothness=result_data.get('path_smoothness', 0.0),
        mean_speed_kmh=result_data.get('mean_speed_kmh', 0.0),
        max_speed_kmh=result_data.get('max_speed_kmh', 0.0),
        decision_proceed=result_data.get('decision_proceed', 1.0),
        decision_caution=result_data.get('decision_caution', 0.0),
        decision_slow=result_data.get('decision_slow', 0.0),
        decision_brake=result_data.get('decision_brake', 0.0),
        decision_emergency_stop=em_stop,
        traj_x=result_data.get('traj_x', []),
        traj_y=result_data.get('traj_y', []),
        explanation=explanation
    )
