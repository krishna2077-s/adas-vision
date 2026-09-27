from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal

class Metadata(BaseModel):
    name: str
    description: str
    source: Literal["nlp", "template", "builder", "variant", "sweep"]
    created_at: str
    tags: List[str] = []
    original_text: Optional[str] = None
    version: int = 1

class Road(BaseModel):
    type: Literal["village_road", "urban_street", "highway", "market_street", "intersection", "custom"]
    lane_half_width_m: float = Field(ge=1.5, le=10.0)
    length_m: float = Field(ge=50.0, le=500.0)
    markings: Optional[Literal["none", "broken_center", "double_center", "full"]] = "none"
    surface: Optional[Literal["tarmac", "concrete", "dirt", "broken_tarmac"]] = "tarmac"
    has_shoulder: bool = False

class Environment(BaseModel):
    time_of_day: Optional[Literal["day", "dusk", "night"]] = "day"
    weather: Optional[Literal["clear", "overcast", "light_rain", "heavy_rain", "fog"]] = "clear"
    visibility_m: Optional[float] = Field(None, ge=20.0, le=500.0)

class EgoVehicle(BaseModel):
    initial_speed_kmh: float = Field(ge=0.0, le=120.0)
    initial_x_m: float = 0.0
    initial_y_m: float = 0.0
    initial_heading_deg: float = 0.0
    goal_x_m: float
    goal_y_m: float = 0.0
    goal_tolerance_m: float = 4.0
    mode: Literal["adaptive", "baseline"] = "adaptive"

class Actor(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    actor_class: Literal[
        "cow", "buffalo", "dog", "person", "child", 
        "motorcycle", "scooter", "bicycle", 
        "auto_rickshaw", "e_rickshaw", 
        "pushcart", "thela", 
        "car", "suv", "bus", "truck", "tractor"
    ] = Field(alias="class")
    initial_x_m: float
    initial_y_m: float
    initial_vx_mps: float = 0.0
    initial_vy_mps: float = 0.0
    width_m: Optional[float] = None
    length_m: Optional[float] = None
    behaviour: Literal["static", "constant_velocity", "sudden_stop", "crossing", "wrong_side", "cut_in", "gradual"] = "constant_velocity"
    behaviour_trigger_s: Optional[float] = None
    behaviour_y_target_m: Optional[float] = None

class SimulationParameters(BaseModel):
    dt_s: float = 0.033
    max_time_s: float = Field(35.0, ge=10.0, le=120.0)

class SweepSpec(BaseModel):
    sweep_parameter: str
    sweep_path: str
    values: List[float]

class Scenario(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    scenario_id: str
    parent_id: Optional[str] = None
    metadata: Metadata
    road: Road
    environment: Optional[Environment] = None
    ego_vehicle: EgoVehicle
    actors: List[Actor] = Field(min_length=1, max_length=8)
    simulation_parameters: SimulationParameters = SimulationParameters()
    sweep_spec: Optional[SweepSpec] = None

class ResultExplanation(BaseModel):
    outcome_summary: str
    critical_events: List[str]
    avoidance_successful: bool
    failure_reason: Optional[str] = None

class SimulationResult(BaseModel):
    result_id: str
    scenario_id: str
    executed_at: str
    mode: str
    arrived: bool
    t_total_s: float
    status: str
    collisions: int
    min_clearance_m: float
    replans: int
    mean_replan_ms: float
    max_replan_ms: float
    path_smoothness: float
    mean_speed_kmh: float
    max_speed_kmh: float
    decision_proceed: float
    decision_caution: float
    decision_slow: float
    decision_brake: float
    decision_emergency_stop: float
    traj_x: List[float]
    traj_y: List[float]
    explanation: ResultExplanation

