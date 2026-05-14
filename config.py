"""Configuration management for the Field Survey Agent."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TerrainConfig:
    """Terrain analysis parameters."""

    dem_resolution_m: float = 30.0
    max_slope_degrees: float = 45.0  # slopes steeper than this are impassable
    slope_cost_exponent: float = 2.0  # how aggressively slope increases cost
    base_speed_flat_ms: float = 1.4  # m/s on flat open terrain (≈5 km/h)
    forest_speed_factor: float = 0.5
    shrub_speed_factor: float = 0.7
    grassland_speed_factor: float = 0.9
    wetland_speed_factor: float = 0.3
    bare_rock_speed_factor: float = 0.6
    water_buffer_m: float = 50.0  # avoid water by this margin


@dataclass
class WeatherConfig:
    """Weather integration parameters."""

    enabled: bool = True
    max_wind_speed_ms: float = 15.0  # abort threshold
    max_precipitation_mmh: float = 10.0
    min_visibility_m: float = 100.0
    temperature_range_c: tuple[float, float] = (-10.0, 40.0)
    lightning_risk_abort: bool = True
    weather_cost_weight: float = 0.3


@dataclass
class RoutePlannerConfig:
    """Route planning algorithm parameters."""

    algorithm: str = "astar"  # "astar", "dijkstra", "rrt", "multi_objective"
    diagonal_movement: bool = True
    max_route_distance_m: float = 50_000.0
    max_ascent_per_day_m: float = 1500.0
    rest_interval_minutes: int = 120  # rest every N minutes
    rest_duration_minutes: int = 15
    multi_objective_population: int = 50
    multi_objective_generations: int = 100
    smoothing_sigma: float = 1.0  # path smoothing


@dataclass
class RiskConfig:
    """Risk assessment thresholds."""

    slope_risk_threshold: float = 30.0  # degrees
    weather_risk_enabled: bool = True
    isolation_risk_distance_m: float = 5000.0  # distance from roads/help
    night_travel_allowed: bool = False
    communication_check_interval_m: float = 2000.0  # check comms at interval


@dataclass
class AgentConfig:
    """Main agent configuration."""

    name: str = "FieldSurveyAgent"
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    route_planner: RoutePlannerConfig = field(default_factory=RoutePlannerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data_sources: list[str] = field(default_factory=lambda: [
        "dem", "landcover", "weather", "hydrology", "roads", "hazards"
    ])
    output_dir: str = "./survey_output"
    log_level: str = "INFO"
    cache_dir: Optional[str] = None
