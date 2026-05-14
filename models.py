"""Data models for the Field Survey Agent."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt


class TerrainType(Enum):
    """Land cover / terrain classification."""

    OPEN = 0
    FOREST = 1
    SHRUB = 2
    GRASSLAND = 3
    WETLAND = 4
    WATER = 5
    BARE_ROCK = 6
    URBAN = 7
    SNOW_ICE = 8
    CROPLAND = 9


class WeatherCondition(Enum):
    """Weather severity categories."""

    CLEAR = "clear"
    CLOUDY = "cloudy"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    FOG = "fog"
    THUNDERSTORM = "thunderstorm"
    HIGH_WIND = "high_wind"


class RiskLevel(Enum):
    """Risk assessment levels."""

    NEGLIGIBLE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GeoPoint:
    """A geographic point with elevation."""

    lat: float
    lon: float
    elevation: float = 0.0

    def distance_to(self, other: "GeoPoint") -> float:
        """Haversine distance in meters."""
        R = 6_371_000
        lat1, lon1 = np.radians(self.lat), np.radians(self.lon)
        lat2, lon2 = np.radians(other.lat), np.radians(other.lon)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


@dataclass
class SurveyTarget(GeoPoint):
    """A survey / sampling target point."""

    target_id: str = ""
    priority: float = 1.0  # 0-1, higher = more important
    description: str = ""
    min_stay_minutes: int = 30
    required_equipment: list[str] = field(default_factory=list)


@dataclass
class TerrainCell:
    """A single cell in the terrain cost grid."""

    lat: float
    lon: float
    elevation: float
    slope_degrees: float
    aspect_degrees: float
    terrain_type: TerrainType
    traversability: float  # 0=impassable, 1=effortless
    roughness: float  # surface roughness index


@dataclass
class WeatherCell:
    """Weather data at a grid cell."""

    lat: float
    lon: float
    condition: WeatherCondition
    temperature_c: float
    wind_speed_ms: float
    wind_gust_ms: float
    precipitation_mmh: float
    visibility_m: float
    cloud_cover_pct: float


@dataclass
class CostLayer:
    """A single cost layer for route planning."""

    name: str
    grid: npt.NDArray[np.float64]  # 2D cost array
    weight: float = 1.0
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    resolution_m: float = 30.0  # cell size in meters
    rows: int = 0
    cols: int = 0

    def __post_init__(self):
        if self.rows == 0 or self.cols == 0:
            self.rows, self.cols = self.grid.shape


@dataclass
class Route:
    """A planned survey route."""

    waypoints: list[GeoPoint] = field(default_factory=list)
    total_distance_m: float = 0.0
    total_ascent_m: float = 0.0
    total_descent_m: float = 0.0
    estimated_time_minutes: float = 0.0
    risk_score: float = 0.0
    cumulative_cost: float = 0.0
    visited_targets: list[str] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)


@dataclass
class SurveyPlan:
    """Complete survey plan with multiple routes."""

    routes: list[Route] = field(default_factory=list)
    targets: list[SurveyTarget] = field(default_factory=list)
    start_point: Optional[GeoPoint] = None
    end_point: Optional[GeoPoint] = None
    total_distance_m: float = 0.0
    total_time_hours: float = 0.0
    overall_risk: RiskLevel = RiskLevel.NEGLIGIBLE
    metadata: dict = field(default_factory=dict)


@dataclass
class FusionResult:
    """Output of the data fusion engine."""

    combined_cost: npt.NDArray[np.float64]
    confidence: npt.NDArray[np.float64]  # per-cell confidence 0-1
    layers_used: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
