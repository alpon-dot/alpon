"""Field Survey Agent — Multi-source Data Fusion & Intelligent Route Planning.

A comprehensive agent for field survey / reconnaissance planning that:
- Fuses terrain (DEM), land cover, weather, hydrology, hazard, and road data
- Plans optimal routes using A* and multi-objective evolutionary optimization
- Assesses risk across terrain, weather, isolation, and communication dimensions
- Exports plans to JSON and generates publication-quality visualizations

Quick start
-----------
>>> from field_survey_agent import FieldSurveyAgent
>>> agent = FieldSurveyAgent()
>>> dem = agent.generate_synthetic_dem(200, 200)
>>> terrain = agent.generate_synthetic_terrain(200, 200, dem)
>>> plan = agent.plan_survey(dem, terrain, start, targets)
>>> print(agent.plan_summary(plan))
"""

from .agent import FieldSurveyAgent
from .config import AgentConfig, RoutePlannerConfig, RiskConfig, TerrainConfig, WeatherConfig
from .data_fusion import DataFusionEngine
from .models import (
    CostLayer,
    FusionResult,
    GeoPoint,
    RiskLevel,
    Route,
    SurveyPlan,
    SurveyTarget,
    TerrainCell,
    TerrainType,
    WeatherCell,
    WeatherCondition,
)
from .risk_assessment import RiskAssessor
from .route_planner import AStarPlanner, MultiObjectivePlanner, SurveyRoutePlanner
from .terrain_analysis import TerrainAnalyzer
from .visualization import SurveyVisualizer
from .weather import WeatherIntegrator

__all__ = [
    # Agent
    "FieldSurveyAgent",
    # Config
    "AgentConfig",
    "TerrainConfig",
    "WeatherConfig",
    "RoutePlannerConfig",
    "RiskConfig",
    # Models
    "GeoPoint",
    "SurveyTarget",
    "SurveyPlan",
    "Route",
    "CostLayer",
    "FusionResult",
    "TerrainCell",
    "WeatherCell",
    "TerrainType",
    "WeatherCondition",
    "RiskLevel",
    # Core modules
    "DataFusionEngine",
    "TerrainAnalyzer",
    "WeatherIntegrator",
    "RiskAssessor",
    "SurveyRoutePlanner",
    "AStarPlanner",
    "MultiObjectivePlanner",
    # Visualization
    "SurveyVisualizer",
]
__version__ = "1.0.0"
