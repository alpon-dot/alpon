"""Field Survey Agent — main orchestrator.

Integrates multi-source data fusion, terrain analysis, weather,
risk assessment, and intelligent route planning for field surveys.
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import AgentConfig
from .data_fusion import DataFusionEngine
from .models import (
    CostLayer,
    FusionResult,
    GeoPoint,
    RiskLevel,
    Route,
    SurveyPlan,
    SurveyTarget,
    TerrainType,
    WeatherCondition,
)
from .risk_assessment import RiskAssessor
from .route_planner import SurveyRoutePlanner
from .terrain_analysis import TerrainAnalyzer
from .weather import WeatherIntegrator

logger = logging.getLogger(__name__)


class FieldSurveyAgent:
    """Autonomous agent for field survey route planning with multi-source data fusion.

    Usage
    -----
    >>> agent = FieldSurveyAgent()
    >>> dem = agent.generate_synthetic_dem(200, 200)
    >>> terrain = agent.generate_synthetic_terrain(200, 200)
    >>> plan = agent.plan_survey(dem, terrain, start, targets)
    >>> agent.export_plan(plan, "survey_plan.json")
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._setup_logging()

        self.terrain_analyzer = TerrainAnalyzer(self.config.terrain)
        self.weather_integrator = WeatherIntegrator(self.config.weather)
        self.fusion_engine = DataFusionEngine(
            self.config.terrain, self.config.weather
        )
        self.route_planner = SurveyRoutePlanner(self.config.route_planner)
        self.risk_assessor = RiskAssessor(self.config.risk)

        self._state: dict = {}
        logger.info("FieldSurveyAgent '%s' initialized", self.config.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_survey(self,
                    dem: NDArray[np.float64],
                    terrain_grid: NDArray[np.int_],
                    start: GeoPoint,
                    targets: list[SurveyTarget],
                    end: Optional[GeoPoint] = None,
                    weather_cells: Optional[list[list]] = None,
                    water_mask: Optional[NDArray[np.bool_]] = None,
                    hazard_points: Optional[list[tuple[int, int, float]]] = None,
                    road_pixels: Optional[list[tuple[int, int]]] = None,
                    fusion_strategy: str = "weighted_sum",
                    optimize_order: bool = True) -> SurveyPlan:
        """Plan a complete field survey.

        Parameters
        ----------
        dem : 2D numpy array of elevations (meters).
        terrain_grid : 2D numpy array of TerrainType enum values.
        start : Starting GeoPoint.
        targets : List of survey targets to visit.
        end : Optional end point (defaults to start).
        weather_cells : Optional 2D grid of WeatherCell objects.
        water_mask : Optional boolean mask of water bodies.
        hazard_points : Optional list of (row, col, radius_m) hazards.
        road_pixels : Optional list of (row, col) road pixels.
        fusion_strategy : One of "weighted_sum", "multiplicative", "fuzzy".
        optimize_order : Use evolutionary optimization for target order.

        Returns
        -------
        SurveyPlan with routes, metrics, and risk assessment.
        """
        self._state["phase"] = "fusing_data"
        logger.info("Starting survey plan: %d targets, %s fusion",
                     len(targets), fusion_strategy)

        # --- Data fusion ---
        weather_cost = None
        if weather_cells is not None:
            rows, cols = dem.shape
            weather_cost = self.weather_integrator.weather_cost_grid(
                weather_cells, rows, cols
            )

        fusion_result = self.fusion_engine.build_fused_cost(
            dem=dem,
            terrain_grid=terrain_grid,
            water_mask=water_mask,
            weather_cost_grid=weather_cost,
            hazard_points=hazard_points,
            road_pixels=road_pixels,
            origin=start,
            fusion_strategy=fusion_strategy,
        )

        cost_layer = CostLayer(
            name="fused_cost",
            grid=fusion_result.combined_cost,
            resolution_m=self.config.terrain.dem_resolution_m,
            origin_lat=start.lat,
            origin_lon=start.lon,
            rows=dem.shape[0],
            cols=dem.shape[1],
        )

        # --- Route planning ---
        self._state["phase"] = "planning_routes"
        end = end or start

        if len(targets) == 0:
            route = self.route_planner.plan_single(cost_layer, start, end, dem)
            routes = [route] if route.waypoints else []
        else:
            main_route = self.route_planner.plan_multi_target(
                cost_layer, start, targets, dem,
                optimize_order=optimize_order and len(targets) > 2,
            )
            # Return leg
            if main_route.waypoints:
                last = main_route.waypoints[-1]
                return_route = self.route_planner.plan_single(
                    cost_layer, last, end, dem
                )
                routes = [main_route]
                if return_route.waypoints:
                    routes.append(return_route)
            else:
                routes = []

        # --- Risk assessment ---
        self._state["phase"] = "assessing_risk"
        overall_risk = RiskLevel.NEGLIGIBLE
        for route in routes:
            score, level, warnings = self.risk_assessor.assess_route(
                route, cost_layer
            )
            route.risk_score = score
            if level.value > overall_risk.value:
                overall_risk = level
            if warnings:
                logger.warning("Route risk: %s", "; ".join(warnings))

        # --- Compile plan ---
        plan = SurveyPlan(
            routes=routes,
            targets=targets,
            start_point=start,
            end_point=end,
            total_distance_m=sum(r.total_distance_m for r in routes),
            total_time_hours=sum(r.estimated_time_minutes for r in routes) / 60.0,
            overall_risk=overall_risk,
            metadata={
                "fusion_strategy": fusion_strategy,
                "layers_used": fusion_result.layers_used,
                "timestamp": datetime.now().isoformat(),
                "agent": self.config.name,
                "num_targets": len(targets),
                "optimize_order": optimize_order,
            },
        )

        self._state["phase"] = "complete"
        logger.info(
            "Plan complete: %.1f km, %.1f h, risk=%s",
            plan.total_distance_m / 1000,
            plan.total_time_hours,
            plan.overall_risk.name,
        )
        return plan

    # ------------------------------------------------------------------
    # Synthetic data generation (for testing / demo)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_dem(rows: int, cols: int,
                               seed: int = 42) -> NDArray[np.float64]:
        """Generate a realistic synthetic DEM using fractal noise."""
        rng = np.random.default_rng(seed)

        # Fractal / multi-octave noise
        dem = np.zeros((rows, cols), dtype=np.float64)
        for octave, scale in enumerate([64, 32, 16, 8, 4], 1):
            freq = 1.0 / scale
            coarse_rows = max(2, rows // scale)
            coarse_cols = max(2, cols // scale)
            coarse = rng.normal(0, 50.0 / octave, (coarse_rows, coarse_cols))

            # Upsample
            from scipy.ndimage import zoom
            zoom_r = rows / coarse_rows
            zoom_c = cols / coarse_cols
            fine = zoom(coarse, (zoom_r, zoom_c), order=1)
            h, w = fine.shape
            dem[:h, :w] += fine[:h, :w]

        # Add a ridge and valley
        ridge_x = np.linspace(0, cols - 1, cols)
        ridge = 200 * np.sin(ridge_x / cols * np.pi * 3) ** 2
        for r in range(rows):
            dem[r, :] += ridge * (1 - abs(r - rows // 2) / (rows // 2))

        # Normalize to reasonable elevation range
        dem = (dem - dem.min()) / (dem.max() - dem.min()) * 800 + 100
        return dem.astype(np.float64)

    @staticmethod
    def generate_synthetic_terrain(rows: int, cols: int,
                                   dem: Optional[NDArray[np.float64]] = None,
                                   seed: int = 43) -> NDArray[np.int_]:
        """Generate a synthetic land cover grid."""
        rng = np.random.default_rng(seed)
        terrain = np.full((rows, cols), TerrainType.OPEN.value, dtype=np.int_)

        # Large contiguous patches
        n_patches = 8
        for _ in range(n_patches):
            pr, pc = rng.integers(0, rows), rng.integers(0, cols)
            max_radius = max(12, min(rows, cols) // 6)
            radius = rng.integers(10, max_radius)
            terrain_type = rng.choice([
                TerrainType.FOREST, TerrainType.SHRUB, TerrainType.GRASSLAND,
                TerrainType.BARE_ROCK, TerrainType.WETLAND,
            ])
            rr, cc = np.mgrid[0:rows, 0:cols]
            dist = np.sqrt((rr - pr) ** 2 + (cc - pc) ** 2)
            mask = dist < radius
            terrain[mask] = terrain_type.value

        # Water bodies in low elevations (small fraction to avoid blocking)
        if dem is not None:
            low_mask = dem < np.percentile(dem, 8)
            # Only mark as water if in contiguous low region (avoid scattered blocking)
            from scipy import ndimage
            labeled, n_features = ndimage.label(low_mask)
            for feat_id in range(1, n_features + 1):
                region = labeled == feat_id
                if np.sum(region) < 5:  # skip tiny isolated lows
                    low_mask[region] = False
            terrain[low_mask] = TerrainType.WATER.value

        return terrain

    def generate_synthetic_weather(self, rows: int, cols: int,
                                   seed: int = 44) -> list[list]:
        """Generate synthetic weather grid."""
        return WeatherIntegrator.generate_synthetic_grid(rows, cols, seed=seed)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def export_plan(self, plan: SurveyPlan, filepath: str) -> str:
        """Export a survey plan to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        data = {
            "metadata": plan.metadata,
            "overall_risk": plan.overall_risk.name,
            "total_distance_km": round(plan.total_distance_m / 1000, 2),
            "total_time_hours": round(plan.total_time_hours, 1),
            "start": {
                "lat": plan.start_point.lat if plan.start_point else 0,
                "lon": plan.start_point.lon if plan.start_point else 0,
            },
            "targets": [
                {
                    "id": t.target_id,
                    "lat": t.lat,
                    "lon": t.lon,
                    "priority": t.priority,
                    "min_stay_min": t.min_stay_minutes,
                }
                for t in plan.targets
            ],
            "routes": [],
        }

        for i, route in enumerate(plan.routes):
            route_data = {
                "index": i,
                "distance_km": round(route.total_distance_m / 1000, 2),
                "ascent_m": round(route.total_ascent_m, 0),
                "descent_m": round(route.total_descent_m, 0),
                "time_min": round(route.estimated_time_minutes, 0),
                "risk_score": round(route.risk_score, 2),
                "visited_targets": route.visited_targets,
                "waypoints": [
                    {"lat": wp.lat, "lon": wp.lon, "elev": wp.elevation}
                    for wp in route.waypoints[::max(1, len(route.waypoints) // 200)]
                ],
            }
            data["routes"].append(route_data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Plan exported to %s", filepath)
        return filepath

    def plan_summary(self, plan: SurveyPlan) -> str:
        """Return a human-readable summary of the plan."""
        lines = [
            "=" * 60,
            f"  Survey Plan — {plan.metadata.get('timestamp', '')}",
            "=" * 60,
            f"  Targets: {len(plan.targets)}",
            f"  Total distance: {plan.total_distance_m / 1000:.2f} km",
            f"  Total time: {plan.total_time_hours:.1f} hours",
            f"  Overall risk: {plan.overall_risk.name}",
            f"  Routes: {len(plan.routes)}",
            "-" * 60,
        ]
        for i, route in enumerate(plan.routes):
            lines.append(
                f"  Route {i + 1}: {route.total_distance_m / 1000:.2f} km, "
                f"{route.estimated_time_minutes:.0f} min, "
                f"ascent {route.total_ascent_m:.0f} m, "
                f"risk {route.risk_score:.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_logging(self):
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
