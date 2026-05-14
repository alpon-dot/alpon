"""Risk assessment for survey routes and field conditions."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import RiskConfig
from .models import (
    CostLayer,
    GeoPoint,
    RiskLevel,
    Route,
    SurveyTarget,
    TerrainCell,
    WeatherCell,
    WeatherCondition,
)


class RiskAssessor:
    """Evaluates risk for routes, waypoints, and survey plans."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    # ------------------------------------------------------------------
    # Point-level risks
    # ------------------------------------------------------------------

    def terrain_risk(self, cell: TerrainCell) -> float:
        """Risk score 0–1 for a terrain cell."""
        risk = 0.0
        risk += min(cell.slope_degrees / self.config.slope_risk_threshold, 1.0) * 0.5
        risk += (1.0 - cell.traversability) * 0.3
        risk += (cell.roughness / 10.0) * 0.2  # normalize roughness
        return min(risk, 1.0)

    def weather_risk(self, weather: WeatherCell) -> float:
        """Risk score 0–1 for weather conditions."""
        if not self.config.weather_risk_enabled:
            return 0.0

        risk = 0.0
        if weather.condition == WeatherCondition.THUNDERSTORM:
            return 1.0
        elif weather.condition in (WeatherCondition.HEAVY_RAIN, WeatherCondition.HIGH_WIND):
            risk += 0.6
        elif weather.condition in (WeatherCondition.SNOW, WeatherCondition.FOG):
            risk += 0.4
        elif weather.condition == WeatherCondition.LIGHT_RAIN:
            risk += 0.15

        wind_factor = weather.wind_speed_ms / 20.0
        risk += wind_factor * 0.3
        risk += (weather.precipitation_mmh / 15.0) * 0.2

        return min(risk, 1.0)

    # ------------------------------------------------------------------
    # Route-level risks
    # ------------------------------------------------------------------

    def assess_route(self, route: Route,
                     cost_layer: Optional[CostLayer] = None) -> tuple[float, RiskLevel, list[str]]:
        """Assess risk for an entire route. Returns (score, level, warnings)."""
        warnings: list[str] = []
        score = 0.0

        # Distance risk
        if route.total_distance_m > 30000:
            score += 0.4
            warnings.append("Route exceeds 30 km")
        elif route.total_distance_m > 20000:
            score += 0.2
            warnings.append("Route exceeds 20 km")
        elif route.total_distance_m > 10000:
            score += 0.1

        # Ascent risk
        if route.total_ascent_m > self.config.isolation_risk_distance_m / 1000 * 300:
            score += 0.3
            warnings.append(f"High cumulative ascent: {route.total_ascent_m:.0f}m")

        # Time risk (fatigue)
        if route.estimated_time_minutes > 600:
            score += 0.3
            warnings.append("Estimated time > 10 hours")
        elif route.estimated_time_minutes > 480:
            score += 0.15

        # Steep segment check
        n_steep = sum(
            1 for seg in route.segments
            if seg.get("max_slope", 0) > self.config.slope_risk_threshold
        )
        if n_steep > 0:
            score += min(0.3, n_steep * 0.02)  # cap steep-segment contribution
            warnings.append(
                f"{n_steep} segments exceed {self.config.slope_risk_threshold}° slope"
            )

        # Night travel
        if not self.config.night_travel_allowed and route.estimated_time_minutes > 720:
            score += 0.3
            warnings.append("Route may require night travel")

        score = min(score, 1.0)

        if score >= 0.7:
            level = RiskLevel.CRITICAL
        elif score >= 0.5:
            level = RiskLevel.HIGH
        elif score >= 0.3:
            level = RiskLevel.MODERATE
        elif score >= 0.15:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.NEGLIGIBLE

        return score, level, warnings

    def assess_waypoint(self, point: GeoPoint,
                        terrain_cells: Optional[NDArray] = None,
                        weather: Optional[WeatherCell] = None,
                        cost_layer: Optional[CostLayer] = None) -> dict:
        """Assess risk at a single waypoint."""
        result: dict = {"safe": True, "risks": [], "score": 0.0}

        # Terrain risk from cost surface
        if cost_layer is not None:
            try:
                cell_cost = cost_layer.grid[int(point.lat), int(point.lon)]
                if cell_cost > 100:
                    result["safe"] = False
                    result["risks"].append("High terrain cost")
                    result["score"] += 0.5
            except IndexError:
                pass

        # Weather risk
        if weather is not None:
            abort, reason = WeatherCell.check_abort(weather)
            if abort:
                result["safe"] = False
                result["risks"].append(reason)
                result["score"] += 0.5

        return result

    # ------------------------------------------------------------------
    # Communication gap detection
    # ------------------------------------------------------------------

    def communication_gaps(self, route: Route,
                           comm_range_m: float = 2000.0) -> list[tuple[int, int]]:
        """Find segments where consecutive waypoints exceed communication range."""
        gaps: list[tuple[int, int]] = []
        for i in range(len(route.waypoints) - 1):
            d = route.waypoints[i].distance_to(route.waypoints[i + 1])
            if d > comm_range_m:
                gaps.append((i, i + 1))
        return gaps

    def isolation_score(self, route: Route,
                        road_points: list[GeoPoint]) -> float:
        """Score 0–1 how isolated the route is from roads/help."""
        if not road_points:
            return 0.0
        max_dist = 0.0
        for wp in route.waypoints:
            min_d = min(wp.distance_to(rp) for rp in road_points)
            max_dist = max(max_dist, min_d)
        return min(max_dist / self.config.isolation_risk_distance_m, 1.0)
