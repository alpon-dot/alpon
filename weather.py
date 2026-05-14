"""Weather data integration and weather-based cost adjustment."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import WeatherConfig
from .models import WeatherCell, WeatherCondition, CostLayer


class WeatherIntegrator:
    """Integrates weather data and produces weather cost modifiers."""

    def __init__(self, config: Optional[WeatherConfig] = None):
        self.config = config or WeatherConfig()

    # ------------------------------------------------------------------
    # Weather cost
    # ------------------------------------------------------------------

    def weather_cost_multiplier(self, weather: WeatherCell) -> float:
        """Convert weather conditions to a route cost multiplier (>=1.0)."""
        multiplier = 1.0

        # Wind
        wind_factor = weather.wind_speed_ms / self.config.max_wind_speed_ms
        multiplier += wind_factor * 2.0

        # Gust penalty
        gust_excess = max(0, weather.wind_gust_ms - weather.wind_speed_ms)
        multiplier += gust_excess * 0.1

        # Precipitation
        precip_factor = weather.precipitation_mmh / self.config.max_precipitation_mmh
        multiplier += precip_factor * 3.0

        # Visibility
        if weather.visibility_m < self.config.min_visibility_m:
            multiplier += 5.0
        elif weather.visibility_m < 500:
            multiplier += 2.0

        # Temperature extremes
        t_min, t_max = self.config.temperature_range_c
        if weather.temperature_c < t_min or weather.temperature_c > t_max:
            multiplier += 3.0
        elif weather.temperature_c < t_min + 5 or weather.temperature_c > t_max - 5:
            multiplier += 1.0

        # Condition-based penalties
        condition_penalty = {
            WeatherCondition.CLEAR: 0.0,
            WeatherCondition.CLOUDY: 0.1,
            WeatherCondition.LIGHT_RAIN: 0.5,
            WeatherCondition.FOG: 2.0,
            WeatherCondition.HEAVY_RAIN: 4.0,
            WeatherCondition.SNOW: 3.0,
            WeatherCondition.THUNDERSTORM: 10.0,
            WeatherCondition.HIGH_WIND: 4.0,
        }
        multiplier += condition_penalty.get(weather.condition, 0.0)

        return max(1.0, multiplier)

    def is_abort_condition(self, weather: WeatherCell) -> tuple[bool, str]:
        """Check if weather is dangerous enough to abort."""
        if weather.wind_speed_ms > self.config.max_wind_speed_ms:
            return True, f"Wind speed {weather.wind_speed_ms:.1f} m/s exceeds limit"
        if weather.precipitation_mmh > self.config.max_precipitation_mmh:
            return True, f"Precipitation {weather.precipitation_mmh:.1f} mm/h exceeds limit"
        if weather.condition == WeatherCondition.THUNDERSTORM and self.config.lightning_risk_abort:
            return True, "Thunderstorm — lightning risk"
        if weather.visibility_m < self.config.min_visibility_m:
            return True, f"Visibility {weather.visibility_m:.0f}m below minimum"
        return False, ""

    def weather_cost_grid(self, weather_grid: list[list[WeatherCell]],
                          rows: int, cols: int) -> NDArray[np.float64]:
        """Produce a 2D weather cost multiplier grid."""
        cost = np.ones((rows, cols), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                cost[r, c] = self.weather_cost_multiplier(weather_grid[r][c])
        return cost

    # ------------------------------------------------------------------
    # Synthetic / simulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_grid(rows: int, cols: int,
                                base_condition: WeatherCondition = WeatherCondition.CLEAR,
                                seed: int = 42) -> list[list[WeatherCell]]:
        """Generate synthetic weather data for testing."""
        rng = np.random.default_rng(seed)
        conditions = list(WeatherCondition)
        grid: list[list[WeatherCell]] = []
        for r in range(rows):
            row: list[WeatherCell] = []
            for c in range(cols):
                p = rng.random()
                if p < 0.7:
                    cond = base_condition
                else:
                    cond = conditions[rng.integers(0, len(conditions))]
                row.append(WeatherCell(
                    lat=0.0, lon=0.0,
                    condition=cond,
                    temperature_c=rng.normal(20, 5),
                    wind_speed_ms=max(0, rng.normal(5, 3)),
                    wind_gust_ms=max(0, rng.normal(8, 4)),
                    precipitation_mmh=max(0, rng.exponential(2)),
                    visibility_m=max(50, rng.normal(5000, 2000)),
                    cloud_cover_pct=np.clip(rng.normal(40, 30), 0, 100),
                ))
            grid.append(row)
        return grid
