"""Multi-source data fusion engine.

Combines terrain, weather, hazard, hydrology, and human-activity layers
into a unified cost surface for route planning.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .config import TerrainConfig, WeatherConfig
from .models import CostLayer, FusionResult, GeoPoint
from .terrain_analysis import TerrainAnalyzer
from .weather import WeatherIntegrator


class DataFusionEngine:
    """Fuses multiple cost layers into a single decision surface."""

    def __init__(self,
                 terrain_config: Optional[TerrainConfig] = None,
                 weather_config: Optional[WeatherConfig] = None):
        self.terrain_config = terrain_config or TerrainConfig()
        self.weather_config = weather_config or WeatherConfig()
        self.terrain_analyzer = TerrainAnalyzer(self.terrain_config)
        self.weather_integrator = WeatherIntegrator(self.weather_config)

    # ------------------------------------------------------------------
    # Individual cost layers
    # ------------------------------------------------------------------

    def water_barrier_layer(self, dem: NDArray[np.float64],
                            water_mask: NDArray[np.bool_],
                            resolution_m: float) -> CostLayer:
        """Create a cost layer from water bodies with buffer.

        Uses high (but not infinite) cost so paths can cross water if unavoidable.
        """
        cost = np.ones_like(dem, dtype=np.float64)
        buffer_px = int(self.terrain_config.water_buffer_m / resolution_m)
        if buffer_px > 0:
            dilated = ndimage.binary_dilation(water_mask, iterations=buffer_px)
            cost[dilated] = 5000.0
        cost[water_mask] = 10000.0
        return CostLayer(name="water_barrier", grid=cost, resolution_m=resolution_m)

    def slope_cost_layer(self, dem: NDArray[np.float64],
                         resolution_m: float) -> CostLayer:
        """Cost layer based purely on slope."""
        slope = self.terrain_analyzer.slope_degrees(dem, resolution_m)
        return CostLayer(
            name="slope_cost",
            grid=self.terrain_analyzer.slope_cost(slope),
            resolution_m=resolution_m,
        )

    def terrain_type_layer(self, dem: NDArray[np.float64],
                           terrain_grid: NDArray[np.int_],
                           resolution_m: float) -> CostLayer:
        """Cost layer from land cover / terrain type."""
        speed = self.terrain_analyzer.terrain_speed_factor(terrain_grid)
        cost = np.where(speed > 0, 1.0 / speed, 50000)
        return CostLayer(name="terrain_type", grid=cost, resolution_m=resolution_m)

    def weather_layer(self, weather_cost_grid: NDArray[np.float64],
                      resolution_m: float,
                      origin_lat: float = 0.0,
                      origin_lon: float = 0.0) -> CostLayer:
        """Cost layer from weather conditions."""
        return CostLayer(
            name="weather",
            grid=weather_cost_grid,
            weight=self.weather_config.weather_cost_weight,
            resolution_m=resolution_m,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )

    def hazard_layer(self, dem_shape: tuple[int, int],
                     hazard_points: list[tuple[int, int, float]],
                     resolution_m: float) -> CostLayer:
        """Cost layer from known hazards with distance-based decay.

        hazard_points: list of (row, col, radius_m).
        """
        cost = np.ones(dem_shape, dtype=np.float64)
        rows, cols = dem_shape
        rr, cc = np.mgrid[0:rows, 0:cols]
        for hr, hc, radius_m in hazard_points:
            radius_px = radius_m / resolution_m
            dist = np.sqrt((rr - hr) ** 2 + (cc - hc) ** 2)
            penalty = np.exp(-dist / radius_px) * 10.0
            cost += penalty
        return CostLayer(name="hazards", grid=cost, resolution_m=resolution_m)

    def road_proximity_layer(self, dem_shape: tuple[int, int],
                             road_pixels: list[tuple[int, int]],
                             resolution_m: float,
                             max_distance_m: float = 5000.0) -> CostLayer:
        """Bonus (cost < 1) near roads; fades to 1 at max_distance_m.

        Roads make travel easier — this creates attractor corridors.
        """
        rows, cols = dem_shape
        rr, cc = np.mgrid[0:rows, 0:cols]
        dist = np.full(dem_shape, np.inf, dtype=np.float64)
        for rr_idx, cc_idx in road_pixels:
            d = np.sqrt((rr - rr_idx) ** 2 + (cc - cc_idx) ** 2)
            dist = np.minimum(dist, d)
        max_px = max_distance_m / resolution_m
        benefit = 0.5 * np.exp(-dist / (max_px / 3))  # max 50% cost reduction near road
        cost = 1.0 - np.where(np.isfinite(dist), benefit, 0.0)
        return CostLayer(name="road_proximity", grid=np.clip(cost, 0.5, 1.0),
                         resolution_m=resolution_m)

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------

    def weighted_sum(self, layers: list[CostLayer]) -> FusionResult:
        """Weighted linear combination of cost layers."""
        if not layers:
            raise ValueError("No layers to fuse")

        shape = layers[0].grid.shape
        combined = np.zeros(shape, dtype=np.float64)
        total_weight = 0.0

        for layer in layers:
            if layer.grid.shape != shape:
                raise ValueError(f"Layer {layer.name} shape mismatch: "
                                 f"{layer.grid.shape} vs {shape}")
            if not np.all(np.isfinite(layer.grid)):
                continue
            combined += layer.weight * layer.grid
            total_weight += layer.weight

        combined /= max(total_weight, 1e-9)

        # Confidence: higher where layers agree
        normalized_layers = []
        for layer in layers:
            g = layer.grid.copy()
            g[~np.isfinite(g)] = 50000
            if g.max() > g.min():
                normalized_layers.append((g - g.min()) / (g.max() - g.min()))
            else:
                normalized_layers.append(np.zeros_like(g))

        if len(normalized_layers) >= 2:
            stacked = np.stack(normalized_layers, axis=-1)
            confidence = 1.0 - np.std(stacked, axis=-1)
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            confidence = np.ones(shape, dtype=np.float64)

        return FusionResult(
            combined_cost=combined,
            confidence=confidence,
            layers_used=[l.name for l in layers],
        )

    def multiplicative_fusion(self, layers: list[CostLayer]) -> FusionResult:
        """Multiply costs — useful when all layers are >= 1 multipliers."""
        shape = layers[0].grid.shape
        combined = np.ones(shape, dtype=np.float64)
        for layer in layers:
            combined *= np.maximum(layer.grid, 1.0) ** layer.weight
        return FusionResult(
            combined_cost=combined,
            confidence=np.ones(shape, dtype=np.float64),
            layers_used=[l.name for l in layers],
        )

    def fuzzy_overlay(self, layers: list[CostLayer]) -> FusionResult:
        """Fuzzy-logic AND overlay: combined = max of normalized costs."""
        shape = layers[0].grid.shape
        combined = np.zeros(shape, dtype=np.float64)
        for layer in layers:
            g = layer.grid.copy()
            g[~np.isfinite(g)] = 50000
            if g.max() > 0:
                combined = np.maximum(combined, g / g.max() * layer.weight)
        return FusionResult(
            combined_cost=combined * 100.0 + 1.0,  # rescale
            confidence=np.ones(shape, dtype=np.float64),
            layers_used=[l.name for l in layers],
        )

    def fuse(self, layers: list[CostLayer],
             strategy: str = "weighted_sum") -> FusionResult:
        """Fuse multiple cost layers using the specified strategy."""
        strategies = {
            "weighted_sum": self.weighted_sum,
            "multiplicative": self.multiplicative_fusion,
            "fuzzy": self.fuzzy_overlay,
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown fusion strategy: {strategy}. "
                             f"Choose from {list(strategies)}")
        return strategies[strategy](layers)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build_fused_cost(self,
                         dem: NDArray[np.float64],
                         terrain_grid: NDArray[np.int_],
                         water_mask: Optional[NDArray[np.bool_]] = None,
                         weather_cost_grid: Optional[NDArray[np.float64]] = None,
                         hazard_points: Optional[list[tuple[int, int, float]]] = None,
                         road_pixels: Optional[list[tuple[int, int]]] = None,
                         origin: Optional[GeoPoint] = None,
                         resolution_m: Optional[float] = None,
                         fusion_strategy: str = "weighted_sum") -> FusionResult:
        """Full pipeline: build all layers and fuse them.

        Returns a FusionResult with the combined cost surface.
        """
        res = resolution_m or self.terrain_config.dem_resolution_m
        origin = origin or GeoPoint(0, 0, 0)

        layers: list[CostLayer] = [
            self.terrain_analyzer.build_cost_layer(dem, terrain_grid, origin, res),
        ]

        # Slope emphasis
        slope_layer = self.slope_cost_layer(dem, res)
        slope_layer.origin_lat = origin.lat
        slope_layer.origin_lon = origin.lon
        slope_layer.weight = 0.3
        layers.append(slope_layer)

        # Water barriers
        if water_mask is not None:
            wl = self.water_barrier_layer(dem, water_mask, res)
            wl.weight = 5.0
            layers.append(wl)

        # Weather
        if weather_cost_grid is not None and self.weather_config.enabled:
            wl = self.weather_layer(weather_cost_grid, res, origin.lat, origin.lon)
            layers.append(wl)

        # Hazards
        if hazard_points:
            hl = self.hazard_layer(dem.shape, hazard_points, res)
            hl.weight = 3.0
            layers.append(hl)

        # Road attractors
        if road_pixels:
            rl = self.road_proximity_layer(dem.shape, road_pixels, res)
            rl.weight = 0.5
            layers.append(rl)

        return self.fuse(layers, strategy=fusion_strategy)
