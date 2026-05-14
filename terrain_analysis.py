"""Terrain analysis: slope, aspect, traversability, cost surfaces."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .config import TerrainConfig
from .models import GeoPoint, TerrainCell, TerrainType, CostLayer


class TerrainAnalyzer:
    """Analyzes terrain data to produce cost and traversability surfaces."""

    def __init__(self, config: Optional[TerrainConfig] = None):
        self.config = config or TerrainConfig()

    # ------------------------------------------------------------------
    # Digital Elevation Model processing
    # ------------------------------------------------------------------

    def slope_degrees(self, dem: NDArray[np.float64], resolution_m: float) -> NDArray[np.float64]:
        """Compute slope in degrees from a DEM using Horn's method."""
        dzdx = ndimage.sobel(dem, axis=1) / (8.0 * resolution_m)
        dzdy = ndimage.sobel(dem, axis=0) / (8.0 * resolution_m)
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        return np.degrees(slope_rad)

    def aspect(self, dem: NDArray[np.float64], resolution_m: float) -> NDArray[np.float64]:
        """Compute aspect (0–360°) from a DEM."""
        dzdx = ndimage.sobel(dem, axis=1) / (8.0 * resolution_m)
        dzdy = ndimage.sobel(dem, axis=0) / (8.0 * resolution_m)
        aspect_rad = np.arctan2(-dzdx, dzdy)
        return (np.degrees(aspect_rad) + 360.0) % 360.0

    def hillshade(self, dem: NDArray[np.float64], resolution_m: float,
                  azimuth: float = 315.0, altitude: float = 45.0) -> NDArray[np.float64]:
        """Compute hillshade for visualization."""
        slope = np.radians(self.slope_degrees(dem, resolution_m))
        asp = np.radians(self.aspect(dem, resolution_m))
        az_rad = np.radians(360.0 - azimuth + 90.0)
        alt_rad = np.radians(altitude)
        return (np.cos(alt_rad) * np.cos(slope)
                + np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - asp))

    def roughness(self, dem: NDArray[np.float64]) -> NDArray[np.float64]:
        """Terrain roughness index (TRI): mean absolute difference to neighbours."""
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        neighbor_mean = ndimage.convolve(dem, kernel) / 8.0
        return np.abs(dem - neighbor_mean)

    # ------------------------------------------------------------------
    # Traversability & cost
    # ------------------------------------------------------------------

    def terrain_speed_factor(self, terrain_grid: NDArray[np.int_]) -> NDArray[np.float64]:
        """Map terrain type codes to speed multipliers."""
        speed_map = np.ones(10, dtype=np.float64)
        speed_map[TerrainType.OPEN.value] = 1.0
        speed_map[TerrainType.FOREST.value] = self.config.forest_speed_factor
        speed_map[TerrainType.SHRUB.value] = self.config.shrub_speed_factor
        speed_map[TerrainType.GRASSLAND.value] = self.config.grassland_speed_factor
        speed_map[TerrainType.WETLAND.value] = self.config.wetland_speed_factor
        speed_map[TerrainType.BARE_ROCK.value] = self.config.bare_rock_speed_factor
        speed_map[TerrainType.URBAN.value] = 1.0
        speed_map[TerrainType.CROPLAND.value] = 0.85
        speed_map[TerrainType.WATER.value] = 0.0  # impassable
        speed_map[TerrainType.SNOW_ICE.value] = 0.3
        return speed_map[terrain_grid]

    def slope_cost(self, slope_deg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert slope to a cost multiplier.

        Flat terrain = 1.0; max configured slope = infinite cost.
        """
        max_s = self.config.max_slope_degrees
        clipped = np.clip(slope_deg, 0, max_s)
        normalized = clipped / max_s
        return 1.0 + (np.power(normalized, self.config.slope_cost_exponent) * 100.0)

    def traversability(self, slope_deg: NDArray[np.float64],
                       terrain_grid: NDArray[np.int_]) -> NDArray[np.float64]:
        """Per-cell traversability score (0 = impassable, 1 = effortless)."""
        speed = self.terrain_speed_factor(terrain_grid)
        slope_c = self.slope_cost(slope_deg)
        traversable = speed / np.maximum(slope_c, 1.0)
        traversable[terrain_grid == TerrainType.WATER.value] = 0.0
        traversable[slope_deg > self.config.max_slope_degrees] = 0.0
        return np.clip(traversable, 0.0, 1.0)

    def cost_surface(self, dem: NDArray[np.float64], terrain_grid: NDArray[np.int_],
                     resolution_m: float) -> CostLayer:
        """Build the base terrain cost layer from DEM and land cover."""
        slope = self.slope_degrees(dem, resolution_m)
        trav = self.traversability(slope, terrain_grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = np.where(trav > 0, 1.0 / trav, 10000.0)
        cost[~np.isfinite(cost)] = 10000.0
        return CostLayer(
            name="terrain_cost",
            grid=cost.astype(np.float64),
            resolution_m=resolution_m,
            rows=cost.shape[0],
            cols=cost.shape[1],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def latlon_to_grid(self, lat: float, lon: float, origin_lat: float, origin_lon: float,
                       resolution_m: float) -> tuple[int, int]:
        """Approximate lat/lon → grid indices (assumes small area, planar approx)."""
        row = int((origin_lat - lat) / (resolution_m / 111_320.0))
        col = int((lon - origin_lon) / (resolution_m / (111_320.0 * np.cos(np.radians(lat)))))
        return row, col

    def grid_to_latlon(self, row: int, col: int, origin_lat: float, origin_lon: float,
                       resolution_m: float) -> tuple[float, float]:
        """Grid indices → approximate lat/lon."""
        lat = origin_lat - row * (resolution_m / 111_320.0)
        lon = origin_lon + col * (resolution_m / (111_320.0 * np.cos(np.radians(lat))))
        return lat, lon

    def build_cost_layer(self, dem: NDArray[np.float64], terrain_grid: NDArray[np.int_],
                         origin: GeoPoint, resolution_m: Optional[float] = None) -> CostLayer:
        """Convenience: build a full CostLayer from raw grids."""
        res = resolution_m or self.config.dem_resolution_m
        cost = self.cost_surface(dem, terrain_grid, res)
        cost.origin_lat = origin.lat
        cost.origin_lon = origin.lon
        return cost
