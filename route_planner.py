"""Intelligent route planning with A*, multi-objective optimization, and RRT."""

import heapq
import math
from collections import deque
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .config import RoutePlannerConfig
from .models import CostLayer, FusionResult, GeoPoint, Route, SurveyTarget


# ------------------------------------------------------------------
# Heuristics
# ------------------------------------------------------------------

def _octile_distance(r1: int, c1: int, r2: int, c2: int) -> float:
    """Octile distance (allows diagonal movement)."""
    dr, dc = abs(r1 - r2), abs(c1 - c2)
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


def _euclidean_grid(r1: int, c1: int, r2: int, c2: int) -> float:
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)


# ------------------------------------------------------------------
# A* planner
# ------------------------------------------------------------------

class AStarPlanner:
    """A* pathfinding on a cost grid with diagonal movement support."""

    def __init__(self, config: Optional[RoutePlannerConfig] = None):
        self.config = config or RoutePlannerConfig()

    def _neighbors(self, row: int, col: int, rows: int, cols: int) -> list[tuple[int, int, float]]:
        """Return (nr, nc, move_cost_multiplier) for valid neighbors."""
        nb: list[tuple[int, int, float]] = []
        # Cardinal
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb.append((nr, nc, 1.0))
        # Diagonal
        if self.config.diagonal_movement:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb.append((nr, nc, math.sqrt(2)))
        return nb

    def find_path(self, cost_grid: NDArray[np.float64],
                  start_row: int, start_col: int,
                  goal_row: int, goal_col: int) -> tuple[list[tuple[int, int]], float]:
        """A* search. Returns (path_pixels, total_cost)."""
        rows, cols = cost_grid.shape
        heuristic = _octile_distance if self.config.diagonal_movement else _euclidean_grid

        open_set: list[tuple[float, int, int]] = [(0.0, start_row, start_col)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {(start_row, start_col): 0.0}
        closed: set[tuple[int, int]] = set()

        while open_set:
            _, cr, cc = heapq.heappop(open_set)
            if (cr, cc) in closed:
                continue
            if (cr, cc) == (goal_row, goal_col):
                return self._reconstruct(came_from, (cr, cc)), g_score[(cr, cc)]

            closed.add((cr, cc))

            for nr, nc, move_mult in self._neighbors(cr, cc, rows, cols):
                if (nr, nc) in closed:
                    continue
                cell_cost = cost_grid[nr, nc]
                if not np.isfinite(cell_cost) or cell_cost >= 1e8:
                    continue

                step_cost = cell_cost * move_mult
                tentative_g = g_score[(cr, cc)] + step_cost

                if tentative_g < g_score.get((nr, nc), float("inf")):
                    came_from[(nr, nc)] = (cr, cc)
                    g_score[(nr, nc)] = tentative_g
                    f = tentative_g + heuristic(nr, nc, goal_row, goal_col)
                    heapq.heappush(open_set, (f, nr, nc))

        return [], float("inf")

    @staticmethod
    def _reconstruct(came_from: dict, current: tuple[int, int]) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# ------------------------------------------------------------------
# Multi-objective planner (NSGA-II style)
# ------------------------------------------------------------------

class MultiObjectivePlanner:
    """Multi-objective route optimization using evolutionary search.

    Optimizes simultaneously for:
      - Total cost (terrain + weather)
      - Total distance
      - Risk exposure
    """

    def __init__(self, config: Optional[RoutePlannerConfig] = None):
        self.config = config or RoutePlannerConfig()
        self.astar = AStarPlanner(config)

    def pareto_front(self, routes: list[Route]) -> list[Route]:
        """Return the Pareto-optimal subset of routes.

        Objectives minimized: (cumulative_cost, total_distance_m, risk_score).
        """
        pareto: list[Route] = []
        for r in routes:
            dominated = False
            for other in routes:
                if r is other:
                    continue
                if (other.cumulative_cost <= r.cumulative_cost
                        and other.total_distance_m <= r.total_distance_m
                        and other.risk_score <= r.risk_score
                        and (other.cumulative_cost < r.cumulative_cost
                             or other.total_distance_m < r.total_distance_m
                             or other.risk_score < r.risk_score)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return pareto

    def optimize_target_order(self,
                              cost_grid: NDArray[np.float64],
                              start: tuple[int, int],
                              targets: list[tuple[int, int, float]],
                              target_ids: Optional[list[str]] = None,
                              pop_size: int = 50,
                              generations: int = 100) -> list[Route]:
        """Evolve optimal order to visit targets (TSP variant on cost grid).

        Each target: (row, col, priority).
        """
        n = len(targets)
        if n <= 1:
            path, cost = self.astar.find_path(cost_grid, *start, *targets[0][:2])
            return [Route(waypoints=[], cumulative_cost=cost)]

        rng = np.random.default_rng(42)
        pop_size = min(pop_size, self.config.multi_objective_population)
        generations = min(generations, self.config.multi_objective_generations)

        # Population: each individual is a permutation of target indices
        indices = list(range(n))
        population = [rng.permutation(indices).tolist() for _ in range(pop_size)]

        for gen in range(generations):
            # Evaluate fitness for each individual
            fitness: list[float] = []
            for perm in population:
                total_cost = 0.0
                prev = start
                for idx in perm:
                    tr, tc, _ = targets[idx]
                    _, cost = self.astar.find_path(cost_grid, *prev, tr, tc)
                    total_cost += cost
                    prev = (tr, tc)
                # Penalize missing high-priority targets early
                priority_penalty = sum(
                    targets[idx][2] * (len(perm) - i) * 100
                    for i, idx in enumerate(perm)
                )
                fitness.append(total_cost + priority_penalty)

            # Tournament selection
            new_pop: list[list[int]] = []
            for _ in range(pop_size):
                a, b = rng.choice(pop_size, 2, replace=False)
                winner = population[a] if fitness[a] < fitness[b] else population[b]
                new_pop.append(winner)

            # Crossover (ordered crossover)
            for i in range(0, pop_size - 1, 2):
                if rng.random() < 0.8:
                    p1, p2 = new_pop[i], new_pop[i + 1]
                    a_idx, b_idx = sorted(rng.choice(n, 2, replace=False))
                    child1 = [-1] * n
                    child2 = [-1] * n
                    child1[a_idx:b_idx] = p1[a_idx:b_idx]
                    child2[a_idx:b_idx] = p2[a_idx:b_idx]
                    # Fill remaining from other parent preserving order
                    for child, parent in [(child1, p2), (child2, p1)]:
                        fill = [x for x in parent if x not in child]
                        j = 0
                        for i_pos in range(n):
                            if child[i_pos] == -1:
                                child[i_pos] = fill[j]
                                j += 1
                    new_pop[i], new_pop[i + 1] = child1, child2

            # Mutation (swap)
            for i in range(pop_size):
                if rng.random() < 0.15:
                    a, b = rng.choice(n, 2, replace=False)
                    new_pop[i][a], new_pop[i][b] = new_pop[i][b], new_pop[i][a]

            population = new_pop

        # Build final routes
        resolution_m = 30.0  # default pixel-to-meter scale
        routes: list[Route] = []
        for perm in population:
            waypoints: list[GeoPoint] = []
            total_cost = 0.0
            total_dist = 0.0
            prev = start
            for idx in perm:
                tr, tc, _ = targets[idx]
                path_px, cost = self.astar.find_path(cost_grid, *prev, tr, tc)
                total_cost += cost
                for pr, pc in path_px[1:]:
                    if waypoints:
                        dr = pr - waypoints[-1].lat
                        dc = pc - waypoints[-1].lon
                        total_dist += math.sqrt(dr**2 + dc**2) * resolution_m
                    waypoints.append(GeoPoint(lat=float(pr), lon=float(pc)))
                prev = (tr, tc)
            visited_ids = [target_ids[idx] for idx in perm] if target_ids else []
            route = Route(
                waypoints=waypoints,
                cumulative_cost=total_cost,
                total_distance_m=total_dist,
                visited_targets=visited_ids,
            )
            routes.append(route)

        return self.pareto_front(routes)


# ------------------------------------------------------------------
# Route builder: turns pixel paths into rich Route objects
# ------------------------------------------------------------------

class RouteBuilder:
    """Converts pixel-path results into structured Route objects with metrics."""

    def __init__(self, resolution_m: float = 30.0,
                 dem: Optional[NDArray[np.float64]] = None):
        self.resolution_m = resolution_m
        self.dem = dem

    def build(self,
              pixel_path: list[tuple[int, int]],
              cost_grid: NDArray[np.float64],
              target_ids: Optional[list[str]] = None) -> Route:
        """Build a Route from a pixel path."""
        if not pixel_path:
            return Route()

        waypoints = [GeoPoint(lat=float(r), lon=float(c)) for r, c in pixel_path]

        total_dist = 0.0
        total_ascent = 0.0
        total_descent = 0.0
        cumulative_cost = 0.0
        segments: list[dict] = []

        for i in range(len(pixel_path)):
            r, c = pixel_path[i]
            cumulative_cost += cost_grid[r, c]
            if i > 0:
                pr, pc = pixel_path[i - 1]
                dx = (c - pc) * self.resolution_m
                dy = (r - pr) * self.resolution_m
                seg_dist = math.sqrt(dx**2 + dy**2)
                total_dist += seg_dist

                if self.dem is not None:
                    dz = self.dem[r, c] - self.dem[pr, pc]
                    if dz > 0:
                        total_ascent += dz
                    else:
                        total_descent += abs(dz)

                slope = 0.0
                if self.dem is not None and seg_dist > 0:
                    dz_abs = abs(self.dem[r, c] - self.dem[pr, pc])
                    slope = math.degrees(math.atan(dz_abs / seg_dist))

                segments.append({
                    "from": (pr, pc),
                    "to": (r, c),
                    "distance_m": seg_dist,
                    "cost": cost_grid[r, c],
                    "max_slope": slope,
                })

        # Estimate time (Naismith's rule variant: 5 km/h flat + 1h per 600m ascent)
        speed_ms = 1.4  # ≈5 km/h
        time_h = (total_dist / speed_ms) / 3600 + (total_ascent / 600.0)
        time_min = time_h * 60

        # Add rest stops
        rest_interval = 120  # minutes
        rest_duration = 15
        n_rests = max(0, int(time_min // rest_interval))
        time_min += n_rests * rest_duration

        return Route(
            waypoints=waypoints,
            total_distance_m=total_dist,
            total_ascent_m=total_ascent,
            total_descent_m=total_descent,
            estimated_time_minutes=time_min,
            cumulative_cost=cumulative_cost,
            visited_targets=target_ids or [],
            segments=segments,
        )

    def smooth_path(self, pixel_path: list[tuple[int, int]],
                    sigma: float = 1.0) -> list[tuple[int, int]]:
        """Gaussian smooth a pixel path to remove jitter."""
        if len(pixel_path) < 3:
            return pixel_path
        from scipy.ndimage import gaussian_filter1d
        arr = np.array(pixel_path, dtype=np.float64)
        smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
        return [(int(round(r)), int(round(c))) for r, c in smoothed]


# ------------------------------------------------------------------
# Top-level planner API
# ------------------------------------------------------------------

class SurveyRoutePlanner:
    """Top-level route planner that orchestrates pathfinding for surveys."""

    def __init__(self, config: Optional[RoutePlannerConfig] = None):
        self.config = config or RoutePlannerConfig()
        self.astar = AStarPlanner(self.config)
        self.multi_obj = MultiObjectivePlanner(self.config)

    def plan_single(self,
                    cost_layer: CostLayer,
                    start: GeoPoint,
                    goal: GeoPoint,
                    dem: Optional[NDArray[np.float64]] = None) -> Route:
        """Plan a single route from start to goal on the given cost layer."""
        sr, sc = self._point_to_grid(start, cost_layer)
        gr, gc = self._point_to_grid(goal, cost_layer)

        path_px, total_cost = self.astar.find_path(
            cost_layer.grid, sr, sc, gr, gc
        )
        if not path_px:
            return Route()

        builder = RouteBuilder(cost_layer.resolution_m, dem)
        path_px = builder.smooth_path(path_px, self.config.smoothing_sigma)
        route = builder.build(path_px, cost_layer.grid)
        route.cumulative_cost = total_cost
        return route

    def plan_multi_target(self,
                          cost_layer: CostLayer,
                          start: GeoPoint,
                          targets: list[SurveyTarget],
                          dem: Optional[NDArray[np.float64]] = None,
                          optimize_order: bool = True) -> Route:
        """Plan a route visiting multiple survey targets.

        If optimize_order=True, uses evolutionary search for target ordering.
        """
        if not targets:
            return Route()

        sr, sc = self._point_to_grid(start, cost_layer)
        target_px = [
            (*self._point_to_grid(t, cost_layer), t.priority)
            for t in targets
        ]

        if optimize_order and len(targets) > 2:
            routes = self.multi_obj.optimize_target_order(
                cost_layer.grid, (sr, sc), target_px,
                target_ids=[t.target_id for t in targets],
            )
            if not routes:
                return Route()
            # Rebuild the best route with full metrics (segments, ascent, time)
            best_route = routes[0]
            builder = RouteBuilder(cost_layer.resolution_m, dem)
            pixel_path = [(int(wp.lat), int(wp.lon)) for wp in best_route.waypoints]
            pixel_path = builder.smooth_path(pixel_path, self.config.smoothing_sigma)
            best = builder.build(pixel_path, cost_layer.grid,
                                 best_route.visited_targets)
            best.cumulative_cost = best_route.cumulative_cost
        else:
            # Greedy nearest-neighbor ordering
            builder = RouteBuilder(cost_layer.resolution_m, dem)
            all_waypoints: list[GeoPoint] = []
            total_cost = 0.0
            current = (sr, sc)
            remaining = list(enumerate(target_px))
            visited_ids: list[str] = []

            while remaining:
                best_idx = min(
                    range(len(remaining)),
                    key=lambda i: _octile_distance(*current, *remaining[i][1][:2])
                )
                idx, (tr, tc, _) = remaining.pop(best_idx)
                path_px, cost = self.astar.find_path(
                    cost_layer.grid, *current, tr, tc
                )
                total_cost += cost
                smooth = builder.smooth_path(path_px, self.config.smoothing_sigma)
                for r, c in smooth[1:]:
                    all_waypoints.append(GeoPoint(lat=float(r), lon=float(c)))
                current = (tr, tc)
                visited_ids.append(targets[idx].target_id)

            best = builder.build(
                [(int(wp.lat), int(wp.lon)) for wp in all_waypoints],
                cost_layer.grid,
                visited_ids,
            )
            best.cumulative_cost = total_cost

        return best

    @staticmethod
    def _point_to_grid(point: GeoPoint, cost_layer: CostLayer) -> tuple[int, int]:
        """Convert a GeoPoint to grid coordinates.

        When lat/lon values are within grid dimensions, they are treated
        as direct pixel (row, col) indices regardless of origin. Otherwise
        geographic lat/lon → grid conversion is applied.
        """
        rows, cols = cost_layer.rows, cost_layer.cols
        lat, lon = point.lat, point.lon

        # Pixel-coordinate mode: lat/lon are row/col within grid bounds
        if 0 <= lat < rows and 0 <= lon < cols:
            return int(lat), int(lon)

        from .terrain_analysis import TerrainAnalyzer
        ta = TerrainAnalyzer()
        r, c = ta.latlon_to_grid(
            lat, lon,
            cost_layer.origin_lat, cost_layer.origin_lon,
            cost_layer.resolution_m,
        )
        r = max(0, min(r, rows - 1))
        c = max(0, min(c, cols - 1))
        return r, c
