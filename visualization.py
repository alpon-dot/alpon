"""Visualization utilities for survey plans and cost surfaces."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .models import CostLayer, FusionResult, GeoPoint, Route, SurveyPlan, SurveyTarget


class SurveyVisualizer:
    """Matplotlib-based visualization of survey plans and data layers."""

    @staticmethod
    def plot_cost_surface(cost_layer: CostLayer,
                          title: str = "Fused Cost Surface",
                          cmap: str = "YlOrRd",
                          alpha: float = 0.85,
                          figsize: tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None):
        """Plot a 2D cost surface."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        grid = cost_layer.grid.copy()
        grid[~np.isfinite(grid)] = np.nan
        vmax = np.nanpercentile(grid, 95) if np.any(np.isfinite(grid)) else 1

        im = ax.imshow(grid, cmap=cmap, origin="upper", alpha=alpha,
                       vmin=1, vmax=vmax, interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="Cost", shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_route_on_cost(cost_layer: CostLayer,
                           route: Route,
                           targets: Optional[list[SurveyTarget]] = None,
                           title: str = "Survey Route",
                           figsize: tuple[int, int] = (12, 9),
                           save_path: Optional[str] = None):
        """Plot a route overlaid on the cost surface."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        # Cost surface background
        grid = cost_layer.grid.copy()
        grid[~np.isfinite(grid)] = np.nan
        vmax = np.nanpercentile(grid, 95) if np.any(np.isfinite(grid)) else 1
        ax.imshow(grid, cmap="YlOrRd", origin="upper", alpha=0.6,
                  vmin=1, vmax=vmax, interpolation="bilinear")

        # Route
        if route.waypoints:
            lats = [wp.lat for wp in route.waypoints]
            lons = [wp.lon for wp in route.waypoints]
            ax.plot(lons, lats, "b-", linewidth=2, label="Route", zorder=3)
            ax.plot(lons[0], lats[0], "go", markersize=10, label="Start", zorder=4)
            ax.plot(lons[-1], lats[-1], "ro", markersize=10, label="End", zorder=4)

        # Targets
        if targets:
            t_lats = [t.lat for t in targets]
            t_lons = [t.lon for t in targets]
            ax.scatter(t_lons, t_lats, c="cyan", s=60, marker="D",
                       edgecolors="black", linewidth=0.5,
                       label="Targets", zorder=5)
            for t in targets:
                ax.annotate(t.target_id, (t.lon, t.lat),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=8, color="white",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

        ax.set_title(title)
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")
        ax.legend(loc="upper right")
        ax.invert_yaxis()  # match image origin

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_dem_3d(dem: NDArray[np.float64],
                    route: Optional[Route] = None,
                    title: str = "Terrain & Route",
                    figsize: tuple[int, int] = (12, 9),
                    save_path: Optional[str] = None):
        """3D terrain view with optional route overlay."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        rows, cols = dem.shape
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)

        # Downsample for performance
        stride = max(1, min(rows, cols) // 150)
        ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride],
                        dem[::stride, ::stride],
                        cmap="terrain", alpha=0.85, linewidth=0,
                        antialiased=True)

        if route and route.waypoints:
            rlats = np.array([wp.lat for wp in route.waypoints])
            rlons = np.array([wp.lon for wp in route.waypoints])
            # Clamp to valid range
            valid = (rlats >= 0) & (rlats < rows) & (rlons >= 0) & (rlons < cols)
            rlats, rlons = rlats[valid], rlons[valid]
            if len(rlats) > 0:
                elevations = dem[rlats.astype(int), rlons.astype(int)]
                ax.plot(rlons, rlats, elevations + 5,
                        "r-", linewidth=3, label="Route", zorder=5)

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Elevation (m)")
        ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_fusion_layers(fusion_result: FusionResult,
                           layers: list[CostLayer],
                           figsize: tuple[int, int] = (16, 12),
                           save_path: Optional[str] = None):
        """Multi-panel plot showing input layers and fused result."""
        import matplotlib.pyplot as plt

        n_plots = 1 + len(layers)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for i, layer in enumerate(layers):
            grid = layer.grid.copy()
            grid[~np.isfinite(grid)] = np.nan
            vmax = np.nanpercentile(grid, 95) if np.any(np.isfinite(grid)) else 1
            im = axes[i].imshow(grid, cmap="YlOrRd", origin="upper",
                                vmin=0, vmax=vmax)
            axes[i].set_title(f"{layer.name} (w={layer.weight:.2f})")
            plt.colorbar(im, ax=axes[i], shrink=0.8)

        # Fused result
        fused = fusion_result.combined_cost
        fused[~np.isfinite(fused)] = np.nan
        vmax = np.nanpercentile(fused, 95) if np.any(np.isfinite(fused)) else 1
        im = axes[len(layers)].imshow(fused, cmap="YlOrRd", origin="upper",
                                      vmin=0, vmax=vmax)
        axes[len(layers)].set_title("Fused Cost + Confidence")
        plt.colorbar(im, ax=axes[len(layers)], shrink=0.8)

        # Overlay confidence contours
        if len(layers) > 0:
            conf = fusion_result.confidence
            axes[len(layers)].contour(conf, levels=[0.3, 0.5, 0.7],
                                      colors=["red", "orange", "green"],
                                      linewidths=1, alpha=0.6)

        # Hide unused axes
        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Data Fusion: Input Layers & Result", fontsize=14)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_plan_overview(plan: SurveyPlan,
                           cost_layer: Optional[CostLayer] = None,
                           figsize: tuple[int, int] = (14, 10),
                           save_path: Optional[str] = None):
        """Complete plan overview with routes, targets, and stats."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        # Cost background
        if cost_layer is not None:
            grid = cost_layer.grid.copy()
            grid[~np.isfinite(grid)] = np.nan
            vmax = np.nanpercentile(grid, 95) if np.any(np.isfinite(grid)) else 1
            ax.imshow(grid, cmap="YlOrRd", origin="upper", alpha=0.5,
                      vmin=1, vmax=vmax, interpolation="bilinear")

        colors = ["blue", "orange", "green", "purple"]
        for i, route in enumerate(plan.routes):
            if route.waypoints:
                lats = [wp.lat for wp in route.waypoints]
                lons = [wp.lon for wp in route.waypoints]
                color = colors[i % len(colors)]
                ax.plot(lons, lats, "-", color=color, linewidth=2.5,
                        label=f"Route {i + 1}", zorder=3)
                if i == 0:
                    ax.plot(lons[0], lats[0], "go", markersize=12,
                            markeredgecolor="black", zorder=4)
                ax.plot(lons[-1], lats[-1], "rs", markersize=8, zorder=4)

        # Targets
        if plan.targets:
            t_lats = [t.lat for t in plan.targets]
            t_lons = [t.lon for t in plan.targets]
            sizes = [30 + t.priority * 70 for t in plan.targets]
            ax.scatter(t_lons, t_lats, c="cyan", s=sizes, marker="D",
                       edgecolors="black", linewidth=0.8,
                       label=f"Targets ({len(plan.targets)})", zorder=5)

        # Info box
        info = (
            f"Distance: {plan.total_distance_m / 1000:.1f} km\n"
            f"Time: {plan.total_time_hours:.1f} h\n"
            f"Risk: {plan.overall_risk.name}\n"
            f"Targets: {len(plan.targets)}"
        )
        ax.text(0.02, 0.98, info, transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))

        ax.set_title("Field Survey Plan Overview", fontsize=14)
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")
        ax.legend(loc="lower right")
        ax.invert_yaxis()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
