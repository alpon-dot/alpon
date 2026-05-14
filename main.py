#!/usr/bin/env python3
"""Field Survey Agent — Demo & CLI entry point.

Usage
-----
    python main.py                          # Run demo with synthetic data
    python main.py --rows 300 --cols 300    # Larger terrain
    python main.py --targets 8              # More survey targets
    python main.py --export plan.json       # Export to file
    python main.py --fusion fuzzy           # Use fuzzy fusion strategy
    python main.py --no-optimize            # Disable target order optimization
"""

import argparse
import sys
from pathlib import Path

# Allow running from the package directory or as a module
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from field_survey_agent import (
    AgentConfig,
    FieldSurveyAgent,
    GeoPoint,
    SurveyTarget,
    SurveyVisualizer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Field Survey Agent — Multi-source Data Fusion & Route Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rows", type=int, default=200, help="DEM rows")
    p.add_argument("--cols", type=int, default=200, help="DEM cols")
    p.add_argument("--targets", type=int, default=5, help="Number of survey targets")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--fusion", choices=["weighted_sum", "multiplicative", "fuzzy"],
                   default="weighted_sum", help="Fusion strategy")
    p.add_argument("--no-optimize", action="store_true",
                   help="Disable evolutionary target order optimization")
    p.add_argument("--export", type=str, default="",
                   help="Export plan to JSON file")
    p.add_argument("--plot", type=str, default="",
                   help="Save overview plot to file")
    p.add_argument("--plot-3d", type=str, default="",
                   help="Save 3D terrain plot to file")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Field Survey Agent — Multi-source Data Fusion & Route Planning")
    print("=" * 60)

    # --- Initialize agent ---
    config = AgentConfig()
    config.route_planner.multi_objective_population = 50
    config.route_planner.multi_objective_generations = 80
    agent = FieldSurveyAgent(config)

    # --- Generate synthetic data ---
    print(f"\n[1/5] Generating synthetic DEM ({args.rows}x{args.cols}) ...")
    dem = agent.generate_synthetic_dem(args.rows, args.cols, seed=args.seed)

    print("[2/5] Generating synthetic terrain / land cover ...")
    terrain = agent.generate_synthetic_terrain(args.rows, args.cols, dem, seed=args.seed + 1)

    print("[3/5] Generating synthetic weather ...")
    weather = agent.generate_synthetic_weather(args.rows, args.cols, seed=args.seed + 2)

    # --- Create survey targets ---
    print(f"[4/5] Creating {args.targets} survey targets ...")
    rng = np.random.default_rng(args.seed + 3)
    targets = []
    for i in range(args.targets):
        tr, tc = rng.integers(10, args.rows - 10), rng.integers(10, args.cols - 10)
        targets.append(SurveyTarget(
            target_id=f"T{i + 1:03d}",
            lat=float(tr),
            lon=float(tc),
            elevation=dem[tr, tc],
            priority=round(float(rng.random()), 2),
            min_stay_minutes=int(rng.integers(15, 60)),
            description=f"Survey point {i + 1}",
        ))

    start = GeoPoint(lat=float(args.rows - 20), lon=20.0, elevation=dem[args.rows - 20, 20])

    # --- Plan survey ---
    print(f"[5/5] Planning survey route (fusion={args.fusion}, "
          f"optimize={not args.no_optimize}) ...")
    plan = agent.plan_survey(
        dem=dem,
        terrain_grid=terrain,
        start=start,
        targets=targets,
        weather_cells=weather,
        fusion_strategy=args.fusion,
        optimize_order=not args.no_optimize,
    )

    # --- Output ---
    print()
    print(agent.plan_summary(plan))

    if args.export:
        path = agent.export_plan(plan, args.export)
        print(f"\nPlan exported to: {path}")

    # --- Visualization ---
    if args.plot or args.plot_3d:
        from field_survey_agent.models import CostLayer
        from field_survey_agent.data_fusion import DataFusionEngine

        # Rebuild cost layer for visualization context
        engine = DataFusionEngine()
        fusion = engine.build_fused_cost(dem, terrain, origin=start)
        cost_layer = CostLayer(
            name="fused", grid=fusion.combined_cost,
            origin_lat=start.lat, origin_lon=start.lon,
            rows=args.rows, cols=args.cols,
        )

        viz = SurveyVisualizer()

        if args.plot:
            viz.plot_plan_overview(plan, cost_layer, save_path=args.plot)
            print(f"Overview plot saved to: {args.plot}")

        if args.plot_3d:
            route = plan.routes[0] if plan.routes else None
            viz.plot_dem_3d(dem, route, save_path=args.plot_3d)
            print(f"3D plot saved to: {args.plot_3d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
