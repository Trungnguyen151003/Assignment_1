"""Command-line runner for paired Assignment 2 experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from rich.console import Console

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assignments.assignment_2.a2_config import (
    DEFAULT_OUTPUT_ROOT,
    ExperimentConfig,
    Variant,
)
from assignments.assignment_2.a2_operators import initial_weight_population
from assignments.assignment_2.a2_run_ea import run_variant  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the experiment command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("fixed", "adaptive", "random"),
        default=("fixed", "adaptive", "random"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--offspring-count", type=int, default=50)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fixed-sigma", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run requested variants sequentially with paired initial populations."""
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        population_size=args.population_size,
        offspring_count=args.offspring_count,
        generations=args.generations,
        tournament_size=args.tournament_size,
        simulation_duration=args.duration,
        fixed_sigma=args.fixed_sigma,
    )
    variants = cast("list[Variant]", args.variants)
    console = Console()
    console.print(
        f"Running {len(variants)} variants x {len(args.seeds)} seeds; "
        f"budget={config.evaluation_budget:,} evaluations per run",
    )
    for seed in args.seeds:
        shared_weights = initial_weight_population(config, seed)
        for variant in variants:
            console.print(f"[cyan]{variant}[/cyan] seed={seed}")
            result = run_variant(
                variant,
                seed,
                config,
                args.output,
                shared_initial_weights=shared_weights,
                quiet=args.quiet,
            )
            console.print(f"  best final distance: {result.best_fitness:.6f}")


if __name__ == "__main__":
    main()
