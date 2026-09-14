"""Run both EA variants and the random-search baseline over all seeds."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from a1_config import (
    CROSSOVER_PROBABILITY,
    EA_VARIANTS,
    GENERATIONS,
    MAX_TOTAL_NODES,
    MODULE_BUDGET,
    POPULATION_SIZE,
    RESULTS_ROOT,
    SEEDS,
    TOURNAMENT_SIZE,
    evaluation_budget,
)
from a1_ea import EAVariant, run_ea
from a1_random_search import run_random_search


type RunSummary = dict[str, float | int | str]


def create_result_directory() -> Path:
    """Create a unique directory so previous experiments are not overwritten."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = RESULTS_ROOT / timestamp
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def save_experiment_configuration(result_dir: Path) -> None:
    """Store the settings needed to reproduce the experiment."""
    configuration: dict[str, Any] = {
        "research_question": (
            "Does using crossover improve the performance of an evolutionary "
            "algorithm for evolving robot body phenotypes compared with "
            "mutation-only evolution?"
        ),
        "encoding": "tree",
        "module_budget_argument": MODULE_BUDGET,
        "maximum_total_nodes": MAX_TOTAL_NODES,
        "population_size": POPULATION_SIZE,
        "generations": GENERATIONS,
        "seeds": SEEDS,
        "tournament_size": TOURNAMENT_SIZE,
        "crossover_probability": CROSSOVER_PROBABILITY,
        "mutation_probability_per_child": 1.0,
        "evaluation_budget_per_run": evaluation_budget(),
        "fitness_direction": "minimisation",
        "fitness": "mean target tree-edit distance + standard deviation",
    }
    path = result_dir / "experiment_config.json"
    path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def print_summary(summary: RunSummary) -> None:
    """Print a concise completion line for one run."""
    print(
        f"{summary['variant']}, seed={summary['seed']}, "
        f"best={float(summary['best_fitness']):.4f}, "
        f"evaluations={summary['evaluations']}",
    )


def main() -> None:
    """Run 2 EA variants and random search using identical seed sets."""
    result_dir = create_result_directory()
    csv_path = result_dir / "results.csv"
    save_experiment_configuration(result_dir)

    print("Assignment 1 experiments")
    print(f"Results directory: {result_dir}")
    print(f"Evaluation budget per run: {evaluation_budget()}\n")

    for variant_name in EA_VARIANTS:
        variant = cast(EAVariant, variant_name)
        for seed in SEEDS:
            print(f"Running {variant}, seed={seed}...")
            print_summary(run_ea(variant, seed, result_dir, csv_path))

    for seed in SEEDS:
        print(f"Running random_search, seed={seed}...")
        print_summary(run_random_search(seed, result_dir, csv_path))

    print("\nAll experiments completed.")
    print(f"Results CSV: {csv_path}")
    print(
        "Plot command:\n"
        "uv run assignments/assignment_1/a1_plot_results.py "
        f'"{csv_path}"',
    )


if __name__ == "__main__":
    main()
