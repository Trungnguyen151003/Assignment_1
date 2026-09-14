"""Aggregate independent runs, create plots, and summarize final fitness."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib.pyplot as plt

from a1_config import ALL_VARIANTS


VARIANT_LABELS = {
    "mutation_only": "Mutation only",
    "crossover": "Mutation + crossover",
    "random_search": "Random search",
}
VARIANT_COLORS = {
    "mutation_only": "tab:blue",
    "crossover": "tab:orange",
    "random_search": "tab:green",
}


def load_results(csv_path: Path) -> list[dict[str, str]]:
    """Read experiment rows from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def aggregate_metric(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[str, dict[int, tuple[float, float]]]:
    """Compute mean and population SD across seeds at every generation."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], int(row["generation"]))].append(
            float(row[metric]),
        )

    aggregated: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for (variant, generation), values in grouped.items():
        aggregated[variant][generation] = (fmean(values), pstdev(values))
    return aggregated


def plot_metric(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot cross-run mean with a shaded plus/minus-one-SD band."""
    aggregated = aggregate_metric(rows, metric)
    plt.figure(figsize=(8, 5))

    for variant in ALL_VARIANTS:
        data = aggregated.get(variant)
        if not data:
            continue
        generations = sorted(data)
        means = [data[generation][0] for generation in generations]
        deviations = [data[generation][1] for generation in generations]
        lower = [mean - sd for mean, sd in zip(means, deviations, strict=True)]
        upper = [mean + sd for mean, sd in zip(means, deviations, strict=True)]
        color = VARIANT_COLORS[variant]
        plt.plot(
            generations,
            means,
            color=color,
            linewidth=2,
            label=VARIANT_LABELS[variant],
        )
        plt.fill_between(generations, lower, upper, color=color, alpha=0.2)

    plt.xlabel("Generation / random-search checkpoint")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_final_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the latest row for each variant and seed."""
    latest: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["variant"], int(row["seed"]))
        if key not in latest or int(row["generation"]) > int(
            latest[key]["generation"],
        ):
            latest[key] = row
    return list(latest.values())


def write_final_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write and print final-best descriptive statistics per method."""
    final_rows = get_final_rows(rows)
    summaries: list[dict[str, Any]] = []

    for variant in ALL_VARIANTS:
        values = [
            float(row["best_fitness"])
            for row in final_rows
            if row["variant"] == variant
        ]
        if not values:
            continue
        summaries.append(
            {
                "variant": variant,
                "number_of_runs": len(values),
                "mean_final_best_fitness": fmean(values),
                "std_final_best_fitness": pstdev(values),
                "minimum_final_best_fitness": min(values),
                "maximum_final_best_fitness": max(values),
            },
        )

    fields = [
        "variant",
        "number_of_runs",
        "mean_final_best_fitness",
        "std_final_best_fitness",
        "minimum_final_best_fitness",
        "maximum_final_best_fitness",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print("Final best fitness (lower is better):")
    for summary in summaries:
        print(
            f"  {VARIANT_LABELS[summary['variant']]}: "
            f"{summary['mean_final_best_fitness']:.4f} +/- "
            f"{summary['std_final_best_fitness']:.4f}",
        )


def main() -> None:
    """Generate three figures and one final summary CSV."""
    parser = argparse.ArgumentParser(description="Plot Assignment 1 results.")
    parser.add_argument("csv_path", type=Path, help="Path to results.csv")
    args = parser.parse_args()

    rows = load_results(args.csv_path)
    output_dir = args.csv_path.parent
    plot_metric(
        rows,
        "best_fitness",
        "Best fitness across generations",
        "Best fitness (lower is better)",
        output_dir / "best_fitness.png",
    )
    plot_metric(
        rows,
        "mean_fitness",
        "Mean fitness across generations",
        "Mean fitness (lower is better)",
        output_dir / "mean_fitness.png",
    )
    plot_metric(
        rows,
        "std_fitness",
        "Fitness variation across generations",
        "Within-population fitness standard deviation",
        output_dir / "std_fitness.png",
    )
    write_final_summary(rows, output_dir / "final_summary.csv")
    print(f"Plots and summary saved in: {output_dir}")


if __name__ == "__main__":
    main()
