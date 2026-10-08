"""Analyse completed Assignment 2 runs and create report-ready outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assignments.assignment_2.a2_config import DEFAULT_OUTPUT_ROOT  # noqa: E402


@dataclass(frozen=True)
class RunCurve:
    """Generation-level observations for one variant and seed."""

    variant: str
    seed: int
    evaluations: np.ndarray
    best_fitness: np.ndarray
    mean_sigma: np.ndarray

    @property
    def final_fitness(self) -> float:
        return float(self.best_fitness[-1])

    @property
    def normalized_auc(self) -> float:
        width = float(self.evaluations[-1] - self.evaluations[0])
        if width <= 0.0:
            msg = "AUC requires at least two distinct evaluation counts"
            raise ValueError(
                msg,
            )
        return float(np.trapezoid(self.best_fitness, self.evaluations) / width)


def _read_curve(path: Path) -> RunCurve:
    variant = path.parent.parent.name
    seed = int(path.parent.name.removeprefix("seed_"))
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        msg = f"run has insufficient generations: {path}"
        raise ValueError(msg)
    return RunCurve(
        variant=variant,
        seed=seed,
        evaluations=np.asarray([int(row["evaluations"]) for row in rows]),
        best_fitness=np.asarray([float(row["best_fitness"]) for row in rows]),
        mean_sigma=np.asarray([float(row["mean_sigma"]) for row in rows]),
    )


def load_runs(results_root: Path) -> list[RunCurve]:
    """Load every completed run below a result root."""
    paths = sorted(results_root.glob("*/seed_*/generations.csv"))
    if not paths:
        msg = f"no generations.csv files found below {results_root}"
        raise FileNotFoundError(
            msg,
        )
    return [_read_curve(path) for path in paths]


def paired_permutation_test(differences: np.ndarray) -> float:
    """Return a two-sided paired sign-flip permutation p-value."""
    differences = np.asarray(differences, dtype=float)
    if len(differences) == 0:
        msg = "at least one paired difference is required"
        raise ValueError(msg)
    observed = abs(float(np.mean(differences)))
    if len(differences) <= 20:
        statistics = (
            abs(float(np.mean(differences * np.asarray(signs))))
            for signs in product((-1.0, 1.0), repeat=len(differences))
        )
        extreme = sum(statistic >= observed - 1e-12 for statistic in statistics)
        return extreme / (2 ** len(differences))

    rng = np.random.default_rng(20261002)
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(differences)))
    statistics = np.abs(np.mean(signs * differences, axis=1))
    return float((np.count_nonzero(statistics >= observed) + 1) / 100_001)


def bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a deterministic percentile bootstrap CI for a paired mean."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(20261002)
    samples = rng.choice(values, size=(20_000, len(values)), replace=True)
    means = np.mean(samples, axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def _sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _paired_analysis(
    by_variant: dict[str, list[RunCurve]],
    metric: str,
) -> dict[str, object]:
    fixed = {run.seed: run for run in by_variant.get("fixed", [])}
    adaptive = {run.seed: run for run in by_variant.get("adaptive", [])}
    seeds = sorted(set(fixed) & set(adaptive))
    if not seeds:
        msg = "paired analysis requires matching fixed and adaptive seeds"
        raise ValueError(
            msg,
        )
    fixed_values = np.asarray([getattr(fixed[seed], metric) for seed in seeds])
    adaptive_values = np.asarray([
        getattr(adaptive[seed], metric) for seed in seeds
    ])
    differences = adaptive_values - fixed_values
    ci_low, ci_high = bootstrap_mean_ci(differences)
    return {
        "difference_definition": "adaptive_minus_fixed; negative favours adaptive",
        "paired_seeds": seeds,
        "n": len(seeds),
        "fixed_mean": float(np.mean(fixed_values)),
        "adaptive_mean": float(np.mean(adaptive_values)),
        "mean_paired_difference": float(np.mean(differences)),
        "median_paired_difference": float(np.median(differences)),
        "paired_difference_95pct_bootstrap_ci": [ci_low, ci_high],
        "two_sided_paired_permutation_p": paired_permutation_test(differences),
    }


def _write_final_results(runs: list[RunCurve], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("variant", "seed", "final_fitness", "normalized_auc"),
        )
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (item.variant, item.seed)):
            writer.writerow({
                "variant": run.variant,
                "seed": run.seed,
                "final_fitness": run.final_fitness,
                "normalized_auc": run.normalized_auc,
            })


def _plot_convergence(
    by_variant: dict[str, list[RunCurve]], output: Path,
) -> None:
    colors = {"fixed": "#0072B2", "adaptive": "#D55E00", "random": "#666666"}
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for variant in ("fixed", "adaptive", "random"):
        curves = by_variant.get(variant, [])
        if not curves:
            continue
        reference = curves[0].evaluations
        if any(
            not np.array_equal(run.evaluations, reference) for run in curves
        ):
            msg = f"evaluation checkpoints do not align for {variant}"
            raise ValueError(
                msg,
            )
        values = np.vstack([run.best_fitness for run in curves])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=0)
        color = colors[variant]
        axis.plot(reference, mean, label=variant, color=color, linewidth=2)
        axis.fill_between(
            reference, mean - std, mean + std, color=color, alpha=0.18,
        )
    axis.set_xlabel("Fitness evaluations")
    axis.set_ylabel("Best target distance (m)")
    axis.set_title("Convergence (mean ± population SD across runs)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_final_boxplot(
    by_variant: dict[str, list[RunCurve]], output: Path,
) -> None:
    variants = [
        variant
        for variant in ("fixed", "adaptive", "random")
        if variant in by_variant
    ]
    values = [
        [run.final_fitness for run in by_variant[variant]]
        for variant in variants
    ]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.boxplot(values, tick_labels=variants, showmeans=True)
    axis.set_ylabel("Final best target distance (m)")
    axis.set_title("Final performance across independent runs")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_adaptive_sigma(
    by_variant: dict[str, list[RunCurve]], output: Path,
) -> None:
    curves = by_variant.get("adaptive", [])
    if not curves:
        return
    reference = curves[0].evaluations
    values = np.vstack([run.mean_sigma for run in curves])
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(reference, mean, color="#D55E00", linewidth=2)
    axis.fill_between(
        reference, mean - std, mean + std, color="#D55E00", alpha=0.18,
    )
    axis.set_xlabel("Fitness evaluations")
    axis.set_ylabel("Mean self-adaptive sigma")
    axis.set_title("Evolution of mutation strength (mean ± population SD)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def analyse(results_root: Path, output_directory: Path) -> None:
    """Create descriptive summaries, paired tests, and figures."""
    runs = load_runs(results_root)
    by_variant: dict[str, list[RunCurve]] = {}
    for run in runs:
        by_variant.setdefault(run.variant, []).append(run)
    for curves in by_variant.values():
        curves.sort(key=lambda run: run.seed)

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_final_results(runs, output_directory / "final_results.csv")

    descriptive: dict[str, dict[str, float | int]] = {}
    for variant, curves in by_variant.items():
        final = np.asarray([run.final_fitness for run in curves])
        auc = np.asarray([run.normalized_auc for run in curves])
        descriptive[variant] = {
            "n": len(curves),
            "final_mean": float(np.mean(final)),
            "final_sample_sd": _sample_std(final),
            "final_median": float(np.median(final)),
            "final_min": float(np.min(final)),
            "final_max": float(np.max(final)),
            "auc_mean": float(np.mean(auc)),
            "auc_sample_sd": _sample_std(auc),
        }

    report = {
        "descriptive": descriptive,
        "final_fitness_paired_test": _paired_analysis(
            by_variant, "final_fitness",
        ),
        "convergence_auc_paired_test": _paired_analysis(
            by_variant, "normalized_auc",
        ),
        "notes": [
            "The independent experimental unit is one seeded run.",
            "Generation-level observations are not treated as independent samples.",
            "Final fitness is the primary outcome; normalized AUC is the convergence outcome.",
        ],
    }
    (output_directory / "statistical_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )
    _plot_convergence(by_variant, output_directory / "convergence.png")
    _plot_final_boxplot(by_variant, output_directory / "final_boxplot.png")
    _plot_adaptive_sigma(by_variant, output_directory / "adaptive_sigma.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "analysis",
    )
    args = parser.parse_args(argv)
    analyse(args.results, args.output)


if __name__ == "__main__":
    main()
