"""Run one fixed, adaptive, or random-search Assignment 2 experiment."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ariel.ec import EA, EAOperation, Population, set_seed
from assignments.assignment_2.a2_controller import genotype_sigma
from assignments.assignment_2.a2_evaluator import ControllerEvaluator
from assignments.assignment_2.a2_operators import (
    evaluate_population,
    initial_weight_population,
    make_initial_population,
    random_reproduce,
    reproduce,
    survivor_selection,
)

if TYPE_CHECKING:
    from pathlib import Path

    from assignments.assignment_2.a2_config import ExperimentConfig, Variant


@dataclass(frozen=True)
class RunResult:
    """Key output paths and final score from a completed run."""

    variant: Variant
    seed: int
    best_fitness: float
    run_directory: Path
    database_path: Path


class GenerationRecorder:
    """Write reproducible generation-level summaries incrementally."""

    fieldnames = (
        "generation",
        "evaluations",
        "best_fitness",
        "mean_fitness",
        "std_fitness",
        "worst_fitness",
        "mean_sigma",
        "std_sigma",
    )

    def __init__(self, output_path: Path, config: ExperimentConfig) -> None:
        self.output_path = output_path
        self.config = config
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
            writer.writeheader()

    def record(self, generation: int, population: Population) -> None:
        """Append statistics for the alive population at one generation."""
        alive = population.alive.evaluated.to_list()
        fitness = np.asarray(
            [individual.fitness_ for individual in alive], dtype=float,
        )
        sigmas = np.asarray(
            [
                genotype_sigma(cast("dict[str, Any]", individual.genotype))
                for individual in alive
            ],
            dtype=float,
        )
        row = {
            "generation": generation,
            "evaluations": (
                self.config.population_size
                + generation * self.config.offspring_count
            ),
            "best_fitness": float(np.min(fitness)),
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness, ddof=0)),
            "worst_fitness": float(np.max(fitness)),
            "mean_sigma": float(np.mean(sigmas)),
            "std_sigma": float(np.std(sigmas, ddof=0)),
        }
        with self.output_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.fieldnames).writerow(row)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8",
    )


def run_variant(
    variant: Variant,
    seed: int,
    config: ExperimentConfig,
    output_root: Path,
    *,
    shared_initial_weights: np.ndarray | None = None,
    quiet: bool = False,
) -> RunResult:
    """Execute one complete run and persist its population and summaries."""
    if variant not in {"fixed", "adaptive", "random"}:
        msg = f"unknown variant {variant!r}"
        raise ValueError(msg)

    run_directory = output_root / variant / f"seed_{seed:03d}"
    run_directory.mkdir(parents=True, exist_ok=True)
    database_path = run_directory / "population.db"
    if database_path.exists():
        msg = f"refusing to overwrite existing run database: {database_path}"
        raise FileExistsError(
            msg,
        )

    weights = (
        initial_weight_population(config, seed)
        if shared_initial_weights is None
        else np.asarray(shared_initial_weights, dtype=np.float64).copy()
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xA2, 1]))
    set_seed(seed)

    evaluator = ControllerEvaluator(config)
    initial = make_initial_population(weights, config)
    evaluate_population(initial, evaluator)

    if variant == "random":
        generation_operations = [
            EAOperation(random_reproduce, config=config, rng=rng),
            EAOperation(evaluate_population, evaluator=evaluator),
            EAOperation(
                survivor_selection,
                population_size=config.population_size,
                rng=rng,
            ),
        ]
    else:
        generation_operations = [
            EAOperation(
                reproduce,
                variant=variant,
                config=config,
                rng=rng,
            ),
            EAOperation(evaluate_population, evaluator=evaluator),
            EAOperation(
                survivor_selection,
                population_size=config.population_size,
                rng=rng,
            ),
        ]

    ea = EA(
        initial,
        generation_operations,
        num_steps=config.generations,
        is_maximisation=False,
        quiet=quiet,
        db_file_path=database_path,
        db_handling="halt",
    )
    recorder = GenerationRecorder(run_directory / "generations.csv", config)
    # EA commits expire the in-memory SQLModel objects when its short-lived
    # database session closes. Refresh before reading statistics outside the
    # engine so no detached object attempts a lazy database load.
    ea.fetch_population()
    recorder.record(0, ea.population)
    for _ in range(config.generations):
        ea.step()
        ea.fetch_population()
        recorder.record(ea.current_generation, ea.population)

    best = ea.get_solution("best")
    if best.fitness_ is None:
        msg = "completed run has no evaluated best individual"
        raise RuntimeError(msg)
    best_genotype = cast("dict[str, Any]", best.genotype)
    _write_json(run_directory / "best_genotype.json", best_genotype)
    _write_json(
        run_directory / "config.json",
        {
            "variant": variant,
            "seed": seed,
            **config.to_dict(),
        },
    )
    _write_json(
        run_directory / "final_summary.json",
        {
            "variant": variant,
            "seed": seed,
            "evaluations": config.evaluation_budget,
            "best_fitness": best.fitness_,
            "best_sigma": genotype_sigma(best_genotype),
            "best_diagnostics": best.tags,
        },
    )
    return RunResult(
        variant=variant,
        seed=seed,
        best_fitness=best.fitness_,
        run_directory=run_directory,
        database_path=database_path,
    )
