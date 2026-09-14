"""Equal-evaluation-budget random-search baseline."""

from __future__ import annotations

import copy
from pathlib import Path
from statistics import fmean, pstdev

from a1_common import append_result, calculate_fitness, random_genome, set_seed
from a1_config import GENERATIONS, POPULATION_SIZE


type RunSummary = dict[str, float | int | str]


def run_random_search(
    seed: int,
    result_dir: Path,
    csv_path: Path,
) -> RunSummary:
    """Sample random bodies using exactly the same budget as one EA run."""
    set_seed(seed)
    best_genome = None
    best_fitness = float("inf")
    evaluations = 0

    # Generation 0 matches the EA's initial population. Every later checkpoint
    # adds the same number of evaluations as one EA generation.
    for generation in range(GENERATIONS + 1):
        batch_fitness: list[float] = []
        for _ in range(POPULATION_SIZE):
            genome = random_genome()
            fitness = calculate_fitness(genome)
            evaluations += 1
            batch_fitness.append(fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_genome = copy.deepcopy(genome)

        if best_genome is None:
            raise RuntimeError("Random search did not generate a genome.")

        append_result(
            csv_path,
            {
                "variant": "random_search",
                "seed": seed,
                "generation": generation,
                "evaluations": evaluations,
                "best_fitness": best_fitness,
                "mean_fitness": fmean(batch_fitness),
                "std_fitness": pstdev(batch_fitness),
                "worst_fitness": max(batch_fitness),
                "best_body_size": len(best_genome.nodes),
            },
        )

    best_path = (
        result_dir / "best_genomes" / f"random_search_seed_{seed}.json"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    if best_genome is None:
        raise RuntimeError("Random search did not generate a genome.")
    best_genome.save_json(str(best_path))
    return {
        "variant": "random_search",
        "seed": seed,
        "best_fitness": best_fitness,
        "evaluations": evaluations,
    }
