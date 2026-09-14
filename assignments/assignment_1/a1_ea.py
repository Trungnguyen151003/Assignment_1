"""Mutation-only and subtree-crossover evolutionary algorithms."""

import copy
import random
from pathlib import Path
from typing import Literal

from a1_common import (
    append_result,
    calculate_fitness,
    calculate_population_statistics,
    create_individual,
    crossover_genomes,
    genome_from_individual,
    mutate_genome,
    set_seed,
    tournament_selection,
)
from a1_config import (
    CROSSOVER_PROBABILITY,
    GENERATIONS,
    POPULATION_SIZE,
    TOURNAMENT_SIZE,
)
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome


type EAVariant = Literal["mutation_only", "crossover"]
type RunSummary = dict[str, float | int | str]


class TreeEvolution:
    """One independent run of either EA variant."""

    def __init__(
        self,
        variant: EAVariant,
        seed: int,
        result_dir: Path,
        csv_path: Path,
    ) -> None:
        self.variant = variant
        self.seed = seed
        self.result_dir = result_dir
        self.csv_path = csv_path
        self.generation = 0
        self.evaluations = 0

    def evaluate(self, population: Population) -> Population:
        """Evaluate only newly created individuals."""
        for individual in population:
            if individual.alive and individual.requires_eval:
                individual.fitness = calculate_fitness(
                    genome_from_individual(individual),
                )
                self.evaluations += 1
        return population

    @staticmethod
    def select_parent(parents: list[Individual]) -> Individual:
        """Select one parent using minimising tournament selection."""
        return tournament_selection(parents, TOURNAMENT_SIZE)

    def mutation_only_child(self, parents: list[Individual]) -> TreeGenome:
        """Copy one selected parent and mutate the copy."""
        parent = self.select_parent(parents)
        copied_genome = copy.deepcopy(genome_from_individual(parent))
        return mutate_genome(copied_genome)

    def crossover_child(self, parents: list[Individual]) -> TreeGenome:
        """Optionally cross two parents, then apply the same mutation."""
        parent1 = self.select_parent(parents)
        genome1 = genome_from_individual(parent1)

        if len(parents) >= 2 and random.random() < CROSSOVER_PROBABILITY:
            alternatives = [ind for ind in parents if ind is not parent1]
            parent2 = self.select_parent(alternatives)
            genome2 = genome_from_individual(parent2)
            child = crossover_genomes(genome1, genome2)
        else:
            child = copy.deepcopy(genome1)

        return mutate_genome(child)

    def reproduce(self, population: Population) -> Population:
        """Create exactly POPULATION_SIZE new offspring."""
        parents = [
            ind
            for ind in population
            if ind.alive and ind.fitness_ is not None
        ]
        if not parents:
            raise RuntimeError("No evaluated parents are available.")

        children: list[Individual] = []
        while len(children) < POPULATION_SIZE:
            if self.variant == "mutation_only":
                genome = self.mutation_only_child(parents)
            else:
                genome = self.crossover_child(parents)
            children.append(create_individual(genome))

        population.extend(children)
        return population

    def survivor_selection(self, population: Population) -> Population:
        """Use elitist (mu + lambda) selection to retain the best candidates."""
        candidates = [
            ind
            for ind in population
            if ind.alive and ind.fitness_ is not None
        ]
        ranked = sorted(candidates, key=lambda ind: float(ind.fitness_))
        survivors = ranked[:POPULATION_SIZE]

        # New offspring have no database ID until the generation is committed.
        survivor_objects = {id(ind) for ind in survivors}
        for individual in population:
            if individual.alive and id(individual) not in survivor_objects:
                individual.alive = False

        self.generation += 1
        self.record_generation(survivors)
        return population

    def record_generation(self, individuals: list[Individual]) -> None:
        """Record one population snapshot in results.csv."""
        append_result(
            self.csv_path,
            {
                "variant": self.variant,
                "seed": self.seed,
                "generation": self.generation,
                "evaluations": self.evaluations,
                **calculate_population_statistics(individuals),
            },
        )

    def run(self) -> RunSummary:
        """Execute one complete, independently seeded EA run."""
        set_seed(self.seed)
        initial = Population(
            [create_individual() for _ in range(POPULATION_SIZE)],
        )
        initial = self.evaluate(initial)
        self.record_generation(list(initial))

        database_path = (
            self.result_dir
            / "databases"
            / f"{self.variant}_seed_{self.seed}.db"
        )
        operations: list[EAOperation] = [
            EAOperation(self.reproduce),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
        ]
        ea = EA(
            initial,
            operations=operations,
            num_steps=GENERATIONS,
            is_maximisation=False,
            db_file_path=database_path,
            db_handling="halt",
            quiet=True,
        )
        ea.run()

        best = ea.get_solution("best", only_alive=True)
        best_genome = genome_from_individual(best)
        best_path = (
            self.result_dir
            / "best_genomes"
            / f"{self.variant}_seed_{self.seed}.json"
        )
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_genome.save_json(str(best_path))

        return {
            "variant": self.variant,
            "seed": self.seed,
            "best_fitness": best.fitness,
            "evaluations": self.evaluations,
        }


def run_ea(
    variant: EAVariant,
    seed: int,
    result_dir: Path,
    csv_path: Path,
) -> RunSummary:
    """Entry point used by the all-experiments runner."""
    return TreeEvolution(variant, seed, result_dir, csv_path).run()
