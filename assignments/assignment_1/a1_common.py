from __future__ import annotations
import copy
import csv
import random
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, cast
import networkx as nx
from A1_template_2026 import fitness_function, load_targets
from a1_config import (
    CROSSOVER_ATTEMPTS,
    CSV_FIELDS,
    MAX_TOTAL_NODES,
    MODULE_BUDGET,
    MUTATION_ATTEMPTS,
    TARGET_DIR,
)
from ariel.ec import Individual
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

TARGETS = load_targets(TARGET_DIR)

def set_seed(seed: int) -> None:
    random.seed(seed)

def random_genome() -> TreeGenome:
    return random_tree(max_modules=MODULE_BUDGET)

def calculate_fitness(genome: TreeGenome) -> float:
    return fitness_function(genome.to_networkx(), TARGETS)

def is_valid_genome(genome: TreeGenome) -> bool:
    if not genome.nodes or len(genome.nodes) > MAX_TOTAL_NODES:
        return False
    try:
        validate_genome_dict(genome.to_dict())
        graph = genome.to_networkx()
        return (
            graph.number_of_nodes() > 0
            and 0 in graph
            and graph.in_degree(0) == 0
            and nx.is_arborescence(graph)
        )
    except (KeyError, TypeError, ValueError, nx.NetworkXException):
        return False

def mutate_genome(parent: TreeGenome) -> TreeGenome:
    parent_dict = parent.to_dict()
    for _ in range(MUTATION_ATTEMPTS):
        child = copy.deepcopy(parent)
        choice = random.random()

        if choice < 0.40:
            mutate_replace_node(child)
        elif choice < 0.75:
            mutate_subtree_replacement(child, max_modules=MODULE_BUDGET)
        elif choice < 0.90:
            mutate_shrink(child)
        else:
            mutate_hoist(child)
        if child.to_dict() != parent_dict and is_valid_genome(child):
            return child
    return copy.deepcopy(parent)

def crossover_genomes(parent1: TreeGenome, parent2: TreeGenome) -> TreeGenome:
    for _ in range(CROSSOVER_ATTEMPTS):
        child1, child2 = crossover_subtree(parent1, parent2)
        valid = [child for child in (child1, child2) if is_valid_genome(child)]
        if valid:
            return copy.deepcopy(random.choice(valid))
    return copy.deepcopy(random.choice([parent1, parent2]))

def create_individual(genome: TreeGenome | None = None) -> Individual:
    if genome is None:
        genome = random_genome()
    individual = Individual()
    individual.genotype = genome.to_dict()
    return individual

def genome_from_individual(individual: Individual) -> TreeGenome:
    genotype = individual.genotype
    if not isinstance(genotype, dict):
        raise TypeError("Expected a dictionary for a tree genotype.")
    return TreeGenome.from_dict(cast(dict[str, Any], genotype))

def tournament_selection(
    individuals: list[Individual],
    tournament_size: int,
) -> Individual:
    if not individuals:
        raise ValueError("Cannot select from an empty population.")

    competitors = random.sample(
        individuals,
        k=min(tournament_size, len(individuals)),
    )
    evaluated = [ind for ind in competitors if ind.fitness_ is not None]
    if not evaluated:
        raise RuntimeError("Tournament contains no evaluated individuals.")
    return min(evaluated, key=lambda ind: float(ind.fitness_))

def append_result(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def calculate_population_statistics(
    individuals: list[Individual],
) -> dict[str, float | int]:
    evaluated = [ind for ind in individuals if ind.fitness_ is not None]
    if not evaluated:
        raise ValueError("Cannot summarize an unevaluated population.")
    fitness_values = [float(ind.fitness_) for ind in evaluated]
    best = min(evaluated, key=lambda ind: float(ind.fitness_))
    best_genome = genome_from_individual(best)
    return {
        "best_fitness": min(fitness_values),
        "mean_fitness": fmean(fitness_values),
        "std_fitness": pstdev(fitness_values),
        "worst_fitness": max(fitness_values),
        "best_body_size": len(best_genome.nodes),
    }
