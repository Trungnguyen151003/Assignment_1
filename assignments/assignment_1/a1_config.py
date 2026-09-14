"""Configuration shared by all Assignment 1 experiments."""

from pathlib import Path


HERE = Path(__file__).parent
TARGET_DIR = HERE / "target_bodies"
RESULTS_ROOT = HERE / "results"

# In the current ARIEL implementation, random_tree(20) creates one CORE plus
# at most 20 non-core modules. We therefore enforce 21 total graph nodes.
MODULE_BUDGET = 20
MAX_TOTAL_NODES = MODULE_BUDGET + 1

POPULATION_SIZE = 50
GENERATIONS = 100
SEEDS = [0, 1, 2, 3, 4]

# POPULATION_SIZE = 10
# GENERATIONS = 3
# SEEDS = [0]

TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.8
MUTATION_ATTEMPTS = 20
CROSSOVER_ATTEMPTS = 10

EA_VARIANTS = ["mutation_only", "crossover"]
ALL_VARIANTS = [*EA_VARIANTS, "random_search"]

CSV_FIELDS = [
    "variant",
    "seed",
    "generation",
    "evaluations",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "worst_fitness",
    "best_body_size",
]


def evaluation_budget() -> int:
    """Return the number of fitness evaluations in one complete EA run."""
    return POPULATION_SIZE + GENERATIONS * POPULATION_SIZE
