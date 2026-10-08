"""Evolutionary operators shared by all Assignment 2 variants."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ariel.ec import Individual, Population
from assignments.assignment_2.a2_controller import (
    Genotype,
    genotype_sigma,
    genotype_weights,
    make_genotype,
)

if TYPE_CHECKING:
    from assignments.assignment_2.a2_config import ExperimentConfig, Variant
    from assignments.assignment_2.a2_evaluator import ControllerEvaluator


def initial_weight_population(
    config: ExperimentConfig,
    seed: int,
) -> np.ndarray:
    """Generate the common initial population used by paired variants."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xA2, 0]))
    weights = rng.normal(
        loc=0.0,
        scale=config.initial_weight_std,
        size=(config.population_size, config.network.num_weights),
    )
    return np.clip(weights, config.weight_min, config.weight_max)


def make_initial_population(
    weights: np.ndarray,
    config: ExperimentConfig,
) -> Population:
    """Create unevaluated ARIEL individuals from a shared weight matrix."""
    expected_shape = (config.population_size, config.network.num_weights)
    if weights.shape != expected_shape:
        msg = f"initial weights have shape {weights.shape}, expected {expected_shape}"
        raise ValueError(
            msg,
        )
    individuals: list[Individual] = []
    for row in weights:
        individual = Individual()
        individual.genotype = make_genotype(row, config.fixed_sigma)
        individual.tags = {"origin": "initial"}
        individuals.append(individual)
    return Population(individuals)


def tournament_select(
    population: Population,
    tournament_size: int,
    rng: np.random.Generator,
) -> Individual:
    """Select the lowest-fitness individual from a random tournament."""
    candidates = population.alive.evaluated.to_list()
    if tournament_size > len(candidates):
        msg = "tournament_size exceeds the evaluated population"
        raise ValueError(msg)
    indices = rng.choice(len(candidates), size=tournament_size, replace=False)
    tournament = [candidates[int(index)] for index in indices]
    rng.shuffle(tournament)
    return min(
        tournament,
        key=lambda individual: (
            float("inf") if individual.fitness_ is None else individual.fitness_
        ),
    )


def mutate_genotype(
    parent_genotype: dict[str, Any],
    variant: Variant,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> Genotype:
    """Apply fixed or self-adaptive Gaussian mutation to a copied genotype."""
    if variant not in {"fixed", "adaptive"}:
        msg = f"mutation is not defined for variant {variant!r}"
        raise ValueError(msg)

    parent_weights = genotype_weights(parent_genotype, config.network)
    strategy_noise = float(rng.normal())
    weight_noise = rng.normal(size=config.network.num_weights)

    if variant == "adaptive":
        parent_log_sigma = float(np.log(genotype_sigma(parent_genotype)))
        log_sigma = np.clip(
            parent_log_sigma + config.adaptation_tau * strategy_noise,
            np.log(config.sigma_min),
            np.log(config.sigma_max),
        )
        sigma = float(np.exp(log_sigma))
    else:
        sigma = config.fixed_sigma

    child_weights = np.clip(
        parent_weights + sigma * weight_noise,
        config.weight_min,
        config.weight_max,
    )
    return make_genotype(child_weights, sigma)


def reproduce(
    population: Population,
    variant: Variant,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> Population:
    """Append lambda independently selected and mutated offspring."""
    if variant not in {"fixed", "adaptive"}:
        msg = f"reproduce does not support variant {variant!r}"
        raise ValueError(msg)
    children: list[Individual] = []
    for _ in range(config.offspring_count):
        parent = tournament_select(population, config.tournament_size, rng)
        parent_genotype = cast("dict[str, Any]", deepcopy(parent.genotype))
        child = Individual()
        child.genotype = mutate_genotype(parent_genotype, variant, config, rng)
        child.tags = {
            "origin": "offspring",
            "variant": variant,
            "parent_id": parent.id if parent.id is not None else -1,
        }
        children.append(child)
    population.extend(children)
    return population


def random_reproduce(
    population: Population,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> Population:
    """Append lambda independent samples for the equal-budget baseline."""
    children: list[Individual] = []
    for _ in range(config.offspring_count):
        weights = np.clip(
            rng.normal(
                loc=0.0,
                scale=config.initial_weight_std,
                size=config.network.num_weights,
            ),
            config.weight_min,
            config.weight_max,
        )
        child = Individual()
        child.genotype = make_genotype(weights, config.fixed_sigma)
        child.tags = {"origin": "random", "variant": "random"}
        children.append(child)
    population.extend(children)
    return population


def evaluate_population(
    population: Population,
    evaluator: ControllerEvaluator,
) -> Population:
    """Evaluate every individual whose genotype changed or is new."""
    for individual in population.unevaluated:
        genotype = cast("dict[str, Any]", individual.genotype)
        result = evaluator.evaluate(genotype)
        individual.fitness = result.fitness
        individual.tags = result.diagnostics()
        individual.tags = {"sigma": genotype_sigma(genotype)}
    return population


def survivor_selection(
    population: Population,
    population_size: int,
    rng: np.random.Generator,
) -> Population:
    """Keep the mu lowest-fitness individuals, randomizing exact ties."""
    evaluated = population.alive.evaluated.to_list()
    if len(evaluated) < population_size:
        msg = "fewer evaluated candidates than requested survivors"
        raise ValueError(msg)
    rng.shuffle(evaluated)
    evaluated.sort(
        key=lambda individual: (
            float("inf") if individual.fitness_ is None else individual.fitness_
        ),
    )
    survivor_ids = {
        id(individual) for individual in evaluated[:population_size]
    }
    for individual in population.alive:
        if id(individual) not in survivor_ids:
            individual.alive = False
    return population
