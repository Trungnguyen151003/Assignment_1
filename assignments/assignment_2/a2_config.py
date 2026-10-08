"""Configuration for the Assignment 2 neuroevolution experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

type Variant = Literal["fixed", "adaptive", "random"]


@dataclass(frozen=True)
class NetworkSpec:
    """Shape and normalization constants for the neural controller."""

    joint_count: int = 6
    hidden_size: int = 8
    clock_period: float = 1.5
    velocity_scale: float = 5.0

    @property
    def input_size(self) -> int:
        """Number of controller inputs: q, qdot, target vector, and clock."""
        return 2 * self.joint_count + 4

    @property
    def output_size(self) -> int:
        """One output per actuated hinge."""
        return self.joint_count

    @property
    def num_weights(self) -> int:
        """Total number of weights and biases in the two-layer network."""
        return (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.output_size
            + self.output_size
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """All choices held fixed across the experimental variants."""

    population_size: int = 50
    offspring_count: int = 50
    generations: int = 150
    tournament_size: int = 3
    simulation_duration: float = 15.0
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.1)
    target_position: tuple[float, float, float] = (2.0, 0.0, 0.1)
    initial_weight_std: float = 0.5
    fixed_sigma: float = 0.1
    sigma_min: float = 0.005
    sigma_max: float = 1.0
    weight_min: float = -3.0
    weight_max: float = 3.0
    network: NetworkSpec = NetworkSpec()

    def __post_init__(self) -> None:
        """Reject configurations that would invalidate the experiment."""
        if self.population_size < 2:
            msg = "population_size must be at least 2"
            raise ValueError(msg)
        if self.offspring_count < 1:
            msg = "offspring_count must be positive"
            raise ValueError(msg)
        if not 1 <= self.tournament_size <= self.population_size:
            msg = "tournament_size must be in [1, population_size]"
            raise ValueError(msg)
        if self.generations < 1:
            msg = "generations must be positive"
            raise ValueError(msg)
        if self.simulation_duration <= 0.0:
            msg = "simulation_duration must be positive"
            raise ValueError(msg)
        if not 0.0 < self.sigma_min <= self.fixed_sigma <= self.sigma_max:
            msg = "sigma bounds must contain fixed_sigma"
            raise ValueError(msg)
        if self.weight_min >= self.weight_max:
            msg = "weight_min must be smaller than weight_max"
            raise ValueError(msg)

    @property
    def evaluation_budget(self) -> int:
        """Total number of evaluated controllers in one run."""
        return self.population_size + self.generations * self.offspring_count

    @property
    def adaptation_tau(self) -> float:
        """Learning rate for one global self-adaptive mutation strength."""
        return 1.0 / (2.0 * self.network.num_weights) ** 0.5

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable configuration dictionary."""
        result = asdict(self)
        result["evaluation_budget"] = self.evaluation_budget
        result["adaptation_tau"] = self.adaptation_tau
        return result


DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "results"
