"""Deterministic MuJoCo evaluation of evolved neural controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner
from assignments.assignment_2.a2_controller import (
    decode_weights,
    genotype_weights,
    nn_controller,
)

if TYPE_CHECKING:
    from assignments.assignment_2.a2_config import ExperimentConfig


@dataclass(frozen=True)
class EvaluationResult:
    """Fitness and diagnostic measurements for one simulated controller."""

    fitness: float
    initial_position: tuple[float, float, float]
    final_position: tuple[float, float, float]
    minimum_core_height: float
    total_control_effort: float

    def diagnostics(self) -> dict[str, float | list[float]]:
        """Return diagnostics in JSON-serializable form."""
        return {
            "initial_position": list(self.initial_position),
            "final_position": list(self.final_position),
            "minimum_core_height": self.minimum_core_height,
            "total_control_effort": self.total_control_effort,
        }


class ControllerEvaluator:
    """Compile the fixed world once and evaluate controller genotypes serially."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        world = SimpleFlatWorld()
        world.spawn(
            gecko().spec,
            position=list(config.spawn_position),
            correct_collision_with_floor=True,
        )
        self.model = world.spec.compile()
        expected = config.network
        if self.model.nu != expected.output_size:
            msg = (
                f"John-set gecko exposes {self.model.nu} actuators, "
                f"but NetworkSpec expects {expected.output_size}"
            )
            raise ValueError(
                msg,
            )
        if self.model.nq != 7 + expected.joint_count:
            msg = f"unexpected qpos size {self.model.nq}; expected {7 + expected.joint_count}"
            raise ValueError(
                msg,
            )

    def evaluate(self, genotype: dict[str, Any]) -> EvaluationResult:
        """Run one headless simulation and return final planar target distance."""
        weights = decode_weights(
            genotype_weights(genotype, self.config.network), self.config.network,
        )
        data = mj.MjData(self.model)
        mj.mj_resetData(self.model, data)
        mj.mj_forward(self.model, data)

        initial = np.asarray(data.qpos[:3], dtype=np.float64).copy()
        target = np.asarray(self.config.target_position, dtype=np.float64)
        initial_target_distance = float(
            np.linalg.norm(initial[:2] - target[:2]),
        )
        minimum_height = float(initial[2])
        total_effort = 0.0

        def control_callback(
            model: mj.MjModel, callback_data: mj.MjData,
        ) -> None:
            nonlocal minimum_height, total_effort
            actions = nn_controller(
                model,
                callback_data,
                weights,
                target,
                initial_target_distance,
                self.config.network,
            )
            callback_data.ctrl[:] = actions
            minimum_height = min(minimum_height, float(callback_data.qpos[2]))
            total_effort += float(
                np.sum(np.square(actions)) * model.opt.timestep,
            )

        mj.set_mjcb_control(control_callback)
        try:
            simple_runner(
                self.model,
                data,
                duration=self.config.simulation_duration,
            )
        finally:
            mj.set_mjcb_control(None)

        final = np.asarray(data.qpos[:3], dtype=np.float64).copy()
        fitness = float(np.linalg.norm(final[:2] - target[:2]))
        if not np.isfinite(fitness):
            fitness = float("inf")
        return EvaluationResult(
            fitness=fitness,
            initial_position=tuple(float(value) for value in initial),
            final_position=tuple(float(value) for value in final),
            minimum_core_height=minimum_height,
            total_control_effort=total_effort,
        )
