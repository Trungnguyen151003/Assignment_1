"""Genotype representation and neural controller for Assignment 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import mujoco as mj

    from assignments.assignment_2.a2_config import NetworkSpec

type Genotype = dict[str, float | list[float]]
type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DecodedWeights:
    """Matrices and biases consumed by the controller forward pass."""

    input_to_hidden: FloatArray
    hidden_bias: FloatArray
    hidden_to_output: FloatArray
    output_bias: FloatArray


def make_genotype(weights: npt.ArrayLike, sigma: float) -> Genotype:
    """Create a JSON-serializable genotype with a log-scaled strategy gene."""
    flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    if sigma <= 0.0 or not np.isfinite(sigma):
        msg = "sigma must be finite and positive"
        raise ValueError(msg)
    if not np.all(np.isfinite(flat)):
        msg = "all weights must be finite"
        raise ValueError(msg)
    return {"weights": flat.tolist(), "log_sigma": float(np.log(sigma))}


def genotype_weights(genotype: dict[str, Any], spec: NetworkSpec) -> FloatArray:
    """Validate and return the flat neural-network weight vector."""
    if "weights" not in genotype:
        msg = "genotype is missing 'weights'"
        raise ValueError(msg)
    weights = np.asarray(genotype["weights"], dtype=np.float64).reshape(-1)
    if len(weights) != spec.num_weights:
        msg = f"expected {spec.num_weights} weights, received {len(weights)}"
        raise ValueError(
            msg,
        )
    if not np.all(np.isfinite(weights)):
        msg = "genotype contains non-finite weights"
        raise ValueError(msg)
    return weights


def genotype_sigma(genotype: dict[str, Any]) -> float:
    """Return the positive mutation strength encoded by a genotype."""
    try:
        log_sigma = float(genotype["log_sigma"])
    except (KeyError, TypeError, ValueError) as exc:
        msg = "genotype has an invalid 'log_sigma'"
        raise ValueError(msg) from exc
    sigma = float(np.exp(log_sigma))
    if not np.isfinite(sigma) or sigma <= 0.0:
        msg = "decoded sigma must be finite and positive"
        raise ValueError(msg)
    return sigma


def decode_weights(weights: npt.ArrayLike, spec: NetworkSpec) -> DecodedWeights:
    """Decode a flat genotype into network matrices without copying semantics."""
    flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(flat) != spec.num_weights:
        msg = f"expected {spec.num_weights} weights, received {len(flat)}"
        raise ValueError(
            msg,
        )

    cursor = 0
    w1_size = spec.input_size * spec.hidden_size
    w1 = flat[cursor : cursor + w1_size].reshape(
        spec.input_size, spec.hidden_size,
    )
    cursor += w1_size
    b1 = flat[cursor : cursor + spec.hidden_size]
    cursor += spec.hidden_size
    w2_size = spec.hidden_size * spec.output_size
    w2 = flat[cursor : cursor + w2_size].reshape(
        spec.hidden_size, spec.output_size,
    )
    cursor += w2_size
    b2 = flat[cursor : cursor + spec.output_size]

    return DecodedWeights(w1, b1, w2, b2)


def _world_vector_to_body_frame(
    vector_xy: FloatArray, quaternion: FloatArray,
) -> FloatArray:
    """Rotate a horizontal world-frame vector into the robot core frame."""
    w, x, y, z = quaternion
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    dx, dy = vector_xy
    return np.array(
        [cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy],
        dtype=np.float64,
    )


def controller_inputs(
    data: mj.MjData,
    target_position: npt.ArrayLike,
    initial_target_distance: float,
    spec: NetworkSpec,
) -> FloatArray:
    """Build the normalized 16-value state, task, and clock input vector."""
    joint_positions = np.asarray(data.qpos[7:], dtype=np.float64)
    joint_velocities = np.asarray(data.qvel[6:], dtype=np.float64)
    if len(joint_positions) != spec.joint_count:
        msg = f"expected {spec.joint_count} joint positions, got {len(joint_positions)}"
        raise ValueError(
            msg,
        )
    if len(joint_velocities) != spec.joint_count:
        msg = f"expected {spec.joint_count} joint velocities, got {len(joint_velocities)}"
        raise ValueError(
            msg,
        )

    normalized_positions = np.clip(joint_positions / (np.pi / 2.0), -1.0, 1.0)
    normalized_velocities = np.clip(
        joint_velocities / spec.velocity_scale, -1.0, 1.0,
    )

    target = np.asarray(target_position, dtype=np.float64)
    core_xy = np.asarray(data.qpos[:2], dtype=np.float64)
    relative_world = target[:2] - core_xy
    relative_body = _world_vector_to_body_frame(
        relative_world, np.asarray(data.qpos[3:7], dtype=np.float64),
    )
    distance_scale = max(float(initial_target_distance), 1e-9)
    normalized_target = np.clip(relative_body / distance_scale, -2.0, 2.0)

    phase = 2.0 * np.pi * float(data.time) / spec.clock_period
    clock = np.array([np.sin(phase), np.cos(phase)], dtype=np.float64)
    inputs = np.concatenate([
        normalized_positions,
        normalized_velocities,
        normalized_target,
        clock,
    ])
    if len(inputs) != spec.input_size:
        msg = f"constructed {len(inputs)} inputs, expected {spec.input_size}"
        raise RuntimeError(
            msg,
        )
    return inputs


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: DecodedWeights,
    target_position: npt.ArrayLike,
    initial_target_distance: float,
    spec: NetworkSpec,
) -> FloatArray:
    """Map state and task information to bounded hinge position commands."""
    if model.nu != spec.output_size:
        msg = f"model has {model.nu} actuators, expected {spec.output_size}"
        raise ValueError(
            msg,
        )
    inputs = controller_inputs(
        data, target_position, initial_target_distance, spec,
    )
    hidden = np.tanh(inputs @ weights.input_to_hidden + weights.hidden_bias)
    outputs = np.tanh(hidden @ weights.hidden_to_output + weights.output_bias)
    actions = np.asarray(outputs * (np.pi / 2.0), dtype=np.float64)
    if not np.all(np.isfinite(actions)):
        msg = "controller produced non-finite actions"
        raise FloatingPointError(msg)
    return actions
