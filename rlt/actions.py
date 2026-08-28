"""D1 normalized-delta conversion and final command safety."""

from __future__ import annotations

import numpy as np


ARM_LOW = np.deg2rad([-150, -90, -90, -150, -100, -150]).astype(np.float32)
ARM_HIGH = np.deg2rad([150, 90, 90, 150, 85, 150]).astype(np.float32)


def normalized_to_absolute(normalized, state, q01, q99) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float32)
    physical = (normalized + 1) / 2 * (
        np.asarray(q99) - np.asarray(q01) + 1e-6
    ) + np.asarray(q01)
    physical[..., :6] += np.asarray(state, dtype=np.float32)[:6]
    return physical[..., :12].astype(np.float32)


def absolute_to_normalized(absolute, state, q01, q99) -> np.ndarray:
    physical = np.asarray(absolute, dtype=np.float32).copy()
    physical[..., :6] -= np.asarray(state, dtype=np.float32)[:6]
    return (
        (physical - np.asarray(q01)) / (np.asarray(q99) - np.asarray(q01) + 1e-6) * 2
        - 1
    ).astype(np.float32)


def safe_command(
    command,
    current,
    max_arm_delta_deg: float = 12.0,
    arm_low: np.ndarray = ARM_LOW,
    arm_high: np.ndarray = ARM_HIGH,
) -> np.ndarray:
    command = np.asarray(command, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if (
        command.shape != (12,)
        or current.shape != (12,)
        or not np.isfinite(command).all()
    ):
        raise ValueError("D1 command/current must be finite 12-D vectors")
    max_delta = np.deg2rad(float(max_arm_delta_deg))
    arm = np.clip(command[:6], current[:6] - max_delta, current[:6] + max_delta)
    arm = np.clip(arm, arm_low, arm_high)
    return np.concatenate([arm, np.clip(command[6:], 0, 1)]).astype(np.float32)
