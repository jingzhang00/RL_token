#!/usr/bin/env python3
"""D1 hardware wrapper for the generic online actor-critic loop."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np

from rlt.actions import ARM_HIGH, ARM_LOW
from rlt.online_rl import run


HEAD = np.deg2rad([-15, -60]).astype(np.float32)


def _read_state(head_arm, hand):
    headarm = np.asarray(head_arm.get_positions(), dtype=np.float32)
    hand_state = np.asarray(hand.read_joint_pos(), dtype=np.float32)
    if (
        headarm.shape != (8,)
        or hand_state.shape != (6,)
        or not np.isfinite(headarm).all()
        or not np.isfinite(hand_state).all()
    ):
        raise RuntimeError("invalid D1 state")
    return np.concatenate([headarm[2:], hand_state]).astype(np.float32)


class D1Hardware:
    """Adapter implementing the hardware protocol expected by ``rlt.online_rl``."""

    proprio_dim = 24

    def __init__(self, args):
        self.args, self.robot = args, None
        self.exo = self.glove = None
        self.exo_zero = self.takeover_arm = self.glove_zero = self.takeover_hand = None

    def __enter__(self):
        from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path
        from robots.d1.d1_robot import D1Robot

        self.robot = D1Robot(
            urdf_path=get_default_urdf_path(),
            arm_dev=self.args.arm_dev,
            arm_baud=self.args.arm_baud,
            hand_type=self.args.hand_type,
            hand_can=self.args.hand_can,
            hand_baud=self.args.hand_baud,
            vision_device=self.args.camera_device,
        )
        self.robot.__enter__()
        self.robot.vision.view = "left"
        self.robot.vision.start()
        if self.args.human:
            from beingbeyond_d1_edu_sdk.exo import ExoDriver, ExoDriverCfg
            from beingbeyond_d1_edu_sdk.glove_driver import GloveConfig, GloveReader

            self.exo = ExoDriver(
                ExoDriverCfg(port=self.args.exo_port, read_hz=self.args.control_hz)
            )
            self.exo.open()
            self.exo.release_torque()
            self.glove = GloveReader(
                GloveConfig(
                    port=self.args.glove_port,
                    timeout=0.01,
                    use_auto_calib=True,
                    offset_deg=[60, -70, 120, 120, 120, 120],
                )
            )
        return self

    def read(self):
        bgr = self.robot.vision.read_view()
        state = _read_state(self.robot.head_arm, self.robot.hand)
        return np.ascontiguousarray(np.asarray(bgr, dtype=np.uint8)[:, :, ::-1]), state

    def rebase_human(self, state):
        if self.exo is None or self.glove is None:
            raise RuntimeError("human takeover requires --human")
        frame = self.exo.read_frame()
        if not frame.ok:
            raise RuntimeError("cannot rebase: invalid exoskeleton frame")
        self.exo_zero = np.asarray(frame.angles_deg, np.float32)
        self.glove_zero = np.asarray(self.glove.get_norm_values(), np.float32)
        self.takeover_arm, self.takeover_hand = state[:6].copy(), state[6:].copy()

    def human_target(self):
        frame = self.exo.read_frame()
        if sum(read.ok for read in frame.reads) < 4:
            raise RuntimeError("too few valid exoskeleton joints")
        angles = np.asarray(
            [
                read.angle_deg if read.angle_deg is not None else zero
                for read, zero in zip(frame.reads, self.exo_zero)
            ],
            np.float32,
        )
        arm = np.clip(
            self.takeover_arm + np.deg2rad(angles - self.exo_zero), ARM_LOW, ARM_HIGH
        )
        hand = np.clip(
            self.takeover_hand
            + np.asarray(self.glove.get_norm_values())
            - self.glove_zero,
            0,
            1,
        )
        return np.concatenate([arm, hand]).astype(np.float32)

    def send(self, command):
        if not self.args.execute:
            return
        self.robot.head_arm.set_positions(
            np.concatenate([HEAD, command[:6]]).astype(float).tolist()
        )
        self.robot.hand.set_joint_pos(command[6:].astype(float).tolist())

    def __exit__(self, *_):
        for obj in (self.glove, self.exo, self.robot):
            with contextlib.suppress(Exception):
                if obj is not None:
                    obj.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--phase",
        choices=("auto", "warmup", "critical", "full", "eval-base", "eval-rlt"),
        default=None,
    )
    parser.add_argument("--eval-task", choices=("critical", "full"), default="critical")
    parser.add_argument("--episodes", type=int, help="Limit the number of episodes.")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human", action="store_true")
    parser.add_argument(
        "--learner-device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Online actor/critic device; auto uses CUDA when available.",
    )
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-arm-delta-deg", type=float)
    parser.add_argument("--arm-dev", default=None)
    parser.add_argument("--arm-baud", type=int, default=None)
    parser.add_argument("--hand-type", default=None)
    parser.add_argument("--hand-can", default=None)
    parser.add_argument("--hand-baud", type=int, default=None)
    parser.add_argument("--camera-device", default=None)
    parser.add_argument("--exo-port", default=None)
    parser.add_argument("--glove-port", default=None)
    args = parser.parse_args()
    run(args, D1Hardware)


if __name__ == "__main__":
    main()
