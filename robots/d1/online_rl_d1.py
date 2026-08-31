#!/usr/bin/env python3
"""D1 hardware wrapper for the generic online actor-critic loop."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import threading

import numpy as np

from rlt.online_runner import run


HEAD = np.deg2rad([-15, -60]).astype(np.float32)
DEFAULT_INITIAL_STATE_FILE = (
    Path(__file__).resolve().parents[2]
    / "rlt"
    / "config"
    / "pick_up_the_doll_initial_state.json"
)


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


def _load_initial_state(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"initial state file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    state = np.asarray(data.get("observation.state"), dtype=np.float32)
    if state.shape != (12,) or not np.isfinite(state).all():
        raise ValueError(f"initial state must be a finite 12-D vector: {path}")
    if np.any((state[6:] < 0) | (state[6:] > 1)):
        raise ValueError(f"initial hand state must be in [0, 1]: {path}")
    return state


class D1Hardware:
    """Adapter implementing the hardware protocol expected by ``rlt.online_rl``."""

    proprio_dim = 24

    def __init__(self, args):
        self.args, self.robot = args, None
        self.window_name = "D1 Online RL"
        self._camera_thread = None
        self._camera_stop = threading.Event()
        self._camera_active = threading.Event()
        self._camera_paused = threading.Event()
        self._camera_paused.set()
        self._camera_ready = threading.Event()
        self._camera_lock = threading.Lock()
        self._latest_bgr = None
        self._camera_error = None
        self._sync_dashboard = None
        self._sync_dashboard_active = False
        self.urdf_path = None
        self.initial_state = _load_initial_state(args.initial_state_file)
        self.exo = self.glove = None
        self.exo_zero = self.takeover_arm = self.glove_zero = self.takeover_hand = None
        self.human_origin_exo = self.human_origin_glove = self.human_origin_robot = None
        self._human_sync_count = 0
        self._human_sync_target = None

    def __enter__(self):
        from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path
        from robots.d1.d1_robot import D1Robot

        self.urdf_path = Path(get_default_urdf_path())
        self.robot = D1Robot(
            urdf_path=str(self.urdf_path),
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

    def start_camera_thread(self):
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._camera_ready.clear()
            self._camera_active.set()
            return
        self._camera_stop.clear()
        self._camera_active.set()
        self._camera_ready.clear()
        self._camera_error = None
        self._camera_thread = threading.Thread(
            target=self._camera_loop, name="d1-camera", daemon=True
        )
        self._camera_thread.start()

    def _camera_loop(self):
        import cv2

        try:
            while not self._camera_stop.is_set():
                if not self._camera_active.is_set():
                    self._camera_ready.clear()
                    self._camera_paused.set()
                    self._camera_active.wait(0.05)
                    continue
                self._camera_paused.clear()
                bgr = self.robot.vision.read_view()
                if not self._camera_active.is_set():
                    continue
                with self._camera_lock:
                    self._latest_bgr = np.ascontiguousarray(bgr)
                self._camera_ready.set()
                cv2.imshow(self.window_name, bgr)
                cv2.waitKey(1)
        except Exception as exc:
            self._camera_error = exc
        finally:
            self._camera_paused.set()
            self._camera_ready.set()

    def read(self):
        if not self._camera_ready.wait(2.0):
            raise RuntimeError("camera thread did not produce a frame")
        if self._camera_error is not None:
            raise RuntimeError("camera thread failed") from self._camera_error
        with self._camera_lock:
            bgr = self._latest_bgr.copy()
        state = _read_state(self.robot.head_arm, self.robot.hand)
        return np.ascontiguousarray(np.asarray(bgr, dtype=np.uint8)[:, :, ::-1]), state

    def _read_human_pose(self):
        if self.exo is None or self.glove is None:
            raise RuntimeError("human takeover requires --human")
        frame = self.exo.read_frame()
        if not frame.ok:
            raise RuntimeError("invalid exoskeleton frame")
        angles = np.asarray(frame.angles_deg, np.float32)
        glove = np.asarray(self.glove.get_norm_values(), np.float32)
        if (
            angles.shape != (6,)
            or glove.shape != (6,)
            or not np.isfinite(angles).all()
            or not np.isfinite(glove).all()
        ):
            raise RuntimeError("invalid human control pose")
        return angles, glove

    def _set_human_rebase(self, state, angles, glove):
        self.exo_zero = angles.copy()
        self.glove_zero = glove.copy()
        self.takeover_arm, self.takeover_hand = state[:6].copy(), state[6:].copy()

    def start_human_episode(self, state):
        if self.exo is None or self.glove is None:
            return
        angles, glove = self._read_human_pose()
        self.human_origin_exo = angles.copy()
        self.human_origin_glove = glove.copy()
        self.human_origin_robot = state.copy()
        self._set_human_rebase(state, angles, glove)

    def rebase_human(self, state):
        angles, glove = self._read_human_pose()
        self._set_human_rebase(state, angles, glove)

    def begin_human_sync(self):
        self._human_sync_count = 0

    def start_human_sync_view(self):
        if self._sync_dashboard is None:
            from rlt.sync_dashboard import SyncDashboard

            self._sync_dashboard = SyncDashboard(self.urdf_path)
        self._sync_dashboard.start()
        self._sync_dashboard_active = True

    def update_human_sync_view(
        self,
        state,
        target,
        arm_error_deg,
        hand_error,
        arm_threshold_deg,
        hand_threshold,
        stable_frames,
        required_frames,
    ):
        if self._sync_dashboard is None or not self._sync_dashboard_active:
            return
        self._sync_dashboard.render(
            state,
            target,
            arm_error_deg,
            hand_error,
            arm_threshold_deg,
            hand_threshold,
            stable_frames,
            required_frames,
        )

    def stop_human_sync_view(self):
        self._sync_dashboard_active = False
        if self._sync_dashboard is not None:
            self._sync_dashboard.stop()

    def human_sync_ready(self, state):
        if self.human_origin_robot is None:
            raise RuntimeError("human episode origin is not initialized")
        try:
            angles, glove = self._read_human_pose()
        except Exception:
            self._human_sync_count = 0
            raise
        target = np.concatenate(
            [
                self.human_origin_robot[:6]
                + np.deg2rad(angles - self.human_origin_exo),
                self.human_origin_robot[6:] + glove - self.human_origin_glove,
            ]
        )
        self._human_sync_target = target.copy()
        arm_threshold = np.deg2rad(
            float(self.args.human_sync_arm_threshold_deg)
        )
        hand_threshold = float(self.args.human_sync_hand_threshold)
        ready = (
            np.all(np.abs(target[:6] - state[:6]) <= arm_threshold)
            and np.all(np.abs(target[6:] - state[6:]) <= hand_threshold)
        )
        self._human_sync_count = self._human_sync_count + 1 if ready else 0
        return self._human_sync_count >= int(self.args.human_sync_frames)

    def human_sync_target(self):
        return self._human_sync_target.copy()

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
        arm_delta = np.deg2rad(angles - self.exo_zero)
        arm_deadband = np.deg2rad(
            float(self.args.human_deadband_arm_deg)
        )
        arm_delta[np.abs(arm_delta) < arm_deadband] = 0
        arm = np.clip(
            self.takeover_arm + arm_delta,
            self.args.arm_low,
            self.args.arm_high,
        )
        hand_delta = (
            np.asarray(self.glove.get_norm_values(), np.float32) - self.glove_zero
        )
        hand_deadband = float(self.args.human_deadband_hand)
        hand_delta[np.abs(hand_delta) < hand_deadband] = 0
        hand = np.clip(
            self.takeover_hand + hand_delta,
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

    def move_to_initial_pose(self):
        if not self.args.execute:
            return
        from beingbeyond_d1_edu_sdk.core._head_arm_core import HeadArmCore

        for home_deg in (
            HeadArmCore.HOME_1_Q_DEG,
            HeadArmCore.HOME_2_Q_DEG,
            HeadArmCore.HOME_3_Q_DEG,
        ):
            home_rad = np.deg2rad(home_deg).tolist()
            home_rad[:2] = HEAD.tolist()
            self.robot.head_arm._core._set_positions_without_limit_clip(home_rad)
            result = self.robot.head_arm.wait_until_reached(
                home_rad,
                active_joint_indices=range(8),
                pos_tol_deg=5.0,
                vel_tol_deg_s=20.0,
                timeout_s=15.0,
            )
            if result is None or result is False:
                raise RuntimeError("robot did not reach the homing pose within 15 seconds")

        headarm_command = np.concatenate([HEAD, self.initial_state[:6]]).astype(float)
        self.robot.head_arm.set_positions(headarm_command.tolist())
        self.robot.hand.set_joint_pos(self.initial_state[6:].astype(float).tolist())
        result = self.robot.head_arm.wait_until_reached(
            headarm_command.tolist(),
            active_joint_indices=range(8),
            pos_tol_deg=5.0,
            vel_tol_deg_s=20.0,
            timeout_s=15.0,
        )
        if result is None or result is False:
            raise RuntimeError("robot did not reach the initial pose within 15 seconds")

    def __exit__(self, *_):
        self.close_camera_thread()
        with contextlib.suppress(Exception):
            import cv2

            cv2.destroyWindow(self.window_name)
        for obj in (self.glove, self.exo, self.robot):
            with contextlib.suppress(Exception):
                if obj is not None:
                    obj.close()

    def stop_camera_thread(self):
        self.stop_human_sync_view()
        self._camera_active.clear()
        self._camera_ready.clear()
        self._camera_paused.wait(2.0)

    def close_camera_thread(self):
        self.stop_human_sync_view()
        self._camera_stop.set()
        self._camera_active.set()
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._camera_thread.join(2.0)
        self._camera_thread = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--phase",
        choices=("critical-only", "full-task", "eval-base", "eval-rlt"),
        required=True,
    )
    parser.add_argument(
        "--eval-task",
        choices=("critical-only", "full-task"),
        default="critical-only",
    )
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
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a learner checkpoint; default uses the latest rollout checkpoint.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-arm-delta-deg", type=float)
    parser.add_argument("--arm-dev", default=None)
    parser.add_argument("--arm-baud", type=int, default=None)
    parser.add_argument("--hand-type", default=None)
    parser.add_argument("--hand-can", default=None)
    parser.add_argument("--hand-baud", type=int, default=None)
    parser.add_argument("--camera-device", default=None)
    parser.add_argument(
        "--initial-state-file",
        type=Path,
        default=DEFAULT_INITIAL_STATE_FILE,
    )
    parser.add_argument("--exo-port", default=None)
    parser.add_argument("--glove-port", default=None)
    args = parser.parse_args()
    run(args, D1Hardware)


if __name__ == "__main__":
    main()
