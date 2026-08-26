#!/usr/bin/env python3
"""Run the OpenPI D1 policy in MuJoCo with a LeRobot video observation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


OPENPI_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = OPENPI_ROOT.parent
OPENPI_INFERENCE_DIR = OPENPI_ROOT / "inference"
OPENPI_CLIENT_DIR = OPENPI_ROOT / "train" / "openpi" / "packages" / "openpi-client" / "src"
for path in (OPENPI_INFERENCE_DIR, OPENPI_CLIENT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi_client import websocket_client_policy  # noqa: E402
from d1_vision_policy import (  # noqa: E402
    DEFAULT_CROP_RATIOS,
    DEFAULT_HEAD_TARGET_RAD,
    DEFAULT_POLICY_IMAGE_SIZE,
    crop_box_for_ratios,
    crop_rgb_by_ratio,
    resize_rgb,
)


ACTION_DIM = 12


DEFAULT_DATASET_ROOT = REPO_ROOT / "data_processed" / "pick_up_the_doll_and_place_it_on_the_plate"
DEFAULT_INITIAL_STATE = np.asarray(
    [0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0] + [0.0] * 6,
    dtype=np.float32,
)
ARM_JOINTS = tuple(f"joint_{index}" for index in range(1, 7))
HEAD_JOINTS = ("joint_7_head_yaw", "joint_8_head_pitch")
# D1 hardware uses negative pitch for looking down; the imported URDF uses positive pitch.
MUJOCO_HEAD_DIRECTIONS = np.asarray([1.0, -1.0], dtype=np.float32)
HAND_JOINTS = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_ip",
    "index_mcp_pitch",
    "index_dip",
    "middle_mcp_pitch",
    "middle_dip",
    "ring_mcp_pitch",
    "ring_dip",
    "pinky_mcp_pitch",
    "pinky_dip",
)
HAND_CHANNELS = (1, 0, 0, 2, 2, 3, 3, 4, 4, 5, 5)
DEFAULT_ARM_DELTA_LIMITS_RAD = np.asarray([0.27, 0.38, 0.26, 0.08, 0.31, 0.08], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class SafetyConfig:
    arm_delta_limits_rad: np.ndarray
    max_abs_arm_rad: float
    hand_min: float
    hand_max: float


def _format_vec(values: np.ndarray, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in np.asarray(values).reshape(-1)) + "]"


def _require_actions(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] < 1:
        raise RuntimeError(f"OpenPI actions must have shape Hx{ACTION_DIM} with H > 0, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("OpenPI actions contain NaN or Inf")
    return actions


def _prepare_policy_image(
    raw_rgb: np.ndarray,
    *,
    image_roi: tuple[int, int, int, int] | None,
    crop_ratios: tuple[float, float, float, float],
    policy_image_size: int,
    no_image_crop: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    raw_rgb = np.asarray(raw_rgb, dtype=np.uint8)
    if raw_rgb.ndim != 3 or raw_rgb.shape[-1] != 3:
        raise RuntimeError(f"Raw RGB image must be HxWx3, got {raw_rgb.shape}")
    if no_image_crop:
        crop_box = (0, 0, raw_rgb.shape[1], raw_rgb.shape[0])
        cropped = raw_rgb
    elif image_roi is None:
        crop_box = crop_box_for_ratios(raw_rgb.shape, crop_ratios)
        cropped = crop_rgb_by_ratio(raw_rgb, crop_ratios)
    else:
        x, y, w, h = image_roi
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI width/height must be positive, got {image_roi}")
        if x < 0 or y < 0 or x + w > raw_rgb.shape[1] or y + h > raw_rgb.shape[0]:
            raise ValueError(f"ROI {image_roi} outside raw image bounds")
        crop_box = image_roi
        cropped = raw_rgb[y : y + h, x : x + w, :]
    policy_rgb = resize_rgb(cropped, (policy_image_size, policy_image_size))
    if policy_rgb.shape != (policy_image_size, policy_image_size, 3):
        raise RuntimeError(f"Policy image must have shape {(policy_image_size, policy_image_size, 3)}, got {policy_rgb.shape}")
    return np.ascontiguousarray(cropped), np.ascontiguousarray(policy_rgb), crop_box


def _build_openpi_observation(image_rgb: np.ndarray, state: np.ndarray, prompt: str) -> dict[str, Any]:
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    state = np.asarray(state, dtype=np.float32)
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise RuntimeError(f"Expected RGB image HxWx3, got {image_rgb.shape}")
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise RuntimeError(f"D1 policy state must be finite with shape {(ACTION_DIM,)}, got {state.shape}")
    return {
        "observation/image": image_rgb,
        "observation/state": state,
        "prompt": str(prompt),
    }


def _command_from_action(
    head_target: np.ndarray,
    arm: np.ndarray,
    action: np.ndarray,
    safety: SafetyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    arm = np.asarray(arm, dtype=np.float32).reshape(6)
    head_target = np.asarray(head_target, dtype=np.float32).reshape(2)
    if not np.isfinite(action).all() or not np.isfinite(arm).all() or not np.isfinite(head_target).all():
        raise RuntimeError("Policy action, arm state, and head target must be finite")
    clipped_delta = np.clip(action[:6], -safety.arm_delta_limits_rad, safety.arm_delta_limits_rad)
    arm_command = np.clip(arm + clipped_delta, -safety.max_abs_arm_rad, safety.max_abs_arm_rad)
    hand_command = np.clip(action[6:12], safety.hand_min, safety.hand_max)
    return (
        np.concatenate([head_target, arm_command], axis=0).astype(float),
        hand_command.astype(float),
        clipped_delta.astype(np.float32),
    )


def _advance_position_profile(
    position: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    *,
    max_velocity: float,
    max_acceleration: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    error = target - position
    desired_velocity = np.sign(error) * np.minimum(
        max_velocity,
        np.sqrt(2.0 * max_acceleration * np.abs(error)),
    )
    next_velocity = velocity + np.clip(
        desired_velocity - velocity,
        -max_acceleration * duration_s,
        max_acceleration * duration_s,
    )
    step = 0.5 * (velocity + next_velocity) * duration_s
    reached = (error * step > 0.0) & (np.abs(step) >= np.abs(error))
    return np.where(reached, target, position + step), np.where(reached, 0.0, next_velocity)


class LeRobotEpisodeVideo:
    def __init__(self, root: Path, episode: int, camera_key: str = "") -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset root not found: {self.root}")

        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.camera_key = camera_key or self._default_camera_key()
        self.episode = int(episode)
        self.fps = float(self.info.get("fps", 0.0))
        if self.fps <= 0:
            raise ValueError(f"Invalid dataset fps in {info_path}: {self.fps}")

        filename = f"episode_{self.episode:06d}.mp4"
        matches = sorted(
            path
            for path in (self.root / "videos").rglob(filename)
            if path.parent.name == self.camera_key
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {self.camera_key}/{filename} under {self.root / 'videos'}, got {len(matches)}"
            )
        self.video_path = matches[0]
        self._container: Any | None = None
        self._frames: Any | None = None
        self.frame_index = 0

    def _default_camera_key(self) -> str:
        collection = self.info.get("d1_collection", {})
        if collection.get("camera_key"):
            return str(collection["camera_key"])
        for name, feature in self.info.get("features", {}).items():
            if feature.get("dtype") == "video":
                return str(name)
        raise KeyError("No video feature found in meta/info.json")

    def read(self) -> np.ndarray | None:
        import av

        if self._container is None:
            self._container = av.open(str(self.video_path))
            self._frames = self._container.decode(video=0)
        try:
            frame = next(self._frames)
        except StopIteration:
            return None
        self.frame_index += 1
        return np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
            self._frames = None


def _find_d1_urdf(path: Path | None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"D1 URDF not found: {resolved}")
        return resolved

    sdk_root = REPO_ROOT.parent / "Beingbeyond_D1_edu"
    patterns = (
        "lib/beingbeyond_d1_edu_sdk-*/beingbeyond_d1_edu_sdk/BeingBeyond_D1_edu/robot_right_hand.urdf",
        ".venv/lib/python*/site-packages/beingbeyond_d1_edu_sdk/BeingBeyond_D1_edu/robot_right_hand.urdf",
    )
    matches = [candidate for pattern in patterns for candidate in sdk_root.glob(pattern)]
    if not matches:
        raise FileNotFoundError("D1 URDF was not found; set D1_URDF to robot_right_hand.urdf")
    return matches[0].resolve()


class D1Mujoco:
    def __init__(self, urdf_path: Path, arm_profile_velocity: float, arm_profile_acceleration: float) -> None:
        import mujoco

        self.mujoco = mujoco
        spec = mujoco.MjSpec.from_file(str(urdf_path))
        spec.worldbody.add_geom(
            name="viewer_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, -0.05],
            size=[0.0, 0.0, 0.05],
            rgba=[0.8, 0.8, 0.8, 1.0],
        )
        spec.worldbody.add_light(
            name="viewer_light",
            pos=[0.0, 0.0, 2.0],
            dir=[0.0, 0.0, -1.0],
            ambient=[0.4, 0.4, 0.4],
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.2, 0.2, 0.2],
        )
        self.model = spec.compile()
        self.model.vis.headlight.ambient[:] = 0.5
        self.model.vis.headlight.diffuse[:] = 0.8
        self.data = mujoco.MjData(self.model)
        self.addresses = {name: self._qpos_address(name) for name in ARM_JOINTS + HEAD_JOINTS + HAND_JOINTS}
        self.arm_profile_velocity = float(arm_profile_velocity)
        self.arm_profile_acceleration = float(arm_profile_acceleration)
        self.arm_velocity = np.zeros(6, dtype=np.float32)
        self.hand = np.zeros(6, dtype=np.float32)
        self.viewer: Any | None = None

    def _qpos_address(self, name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"D1 URDF is missing joint {name!r}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_range(self, name: str) -> tuple[float, float]:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = self.model.jnt_range[joint_id]
        return float(low), float(high)

    def reset(self, state: np.ndarray, head: np.ndarray) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.arm_velocity.fill(0.0)
        self._write_positions(state[:6], state[6:12], head=head)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        arm = np.asarray([self.data.qpos[self.addresses[name]] for name in ARM_JOINTS], dtype=np.float32)
        return arm, self.hand.copy()

    def _write_positions(self, arm: np.ndarray, hand: np.ndarray, head: np.ndarray | None = None) -> None:
        arm = np.asarray(arm, dtype=np.float32).reshape(6)
        hand = np.clip(np.asarray(hand, dtype=np.float32).reshape(6), 0.0, 1.0)
        for name, value in zip(ARM_JOINTS, arm):
            low, high = self._joint_range(name)
            self.data.qpos[self.addresses[name]] = np.clip(value, low, high)
        for name, channel in zip(HAND_JOINTS, HAND_CHANNELS):
            low, high = self._joint_range(name)
            self.data.qpos[self.addresses[name]] = low + float(hand[channel]) * (high - low)
        if head is not None:
            mujoco_head = np.asarray(head, dtype=np.float32).reshape(2) * MUJOCO_HEAD_DIRECTIONS
            for name, value in zip(HEAD_JOINTS, mujoco_head):
                low, high = self._joint_range(name)
                self.data.qpos[self.addresses[name]] = np.clip(value, low, high)
        self.hand = hand
        self.mujoco.mj_forward(self.model, self.data)

    def set_command(
        self,
        arm: np.ndarray,
        hand: np.ndarray,
        *,
        duration_s: float,
        head: np.ndarray | None = None,
    ) -> None:
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        current, _ = self.state()
        target = np.array(arm, dtype=np.float32, copy=True).reshape(6)
        for index, name in enumerate(ARM_JOINTS):
            target[index] = np.clip(target[index], *self._joint_range(name))
        next_arm, self.arm_velocity = _advance_position_profile(
            current,
            self.arm_velocity,
            target,
            max_velocity=self.arm_profile_velocity,
            max_acceleration=self.arm_profile_acceleration,
            duration_s=duration_s,
        )
        self._write_positions(next_arm, hand, head=head)

    def open_viewer(self) -> None:
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.lookat[:] = (0.08, 0.08, 0.26)
        self.viewer.cam.distance = 1.05
        self.viewer.cam.azimuth = 135.0
        self.viewer.cam.elevation = -20.0
        self.viewer.sync()

    def sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        self.viewer.sync()
        return True

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class ImagePreview:
    def __init__(self, enabled: bool, image_size: int) -> None:
        self.enabled = enabled
        self.process: Any | None = None
        if enabled:
            env = os.environ.copy()
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("image_preview.py")),
                    str(image_size),
                    str(image_size),
                ],
                stdin=subprocess.PIPE,
                bufsize=0,
                env=env,
                start_new_session=True,
            )

    def show(self, image_rgb: np.ndarray, frame_index: int) -> bool:
        if not self.enabled:
            return True
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        assert self.process is not None
        if self.process.poll() is not None:
            return False
        try:
            self.process.stdin.write(image_rgb.tobytes())
        except BrokenPipeError:
            return False
        return True

    def close(self) -> None:
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=1.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=1.0)
            self.process = None

    def terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


class VideoPlayback:
    def __init__(
        self,
        replay: LeRobotEpisodeVideo,
        preview: ImagePreview,
        crop_ratios: tuple[float, float, float, float],
        policy_image_size: int,
        no_image_crop: bool,
    ) -> None:
        self.replay = replay
        self.preview = preview
        self.crop_ratios = crop_ratios
        self.policy_image_size = policy_image_size
        self.no_image_crop = no_image_crop
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dataset-video", daemon=True)
        self._latest: tuple[int, np.ndarray] | None = None
        self._finished = False
        self._error: Exception | None = None

    def _run(self) -> None:
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                raw_rgb = self.replay.read()
                if raw_rgb is None:
                    break
                _, policy_rgb, _ = _prepare_policy_image(
                    raw_rgb,
                    image_roi=None,
                    crop_ratios=self.crop_ratios,
                    policy_image_size=self.policy_image_size,
                    no_image_crop=self.no_image_crop,
                )
                frame_index = self.replay.frame_index - 1
                with self._condition:
                    self._latest = frame_index, policy_rgb
                    self._condition.notify_all()
                # ponytail: preview backpressure can slow playback; add a one-frame writer queue if needed.
                if not self.preview.show(policy_rgb, frame_index):
                    break
                next_tick += 1.0 / self.replay.fps
                delay = next_tick - time.monotonic()
                if delay > 0:
                    if self._stop.wait(delay):
                        break
                else:
                    next_tick = time.monotonic()
        except Exception as error:
            with self._condition:
                self._error = error
        finally:
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    def start(self) -> None:
        self._thread.start()

    def snapshot(
        self,
        *,
        after_index: int | None = None,
        timeout: float | None = None,
    ) -> tuple[int, np.ndarray] | None:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._error is not None
                or self._finished
                or self._latest is not None
                and (after_index is None or self._latest[0] > after_index),
                timeout,
            )
            if self._error is not None:
                raise RuntimeError("Dataset video playback failed") from self._error
            if not ready or self._latest is None or after_index is not None and self._latest[0] <= after_index:
                return None
            return self._latest

    @property
    def finished(self) -> bool:
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Dataset video playback failed") from self._error
            return self._finished

    @property
    def frame_index(self) -> int:
        with self._condition:
            return -1 if self._latest is None else self._latest[0]

    def close(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self.preview.terminate()
            self._thread.join()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-episode", type=int, default=0)
    parser.add_argument("--camera-key", default="")
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--prompt", default="Pick up the doll.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 plays the whole episode.")
    parser.add_argument("--control-hz", type=float, default=0.0, help="0 uses the dataset fps.")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-image-crop", action="store_true")
    parser.add_argument("--policy-image-size", type=int, default=int(DEFAULT_POLICY_IMAGE_SIZE[0]))
    parser.add_argument("--image-crop-ratios", nargs=4, type=float, default=list(DEFAULT_CROP_RATIOS))
    parser.add_argument("--chunk-steps", type=int, default=1, help="Actions to execute per synchronous inference.")
    parser.add_argument("--arm-profile-vel-deg", type=float, default=120.0)
    parser.add_argument("--arm-profile-acc-deg", type=float, default=120.0)
    parser.add_argument("--max-abs-arm-rad", type=float, default=3.14)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.dataset_episode < 0:
        raise ValueError("--dataset-episode must be non-negative")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.control_hz < 0 or not np.isfinite(args.control_hz):
        raise ValueError("--control-hz must be finite and non-negative")
    if args.policy_image_size <= 0 or args.log_every <= 0:
        raise ValueError("--policy-image-size and --log-every must be positive")
    if args.chunk_steps < 1:
        raise ValueError("--chunk-steps must be positive")
    if not np.isfinite([args.arm_profile_vel_deg, args.arm_profile_acc_deg]).all():
        raise ValueError("Arm profile velocity/acceleration must be finite")
    if args.arm_profile_vel_deg <= 0 or args.arm_profile_acc_deg <= 0:
        raise ValueError("Arm profile velocity/acceleration must be positive")


def _actions_for_chunk(result: dict[str, Any], chunk_steps: int) -> np.ndarray:
    actions = _require_actions(result["actions"])
    return actions[: min(chunk_steps, actions.shape[0])]


def _self_test(args: argparse.Namespace) -> int:
    replay = LeRobotEpisodeVideo(args.dataset_root, args.dataset_episode, args.camera_key)
    crop_ratios = tuple(float(value) for value in args.image_crop_ratios)
    preview = ImagePreview(False, args.policy_image_size)
    playback = VideoPlayback(replay, preview, crop_ratios, args.policy_image_size, args.no_image_crop)
    profile_velocity = math.radians(args.arm_profile_vel_deg)
    profile_acceleration = math.radians(args.arm_profile_acc_deg)
    sim = D1Mujoco(_find_d1_urdf(args.urdf), profile_velocity, profile_acceleration)
    try:
        state = DEFAULT_INITIAL_STATE.copy()
        playback.start()
        frame = playback.snapshot(timeout=5.0)
        assert frame is not None
        frame_index, policy_rgb = frame
        next_frame = playback.snapshot(after_index=frame_index, timeout=max(1.0, 2.0 / replay.fps))
        assert next_frame is not None and next_frame[0] > frame_index
        sim.reset(state, DEFAULT_HEAD_TARGET_RAD)
        sim_arm, sim_hand = sim.state()
        assert np.allclose(sim_arm, state[:6], atol=1e-5)
        assert np.allclose(sim_hand, state[6:12], atol=1e-5)
        sim_head = np.asarray([sim.data.qpos[sim.addresses[name]] for name in HEAD_JOINTS])
        assert np.allclose(sim_head, DEFAULT_HEAD_TARGET_RAD * MUJOCO_HEAD_DIRECTIONS, atol=1e-5)
        target = state[:6] + np.asarray([0.1, 0.1, -0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        duration_s = 0.1
        sim.set_command(target, state[6:12], duration_s=duration_s)
        sim_arm, _ = sim.state()
        first_step = 0.5 * min(profile_velocity, profile_acceleration * duration_s) * duration_s
        assert np.allclose(sim_arm, state[:6] + np.sign(target - state[:6]) * first_step, atol=1e-5)
        observation = _build_openpi_observation(policy_rgb, state, args.prompt)
        assert observation["observation/image"].shape == (
            args.policy_image_size,
            args.policy_image_size,
            3,
        )
        assert _actions_for_chunk(
            {"actions": np.zeros((16, ACTION_DIM), dtype=np.float32)}, args.chunk_steps
        ).shape == (min(args.chunk_steps, 16), ACTION_DIM)
        print(
            "D1_MUJOCO_DATASET_SELF_TEST_PASS",
            f"video={replay.video_path}",
            f"async_frames={frame_index}->{next_frame[0]}",
            f"policy_shape={policy_rgb.shape}",
            f"model_nq={sim.model.nq}",
        )
        return 0
    finally:
        playback.close()
        preview.close()
        replay.close()
        sim.close()


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.self_test:
        return _self_test(args)

    replay = LeRobotEpisodeVideo(args.dataset_root, args.dataset_episode, args.camera_key)
    control_hz = args.control_hz or replay.fps
    head_target = np.asarray(
        replay.info.get("d1_collection", {}).get("head_target_rad", DEFAULT_HEAD_TARGET_RAD),
        dtype=np.float32,
    )
    if head_target.shape != (2,) or not np.isfinite(head_target).all():
        raise ValueError(f"d1_collection.head_target_rad must contain two finite values, got {head_target}")
    crop_ratios = tuple(float(value) for value in args.image_crop_ratios)
    profile_velocity = math.radians(args.arm_profile_vel_deg)
    profile_acceleration = math.radians(args.arm_profile_acc_deg)
    safety = SafetyConfig(
        arm_delta_limits_rad=DEFAULT_ARM_DELTA_LIMITS_RAD.copy(),
        max_abs_arm_rad=float(args.max_abs_arm_rad),
        hand_min=0.0,
        hand_max=1.0,
    )
    sim = D1Mujoco(_find_d1_urdf(args.urdf), profile_velocity, profile_acceleration)
    sim.reset(DEFAULT_INITIAL_STATE, head_target)
    preview = ImagePreview(args.image_preview, args.policy_image_size)
    playback = VideoPlayback(replay, preview, crop_ratios, args.policy_image_size, args.no_image_crop)
    if args.viewer:
        sim.open_viewer()

    logging.info("Dataset video: %s fps=%.3f", replay.video_path, replay.fps)
    logging.info(
        "Control: %.3f Hz; arm_profile=(%.1f deg/s, %.1f deg/s^2); "
        "head_target=%s; mujoco_head=%s; prompt=%r",
        control_hz,
        args.arm_profile_vel_deg,
        args.arm_profile_acc_deg,
        _format_vec(head_target),
        _format_vec(head_target * MUJOCO_HEAD_DIRECTIONS),
        args.prompt,
    )
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_token,
    )
    logging.info("OpenPI server metadata: %s", policy.get_server_metadata())
    logging.info("Inference mode: synchronous, chunk_steps=%d", args.chunk_steps)
    if args.chunk_steps > 1:
        logging.warning("Chunk actions share the observation used for their synchronous inference.")
    period = 1.0 / control_hz
    next_tick = time.monotonic()
    steps = 0
    actions = np.empty((0, ACTION_DIM), dtype=np.float32)
    chunk_index = 0
    observation_frame_index = -1

    stop = False
    try:
        playback.start()
        while not stop and (args.max_steps == 0 or steps < args.max_steps):
            if playback.finished:
                logging.info("Dataset video playback finished.")
                break
            arm, hand = sim.state()
            needs_observation = chunk_index >= len(actions)
            if needs_observation:
                frame = playback.snapshot()
                if frame is None:
                    logging.info("Dataset video playback finished.")
                    break
                observation_frame_index, policy_rgb = frame
                observation = _build_openpi_observation(
                    policy_rgb,
                    np.concatenate([arm, hand]),
                    args.prompt,
                )
            if needs_observation:
                actions = _actions_for_chunk(policy.infer(observation), args.chunk_steps)
                chunk_index = 0
            action = actions[chunk_index]
            control_status = f"sync chunk={chunk_index + 1}/{len(actions)} obs_frame={observation_frame_index}"
            chunk_index += 1

            if playback.finished:
                logging.info("Dataset video playback finished.")
                break

            headarm_command, hand_command, clipped_delta = _command_from_action(
                head_target,
                arm,
                action,
                safety,
            )
            arm_command = headarm_command[2:8]
            sim.set_command(arm_command, hand_command, duration_s=period)
            sim_arm, _ = sim.state()

            if not sim.sync():
                stop = True
                break
            if steps % args.log_every == 0:
                logging.info(
                    "D1_MUJOCO step=%d video_frame=%d control=%s arm=%s target=%s delta=%s hand=%s",
                    steps,
                    playback.frame_index,
                    control_status,
                    _format_vec(sim_arm),
                    _format_vec(arm_command),
                    _format_vec(clipped_delta),
                    _format_vec(hand_command),
                )
            steps += 1
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        playback.close()
        preview.close()
        replay.close()
        sim.close()

    logging.info("Finished after %d steps.", steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
