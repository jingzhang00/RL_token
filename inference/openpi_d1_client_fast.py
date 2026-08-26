#!/usr/bin/env python3
"""D1 EDU OpenPI websocket client.

Default mode is a fast dry-run path: no policy commands are sent to the robot,
and no images are saved unless ``--record`` is explicitly provided.

Per inference, the observation sent to the policy server is aligned as:
  camera.read_view() -> head_arm/hand state read -> policy preprocessing -> websocket send

Policy action contract:
  action[0:6]   = D1 arm absolute joint target in radians
  action[6:12]  = D1 hand absolute normalized command

Robot command:
  headarm_cmd = [fixed_head_target, action[0:6]]
  hand_cmd    = clip(action[6:12], hand_min, hand_max)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websocket

OPENPI_WORKSPACE = Path(__file__).resolve().parents[1]

from d1_vision_policy import (
    DEFAULT_CAMERA_VIEW,
    DEFAULT_CROP_RATIOS,
    DEFAULT_HEAD_TARGET_RAD,
    DEFAULT_POLICY_IMAGE_SIZE as DEFAULT_POLICY_IMAGE_SHAPE,
    DEFAULT_RAW_IMAGE_SIZE,
    crop_box_for_ratios,
    crop_rgb_by_ratio,
    get_policy_vision_config,
    resize_rgb,
)


D1_ARM_DIM = 6
D1_HAND_DIM = 6
D1_STATE_DIM = D1_ARM_DIM + D1_HAND_DIM
DEFAULT_PROFILE = "d1_edu_policy_20260724_210345_trimmed_no_state_h10"
DEFAULT_ARM_DEV = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115148-if00"
DEFAULT_CAMERA_DEVICE = "/dev/v4l/by-id/usb-SunplusIT_Inc_SPCA2100_PC_Camera-video-index0"
DEFAULT_CONNECT_TIMEOUT_S = 20.0
DEFAULT_INFER_TIMEOUT_S = 120.0
DEFAULT_HEAD_TARGET_DEG = np.rad2deg(DEFAULT_HEAD_TARGET_RAD).astype(np.float32)
DEFAULT_POLICY_IMAGE_SIZE = int(DEFAULT_POLICY_IMAGE_SHAPE[0])
DEFAULT_INITIAL_STATE_FILE = Path(__file__).with_name("pick_up_the_doll_initial_state.json")
PREVIEW_WINDOW = "OpenPI D1 Policy Image"


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and np.dtype(obj.dtype).kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for msgpack numpy payload: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _dict_get(obj: dict, key: str) -> Any:
    return obj[key] if key in obj else obj[key.encode()]


def _unpack_array(obj: dict) -> Any:
    if "__ndarray__" in obj or b"__ndarray__" in obj:
        return np.ndarray(
            buffer=_dict_get(obj, "data"),
            dtype=np.dtype(_dict_get(obj, "dtype")),
            shape=_dict_get(obj, "shape"),
        )
    if "__npgeneric__" in obj or b"__npgeneric__" in obj:
        return np.dtype(_dict_get(obj, "dtype")).type(_dict_get(obj, "data"))
    return obj


_packb = partial(msgpack.packb, default=_pack_array, use_bin_type=True)
_unpackb = partial(msgpack.unpackb, object_hook=_unpack_array, raw=False)


class OpenPIWebsocketClient:
    """Synchronous OpenPI websocket client using packages available in d1_edu."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = 5555,
        *,
        api_key: str | None = None,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        infer_timeout_s: float = DEFAULT_INFER_TIMEOUT_S,
    ) -> None:
        if host.startswith("ws://") or host.startswith("wss://"):
            uri = host
        else:
            uri = f"ws://{host}"
        if port is not None:
            uri = f"{uri}:{int(port)}"
        self.uri = uri
        self.infer_timeout_s = float(infer_timeout_s)
        self._ws: websocket.WebSocket | None = None
        self._server_metadata: dict[str, Any] = {}

        headers = [f"Authorization: Api-Key {api_key}"] if api_key else None
        logging.info("Connecting to OpenPI websocket server at %s", self.uri)
        self._ws = websocket.create_connection(
            self.uri,
            timeout=float(connect_timeout_s),
            header=headers,
            enable_multithread=False,
        )
        self._ws.settimeout(self.infer_timeout_s)
        metadata = self._ws.recv()
        if isinstance(metadata, str):
            raise RuntimeError(f"Expected server metadata bytes, got text: {metadata}")
        self._server_metadata = _unpackb(metadata)
        logging.info("Connected to OpenPI server. metadata=%s", self._server_metadata)

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def infer(self, obs: dict[str, Any], *, wait_callback: Any = None) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("OpenPI websocket is closed")
        self._ws.send_binary(_packb(obs))
        deadline = time.monotonic() + self.infer_timeout_s
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(f"OpenPI inference timed out after {self.infer_timeout_s:.1f}s")
            self._ws.settimeout(min(0.05, remaining_s) if wait_callback is not None else remaining_s)
            try:
                response = self._ws.recv()
                break
            except websocket.WebSocketTimeoutException:
                if wait_callback is None:
                    raise
                wait_callback()
        if isinstance(response, str):
            raise RuntimeError(f"Error in OpenPI inference server:\n{response}")
        return _unpackb(response)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None


def _require_server_action_horizon(metadata: dict[str, Any]) -> int:
    try:
        action_horizon = int(_dict_get(metadata, "action_horizon"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"OpenPI server metadata must contain a positive action_horizon: {metadata}"
        ) from exc
    if action_horizon < 1:
        raise RuntimeError(f"OpenPI server returned invalid action_horizon={action_horizon}")
    return action_horizon


def _require_server_action_dim(metadata: dict[str, Any]) -> int:
    try:
        action_dim = int(_dict_get(metadata, "action_dim"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenPI server metadata must contain a positive action_dim: {metadata}") from exc
    if action_dim < 1:
        raise RuntimeError(f"OpenPI server returned invalid action_dim={action_dim}")
    return action_dim


def _format_vec(values: np.ndarray, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in np.asarray(values).reshape(-1)) + "]"


def _require_finite(name: str, values: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if shape is not None and array.shape != shape:
        raise RuntimeError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    return array


def _require_actions(actions: Any, *, action_dim: int, chunk_steps: int | None = None) -> np.ndarray:
    array = _require_finite("OpenPI actions", np.asarray(actions, dtype=np.float32))
    if array.ndim != 2 or array.shape[1] != action_dim:
        raise RuntimeError(f"OpenPI actions must have shape Hx{action_dim}, got {array.shape}")
    if array.shape[0] < 1:
        raise RuntimeError("OpenPI actions must contain at least one action")
    if chunk_steps is not None and chunk_steps > array.shape[0]:
        raise RuntimeError(f"--chunk-steps={chunk_steps} exceeds returned horizon {array.shape[0]}")
    return array


def _camera_bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    image_bgr = np.asarray(image_bgr, dtype=np.uint8)
    if image_bgr.ndim != 3 or image_bgr.shape[-1] != 3:
        raise RuntimeError(f"Camera image must be HxWx3 BGR, got {image_bgr.shape}")
    return np.ascontiguousarray(image_bgr[:, :, ::-1])


def _crop_rgb_for_policy(
    raw_rgb: np.ndarray,
    *,
    image_roi: tuple[int, int, int, int] | None,
    crop_ratios: tuple[float, float, float, float],
    no_image_crop: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
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
            raise ValueError(f"ROI {image_roi} outside raw image bounds width={raw_rgb.shape[1]}, height={raw_rgb.shape[0]}")
        crop_box = image_roi
        cropped = raw_rgb[y : y + h, x : x + w, :]
    return np.ascontiguousarray(cropped, dtype=np.uint8), crop_box


def _prepare_policy_image(
    raw_rgb: np.ndarray,
    *,
    image_roi: tuple[int, int, int, int] | None,
    crop_ratios: tuple[float, float, float, float],
    policy_image_size: int,
    no_image_crop: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    cropped, crop_box = _crop_rgb_for_policy(
        raw_rgb,
        image_roi=image_roi,
        crop_ratios=crop_ratios,
        no_image_crop=no_image_crop,
    )
    color_rgb = resize_rgb(cropped, (policy_image_size, policy_image_size))
    color_rgb = np.ascontiguousarray(color_rgb, dtype=np.uint8)
    if color_rgb.shape != (policy_image_size, policy_image_size, 3):
        raise RuntimeError(f"Policy image must have shape {(policy_image_size, policy_image_size, 3)}, got {color_rgb.shape}")
    return np.ascontiguousarray(cropped, dtype=np.uint8), color_rgb, crop_box


def _read_policy_state(head_arm: Any, hand: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    headarm = _require_finite("head+arm state", np.asarray(head_arm.get_positions(), dtype=np.float32), (8,))
    hand_q = _require_finite("hand state", np.asarray(hand.read_joint_pos(), dtype=np.float32), (D1_HAND_DIM,))
    head = headarm[:2].copy()
    arm = headarm[2:8].copy()
    state = np.concatenate([arm, hand_q], axis=0).astype(np.float32)
    return head, arm, state


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    except ModuleNotFoundError:
        import cv2

        bgr = np.asarray(image, dtype=np.uint8)[:, :, ::-1]
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Failed to write image: {path}")


def _actions_from_result(
    result: dict[str, Any], *, action_dim: int, chunk_steps: int | None = None
) -> np.ndarray:
    if "actions" not in result:
        raise RuntimeError(f"OpenPI response missing 'actions'. Keys: {sorted(result.keys())}")
    return _require_actions(result["actions"], action_dim=action_dim, chunk_steps=chunk_steps)


def _server_model_ms(result: dict[str, Any]) -> float:
    timing = result.get("policy_timing", result.get("server_timing", {}))
    if not isinstance(timing, dict):
        return -1.0
    return float(timing.get("infer_ms", -1.0))


def _command_from_absolute_action(
    head_target: np.ndarray,
    action: np.ndarray,
    *,
    action_dim: int,
    hand_min: float,
    hand_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    action = _require_finite("policy action", np.asarray(action, dtype=np.float32), (action_dim,))
    if action_dim != D1_STATE_DIM:
        raise RuntimeError(f"D1 client requires model action_dim={D1_STATE_DIM}, got {action_dim}")
    head_target = _require_finite("head target", np.asarray(head_target, dtype=np.float32), (2,))
    arm_cmd = action[:D1_ARM_DIM]
    hand_cmd = np.clip(action[D1_ARM_DIM:D1_STATE_DIM], hand_min, hand_max)
    headarm_cmd = np.concatenate([head_target, arm_cmd], axis=0)
    _require_finite("head+arm command", headarm_cmd, (8,))
    _require_finite("hand command", hand_cmd, (6,))
    return headarm_cmd.astype(float), hand_cmd.astype(float)


def _wait_until(deadline: float | None) -> float:
    if deadline is None:
        return 0.0
    remaining_s = deadline - time.monotonic()
    if remaining_s > 0:
        time.sleep(remaining_s)
        return 0.0
    return -remaining_s * 1000.0


def _add_examples_to_path() -> None:
    default_examples = Path(__file__).resolve().parents[2] / "data_collection" / "d1_collection"
    examples_dir = Path(str(__import__("os").environ.get("D1_EDU_EXAMPLES", default_examples)))
    if not (examples_dir / "D1_robot.py").is_file():
        raise FileNotFoundError(f"Could not find EDU D1_robot.py under {examples_dir}")
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))


def _load_initial_state(path: Path, *, state_dim: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Initial state file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "observation.state" not in data:
        raise KeyError(f"Missing observation.state in {path}")
    return _require_finite(
        "initial state",
        np.asarray(data["observation.state"], dtype=np.float32),
        (state_dim,),
    )


def _move_to_initial_state(
    head_arm: Any,
    hand: Any,
    head_target: np.ndarray,
    initial_state: np.ndarray,
    preview: Any = None,
) -> None:
    headarm_cmd = np.concatenate([head_target, initial_state[:6]], axis=0).astype(float)
    head_arm.set_positions(headarm_cmd.tolist())
    hand.set_joint_pos(initial_state[6:].astype(float).tolist())

    result = [None]
    error = [None]

    def wait_for_robot() -> None:
        try:
            result[0] = head_arm.wait_until_reached(
                headarm_cmd.tolist(),
                active_joint_indices=range(8),
                pos_tol_deg=5.0,
                vel_tol_deg_s=20.0,
                timeout_s=15.0,
            )
        except BaseException as exc:
            error[0] = exc

    if preview is None:
        wait_for_robot()
    else:
        wait_thread = threading.Thread(target=wait_for_robot, name="openpi-initial-pose-wait", daemon=True)
        wait_thread.start()
        try:
            while wait_thread.is_alive():
                preview()
        finally:
            wait_thread.join()

    if error[0] is not None:
        raise error[0]
    if result[0] is None:
        raise RuntimeError("Robot did not reach the initial pose within 15 seconds")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    core = parser.add_argument_group("core")
    vision = parser.add_argument_group("vision")
    runtime = parser.add_argument_group("runtime")
    hardware = parser.add_argument_group("hardware")

    core.add_argument("--profile", choices=(DEFAULT_PROFILE,), default=DEFAULT_PROFILE)
    core.add_argument("--prompt", default="Do the task.")
    core.add_argument("--max-steps", type=int, default=500)
    core.add_argument("--control-hz", type=float, default=5.0)
    core.add_argument(
        "--chunk-steps",
        type=int,
        default=1,
        help="Actions to execute from each policy chunk. Keep at 1 for strict obs/action alignment.",
    )
    core.add_argument("--log-every", type=int, default=10)

    runtime.add_argument("--host", default="127.0.0.1", help="OpenPI websocket host or ws:// URI.")
    runtime.add_argument("--port", type=int, default=5555)
    runtime.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    runtime.add_argument("--connect-timeout-s", type=float, default=DEFAULT_CONNECT_TIMEOUT_S)
    runtime.add_argument("--infer-timeout-s", type=float, default=DEFAULT_INFER_TIMEOUT_S)
    runtime.add_argument("--execute", action="store_true", help="Actually send commands to D1. Omit for dry-run.")
    runtime.add_argument(
        "--return-to-initial-on-interrupt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whenever D1Robot is open, run the homing shutdown sequence on Ctrl+C.",
    )
    runtime.add_argument("--server-smoke-test", action="store_true", help="Check server protocol and exit.")
    runtime.add_argument("--record", action="store_true", help="Save the first policy image sent to the server.")
    runtime.add_argument(
        "--record-dir",
        type=Path,
        default=OPENPI_WORKSPACE / "tools/d1_collection/test/output/live_infer",
    )

    vision.add_argument("--policy-image-size", type=int, default=DEFAULT_POLICY_IMAGE_SIZE)
    vision.add_argument(
        "--image-crop-ratios",
        nargs=4,
        type=float,
        default=list(DEFAULT_CROP_RATIOS),
        metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
    )
    vision.add_argument("--camera-view", choices=["left", "right", "both"], default=DEFAULT_CAMERA_VIEW)
    vision.add_argument(
        "--head-target-deg",
        nargs=2,
        type=float,
        default=DEFAULT_HEAD_TARGET_DEG.tolist(),
        metavar=("YAW", "PITCH"),
    )
    vision.add_argument("--image-roi", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"), help=argparse.SUPPRESS)
    vision.add_argument("--no-image-crop", action="store_true", help=argparse.SUPPRESS)
    vision.add_argument(
        "--image-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the policy RGB image; press q to stop.",
    )

    hardware.add_argument("--arm-dev", default=DEFAULT_ARM_DEV, help=argparse.SUPPRESS)
    hardware.add_argument("--arm-baud", type=int, default=1_000_000, help=argparse.SUPPRESS)
    hardware.add_argument("--hand-type", default="right", help=argparse.SUPPRESS)
    hardware.add_argument("--hand-can", default="can0", help=argparse.SUPPRESS)
    hardware.add_argument("--hand-baud", type=int, default=1_000_000, help=argparse.SUPPRESS)
    hardware.add_argument("--hand-speed", type=float, default=0.2, help=argparse.SUPPRESS)
    hardware.add_argument("--hand-min", type=float, default=0.0, help=argparse.SUPPRESS)
    hardware.add_argument("--hand-max", type=float, default=1.0, help=argparse.SUPPRESS)
    hardware.add_argument("--arm-profile-vel-deg", type=float, default=120.0, help=argparse.SUPPRESS)
    hardware.add_argument("--arm-profile-acc-deg", type=float, default=120.0, help=argparse.SUPPRESS)
    hardware.add_argument(
        "--initial-state-file",
        type=Path,
        default=DEFAULT_INITIAL_STATE_FILE,
        help="JSON file containing the 12-D observation.state reset pose.",
    )
    hardware.add_argument("--move-head-on-start", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(wait_head_on_start=True)
    hardware.add_argument("--wait-head-on-start", dest="wait_head_on_start", action="store_true", help=argparse.SUPPRESS)
    hardware.add_argument("--no-wait-head-on-start", dest="wait_head_on_start", action="store_false", help=argparse.SUPPRESS)
    hardware.add_argument("--open-hand-on-exit", action="store_true", help=argparse.SUPPRESS)
    hardware.add_argument("--camera-device", default=DEFAULT_CAMERA_DEVICE, help=argparse.SUPPRESS)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.chunk_steps < 1:
        raise ValueError("--chunk-steps must be positive")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.control_hz <= 0 or not np.isfinite(args.control_hz):
        raise ValueError("--control-hz must be positive and finite")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.policy_image_size <= 0:
        raise ValueError("--policy-image-size must be positive")
    if args.connect_timeout_s <= 0 or args.infer_timeout_s <= 0:
        raise ValueError("Server timeout values must be positive")
    if not np.isfinite([args.connect_timeout_s, args.infer_timeout_s]).all():
        raise ValueError("Server timeout values must be finite")
    if len(args.image_crop_ratios) != 4:
        raise ValueError("--image-crop-ratios must contain four values")
    if not np.isfinite(args.image_crop_ratios).all():
        raise ValueError("--image-crop-ratios values must be finite")
    if not np.isfinite(args.head_target_deg).all():
        raise ValueError("--head-target-deg values must be finite")
    if not np.isfinite([args.hand_min, args.hand_max]).all() or args.hand_min > args.hand_max:
        raise ValueError("Hand limits must be finite and satisfy --hand-min <= --hand-max")


def _server_smoke_test(args: argparse.Namespace) -> int:
    policy = OpenPIWebsocketClient(
        args.host,
        args.port,
        api_key=args.api_key,
        connect_timeout_s=args.connect_timeout_s,
        infer_timeout_s=args.infer_timeout_s,
    )
    try:
        metadata = policy.get_server_metadata()
        action_horizon = _require_server_action_horizon(metadata)
        action_dim = _require_server_action_dim(metadata)
        obs = {
            "observation/image": np.zeros((*DEFAULT_RAW_IMAGE_SIZE, 3), dtype=np.uint8),
            "observation/state": np.zeros((D1_STATE_DIM,), dtype=np.float32),
            "prompt": args.prompt,
        }
        t0 = time.monotonic()
        result = policy.infer(obs)
        roundtrip_ms = (time.monotonic() - t0) * 1000.0
        actions = _actions_from_result(result, action_dim=action_dim)
        server_ms = _server_model_ms(result)
        logging.info(
            "SERVER_SMOKE_PASS actions_shape=%s roundtrip_ms=%.1f server_model_ms=%.1f first_action=%s",
            actions.shape,
            roundtrip_ms,
            server_ms,
            _format_vec(actions[0]),
        )
        if actions.shape != (action_horizon, action_dim):
            raise RuntimeError(f"Expected actions shape {(action_horizon, action_dim)}, got {actions.shape}")
        return 0
    finally:
        policy.close()


def main() -> int:
    args = _parse_args()
    _validate_args(args)

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.chunk_steps > 1:
        logging.warning(
            "chunk_steps=%d: only chunk 0 is strictly aligned with the observation sent to server; "
            "later chunk commands read fresh state but reuse the same action chunk.",
            args.chunk_steps,
        )

    crop_ratios = tuple(float(v) for v in args.image_crop_ratios)
    image_roi = None if args.image_roi is None else tuple(int(v) for v in args.image_roi)
    head_target = _require_finite(
        "head target",
        np.deg2rad(np.asarray(args.head_target_deg, dtype=np.float32)),
        (2,),
    )
    vision_cfg = get_policy_vision_config(
        camera_view=args.camera_view,
        head_target_rad=head_target,
        crop_ratios=crop_ratios,
        policy_image_size=(args.policy_image_size, args.policy_image_size),
    )

    logging.info("OpenPI D1 fast client. execute=%s record=%s", args.execute, args.record)
    logging.info("Action semantics: arm_cmd=action[:6] (absolute), hand_cmd=clip(action[6:12]).")
    logging.info("Obs order: camera frame -> current robot state -> preprocessing -> websocket send.")
    logging.info("Fixed head target rad: %s", _format_vec(head_target))
    logging.info("Preview vision config: %s", vision_cfg.to_metadata())
    logging.info("Model image input: uncropped camera RGB; server applies the training resize_with_pad.")
    if not args.execute:
        logging.warning("DRY-RUN mode: policy inference runs, but robot commands are not sent.")

    if args.server_smoke_test:
        return _server_smoke_test(args)

    initial_state = None

    _add_examples_to_path()
    from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path
    from D1_robot import D1Robot

    policy = None
    robot = None
    head_arm = None
    hand = None
    camera = None
    preview_cv2 = None
    steps_done = 0
    inference_index = 0
    next_command_deadline = None
    period_s = 1.0 / args.control_hz
    record_dir = args.record_dir.expanduser().resolve()

    try:
        if args.image_preview:
            import cv2

            preview_cv2 = cv2
            logging.info("Image preview enabled: %s (press q to stop).", PREVIEW_WINDOW)

        robot = D1Robot(
            urdf_path=get_default_urdf_path(),
            arm_dev=args.arm_dev,
            arm_baud=args.arm_baud,
            hand_type=args.hand_type,
            hand_can=args.hand_can,
            hand_baud=args.hand_baud,
            vision_device=args.camera_device,
        )
        head_arm = robot.head_arm
        hand = robot.hand
        camera = robot.vision
        camera.view = args.camera_view
        camera.start()

        def refresh_preview() -> None:
            if preview_cv2 is None:
                return
            preview_raw_rgb = _camera_bgr_to_rgb(camera.read_view())
            _, preview_rgb, _ = _prepare_policy_image(
                preview_raw_rgb,
                image_roi=image_roi,
                crop_ratios=crop_ratios,
                policy_image_size=args.policy_image_size,
                no_image_crop=args.no_image_crop,
            )
            preview_cv2.imshow(PREVIEW_WINDOW, preview_cv2.cvtColor(preview_rgb, preview_cv2.COLOR_RGB2BGR))
            if preview_cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        if args.max_steps > 0:
            policy = OpenPIWebsocketClient(
                args.host,
                args.port,
                api_key=args.api_key,
                connect_timeout_s=args.connect_timeout_s,
                infer_timeout_s=args.infer_timeout_s,
            )
            metadata = policy.get_server_metadata()
            action_horizon = _require_server_action_horizon(metadata)
            action_dim = _require_server_action_dim(metadata)
            if action_dim != D1_STATE_DIM:
                raise RuntimeError(f"D1 client requires model action_dim={D1_STATE_DIM}, got {action_dim}")
            if args.chunk_steps > action_horizon:
                raise RuntimeError(
                    f"--chunk-steps={args.chunk_steps} exceeds server action_horizon={action_horizon}"
                )
            logging.info("Server action horizon=%d", action_horizon)

        if args.execute:
            initial_state = _load_initial_state(
                args.initial_state_file.expanduser().resolve(), state_dim=D1_STATE_DIM
            )
            if np.any(initial_state[6:] < args.hand_min) or np.any(initial_state[6:] > args.hand_max):
                raise ValueError("Initial hand state is outside --hand-min/--hand-max")
            logging.info("Initial state file: %s state=%s", args.initial_state_file, _format_vec(initial_state))

        if args.move_head_on_start or args.execute:
            if args.arm_profile_vel_deg > 0 and args.arm_profile_acc_deg > 0:
                try:
                    vels = [math.radians(args.arm_profile_vel_deg)] * 8
                    accs = [math.radians(args.arm_profile_acc_deg)] * 8
                    head_arm.set_profile(vels, accs)
                    logging.info(
                        "Set head/arm profile: vel=%.1f deg/s acc=%.1f deg/s^2",
                        args.arm_profile_vel_deg,
                        args.arm_profile_acc_deg,
                    )
                except Exception:
                    logging.exception("Failed to set head/arm motion profile; continuing with SDK defaults.")
        if args.execute:
            assert initial_state is not None
            hand.set_speed(speed=[args.hand_speed] * 6)
            refresh_preview()
            input("[1/2] Move to initial state, press Enter\nCtrl+C to cancel:")
            logging.info(
                "Moving to initial pose: head=%s arm=%s hand=%s",
                _format_vec(head_target),
                _format_vec(initial_state[:6]),
                _format_vec(initial_state[6:]),
            )
            _move_to_initial_state(
                head_arm,
                hand,
                head_target,
                initial_state,
                preview=refresh_preview if preview_cv2 is not None else None,
            )
            input("[2/2] Initial state reached, press Enter to start execution,\nCtrl+C to cancel:")
        elif args.move_head_on_start:
            current_head, current_arm, _state = _read_policy_state(head_arm, hand)
            headarm_init_cmd = np.concatenate([head_target, current_arm], axis=0).astype(float)
            logging.info("Move/fix head from %s to %s", _format_vec(current_head), _format_vec(head_target))
            head_arm.set_positions(headarm_init_cmd.tolist())
            if args.wait_head_on_start:
                head_arm.wait_until_reached(headarm_init_cmd.tolist(), active_joint_indices=range(2))

        while steps_done < args.max_steps:
            camera_t0 = time.monotonic()
            raw_color_bgr = camera.read_view()
            camera_end = time.monotonic()

            state_t0 = time.monotonic()
            _head, arm, state = _read_policy_state(head_arm, hand)
            state_end = time.monotonic()

            preprocess_t0 = time.monotonic()
            raw_color_rgb = _camera_bgr_to_rgb(raw_color_bgr)
            cropped_rgb, color_rgb, crop_box = _prepare_policy_image(
                raw_color_rgb,
                image_roi=image_roi,
                crop_ratios=crop_ratios,
                policy_image_size=args.policy_image_size,
                no_image_crop=args.no_image_crop,
            )
            preprocess_end = time.monotonic()

            if preview_cv2 is not None:
                preview_cv2.imshow(PREVIEW_WINDOW, preview_cv2.cvtColor(color_rgb, preview_cv2.COLOR_RGB2BGR))
                if preview_cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

            obs = {
                "observation/image": np.asarray(raw_color_rgb, dtype=np.uint8),
                "observation/state": state.astype(np.float32),
                "prompt": args.prompt,
            }
            infer_t0 = time.monotonic()
            if policy is None:
                raise RuntimeError("OpenPI policy client is not connected")
            result = policy.infer(
                obs,
                wait_callback=refresh_preview if preview_cv2 is not None else None,
            )
            infer_end = time.monotonic()
            actions = _actions_from_result(result, action_dim=action_dim, chunk_steps=args.chunk_steps)
            if actions.shape[0] != action_horizon:
                raise RuntimeError(
                    f"Server metadata action_horizon={action_horizon}, got {actions.shape[0]} actions"
                )
            server_ms = _server_model_ms(result)

            for chunk_i in range(args.chunk_steps):
                if steps_done >= args.max_steps:
                    break
                deadline_overrun_ms = _wait_until(next_command_deadline)
                if chunk_i > 0:
                    _head, arm, state = _read_policy_state(head_arm, hand)

                action = actions[chunk_i]
                headarm_cmd, hand_cmd = _command_from_absolute_action(
                    head_target,
                    action,
                    action_dim=action_dim,
                    hand_min=float(args.hand_min),
                    hand_max=float(args.hand_max),
                )

                command_t0 = time.monotonic()
                if args.execute:
                    head_arm.set_positions(headarm_cmd.tolist())
                    hand.set_joint_pos(hand_cmd.tolist())
                command_end = time.monotonic()
                next_command_deadline = command_t0 + period_s

                refresh_preview()

                if args.record and steps_done == 0:
                    record_dir.mkdir(parents=True, exist_ok=True)
                    _save_rgb(record_dir / "inference_000000_policy.png", raw_color_rgb)

                if steps_done % args.log_every == 0:
                    mode = "FAST_EXEC" if args.execute else "FAST_DRY"
                    logging.info(
                        "%s step=%d infer=%d chunk=%d timing_ms(camera=%.1f,state=%.1f,preprocess=%.1f,roundtrip=%.1f,server_model=%.1f,cmd=%.1f,deadline_overrun=%.1f) model_input_shape=%s crop_box=%s preview_shape=%s returned_horizon=%d arm=%s arm_target=%s arm_cmd=%s hand_cmd=%s",
                        mode,
                        steps_done,
                        inference_index,
                        chunk_i,
                        (camera_end - camera_t0) * 1000.0,
                        (state_end - state_t0) * 1000.0,
                        (preprocess_end - preprocess_t0) * 1000.0,
                        (infer_end - infer_t0) * 1000.0,
                        server_ms,
                        (command_end - command_t0) * 1000.0,
                        deadline_overrun_ms,
                        tuple(raw_color_rgb.shape),
                        crop_box,
                        tuple(color_rgb.shape),
                        actions.shape[0],
                        _format_vec(arm),
                        _format_vec(action[:6]),
                        _format_vec(headarm_cmd[2:8]),
                        _format_vec(hand_cmd),
                    )
                steps_done += 1
            inference_index += 1

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        if args.return_to_initial_on_interrupt and robot is not None:
            logging.warning("Press Ctrl+C again to stop immediately.")
            try:
                robot.close()
                robot = None
                hand = None
                head_arm = None
                logging.info("Homing finished.")
            except Exception:
                logging.exception("Homing shutdown failed.")
    finally:
        if policy is not None:
            policy.close()
        if args.open_hand_on_exit and hand is not None:
            try:
                hand.open_hand()
            except Exception:
                logging.exception("Failed to open hand during cleanup.")
        if robot is not None:
            try:
                robot.stop_vision_thread()
            except Exception:
                logging.exception("Failed to stop D1 vision thread.")
        if hand is not None:
            try:
                hand.close_can()
            except Exception:
                logging.exception("Failed to close hand CAN.")
        if head_arm is not None:
            try:
                head_arm.close()
            except Exception:
                logging.exception("Failed to close head/arm.")
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                logging.exception("Failed to stop EDU camera.")
        if preview_cv2 is not None:
            preview_cv2.destroyAllWindows()

    logging.info("Finished after %d steps.", steps_done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
