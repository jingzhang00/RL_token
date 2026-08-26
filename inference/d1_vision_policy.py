"""Shared D1 EDU policy-vision configuration and preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


DEFAULT_HEAD_TARGET_RAD = np.asarray([-0.2617994, -1.0471975], dtype=np.float32)
DEFAULT_CAMERA_VIEW = "left"
# DEFAULT_CROP_RATIOS = (0.20, 0.05, 0.10, 0.10)
DEFAULT_CROP_RATIOS = (0.125, 0.125, 0.0, 0.0)
DEFAULT_POLICY_IMAGE_SIZE = (256, 256)
DEFAULT_RAW_IMAGE_SIZE = (480, 640)


@dataclass(frozen=True)
class D1PolicyVisionConfig:
    camera_view: str = DEFAULT_CAMERA_VIEW
    head_target_rad: tuple[float, float] = (float(DEFAULT_HEAD_TARGET_RAD[0]), float(DEFAULT_HEAD_TARGET_RAD[1]))
    crop_ratios: tuple[float, float, float, float] = DEFAULT_CROP_RATIOS
    policy_image_size: tuple[int, int] = DEFAULT_POLICY_IMAGE_SIZE

    def to_metadata(self) -> dict:
        data = asdict(self)
        data["head_target_deg"] = np.rad2deg(np.asarray(self.head_target_rad, dtype=np.float64)).round(4).tolist()
        return data


def get_policy_vision_config(
    *,
    camera_view: str = DEFAULT_CAMERA_VIEW,
    head_target_rad: tuple[float, float] | list[float] | np.ndarray = DEFAULT_HEAD_TARGET_RAD,
    crop_ratios: tuple[float, float, float, float] | list[float] = DEFAULT_CROP_RATIOS,
    policy_image_size: tuple[int, int] | list[int] = DEFAULT_POLICY_IMAGE_SIZE,
) -> D1PolicyVisionConfig:
    head = tuple(float(v) for v in np.asarray(head_target_rad, dtype=np.float32).reshape(2))
    crop = tuple(float(v) for v in crop_ratios)
    if len(crop) != 4:
        raise ValueError(f"crop_ratios must contain 4 values, got {crop_ratios}")
    size = tuple(int(v) for v in policy_image_size)
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"policy_image_size must be positive height,width, got {policy_image_size}")
    if camera_view not in ("left", "right", "both"):
        raise ValueError(f"camera_view must be left/right/both, got {camera_view!r}")
    return D1PolicyVisionConfig(
        camera_view=camera_view,
        head_target_rad=head,
        crop_ratios=crop,
        policy_image_size=size,
    )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_BGR2RGB)


def crop_box_for_ratios(
    image_shape: tuple[int, int] | tuple[int, int, int],
    crop_ratios: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    left_ratio, right_ratio, top_ratio, bottom_ratio = crop_ratios
    left_ratio = max(0.0, min(float(left_ratio), 0.9))
    right_ratio = max(0.0, min(float(right_ratio), 0.9))
    top_ratio = max(0.0, min(float(top_ratio), 0.9))
    bottom_ratio = max(0.0, min(float(bottom_ratio), 0.9))

    height, width = int(image_shape[0]), int(image_shape[1])
    x0 = int(width * left_ratio)
    x1 = int(width * (1.0 - right_ratio))
    y0 = int(height * top_ratio)
    y1 = int(height * (1.0 - bottom_ratio))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Invalid crop ratios: "
            f"left={left_ratio}, right={right_ratio}, top={top_ratio}, bottom={bottom_ratio}"
        )
    return x0, y0, x1 - x0, y1 - y0


def crop_rgb_by_ratio(
    image: np.ndarray,
    crop_ratios: tuple[float, float, float, float],
) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    x, y, w, h = crop_box_for_ratios(image.shape, crop_ratios)
    return image[y : y + h, x : x + w, :]


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = (int(size[0]), int(size[1]))
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    try:
        import cv2

        return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    except Exception:
        y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
        x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
        return np.ascontiguousarray(image[y_idx][:, x_idx])


def preprocess_policy_rgb(
    raw_rgb: np.ndarray,
    config: D1PolicyVisionConfig | None = None,
) -> np.ndarray:
    cfg = config or get_policy_vision_config()
    cropped = crop_rgb_by_ratio(np.asarray(raw_rgb, dtype=np.uint8), cfg.crop_ratios)
    return resize_rgb(cropped, cfg.policy_image_size)
