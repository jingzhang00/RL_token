import dataclasses

import einops
import numpy as np

from openpi import transforms


STATE_DIM = 12


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _decode_prompt(prompt):
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    if isinstance(prompt, np.ndarray) and prompt.dtype.kind in {"S", "O"} and prompt.shape == ():
        value = prompt.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return prompt


@dataclasses.dataclass(frozen=True)
class D1Inputs(transforms.DataTransformFn):
    """Map D1 LeRobot samples to OpenPI inputs."""

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        zero_wrist = np.zeros_like(base_image)

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": zero_wrist,
                "right_wrist_0_rgb": zero_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = _decode_prompt(data["prompt"])
        return inputs


@dataclasses.dataclass(frozen=True)
class D1Outputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}
