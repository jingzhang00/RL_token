"""Read-only adapter that shares one frozen OpenPI prefix pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


class OpenPIAdapter:
    """Expose selected final VLA image tokens and reference actions."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config_name: str | None = None,
        prompt: str | None = None,
        *,
        num_steps: int | None = None,
        seed: int | None = None,
        camera_names: list[str] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        from rlt.settings import load_config as load_settings

        from robots.d1.train import load_config
        from openpi.policies import policy_config

        settings = load_settings(config_path)
        checkpoint = checkpoint or settings.vla_checkpoint
        config_name = config_name or settings.vla["config_name"]
        prompt = prompt or settings.vla["prompt"]
        num_steps = settings.vla["inference_steps"] if num_steps is None else num_steps
        seed = settings.rl_token["seed"] if seed is None else seed
        self.camera_names = camera_names or list(settings.vla["rl_token_cameras"])
        config = load_config().get_config(config_name)
        self.policy = policy_config.create_trained_policy(
            config, Path(checkpoint), default_prompt=prompt
        )
        if self.policy._is_pytorch_model:
            raise NotImplementedError(
                "RL-token extraction currently requires the JAX OpenPI checkpoint"
            )
        self.model = self.policy._model
        self.num_steps = int(num_steps)
        self.rng = jax.random.key(int(seed))
        self.action_dim = settings.action_dim
        self._shared = self._compile_shared_forward()

    def _compile_shared_forward(self):
        from openpi.models import model as model_api
        from openpi.models.pi0 import make_attn_mask

        graphdef, state = nnx.split(self.model)
        num_steps = self.num_steps

        def run(frozen_state, observation, noise):
            model = nnx.merge(graphdef, frozen_state)
            observation = model_api.preprocess_observation(
                None, observation, train=False
            )
            prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(observation)
            prefix_attn = make_attn_mask(prefix_mask, prefix_ar)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), kv_cache = model.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn, positions=positions
            )
            batch = observation.state.shape[0]
            dt = -1.0 / num_steps

            def step(carry):
                action, time = carry
                suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(
                    observation, action, jnp.broadcast_to(time, batch)
                )
                suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
                prefix_to_suffix = einops.repeat(
                    prefix_mask, "b p -> b s p", s=suffix.shape[1]
                )
                mask = jnp.concatenate([prefix_to_suffix, suffix_attn], axis=-1)
                suffix_positions = (
                    jnp.sum(prefix_mask, axis=-1)[:, None]
                    + jnp.cumsum(suffix_mask, axis=-1)
                    - 1
                )
                (_, suffix_out), _ = model.PaliGemma.llm(
                    [None, suffix],
                    mask=mask,
                    positions=suffix_positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, cond],
                )
                velocity = model.action_out_proj(suffix_out[:, -model.action_horizon :])
                return action + dt * velocity, time + dt

            action, _ = jax.lax.while_loop(
                lambda value: value[1] >= -dt / 2, step, (noise, 1.0)
            )
            return prefix_out, prefix_mask, action

        compiled = jax.jit(run)
        return lambda observation, noise: compiled(state, observation, noise)

    def _inputs(self, observations: list[dict[str, Any]]):
        from openpi.models import model as model_api

        transformed = [
            self.policy._input_transform(jax.tree.map(lambda x: x, obs))
            for obs in observations
        ]
        batch = jax.tree.map(
            lambda *values: jnp.asarray(np.stack(values)), *transformed
        )
        return batch, model_api.Observation.from_dict(batch)

    def infer(
        self,
        observations: dict[str, Any] | list[dict[str, Any]],
        *,
        noise: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        single = isinstance(observations, dict)
        items = [observations] if single else observations
        if not items:
            raise ValueError("observations must not be empty")
        inputs, observation = self._inputs(items)
        batch = len(items)
        if noise is None:
            self.rng, key = jax.random.split(self.rng)
            noise_array = jax.random.normal(
                key, (batch, self.model.action_horizon, self.model.action_dim)
            )
        else:
            noise_array = jnp.asarray(noise)
            if noise_array.ndim == 2:
                noise_array = noise_array[None]
        prefix, prefix_mask, normalized = self._shared(observation, noise_array)

        language_count = (
            0
            if observation.tokenized_prompt is None
            else observation.tokenized_prompt.shape[1]
        )
        image_names = list(observation.images)
        image_count = len(image_names)
        if image_count < 1:
            raise ValueError("VLA observation contains no images")
        image_tokens = (prefix.shape[1] - language_count) // image_count
        blocks = []
        masks = []
        for camera_name in self.camera_names:
            if camera_name not in image_names:
                raise ValueError(f"camera {camera_name!r} is missing from VLA inputs")
            camera_index = image_names.index(camera_name)
            start, stop = camera_index * image_tokens, (camera_index + 1) * image_tokens
            blocks.append(prefix[:, start:stop])
            masks.append(prefix_mask[:, start:stop])
        tokens = np.asarray(jnp.concatenate(blocks, axis=1), dtype=np.float32)
        valid = np.asarray(jnp.concatenate(masks, axis=1), dtype=bool)
        normalized_np = np.asarray(normalized, dtype=np.float32)
        absolute = []
        for index in range(batch):
            output = self.policy._output_transform(
                {
                    "state": np.asarray(inputs["state"][index]),
                    "actions": normalized_np[index].copy(),
                }
            )
            absolute.append(np.asarray(output["actions"], dtype=np.float32))
        result = {
            "tokens": tokens,
            "valid": valid,
            "reference": normalized_np[:, :, : self.action_dim],
            "absolute": np.stack(absolute)[:, :, : self.action_dim],
            "normalized_state": np.asarray(inputs["state"], dtype=np.float32),
        }
        return {key: value[0] for key, value in result.items()} if single else result

    def absolute_actions(
        self, normalized_actions: np.ndarray, normalized_state: np.ndarray
    ) -> np.ndarray:
        actions = np.asarray(normalized_actions, dtype=np.float32)
        output = self.policy._output_transform(
            {
                "state": np.asarray(normalized_state, dtype=np.float32),
                "actions": actions.copy(),
            }
        )
        return np.asarray(output["actions"], dtype=np.float32)[:, : self.action_dim]
