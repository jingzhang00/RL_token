"""D1 post-training config."""

import dataclasses
import logging
import os
import pathlib

import openpi.models.pi0_config as pi0_config
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from d1 import d1_policy

OPENPI_WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
D1_REPO_ID = os.environ.get("D1_REPO_ID")
PI05_BASE_CKPT = os.environ.get("PI05_BASE_CKPT")

@dataclasses.dataclass(frozen=True)
class LeRobotD1DataConfig:
    repo_id: str
    base_config: object | None = None
    use_quantile_norm: bool | None = None

    def create(self, assets_dirs, model_config):
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.d1_rgb",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[d1_policy.D1Inputs()],
            outputs=[d1_policy.D1Outputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(6, -6)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        return dataclasses.replace(
            _create_base_config(self, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=_get_model_transforms(model_config),
            action_sequence_keys=("action",),
        )


def _create_base_config(data_config, model_config):
    from openpi.training.config import DataConfig
    from openpi.training.config import ModelType

    asset_id = "d1"
    return dataclasses.replace(
        data_config.base_config or DataConfig(),
        repo_id=data_config.repo_id,
        asset_id=asset_id,
        norm_stats=_load_norm_stats(data_config.repo_id),
        use_quantile_norm=(
            data_config.use_quantile_norm
            if data_config.use_quantile_norm is not None
            else model_config.model_type != ModelType.PI0
        ),
    )


def _load_norm_stats(dataset_dir: str | None) -> dict[str, _transforms.NormStats] | None:
    if not dataset_dir:
        return None
    try:
        norm_stats = _normalize.load(dataset_dir)
        logging.info("Loaded norm stats from %s", dataset_dir)
        return norm_stats
    except FileNotFoundError:
        logging.info("Norm stats not found in %s, skipping.", dataset_dir)
    return None


def _get_model_transforms(model_config):
    from openpi.training.config import ModelTransformFactory

    return ModelTransformFactory()(model_config)


def get_d1_configs():
    from openpi.training.config import DataConfig
    from openpi.training.config import TrainConfig

    return [
        TrainConfig(
            name="pi05_d1",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=20, discrete_state_input=True),
            data=LeRobotD1DataConfig(
                repo_id=D1_REPO_ID,
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE_CKPT),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            batch_size=128,
            num_train_steps=30_000,
            save_interval=1_000,
            keep_period=15_000,
            # ponytail: single-process stats avoids wrapper pickling; restore workers after an importable stats module.
            num_workers=0,
            fsdp_devices=4,
        )
    ]
