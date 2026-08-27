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
from robots.d1 import d1_policy
from rlt.settings import load_config

OPENPI_WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
RLT_SETTINGS = load_config()
D1_REPO_ID = os.environ.get("D1_REPO_ID", str(RLT_SETTINGS.vla["dataset"]))
PI05_BASE_CKPT = os.environ.get(
    "PI05_BASE_CKPT", str(RLT_SETTINGS.vla["base_checkpoint"])
)
D1_FINETUNE_CLIP_GRADIENT_NORM = 1.0

@dataclasses.dataclass(frozen=True)
class LeRobotD1DataConfig:
    repo_id: str
    action_dim: int
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
            outputs=[d1_policy.D1Outputs(self.action_dim)],
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

    finetune = RLT_SETTINGS.vla_finetune
    action_dim = RLT_SETTINGS.action_dim
    return [
        TrainConfig(
            name=RLT_SETTINGS.vla["config_name"],
            project_name=RLT_SETTINGS.project_name,
            exp_name=finetune["exp_name"],
            model=pi0_config.Pi0Config(
                pi05=RLT_SETTINGS.vla["model"]["pi05"],
                action_dim=action_dim,
                action_horizon=RLT_SETTINGS.action_horizon,
                discrete_state_input=RLT_SETTINGS.vla["model"]["discrete_state_input"],
            ),
            data=LeRobotD1DataConfig(
                repo_id=D1_REPO_ID,
                action_dim=action_dim,
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE_CKPT),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=finetune["warmup_steps"],
                peak_lr=finetune["peak_lr"],
                decay_steps=finetune["decay_steps"],
                decay_lr=finetune["decay_lr"],
            ),
            optimizer=_optimizer.AdamW(
                clip_gradient_norm=D1_FINETUNE_CLIP_GRADIENT_NORM
            ),
            ema_decay=finetune["ema_decay"],
            batch_size=finetune["batch_size"],
            num_train_steps=finetune["steps"],
            log_interval=RLT_SETTINGS.log_interval,
            save_interval=finetune["save_interval"],
            keep_period=finetune["keep_period"],
            checkpoint_base_dir=RLT_SETTINGS.vla["output"],
            overwrite=finetune["overwrite"],
            resume=finetune["resume"],
            wandb_enabled=RLT_SETTINGS.wandb_enabled,
            num_workers=finetune["num_workers"],
            fsdp_devices=finetune["fsdp_devices"],
        )
    ]
