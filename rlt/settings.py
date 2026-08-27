"""Single source of truth for the local RL-token experiment settings."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config") / "config.yaml"


def _steps(seconds: float, hz: float) -> int:
    return max(1, int(math.floor(float(seconds) * float(hz) + 0.5)))


def dataset_action_dim(dataset: str | Path) -> int:
    """Read the per-timestep action width from a local LeRobot dataset."""
    dataset = Path(dataset).expanduser()
    info_path = dataset / "meta" / "info.json"
    if info_path.is_file():
        import json

        info = json.loads(info_path.read_text(encoding="utf-8"))
        shape = info.get("features", {}).get("action", {}).get("shape")
        if isinstance(shape, list) and len(shape) == 1 and int(shape[0]) > 0:
            return int(shape[0])

    import pyarrow.parquet as parquet

    tables = sorted((dataset / "data").glob("chunk-*/episode_*.parquet"))
    if tables:
        table = parquet.read_table(tables[0], columns=["action"])
        value = table["action"][0].as_py()
        if isinstance(value, list) and value:
            return len(value)
    raise FileNotFoundError(f"cannot determine action dimension from dataset {dataset}")


@dataclass(frozen=True)
class MLPConfig:
    actor_hidden_dim: int
    actor_num_layers: int
    critic_hidden_dim: int
    critic_num_layers: int


@dataclass(frozen=True)
class RLTConfig:
    """Parsed config plus values whose units depend on control frequency."""

    path: Path
    robot_path: Path
    task: str
    wandb_enabled: bool
    project_name: str
    log_interval: int
    control_hz: float
    chunk_duration_s: float
    vla_horizon_duration_s: float
    transition_stride_steps: int
    vla: dict[str, Any]
    vla_finetune: dict[str, Any]
    rl_token: dict[str, Any]
    online_rl: dict[str, Any]
    runtime: dict[str, Any]
    hardware: dict[str, Any]
    mlp: MLPConfig

    @property
    def chunk_len(self) -> int:
        return _steps(self.chunk_duration_s, self.control_hz)

    @property
    def transition_horizon(self) -> int:
        return self.chunk_len

    @property
    def vla_horizon(self) -> int:
        return _steps(self.vla_horizon_duration_s, self.control_hz)

    @property
    def vla_checkpoint(self) -> Path:
        """Return the newest OpenPI checkpoint produced by the configured finetune run."""
        output = Path(self.vla["output"])
        if (output / "params").is_dir():
            return output

        run_dir = output / self.vla["config_name"] / self.vla_finetune["exp_name"]
        checkpoints = [
            path
            for path in run_dir.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / "params").is_dir()
        ] if run_dir.is_dir() else []
        if checkpoints:
            return max(checkpoints, key=lambda path: int(path.name))
        raise FileNotFoundError(
            f"no OpenPI checkpoint found under {run_dir}; run VLA finetuning first"
        )

    @property
    def target_discount(self) -> float:
        return float(self.online_rl["gamma"]) ** self.transition_horizon

    @property
    def action_dim(self) -> int:
        return dataset_action_dim(self.vla["dataset"])


def load_config(path: str | Path | None = None, *, task: str | None = None) -> RLTConfig:
    config_path = Path(
        path or os.environ.get("RLT_CONFIG") or DEFAULT_CONFIG
    ).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping")

    robot_path = Path(
        os.environ.get("RLT_ROBOT_CONFIG") or config_path.with_name("robot.yaml")
    ).expanduser().resolve()
    if robot_path.is_file():
        with robot_path.open(encoding="utf-8") as stream:
            robot_data = yaml.safe_load(stream) or {}
    else:
        robot_data = {}
    runtime = {**dict(data.get("runtime", {})), **dict(robot_data.get("runtime", {}))}
    hardware = dict(robot_data.get("hardware", {}))
    control_hz = float(runtime.get("control_hz", 10))
    chunk_duration_s = float(runtime.get("rl_chunk_duration_s", 0.2))
    vla_horizon_duration_s = float(runtime.get("vla_horizon_duration_s", 1.0))
    stride = int(runtime.get("transition_stride_steps", 2))
    if control_hz <= 0 or chunk_duration_s <= 0 or vla_horizon_duration_s <= 0:
        raise ValueError("control_hz and chunk durations must be positive")
    if stride < 1:
        raise ValueError("transition_stride_steps must be positive")

    selected_task = str(task or data.get("task", "d1"))
    task_values = data.get("tasks", {})
    if not isinstance(task_values, dict):
        raise TypeError("tasks must be a YAML mapping")
    default_mlp = dict(task_values.get("default", {}))
    selected_mlp = {**default_mlp, **dict(task_values.get(selected_task, {}))}
    mlp = MLPConfig(
        actor_hidden_dim=int(selected_mlp.get("actor_hidden_dim", 256)),
        actor_num_layers=int(selected_mlp.get("actor_num_layers", 2)),
        critic_hidden_dim=int(selected_mlp.get("critic_hidden_dim", 256)),
        critic_num_layers=int(selected_mlp.get("critic_num_layers", 2)),
    )
    if min(
        mlp.actor_hidden_dim,
        mlp.actor_num_layers,
        mlp.critic_hidden_dim,
        mlp.critic_num_layers,
    ) < 1:
        raise ValueError("MLP dimensions and layer counts must be positive")

    vla = dict(data.get("vla", {}))
    vla_finetune = dict(vla.get("finetune", {}))
    rl_token = dict(data.get("rl_token", {}))
    online_rl = dict(data.get("online_rl", {}))
    if int(data.get("log_interval", 100)) < 1:
        raise ValueError("log_interval must be positive")
    if int(online_rl.get("warmup_transitions", 256)) < 1:
        raise ValueError("warmup_transitions must be positive")
    return RLTConfig(
        path=config_path,
        robot_path=robot_path,
        task=selected_task,
        wandb_enabled=bool(data.get("wandb_enabled", True)),
        project_name=str(data.get("project_name", "openpi")),
        log_interval=int(data.get("log_interval", 100)),
        control_hz=control_hz,
        chunk_duration_s=chunk_duration_s,
        vla_horizon_duration_s=vla_horizon_duration_s,
        transition_stride_steps=stride,
        vla=vla,
        vla_finetune=vla_finetune,
        rl_token=rl_token,
        online_rl=online_rl,
        runtime=runtime,
        hardware=hardware,
        mlp=mlp,
    )
