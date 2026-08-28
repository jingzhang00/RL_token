"""Robot-independent online actor-critic loop."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
import queue
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import torch
from tqdm import tqdm

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from rlt.actions import (
    ARM_HIGH,
    ARM_LOW,
    absolute_to_normalized,
    safe_command,
)
from rlt.models import Actor, Critic
from rlt.rl_core import Learner, Replay, build_transitions, make_proprio
from rlt.settings import load_config


class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self) -> str | None:
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class TrainerThread:
    def __init__(
        self, learner, replay, batch_size, checkpoint, lock, metadata, save_interval,
        wandb_run=None, snapshot_dir=None,
    ):
        self.learner, self.replay, self.batch_size, self.checkpoint, self.lock = (
            learner,
            replay,
            batch_size,
            checkpoint,
            lock,
        )
        self.metadata, self.save_interval = metadata, int(save_interval)
        self.wandb_run = wandb_run
        self.snapshot_dir = Path(snapshot_dir or checkpoint.parent)
        self.trained = False
        self.queue, self.stop = queue.Queue(), threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def notify(self, count):
        for _ in range(count):
            self.queue.put(1)

    def set_wandb_run(self, wandb_run):
        self.wandb_run = wandb_run

    def _save_checkpoint(self, snapshot=False):
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.learner.save(self.checkpoint, self.metadata)
        if snapshot:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            self.learner.save(
                self.snapshot_dir / f"learner_step_{self.learner.critic_steps}.pt",
                self.metadata,
            )

    def run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self.lock:
                updates = self.learner.update_for_new_transition(
                    self.replay, self.batch_size, online=True
                )
                if updates:
                    self.trained = True
                if updates and self.wandb_run is not None:
                    actor_losses = [item["actor_loss"] for item in updates if "actor_loss" in item]
                    self.wandb_run.log(
                        {
                            "critic_step": self.learner.critic_steps,
                            "online/critic_step": self.learner.online_critic_steps,
                            "online/critic_loss": sum(item["critic_loss"] for item in updates)
                            / len(updates),
                            "online/q1_mean": sum(item["q1_mean"] for item in updates)
                            / len(updates),
                            "online/q2_mean": sum(item["q2_mean"] for item in updates)
                            / len(updates),
                            "online/q_min_mean": sum(item["q_min_mean"] for item in updates)
                            / len(updates),
                            "online/target_q_mean": sum(
                                item["target_q_mean"] for item in updates
                            )
                            / len(updates),
                            "online/td_error_abs": sum(
                                item["td_error_abs"] for item in updates
                            )
                            / len(updates),
                            "online/actor_step": self.learner.online_actor_steps,
                            "online/replay_size": len(self.replay),
                            **(
                                {"online/actor_loss": sum(actor_losses) / len(actor_losses)}
                                if actor_losses
                                else {}
                            ),
                        }
                    )
                if updates and self.learner.critic_steps % self.save_interval == 0:
                    self._save_checkpoint(snapshot=True)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        if self.trained:
            self._save_checkpoint()


def _read_episode_label() -> bool | None:
    while True:
        label = input(
            "Episode ended. Enter s for success, f for failure, or q to abort: "
        ).strip().lower()
        if label in {"s", "success"}:
            return True
        if label in {"f", "failure"}:
            return False
        if label in {"q", "quit", "abort"}:
            return None
        print("Please enter s, f, or q.")


def _confirm_offline_warmup() -> bool:
    while True:
        answer = input(
            "Warmup collection is complete. Start offline Actor/Critic warmup? [y/n]: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no", "q", "quit"}:
            return False
        print("Please enter y or n.")


def _confirm_online_rollout() -> bool:
    while True:
        answer = input(
            "Ready for online rollout. Start? [y/n]: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no", "q", "quit"}:
            return False
        print("Please enter y or n.")


def _prepare_episode(hardware, phase: str, episode: int, first: bool) -> None:
    move_to_initial_pose = getattr(hardware, "move_to_initial_pose", None)
    if move_to_initial_pose is None:
        input(f"Reset robot/task for {phase} episode {episode}; press Enter")
        return

    if first:
        input("Calibration: press Enter to run homing, then restore the initial state")
    else:
        input("Episode recorded. Reset robot/task, then press Enter to run homing and restore the initial state")
    move_to_initial_pose()
    start_camera_thread = getattr(hardware, "start_camera_thread", None)
    if start_camera_thread is not None:
        start_camera_thread()
    if first:
        input(f"Initial state reached. Reset robot/task for {phase} episode {episode}; press Enter")
    else:
        input("Initial state reached. Press Enter to start the next episode")


def _init_wandb(settings, args, learner, warmup_transitions):
    import wandb

    wandb_run = wandb.init(
        project=settings.project_name,
        name=settings.online_rl.get("exp_name", f"online_rl_{settings.task}"),
        config={
            "config_path": str(settings.path),
            "task": settings.task,
            "phase": args.phase,
            "control_hz": settings.control_hz,
            "vla_horizon": settings.action_horizon,
            "chunk_length": settings.chunk_length,
            "batch_size": args.batch_size,
            "warmup_transitions": warmup_transitions,
            "offline_warmup_updates": int(settings.online_rl["offline_warmup_updates"]),
            "learner_device": str(learner.device),
            "online_rl": settings.online_rl,
        },
    )
    wandb_run.define_metric("bootstrap/*", step_metric="bootstrap/critic_step")
    wandb_run.define_metric("online/*", step_metric="online/critic_step")
    wandb_run.define_metric("episode/success_rate", step_metric="episode")
    return wandb_run


def _learner_dimensions(
    path: Path, z_dim: int, chunk_len: int, vla_horizon: int
) -> tuple[int, int]:
    """Read dimensions from a learner checkpoint, including old checkpoints."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    saved = data.get("metadata", {})
    saved = saved if isinstance(saved, dict) else {}
    for key, expected in (("C", chunk_len), ("H", vla_horizon)):
        if key in saved and int(saved[key]) != expected:
            raise ValueError(
                f"checkpoint {path} has {key}={saved[key]}, expected {key}={expected}"
            )
    action_dim = saved.get("action_dim")
    proprio_dim = saved.get("proprio_dim")
    if action_dim is None or proprio_dim is None:
        actor = data.get("actor", {})
        weights = [
            value
            for key, value in actor.items()
            if key.endswith("weight") and getattr(value, "ndim", 0) == 2
        ]
        if len(weights) < 2:
            raise ValueError(f"cannot infer actor dimensions from {path}")
        output_dim = int(weights[-1].shape[0])
        if action_dim is None:
            if output_dim % chunk_len:
                raise ValueError(f"checkpoint action shape is incompatible with C={chunk_len}")
            action_dim = output_dim // chunk_len
        if proprio_dim is None:
            proprio_dim = int(weights[0].shape[1]) - z_dim - output_dim
    action_dim, proprio_dim = int(action_dim), int(proprio_dim)
    if action_dim < 1 or proprio_dim < 1:
        raise ValueError(
            f"invalid checkpoint dimensions: action_dim={action_dim}, "
            f"proprio_dim={proprio_dim}"
        )
    return action_dim, proprio_dim


def _offline_warmup(
    learner,
    replay,
    updates: int,
    batch_size: int,
    wandb_run=None,
    log_interval: int = 100,
) -> None:
    if len(replay) < batch_size:
        raise ValueError(
            f"warmup replay has {len(replay)} transitions, need {batch_size}"
        )
    for index in tqdm(
        range(1, updates + 1),
        desc="Offline Actor/Critic warmup",
        unit="update",
    ):
        metrics = learner.update(replay.sample(batch_size))
        if wandb_run is not None and (
            index == 1 or index == updates or index % log_interval == 0
        ):
            payload = {
                "bootstrap/critic_step": index,
                "bootstrap/critic_loss": metrics["critic_loss"],
                "bootstrap/q1_mean": metrics["q1_mean"],
                "bootstrap/q2_mean": metrics["q2_mean"],
                "bootstrap/q_min_mean": metrics["q_min_mean"],
                "bootstrap/target_q_mean": metrics["target_q_mean"],
                "bootstrap/td_error_abs": metrics["td_error_abs"],
            }
            if "actor_loss" in metrics:
                payload["bootstrap/actor_step"] = learner.actor_steps
                payload["bootstrap/actor_loss"] = metrics["actor_loss"]
            wandb_run.log(payload)


def run_episode(args, phase, hardware, feature, learner, model_lock, q01, q99):
    """Collect one episode through the robot adapter's small hardware protocol."""
    warmup = phase == "warmup"
    actor_phase = phase in {"critical-only", "full-task", "eval-rlt"}
    critical = False
    human = phase == "critical-only"
    aborted = False
    ticks = []
    previous = None
    vla_chunk = None
    vla_absolute_chunk = None
    vla_offset = args.chunk_len
    actor_chunk = None
    actor_absolute_chunk = None
    actor_offset = args.chunk_len
    started = time.monotonic()
    timeout = (
        args.critical_timeout_s
        if phase == "critical-only"
        or (phase.startswith("eval") and args.eval_task == "critical-only")
        else args.full_timeout_s
    )
    deadline = None if phase == "critical-only" else started + timeout
    period = 1.0 / args.control_hz
    if phase == "critical-only":
        hardware.rebase_human(hardware.read()[1])
    if warmup:
        controls = "Enter=end episode, s/f=label after ending, q=abort"
    elif phase == "critical-only":
        controls = (
            "Enter=end episode, c=RL critical phase, h=human, "
            "s/f=label after ending, q=abort"
        )
    else:
        controls = (
            "Enter=end episode, c=critical phase, h=human, "
            "s/f=label after ending, q=abort"
        )
    print(f"\n{phase}: {controls} (timeout {timeout}s)")
    with Keys() as keys:
        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            tick_start = time.monotonic()
            key = keys.read()
            if key in {"q", "\x03"}:
                aborted = True
                break
            if key in {"\r", "\n"}:
                if ticks:
                    ticks[-1]["done"] = True
                    break
                if phase == "critical-only":
                    break
                continue

            if key == "c" and not warmup:
                if not (phase == "critical-only" and critical):
                    critical = not critical
                    human = False
                    if phase == "critical-only":
                        deadline = time.monotonic() + timeout if critical else None
                    actor_chunk = None
                    actor_absolute_chunk = None
                    actor_offset = args.chunk_len
                    vla_chunk = None
                    vla_absolute_chunk = None
                    vla_offset = args.chunk_len

            if key == "h" and not warmup:
                if human:
                    if phase == "critical-only" and critical:
                        human = False
                    elif phase == "critical-only":
                        critical = True
                        deadline = time.monotonic() + timeout
                    elif phase != "critical-only":
                        human = False
                        critical = False
                else:
                    human = True
                    if phase in {"critical-only", "full-task"} and not critical:
                        critical = True
                        if phase == "critical-only":
                            deadline = time.monotonic() + timeout
                    hardware.rebase_human(
                        previous if previous is not None else hardware.read()[1]
                    )
                actor_chunk = None
                actor_absolute_chunk = None
                actor_offset = args.chunk_len
                vla_chunk = None
                vla_absolute_chunk = None
                vla_offset = args.chunk_len

            if phase == "critical-only" and not critical:
                image, state = hardware.read()
                previous = state.copy()
                try:
                    command = safe_command(
                        hardware.human_target(),
                        state,
                        args.max_arm_delta_deg,
                        args.arm_low,
                        args.arm_high,
                    )
                    hardware.send(command)
                except Exception:
                    logging.exception("human command failed; command paused")
                time.sleep(max(0, period - (time.monotonic() - tick_start)))
                continue

            image, state = hardware.read()
            proprio = make_proprio(state, previous, dt=period)
            previous = state.copy()
            try:
                features = feature.infer(
                    {
                        "observation/image": image,
                        "observation/state": state,
                        "prompt": args.prompt,
                    }
                )
                z = np.asarray(features["z"], np.float32)
                reference_chunk = np.asarray(features["reference"], np.float32)[
                    : args.vla_horizon
                ]
                absolute_reference = np.asarray(features["absolute"], np.float32)[
                    : args.vla_horizon
                ]
                if reference_chunk.shape[0] < args.vla_horizon:
                    raise ValueError(
                        f"VLA reference has {reference_chunk.shape[0]} steps, "
                        f"need H={args.vla_horizon}"
                    )
                if absolute_reference.shape[0] < args.vla_horizon:
                    raise ValueError(
                        f"VLA absolute actions have {absolute_reference.shape[0]} steps, "
                        f"need H={args.vla_horizon}"
                    )
            except Exception:
                logging.exception("VLA/feature failure: command paused")
                time.sleep(max(0, period - (time.monotonic() - tick_start)))
                continue

            source = "human" if human else "actor" if critical and actor_phase else "vla"
            if (
                source == "vla"
                and (vla_chunk is None or vla_offset >= args.chunk_len)
            ):
                vla_chunk = reference_chunk.copy()
                vla_absolute_chunk = absolute_reference.copy()
                vla_offset = 0
            try:
                if source == "human":
                    absolute = hardware.human_target()
                elif source == "actor":
                    if actor_chunk is None or actor_offset >= args.chunk_len:
                        with model_lock, torch.inference_mode():
                            zt = torch.from_numpy(z[None]).to(learner.device)
                            pt = torch.from_numpy(proprio[None]).to(learner.device)
                            rt = torch.from_numpy(reference_chunk[: args.chunk_len][None]).to(
                                learner.device
                            )
                            sampled_actor_chunk, _ = learner.actor.sample(
                                zt, pt, rt, args.sigma
                            )
                            sampled_actor_chunk = sampled_actor_chunk[0].cpu().numpy()
                            sampled_actor_absolute_chunk = np.asarray(
                                feature.infer(
                                    {
                                        "convert_actions": sampled_actor_chunk,
                                        "normalized_state": features["normalized_state"],
                                    }
                                )["absolute"],
                                dtype=np.float32,
                            )
                            if sampled_actor_absolute_chunk.shape != sampled_actor_chunk.shape:
                                raise ValueError(
                                    "OpenPI actor conversion returned an incompatible action shape"
                                )
                            actor_chunk = sampled_actor_chunk
                            actor_absolute_chunk = sampled_actor_absolute_chunk
                            actor_offset = 0
                    absolute = actor_absolute_chunk[actor_offset]
                else:
                    absolute = vla_absolute_chunk[vla_offset]
                command = safe_command(
                    absolute,
                    state,
                    args.max_arm_delta_deg,
                    args.arm_low,
                    args.arm_high,
                )
                hardware.send(command)
                if source == "actor":
                    actor_offset += 1
                elif source == "vla":
                    vla_offset += 1
            except Exception:
                logging.exception("%s command failed; command paused", source)
                time.sleep(max(0, period - (time.monotonic() - tick_start)))
                continue
            executed_normalized = absolute_to_normalized(
                command[None], state, q01, q99
            )[0]
            ticks.append(
                {
                    "z": z,
                    "reference": reference_chunk[0],
                    "reference_chunk": reference_chunk,
                    "action": executed_normalized,
                    "proprio": proprio,
                    "source": source,
                    "critical": critical,
                    "done": False,
                    "reward": 0.0,
                }
            )
            time.sleep(max(0, period - (time.monotonic() - tick_start)))

    stop_camera_thread = getattr(hardware, "stop_camera_thread", None)
    if stop_camera_thread is not None:
        stop_camera_thread()
    if ticks and not aborted:
        ticks[-1]["done"] = True
        success = _read_episode_label()
        if success is None:
            return ticks, False, time.monotonic() - started, True
        ticks[-1]["reward"] = float(success)
        return ticks, success, time.monotonic() - started, aborted
    return ticks, False, time.monotonic() - started, aborted


def run(args, hardware_cls) -> None:
    """Run online RL using a robot-specific context-manager adapter."""
    settings = load_config(args.config)
    train_phases = {"critical-only", "full-task"}
    eval_phases = {"eval-base", "eval-rlt"}
    if args.phase not in train_phases | eval_phases:
        raise ValueError(
            "--phase is required and must be one of critical-only, full-task, "
            "eval-base or eval-rlt"
        )
    arm_joint_names = (
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
    )
    configured_limits = settings.runtime.get("arm_joint_limits_deg")
    if configured_limits is None:
        args.arm_low, args.arm_high = ARM_LOW, ARM_HIGH
    else:
        try:
            limits_deg = np.asarray(
                [configured_limits[name] for name in arm_joint_names], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "runtime.arm_joint_limits_deg must define joint_1 through joint_6"
            ) from exc
        if (
            limits_deg.shape != (6, 2)
            or not np.isfinite(limits_deg).all()
            or np.any(limits_deg[:, 0] > limits_deg[:, 1])
        ):
            raise ValueError(
                "runtime.arm_joint_limits_deg must contain six finite [low, high] pairs"
            )
        args.arm_low, args.arm_high = np.deg2rad(limits_deg.T)
    for name, value in settings.hardware.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, value)
    args.host = args.host or settings.runtime["feature_host"]
    args.port = args.port or int(settings.runtime["feature_port"])
    args.learner_device = args.learner_device or settings.runtime["learner_device"]
    args.output = Path(args.output or settings.online_rl["output"])
    rollout_output = args.output / "rollout"
    checkpoint = rollout_output / "learner_latest.pt"
    resume_checkpoint = (
        args.resume
        or args.actor_checkpoint
        or settings.online_rl.get("resume")
    )
    if resume_checkpoint is None:
        resume_checkpoint = checkpoint if checkpoint.exists() else None
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint}")
    if args.phase == "eval-rlt" and resume_checkpoint is None:
        raise FileNotFoundError(
            "eval-rlt requires checkpoints/online/rollout/learner_latest.pt or --resume"
        )
    args.batch_size = int(args.batch_size or settings.online_rl["batch_size"])
    warmup_transitions = int(settings.online_rl["warmup_transitions"])
    offline_warmup_updates = int(settings.online_rl["offline_warmup_updates"])
    args.max_arm_delta_deg = float(
        args.max_arm_delta_deg or settings.runtime["max_arm_delta_deg"]
    )
    args.control_hz = settings.control_hz
    args.vla_horizon = settings.action_horizon
    args.chunk_len = settings.chunk_length
    args.prompt = settings.vla["prompt"]
    args.sigma = float(settings.online_rl["sigma"])
    args.critical_timeout_s = float(settings.runtime["critical_timeout_s"])
    args.full_timeout_s = float(settings.runtime["full_timeout_s"])
    if args.episodes is not None and args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if warmup_transitions < args.batch_size:
        raise ValueError(
            "online_rl.warmup_transitions must be at least online_rl.batch_size"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    feature = WebsocketClientPolicy(args.host, args.port)
    metadata = feature.get_server_metadata()
    if args.phase != "eval-base" and not metadata.get("supports_action_conversion", False):
        raise RuntimeError(
            "feature server does not support actor action conversion; "
            "restart it from the current rlt.serve_features"
        )
    q01, q99 = (
        np.asarray(metadata["action_q01"], np.float32),
        np.asarray(metadata["action_q99"], np.float32),
    )
    if int(metadata.get("action_horizon", args.vla_horizon)) != args.vla_horizon:
        raise ValueError(
            f"feature server H={metadata.get('action_horizon')} does not match "
            f"configured H={args.vla_horizon}"
        )
    learner_device = (
        "cuda"
        if args.learner_device == "auto" and torch.cuda.is_available()
        else args.learner_device
    )
    if learner_device == "auto":
        learner_device = "cpu"
    if learner_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--learner-device=cuda requested but torch.cuda.is_available() is false"
        )
    logging.info("Online actor/critic device: %s", learner_device)
    z_dim = int(metadata.get("z_dim", 2048))
    action_dim = int(metadata.get("action_dim", len(q01)))
    hardware_proprio_dim = getattr(hardware_cls, "proprio_dim", None)
    proprio_dim = hardware_proprio_dim
    if resume_checkpoint is not None:
        checkpoint_action_dim, checkpoint_proprio_dim = _learner_dimensions(
            resume_checkpoint, z_dim, args.chunk_len, args.vla_horizon
        )
        if (
            hardware_proprio_dim is not None
            and checkpoint_proprio_dim != hardware_proprio_dim
        ):
            raise ValueError(
                f"checkpoint proprio_dim={checkpoint_proprio_dim} does not match "
                f"hardware proprio_dim={hardware_proprio_dim}"
            )
        action_dim, proprio_dim = checkpoint_action_dim, checkpoint_proprio_dim
    if action_dim != len(q01):
        raise ValueError(
            f"learner action_dim={action_dim} does not match VLA action_dim={len(q01)}"
        )
    if proprio_dim is None:
        raise ValueError(
            "no learner checkpoint proprio_dim; hardware adapter must define proprio_dim"
        )
    actor = Actor(
        z_dim,
        proprio_dim,
        args.chunk_len,
        action_dim,
        settings.mlp.actor_hidden_dim,
        settings.mlp.actor_num_layers,
    )
    critic = Critic(
        z_dim,
        proprio_dim,
        args.chunk_len,
        action_dim,
        settings.mlp.critic_hidden_dim,
        settings.mlp.critic_num_layers,
    )
    learner = Learner(
        actor,
        critic,
        gamma=float(settings.online_rl["gamma"]),
        tau=float(settings.online_rl["tau"]),
        horizon=settings.transition_horizon,
        sigma=args.sigma,
        beta=float(settings.online_rl["beta"]),
        reference_dropout_probability=float(
            settings.online_rl["reference_dropout"]
        ),
        actor_lr=float(settings.online_rl["actor_lr"]),
        critic_lr=float(settings.online_rl["critic_lr"]),
        actor_update_period=int(settings.online_rl["actor_update_period"]),
        critic_updates_per_transition=int(
            settings.online_rl["critic_updates_per_transition"]
        ),
        device=learner_device,
    )
    if resume_checkpoint is not None:
        learner.load(resume_checkpoint)
    learner_metadata = {
        "config": str(settings.path),
        "task": settings.task,
        "control_hz": settings.control_hz,
        "H": args.vla_horizon,
        "C": args.chunk_len,
        "z_dim": z_dim,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "transition_horizon": settings.transition_horizon,
        "target_discount": settings.target_discount,
        "bootstrap_complete": True,
    }
    lock = threading.Lock()
    outcomes = []
    wall_start = time.monotonic()
    episode_count = args.episodes
    if episode_count is None and args.phase in eval_phases:
        episode_count = int(settings.online_rl["evaluation_episodes"])
    wandb_run = None
    replay = None
    warmup_replay = None
    bootstrap = args.phase in train_phases and resume_checkpoint is None
    replay_metadata = {
        "H": args.vla_horizon,
        "C": args.chunk_len,
        "stride": settings.transition_stride_steps,
        "kind": "online",
        "bootstrap_complete": True,
    }
    warmup_metadata = {key: value for key, value in replay_metadata.items() if key != "bootstrap_complete"}
    warmup_metadata["kind"] = "warmup"
    try:
        if args.phase in train_phases and settings.wandb_enabled:
            wandb_run = _init_wandb(settings, args, learner, warmup_transitions)
        with (
            hardware_cls(args) as hardware,
        ):
            first_episode = True
            episode_prepared = False
            if bootstrap:
                online_path = args.output / "replay.sqlite"
                if online_path.exists():
                    existing = Replay(
                        online_path,
                        capacity=int(settings.online_rl["replay_capacity"]),
                        metadata=replay_metadata,
                    )
                    if len(existing):
                        existing.close()
                        raise ValueError(
                            f"{online_path} is not empty without a learner checkpoint; "
                            "archive it before starting bootstrap"
                        )
                    existing.close()
                warmup_replay = Replay(
                    args.output / "warmup_replay.sqlite",
                    capacity=int(settings.online_rl["replay_capacity"]),
                    metadata=warmup_metadata,
                )
                warmup_replay.set_metadata({"bootstrap_complete": False})
                warmup_episode = 0
                while len(warmup_replay) < warmup_transitions:
                    warmup_episode += 1
                    _prepare_episode(hardware, "warmup", warmup_episode, first_episode)
                    first_episode = False
                    ticks, success, elapsed, aborted = run_episode(
                        args, "warmup", hardware, feature, learner, lock, q01, q99
                    )
                    if aborted:
                        continue
                    rows = build_transitions(
                        ticks,
                        horizon=settings.transition_horizon,
                        stride=settings.transition_stride_steps,
                    )
                    rows = [row for row in rows if row.source == "vla"]
                    for row in rows:
                        warmup_replay.add(row)
                    record = {
                        "phase": "warmup",
                        "success": success,
                        "seconds": elapsed,
                        "transitions": len(rows),
                        "replay_size": len(warmup_replay),
                        "robot_hours": (time.monotonic() - wall_start) / 3600,
                    }
                    outcomes.append(record)
                    with (args.output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(
                        f"Warmup transitions: {len(warmup_replay)}/{warmup_transitions}"
                    )

                _prepare_episode(hardware, args.phase, 1, first_episode)
                episode_prepared = True
                if not _confirm_offline_warmup():
                    return
                _offline_warmup(
                    learner,
                    warmup_replay,
                    offline_warmup_updates,
                    args.batch_size,
                    wandb_run,
                    settings.log_interval,
                )
                rollout_output.mkdir(parents=True, exist_ok=True)
                learner.save(checkpoint, learner_metadata)
                warmup_replay.set_metadata({"bootstrap_complete": True})
                warmup_replay.close()
                warmup_replay = None

            if args.phase in train_phases:
                if not _confirm_online_rollout():
                    return
                replay = Replay(
                    args.output / "replay.sqlite",
                    capacity=int(settings.online_rl["replay_capacity"]),
                    metadata=replay_metadata,
                )
            if args.phase in train_phases:
                trainer_context = TrainerThread(
                    learner,
                    replay,
                    args.batch_size,
                    checkpoint,
                    lock,
                    learner_metadata,
                    settings.online_rl["checkpoint_interval"],
                    wandb_run,
                    rollout_output,
                )
            else:
                trainer_context = contextlib.nullcontext(None)
            with trainer_context as trainer:
                episode = 0
                while episode_count is None or episode < episode_count:
                    episode += 1
                    if episode_prepared:
                        episode_prepared = False
                    else:
                        _prepare_episode(hardware, args.phase, episode, first_episode)
                    first_episode = False
                    ticks, success, elapsed, aborted = run_episode(
                        args, args.phase, hardware, feature, learner, lock, q01, q99
                    )
                    if aborted:
                        continue
                    if args.phase in train_phases:
                        target_ticks = [tick for tick in ticks if tick.get("critical")]
                        rows = build_transitions(
                            target_ticks,
                            horizon=settings.transition_horizon,
                            stride=settings.transition_stride_steps,
                        )
                        for row in rows:
                            replay.add(row)
                        if trainer is not None:
                            trainer.notify(len(rows))
                    else:
                        rows = []
                    record = {
                        "phase": args.phase,
                        "success": success,
                        "seconds": elapsed,
                        "transitions": len(rows),
                        "replay_size": len(replay) if replay is not None else 0,
                        "robot_hours": (time.monotonic() - wall_start) / 3600,
                    }
                    outcomes.append(record)
                    if wandb_run is not None:
                        payload = {
                            "episode": len(outcomes),
                            "episode/success_rate": sum(
                                item["success"] for item in outcomes
                            )
                            / len(outcomes),
                        }
                        wandb_run.log(payload)
                    with (args.output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        with contextlib.suppress(Exception):
            feature._ws.close()
        if replay is not None:
            replay.close()
        if warmup_replay is not None:
            warmup_replay.close()
    successes = sum(item["success"] for item in outcomes)
    minutes = sum(item["seconds"] for item in outcomes) / 60
    print(
        json.dumps(
            {
                "episodes": len(outcomes),
                "success_rate": successes / max(1, len(outcomes)),
                "successes_per_10min": successes / max(minutes, 1e-9) * 10,
            },
            indent=2,
        )
    )
