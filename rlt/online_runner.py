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

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from rlt.actions import (
    absolute_to_normalized,
    normalized_to_absolute,
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
                    self.replay, self.batch_size
                )
                if updates:
                    self.trained = True
                if updates and self.wandb_run is not None:
                    actor_losses = [item["actor_loss"] for item in updates if "actor_loss" in item]
                    self.wandb_run.log(
                        {
                            "critic_step": self.learner.critic_steps,
                            "train/critic_loss": sum(item["critic_loss"] for item in updates)
                            / len(updates),
                            "train/q1_mean": sum(item["q1_mean"] for item in updates)
                            / len(updates),
                            "train/q2_mean": sum(item["q2_mean"] for item in updates)
                            / len(updates),
                            "train/q_min_mean": sum(item["q_min_mean"] for item in updates)
                            / len(updates),
                            "train/target_q_mean": sum(
                                item["target_q_mean"] for item in updates
                            )
                            / len(updates),
                            "train/td_error_abs": sum(
                                item["td_error_abs"] for item in updates
                            )
                            / len(updates),
                            "train/actor_step": self.learner.actor_steps,
                            "train/replay_size": len(self.replay),
                            **(
                                {"train/actor_loss": sum(actor_losses) / len(actor_losses)}
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


def _read_episode_label() -> bool:
    while True:
        label = input("Episode ended. Enter s for success or f for failure: ").strip().lower()
        if label in {"s", "success"}:
            return True
        if label in {"f", "failure"}:
            return False
        print("Please enter s or f.")


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


def _confirm_rollout() -> bool:
    while True:
        answer = input("Warmup is complete. Start rollout? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no", "q", "quit", ""}:
            return False
        print("Please enter y or n.")


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
            "action_horizon": settings.action_horizon,
            "batch_size": args.batch_size,
            "warmup_transitions": warmup_transitions,
            "learner_device": str(learner.device),
            "online_rl": settings.online_rl,
        },
    )
    wandb_run.define_metric("train/*", step_metric="critic_step")
    wandb_run.define_metric("episode/*", step_metric="episode")
    return wandb_run


def _learner_dimensions(
    path: Path, z_dim: int, chunk_len: int
) -> tuple[int, int]:
    """Read dimensions from a learner checkpoint, including old checkpoints."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    saved = data.get("metadata", {})
    saved = saved if isinstance(saved, dict) else {}
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


def run_episode(args, phase, hardware, feature, learner, model_lock, q01, q99):
    """Collect one episode through the robot adapter's small hardware protocol."""
    warmup = phase == "warmup" or getattr(args, "auto_warmup", False)
    critical = not warmup and (phase == "critical" or (
        phase == "eval-rlt" and args.eval_task == "critical"
    ))
    human = False
    aborted = False
    ticks = []
    previous = None
    actor_chunk = None
    actor_offset = 0
    started = time.monotonic()
    timeout = (
        args.critical_timeout_s
        if phase == "critical"
        or (phase.startswith("eval") and args.eval_task == "critical")
        else args.full_timeout_s
    )
    period = 1.0 / args.control_hz
    print(
        f"\n{phase}: Enter=end episode, c=enter critical, space=human, "
        f"s/f=label after ending, q=abort "
        f"(timeout {timeout}s)"
    )
    with Keys() as keys:
        while time.monotonic() - started < timeout:
            tick_start = time.monotonic()
            key = keys.read()
            if key in {"q", "\x03"}:
                aborted = True
                break
            if key in {"\r", "\n"}:
                if ticks:
                    ticks[-1]["done"] = True
                    break
                continue
            if key == "c" and not warmup:
                critical = True
                actor_chunk = None
            if key == " ":
                human = not human
                if human:
                    hardware.rebase_human(
                        previous if previous is not None else hardware.read()[1]
                    )
                actor_chunk = None

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
                    : args.chunk_len
                ]
                absolute_reference = np.asarray(features["absolute"], np.float32)
            except Exception:
                logging.exception("VLA/feature failure: command paused")
                time.sleep(max(0, period - (time.monotonic() - tick_start)))
                continue

            source = (
                "human"
                if human
                else (
                    "actor"
                    if critical and phase not in {"warmup", "eval-base"}
                    else "vla"
                )
            )
            try:
                if source == "human":
                    absolute = hardware.human_target()
                elif source == "actor":
                    if actor_chunk is None or actor_offset >= args.chunk_len:
                        with model_lock, torch.inference_mode():
                            zt = torch.from_numpy(z[None]).to(learner.device)
                            pt = torch.from_numpy(proprio[None]).to(learner.device)
                            rt = torch.from_numpy(reference_chunk[None]).to(
                                learner.device
                            )
                            actor_chunk, _ = learner.actor.sample(
                                zt, pt, rt, args.sigma
                            )
                            actor_chunk = actor_chunk[0].cpu().numpy()
                            actor_offset = 0
                    absolute = normalized_to_absolute(actor_chunk, state, q01, q99)[
                        actor_offset
                    ]
                    actor_offset += 1
                else:
                    absolute = absolute_reference[0]
                command = safe_command(absolute, state, args.max_arm_delta_deg)
            except Exception:
                logging.exception(
                    "actor/human command failed; falling back to VLA reference"
                )
                source, command = (
                    "vla",
                    safe_command(absolute_reference[0], state, args.max_arm_delta_deg),
                )

            hardware.send(command)
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
        ticks[-1]["reward"] = float(success)
        return ticks, success, time.monotonic() - started, aborted
    return ticks, False, time.monotonic() - started, aborted


def run(args, hardware_cls) -> None:
    """Run online RL using a robot-specific context-manager adapter."""
    settings = load_config(args.config)
    for name, value in settings.hardware.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, value)
    args.phase = args.phase or "auto"
    args.host = args.host or settings.runtime["feature_host"]
    args.port = args.port or int(settings.runtime["feature_port"])
    args.learner_device = args.learner_device or settings.runtime["learner_device"]
    args.output = Path(args.output or settings.online_rl["output"])
    rollout_output = args.output / "rollout"
    checkpoint = rollout_output / "learner_latest.pt"
    resume_checkpoint = args.resume or args.actor_checkpoint
    if resume_checkpoint is None:
        for candidate in (checkpoint, args.output / "learner.pt"):
            if candidate.exists():
                resume_checkpoint = candidate
                break
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint}")
    args.batch_size = int(args.batch_size or settings.online_rl["batch_size"])
    warmup_transitions = int(settings.online_rl["warmup_transitions"])
    args.max_arm_delta_deg = float(
        args.max_arm_delta_deg or settings.runtime["max_arm_delta_deg"]
    )
    args.control_hz = settings.control_hz
    args.chunk_len = settings.action_horizon
    args.prompt = settings.vla["prompt"]
    args.sigma = float(settings.online_rl["sigma"])
    args.critical_timeout_s = float(settings.runtime["critical_timeout_s"])
    args.full_timeout_s = float(settings.runtime["full_timeout_s"])
    if args.episodes is not None and args.episodes < 1:
        raise ValueError("--episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    feature = WebsocketClientPolicy(args.host, args.port)
    metadata = feature.get_server_metadata()
    q01, q99 = (
        np.asarray(metadata["action_q01"], np.float32),
        np.asarray(metadata["action_q99"], np.float32),
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
    proprio_dim = getattr(hardware_cls, "proprio_dim", None)
    if resume_checkpoint is not None:
        action_dim, proprio_dim = _learner_dimensions(
            resume_checkpoint, z_dim, settings.action_horizon
        )
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
        settings.action_horizon,
        action_dim,
        settings.mlp.actor_hidden_dim,
        settings.mlp.actor_num_layers,
    )
    critic = Critic(
        z_dim,
        proprio_dim,
        settings.action_horizon,
        action_dim,
        settings.mlp.critic_hidden_dim,
        settings.mlp.critic_num_layers,
    )
    replay = Replay(
        args.output / "replay.sqlite",
        capacity=int(settings.online_rl["replay_capacity"]),
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
        "C": settings.action_horizon,
        "z_dim": z_dim,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "transition_horizon": settings.transition_horizon,
        "target_discount": settings.target_discount,
    }
    lock = threading.Lock()
    outcomes = []
    wall_start = time.monotonic()
    episode_count = args.episodes
    if episode_count is None and args.phase.startswith("eval"):
        episode_count = int(settings.online_rl["evaluation_episodes"])
    wandb_run = None
    try:
        rollout_confirmed = args.phase not in {"auto", "warmup"}
        if rollout_confirmed and not args.phase.startswith("eval") and settings.wandb_enabled:
            wandb_run = _init_wandb(settings, args, learner, warmup_transitions)
        with (
            hardware_cls(args) as hardware,
            TrainerThread(
                learner,
                replay,
                args.batch_size,
                checkpoint,
                lock,
                learner_metadata,
                settings.online_rl["checkpoint_interval"],
                wandb_run,
                rollout_output,
            ) as trainer,
        ):
            episode = 0
            first_episode = True
            while episode_count is None or episode < episode_count:
                phase = args.phase
                if (
                    phase == "auto"
                    and not rollout_confirmed
                    and len(replay) >= warmup_transitions
                ):
                    if not _confirm_rollout():
                        return
                    rollout_confirmed = True
                    if settings.wandb_enabled:
                        wandb_run = _init_wandb(
                            settings, args, learner, warmup_transitions
                        )
                        trainer.set_wandb_run(wandb_run)
                args.auto_warmup = phase == "auto" and len(replay) < warmup_transitions
                _prepare_episode(hardware, phase, episode + 1, first_episode)
                first_episode = False
                ticks, success, elapsed, aborted = run_episode(
                    args, phase, hardware, feature, learner, lock, q01, q99
                )
                if aborted:
                    episode += 1
                    continue
                rows = build_transitions(
                    ticks,
                    horizon=settings.transition_horizon,
                    stride=settings.transition_stride_steps,
                )
                if not phase.startswith("eval"):
                    for row in rows:
                        replay.add(row)
                    if rollout_confirmed:
                        trainer.notify(len(rows))
                    else:
                        print(
                            f"Warmup transitions: {len(replay)}/{warmup_transitions}"
                        )
                record = {
                    "phase": phase,
                    "task": args.eval_task if phase.startswith("eval") else phase,
                    "success": success,
                    "seconds": elapsed,
                    "transitions": len(rows),
                    "robot_hours": (time.monotonic() - wall_start) / 3600,
                }
                outcomes.append(record)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "episode": episode + 1,
                            "episode/success": int(success),
                            "episode/success_rate": sum(
                                item["success"] for item in outcomes
                            )
                            / len(outcomes),
                            "episode/seconds": elapsed,
                            "episode/transitions": len(rows),
                            "episode/replay_size": len(replay),
                        }
                    )
                with (args.output / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                episode += 1
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        with contextlib.suppress(Exception):
            feature._ws.close()
        replay.close()
    successes = sum(item["success"] for item in outcomes)
    minutes = sum(item["seconds"] for item in outcomes) / 60
    print(
        json.dumps(
            {
                "episodes": len(outcomes),
                "successes": successes,
                "success_rate": successes / max(1, len(outcomes)),
                "successes_per_10min": successes / max(minutes, 1e-9) * 10,
            },
            indent=2,
        )
    )
