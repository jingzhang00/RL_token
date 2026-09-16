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
from types import SimpleNamespace

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
            "\033[1;33m  Result [s]uccess / [f]ailure / [q]uit: \033[0m"
        ).strip().lower()
        if label in {"s", "success"}:
            return True
        if label in {"f", "failure"}:
            return False
        if label in {"q", "quit", "abort"}:
            return None
        print("\033[31m  Invalid choice. Use s, f, or q.\033[0m")


def _confirm_offline_warmup() -> bool:
    while True:
        answer = input(
            "\033[1;33m  Start offline warmup? [y/n]: \033[0m"
        ).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no", "q", "quit"}:
            return False
        print("\033[31m  Invalid choice. Use y or n.\033[0m")


def _confirm_online_rollout() -> bool:
    while True:
        answer = input(
            "\033[1;33m  Start online rollout? [y/n]: \033[0m"
        ).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no", "q", "quit"}:
            return False
        print("\033[31m  Invalid choice. Use y or n.\033[0m")


def _prepare_episode(
    hardware, phase: str, episode: int, first: bool, load_model=None
) -> None:
    if first:
        input("\033[1;33m  Press Enter to begin homing: \033[0m")
    else:
        input("\033[1;33m  Press Enter to begin homing: \033[0m")
    hardware.move_to_initial_pose()
    hardware.start_camera_thread()
    if first and load_model is not None:
        load_model()
    if first:
        input(
            f"\033[1;32m  ✓ Initial pose reached.\033[0m "
            f"Reset robot/task for {phase} episode {episode}, then press Enter: "
        )
    else:
        input("\033[1;32m  ✓ Initial pose reached.\033[0m Press Enter to start: ")


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
    path: Path, chunk_len: int, vla_horizon: int
) -> tuple[int, int]:
    """Read dimensions from a current learner checkpoint."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    saved = data["metadata"]
    for key, expected in (("C", chunk_len), ("H", vla_horizon)):
        if int(saved[key]) != expected:
            raise ValueError(
                f"checkpoint {path} has {key}={saved[key]}, expected {key}={expected}"
            )
    action_dim, proprio_dim = int(saved["action_dim"]), int(saved["proprio_dim"])
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


SOURCE_IDS = {"vla": 0, "actor": 1, "human": 2}


def _create_episode_dataset(args, settings, hardware_cls, action_dim, proprio_dim):
    from robots.collect_lerobot import (
        DEFAULT_VIDEO_CODEC,
        DEFAULT_VIDEO_CRF,
        DEFAULT_VIDEO_GOP,
        DEFAULT_VIDEO_PIX_FMT,
        LocalLeRobotV21Dataset,
    )

    fps = int(args.control_hz)
    if fps != args.control_hz:
        raise ValueError("LeRobot v2.1 episode recording requires an integer control_hz")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (hardware_cls.state_dim,),
            "names": None,
        },
        "observation.proprio": {
            "dtype": "float32",
            "shape": (proprio_dim,),
            "names": None,
        },
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        "action.base_vla_reference": {
            "dtype": "float32",
            "shape": (args.vla_horizon, action_dim),
            "names": None,
        },
        "control.source": {
            "dtype": "int64",
            "shape": (1,),
            "names": list(SOURCE_IDS),
        },
        "control.critical": {"dtype": "int64", "shape": (1,), "names": None},
        hardware_cls.camera_key: {
            "dtype": "video",
            "shape": hardware_cls.image_shape,
            "names": ["height", "width", "channels"],
        },
    }
    writer_args = SimpleNamespace(
        output_root=args.output / "episodes_lerobot",
        repo_id=f"local/{settings.task}_online",
        robot_type=hardware_cls.robot_type,
        fps=fps,
        use_videos=True,
        append=True,
        image_writer_threads=0,
        video_codec=DEFAULT_VIDEO_CODEC,
        video_pix_fmt=DEFAULT_VIDEO_PIX_FMT,
        video_crf=DEFAULT_VIDEO_CRF,
        video_gop=DEFAULT_VIDEO_GOP,
        video_fast_decode=0,
        encoder_queue_maxsize=30,
        encoder_threads=None,
    )
    return LocalLeRobotV21Dataset(args=writer_args, features=features)


def _finish_episode_dataset(dataset, *, aborted, phase, success, seconds):
    if dataset is None:
        return None
    if aborted or not dataset.has_pending_frames():
        dataset.clear_episode_buffer()
        return None
    episode_index = int(dataset.info["total_episodes"])
    dataset.save_episode(
        {"phase": phase, "success": bool(success), "seconds": float(seconds)}
    )
    return episode_index


def run_episode(
    args, phase, hardware, feature, learner, model_lock, q01, q99,
    episode_dataset=None,
):
    """Collect one episode through the robot adapter's small hardware protocol."""
    warmup = phase == "warmup"
    actor_phase = phase in {"critical-only", "full-task", "eval-rlt"}
    critical = False
    human_available = getattr(hardware, "human_available", True)
    human = phase == "critical-only" and human_available
    human_syncing = False
    human_hold_state = None
    human_sync_stable_frames = 0
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
    deadline = (
        None if phase in {"critical-only", "full-task"} else started + timeout
    )
    human_timeout_remaining = None
    period = 1.0 / args.control_hz

    def start_human_sync_view():
        start = getattr(hardware, "start_human_sync_view", None)
        if start is not None:
            start()

    def update_human_sync_view(state, target, arm_error_deg, hand_error,
                               arm_threshold_deg, hand_threshold):
        update = getattr(hardware, "update_human_sync_view", None)
        if update is not None:
            update(
                state,
                target,
                arm_error_deg,
                hand_error,
                arm_threshold_deg,
                hand_threshold,
                human_sync_stable_frames,
                args.human_sync_frames,
            )

    def stop_human_sync_view():
        stop = getattr(hardware, "stop_human_sync_view", None)
        if stop is not None:
            stop()

    def record_critical_result(keys):
        prompt = (
            "\n\033[1;33m  Critical result "
            "[s]uccess / [f]ailure / [q]uit (no Enter): \033[0m"
        )
        print(prompt, end="", flush=True)
        while True:
            choice = keys.read()
            if choice in {"s", "f", "q", "\x03"}:
                print(choice)
                break
            if choice is not None:
                print("\n\033[31m  Invalid choice. Use s, f, or q.\033[0m")
                print(prompt, end="", flush=True)
            time.sleep(0.01)
        result = True if choice == "s" else False if choice == "f" else None
        if result is not None:
            terminal = next(
                (
                    tick
                    for tick in reversed(ticks)
                    if tick.get("critical") and not tick.get("done")
                ),
                None,
            )
            if terminal is not None:
                terminal["done"] = True
                terminal["reward"] = float(result)
        return result

    hardware.start_human_episode(hardware.read()[1])
    mode = (
        "WARMUP"
        if warmup
        else "HUMAN TELEOP"
        if phase == "critical-only" and human_available
        else "BASE VLA"
    )
    limit = f"LIMIT {timeout:g}s" if warmup else f"CRITICAL LIMIT {timeout:g}s"
    yellow, red, green, magenta = "\033[1;33m", "\033[1;31m", "\033[1;32m", "\033[1;35m"
    def key(label, color):
        text = f"[{label}]"
        return f"{color}{text}\033[0m" + " " * (9 - len(text))

    controls = [
        f"  END     {key('Enter', yellow)}{'finish':<15} "
        f"{key('q', red)}abort",
        f"  RESULT  {key('s', green)}success{'':<8} "
        f"{key('f', red)}failure",
    ]
    if not warmup:
        controls.insert(
            0,
            f"  PHASE   {key('c', yellow)}{'critical / RL':<15} "
            f"{key('h', magenta)}human takeover",
        )
    print(
        f"\n\033[1;36m[{phase.upper()}]\033[0m  {mode}  |  {limit}\n"
        + "\n".join(controls)
    )
    with Keys() as keys:
        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                if (
                    phase in {"critical-only", "full-task"}
                    and critical
                    and ticks
                    and record_critical_result(keys) is None
                ):
                    aborted = True
                break
            tick_start = time.monotonic()
            key = keys.read()
            if key in {"q", "\x03"}:
                aborted = True
                break
            if key in {"\r", "\n"}:
                if ticks:
                    if (
                        phase in {"critical-only", "full-task"}
                        and critical
                        and record_critical_result(keys) is None
                    ):
                        aborted = True
                    break
                if phase == "critical-only":
                    break
                continue

            if phase == "critical-only" and critical and key == "h":
                key = "c"
            if key == "c" and not warmup:
                if (
                    phase in {"critical-only", "full-task"}
                    and critical
                    and record_critical_result(keys) is None
                ):
                    aborted = True
                    break
                critical = not critical
                human = False
                human_syncing = False
                human_hold_state = None
                human_sync_stable_frames = 0
                stop_human_sync_view()
                human_timeout_remaining = None
                deadline = time.monotonic() + timeout if critical else None
                actor_chunk = None
                actor_absolute_chunk = None
                actor_offset = args.chunk_len
                vla_chunk = None
                vla_absolute_chunk = None
                vla_offset = args.chunk_len
                print(
                    "\033[1;32m✅✅✅  RL control activated  ✅✅✅\033[0m\n"
                    if critical
                    else "\033[1;36m🤖🤖🤖  Base VLA control activated  🤖🤖🤖\033[0m\n"
                )

            if key == "h" and not warmup:
                if not human_available:
                    logging.warning("Human takeover unavailable; continuing without it")
                    continue
                if human:
                    if phase == "critical-only":
                        critical = True
                        deadline = time.monotonic() + timeout
                        print(
                            "\033[1;35m✅✅✅  Human control activated  ✅✅✅\033[0m\n"
                        )
                    elif phase != "critical-only":
                        if (
                            phase == "full-task"
                            and critical
                            and record_critical_result(keys) is None
                        ):
                            aborted = True
                            break
                        human = False
                        human_syncing = False
                        human_hold_state = None
                        human_sync_stable_frames = 0
                        stop_human_sync_view()
                        critical = False
                        human_timeout_remaining = None
                        deadline = None
                else:
                    human = True
                    if phase in {"critical-only", "full-task"} and not critical:
                        critical = True
                    human_hold_state = hardware.read()[1].copy()
                    hardware.send(human_hold_state)
                    if deadline is not None:
                        human_timeout_remaining = max(
                            0.0, deadline - time.monotonic()
                        )
                        deadline = None
                    human_syncing = True
                    human_sync_stable_frames = 0
                    hardware.begin_human_sync()
                    start_human_sync_view()
                    print(
                        "\033[1;35m⏳⏳⏳  Human takeover selected; syncing  "
                        "⏳⏳⏳\033[0m\n"
                    )
                actor_chunk = None
                actor_absolute_chunk = None
                actor_offset = args.chunk_len
                vla_chunk = None
                vla_absolute_chunk = None
                vla_offset = args.chunk_len

            if human_syncing:
                image, state = hardware.read()
                previous = state.copy()
                try:
                    sync_ready = hardware.human_sync_ready(state)
                    target = np.asarray(
                        hardware.human_sync_target(), dtype=np.float32
                    )
                    if target.shape != state.shape:
                        raise RuntimeError("invalid human sync target shape")
                    arm_threshold_deg = args.human_sync_arm_threshold_deg
                    hand_threshold = args.human_sync_hand_threshold
                    arm_error_deg = np.rad2deg(target[:6] - state[:6])
                    hand_error = target[6:] - state[6:]
                    in_range = np.all(
                        np.abs(arm_error_deg) <= arm_threshold_deg
                    ) and np.all(np.abs(hand_error) <= hand_threshold)
                    human_sync_stable_frames = (
                        human_sync_stable_frames + 1 if in_range else 0
                    )
                    update_human_sync_view(
                        state,
                        target,
                        arm_error_deg,
                        hand_error,
                        arm_threshold_deg,
                        hand_threshold,
                    )
                    if sync_ready:
                        hardware.rebase_human(state)
                        human_syncing = False
                        human_hold_state = None
                        human_sync_stable_frames = 0
                        stop_human_sync_view()
                        if critical and deadline is None:
                            deadline = time.monotonic() + (
                                human_timeout_remaining
                                if human_timeout_remaining is not None
                                else timeout
                            )
                        human_timeout_remaining = None
                        print("\033[1;32m✅✅✅  Human control activated  ✅✅✅\033[0m\n")
                    else:
                        hardware.send(human_hold_state)
                        time.sleep(max(0, period - (time.monotonic() - tick_start)))
                        continue
                except AttributeError:
                    raise
                except Exception:
                    logging.exception("human sync failed; command paused")
                    if human_hold_state is not None:
                        try:
                            hardware.send(human_hold_state)
                        except Exception:
                            logging.exception("human hold command failed")
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
                z = np.array(features["z"], dtype=np.float32, copy=True)
                reference_chunk = np.array(
                    features["reference"], dtype=np.float32, copy=True
                )[
                    : args.vla_horizon
                ]
                absolute_reference = np.array(
                    features["absolute"], dtype=np.float32, copy=True
                )[
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
            if episode_dataset is not None:
                episode_dataset.add_frame(
                    {
                        "task": args.prompt,
                        "observation.state": state,
                        "observation.proprio": proprio,
                        "action": command,
                        "action.base_vla_reference": reference_chunk,
                        "control.source": np.asarray([SOURCE_IDS[source]], np.int64),
                        "control.critical": np.asarray([critical], np.int64),
                        episode_dataset.camera_keys[0]: image,
                    }
                )
            ticks.append(
                {
                    "z": z,
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

    stop_human_sync_view()
    hardware.stop_camera_thread()
    if ticks and not aborted:
        success = _read_episode_label()
        if success is None:
            return ticks, False, time.monotonic() - started, True
        if not ticks[-1]["done"]:
            ticks[-1]["done"] = True
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
    args.human_sync_arm_threshold_deg = float(
        settings.runtime.get("human_sync_arm_threshold_deg", 8.0)
    )
    args.human_sync_hand_threshold = float(
        settings.runtime.get("human_sync_hand_threshold", 0.1)
    )
    args.human_sync_frames = int(settings.runtime.get("human_sync_frames", 3))
    args.human_deadband_arm_deg = float(
        settings.runtime.get("human_deadband_arm_deg", 1.0)
    )
    args.human_deadband_hand = float(
        settings.runtime.get("human_deadband_hand", 0.02)
    )
    if (
        not np.isfinite(args.human_sync_arm_threshold_deg)
        or args.human_sync_arm_threshold_deg <= 0
        or not np.isfinite(args.human_sync_hand_threshold)
        or args.human_sync_hand_threshold <= 0
        or args.human_sync_frames < 1
        or not np.isfinite(args.human_deadband_arm_deg)
        or args.human_deadband_arm_deg < 0
        or not np.isfinite(args.human_deadband_hand)
        or args.human_deadband_hand < 0
    ):
        raise ValueError(
            "human sync thresholds must be finite and positive, frames must be positive, "
            "and human deadbands must be finite and non-negative"
        )
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
    if args.phase != "eval-base" and not metadata["supports_action_conversion"]:
        raise RuntimeError(
            "feature server does not support actor action conversion; "
            "restart it from the current rlt.serve_features"
        )
    q01, q99 = (
        np.asarray(metadata["action_q01"], np.float32),
        np.asarray(metadata["action_q99"], np.float32),
    )
    if int(metadata["action_horizon"]) != args.vla_horizon:
        raise ValueError(
            f"feature server H={metadata['action_horizon']} does not match "
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
    z_dim = int(metadata["z_dim"])
    action_dim = int(metadata["action_dim"])
    proprio_dim = hardware_cls.proprio_dim
    if resume_checkpoint is not None:
        checkpoint_action_dim, checkpoint_proprio_dim = _learner_dimensions(
            resume_checkpoint, args.chunk_len, args.vla_horizon
        )
        if checkpoint_proprio_dim != proprio_dim:
            raise ValueError(
                f"checkpoint proprio_dim={checkpoint_proprio_dim} does not match "
                f"hardware proprio_dim={proprio_dim}"
            )
        action_dim, proprio_dim = checkpoint_action_dim, checkpoint_proprio_dim
    if action_dim != len(q01):
        raise ValueError(
            f"learner action_dim={action_dim} does not match VLA action_dim={len(q01)}"
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
    episode_dataset = None
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
        if args.phase in train_phases:
            episode_dataset = _create_episode_dataset(
                args, settings, hardware_cls, action_dim, proprio_dim
            )
        if args.phase in train_phases and settings.wandb_enabled:
            wandb_run = _init_wandb(settings, args, learner, warmup_transitions)
        with (
            hardware_cls(args) as hardware,
        ):
            def load_model():
                print("\033[1;33m  Loading model...\033[0m")
                image, state = hardware.read()
                feature.infer(
                    {
                        "observation/image": image,
                        "observation/state": state,
                        "prompt": args.prompt,
                    }
                )
                print("\033[1;32m  ✓ Model ready.\033[0m")

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
                    _prepare_episode(
                        hardware, "warmup", warmup_episode, first_episode, load_model
                    )
                    first_episode = False
                    ticks, success, elapsed, aborted = run_episode(
                        args, "warmup", hardware, feature, learner, lock, q01, q99,
                        episode_dataset,
                    )
                    dataset_episode_index = _finish_episode_dataset(
                        episode_dataset,
                        aborted=aborted,
                        phase="warmup",
                        success=success,
                        seconds=elapsed,
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
                        "dataset_episode_index": dataset_episode_index,
                        "robot_hours": (time.monotonic() - wall_start) / 3600,
                    }
                    outcomes.append(record)
                    with (args.output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(
                        f"Warmup transitions: {len(warmup_replay)}/{warmup_transitions}"
                    )

                _prepare_episode(hardware, args.phase, 1, first_episode, load_model)
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
                        _prepare_episode(
                            hardware, args.phase, episode, first_episode, load_model
                        )
                    first_episode = False
                    ticks, success, elapsed, aborted = run_episode(
                        args, args.phase, hardware, feature, learner, lock, q01, q99,
                        episode_dataset,
                    )
                    dataset_episode_index = _finish_episode_dataset(
                        episode_dataset,
                        aborted=aborted,
                        phase=args.phase,
                        success=success,
                        seconds=elapsed,
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
                        "dataset_episode_index": dataset_episode_index,
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
        if episode_dataset is not None:
            if episode_dataset.has_pending_frames():
                episode_dataset.clear_episode_buffer()
            episode_dataset.finalize()
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
