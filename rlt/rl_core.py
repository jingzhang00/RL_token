"""SQLite replay and asynchronous online actor/critic updates."""

from __future__ import annotations

import io
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from rlt.models import Actor, Critic, actor_loss, reference_dropout


@dataclass
class Transition:
    z: np.ndarray
    reference: np.ndarray
    action: np.ndarray
    reward: float | np.ndarray
    next_z: np.ndarray
    next_reference: np.ndarray
    proprio: np.ndarray
    next_proprio: np.ndarray
    source: str
    done: bool
    rewards: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.rewards is None:
            self.rewards = np.asarray(self.reward, dtype=np.float32).reshape(-1)
        else:
            self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        if self.rewards.size == 0:
            raise ValueError("transition must contain at least one reward")


def _blob(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def _array(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


class Replay:
    def __init__(
        self,
        path: str | Path,
        capacity: int = 100_000,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS transitions(
            id INTEGER PRIMARY KEY, z BLOB, reference BLOB, action BLOB, reward REAL,
            next_z BLOB, next_reference BLOB, proprio BLOB, next_proprio BLOB,
            source TEXT, done INTEGER, rewards BLOB)"""
        )
        columns = {
            row[1]
            for row in self.db.execute("PRAGMA table_info(transitions)").fetchall()
        }
        if "rewards" not in columns:
            self.db.execute("ALTER TABLE transitions ADD COLUMN rewards BLOB")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS replay_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        if metadata is not None:
            expected = {key: json.dumps(value, sort_keys=True) for key, value in metadata.items()}
            stored = dict(self.db.execute("SELECT key, value FROM replay_metadata"))
            if stored:
                if stored != expected:
                    raise ValueError(
                        f"replay metadata mismatch for {path}: stored={stored}, expected={expected}"
                    )
            elif self.db.execute("SELECT 1 FROM transitions LIMIT 1").fetchone() is not None:
                raise ValueError(
                    f"replay {path} has data but no metadata; archive it before changing H/C"
                )
            else:
                self.db.executemany(
                    "INSERT INTO replay_metadata(key, value) VALUES(?, ?)",
                    expected.items(),
                )
        self.db.commit()

    def __len__(self) -> int:
        with self.lock:
            return int(
                self.db.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
            )

    def add(self, item: Transition) -> None:
        arrays = (
            item.z,
            item.reference,
            item.action,
            item.rewards,
            item.next_z,
            item.next_reference,
            item.proprio,
            item.next_proprio,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("transition contains NaN or Inf")
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO transitions "
                "(z,reference,action,reward,next_z,next_reference,proprio,next_proprio,source,done,rewards) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _blob(item.z),
                    _blob(item.reference),
                    _blob(item.action),
                    float(np.asarray(item.rewards).sum()),
                    _blob(item.next_z),
                    _blob(item.next_reference),
                    _blob(item.proprio),
                    _blob(item.next_proprio),
                    item.source,
                    int(item.done),
                    _blob(item.rewards),
                ),
            )
            self.db.execute(
                "DELETE FROM transitions WHERE id IN (SELECT id FROM transitions ORDER BY id LIMIT MAX(0,(SELECT COUNT(*) FROM transitions)-?))",
                (self.capacity,),
            )

    def sample(self, batch_size: int) -> list[Transition]:
        with self.lock:
            rows = self.db.execute(
                "SELECT z,reference,action,reward,next_z,next_reference,proprio,next_proprio,source,done,rewards "
                "FROM transitions ORDER BY RANDOM() LIMIT ?",
                (int(batch_size),),
            ).fetchall()
        if len(rows) < batch_size:
            raise ValueError(f"replay has {len(rows)} rows, need {batch_size}")
        return [
            Transition(
                _array(a),
                _array(b),
                _array(c),
                d,
                _array(e),
                _array(f),
                _array(g),
                    _array(h),
                    i,
                    bool(j),
                    None if k is None else _array(k),
                )
            for a, b, c, d, e, f, g, h, i, j, k in rows
        ]

    def close(self) -> None:
        self.db.close()


def make_proprio(
    position: np.ndarray, previous: np.ndarray | None, dt: float = 0.1
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32)
    velocity = (
        np.zeros_like(position)
        if previous is None
        else (position - previous) / float(dt)
    )
    return np.concatenate([position, velocity]).astype(np.float32)


def _source_segments(ticks: list[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    source = None
    for tick in ticks:
        if current and (tick["source"] != source or current[-1].get("done")):
            segments.append(current)
            current = []
        if not current:
            source = tick["source"]
        current.append(tick)
    if current:
        segments.append(current)
    return segments


def _reference_for(
    tick: dict, action: np.ndarray, horizon: int
) -> np.ndarray:
    if tick["source"] == "human":
        return action.copy()
    value = np.asarray(
        tick.get("reference_chunk", tick["reference"]), dtype=np.float32
    )
    if value.shape == action.shape:
        return value.copy()
    if value.ndim == 2 and value.shape[1] == action.shape[1] and value.shape[0] >= horizon:
        return value[:horizon].copy()
    return np.stack([tick["reference"] for _ in range(horizon)]).astype(np.float32)


def build_transitions(
    ticks: list[dict], horizon: int = 10, stride: int = 2
) -> list[Transition]:
    """Create stride-spaced C-step windows without crossing source segments."""
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    result = []
    for segment in _source_segments(ticks):
        for start in range(0, len(segment), stride):
            window = segment[start : start + horizon]
            terminal_index = next(
                (index for index, tick in enumerate(window) if tick.get("done")), None
            )
            if terminal_index is None and len(window) < horizon:
                break
            if terminal_index is not None:
                window = window[: terminal_index + 1]
            padded = window + [window[-1]] * (horizon - len(window))
            action = np.stack([tick["action"] for tick in padded]).astype(np.float32)
            reference = _reference_for(segment[start], action, horizon)
            terminal = window[-1] if terminal_index is not None else None
            nxt = (
                window[-1]
                if terminal is not None
                else segment[min(start + horizon, len(segment) - 1)]
            )
            next_reference = _reference_for(nxt, action, horizon)
            rewards = np.zeros(horizon, dtype=np.float32)
            rewards[: len(window)] = [float(tick.get("reward", 0.0)) for tick in window]
            result.append(
                Transition(
                    np.asarray(segment[start]["z"], np.float32),
                    reference,
                    action,
                    float(terminal.get("reward", 0.0) if terminal else 0.0),
                    np.asarray(nxt["z"], np.float32),
                    next_reference,
                    np.asarray(segment[start]["proprio"], np.float32),
                    np.asarray(nxt["proprio"], np.float32),
                    segment[start]["source"],
                    terminal is not None,
                    rewards,
                )
            )
    return result


class Learner:
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        horizon: int = 10,
        sigma: float = 0.05,
        beta: float = 1.0,
        reference_dropout_probability: float = 0.5,
        actor_lr: float | None = None,
        critic_lr: float | None = None,
        actor_update_period: int = 2,
        critic_updates_per_transition: int = 5,
        device: str = "cpu",
    ) -> None:
        import copy

        self.actor, self.critic = actor.to(device), critic.to(device)
        self.target_actor, self.target_critic = (
            copy.deepcopy(actor).to(device),
            copy.deepcopy(critic).to(device),
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=lr if actor_lr is None else actor_lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=lr if critic_lr is None else critic_lr
        )
        self.gamma, self.tau, self.horizon, self.sigma, self.beta = (
            gamma,
            tau,
            horizon,
            sigma,
            beta,
        )
        self.reference_dropout_probability = float(reference_dropout_probability)
        self.actor_update_period = int(actor_update_period)
        self.critic_updates_per_transition = int(critic_updates_per_transition)
        if self.actor_update_period < 1 or self.critic_updates_per_transition < 1:
            raise ValueError("update periods must be positive")
        self.device = device
        self.critic_steps = 0
        self.actor_steps = 0
        self.online_critic_steps = 0
        self.online_actor_steps = 0

    @property
    def target_discount(self) -> float:
        return self.gamma**self.horizon

    def _batch(self, rows: Iterable[Transition]) -> dict[str, torch.Tensor]:
        rows = list(rows)

        def tensor(name):
            return torch.as_tensor(
                np.stack([getattr(row, name) for row in rows]), device=self.device
            ).float()

        reward_chunks = []
        for row in rows:
            rewards = np.asarray(row.rewards, dtype=np.float32).reshape(-1)
            if rewards.size == 1 and self.horizon > 1:
                padded = np.zeros(self.horizon, dtype=np.float32)
                padded[0] = rewards[0]
                rewards = padded
            elif rewards.size < self.horizon:
                rewards = np.pad(rewards, (0, self.horizon - rewards.size))
            reward_chunks.append(rewards[: self.horizon])
        return {
            name: tensor(name)
            for name in (
                "z",
                "reference",
                "action",
                "next_z",
                "next_reference",
                "proprio",
                "next_proprio",
            )
        } | {
            "rewards": torch.as_tensor(
                np.stack(reward_chunks), device=self.device
            ).float(),
            "done": torch.tensor(
                [row.done for row in rows], device=self.device
            ).float(),
        }

    def td_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            next_action, _ = self.target_actor.sample(
                batch["next_z"],
                batch["next_proprio"],
                batch["next_reference"],
                self.sigma,
            )
            q1, q2 = self.target_critic(
                batch["next_z"],
                batch["next_proprio"],
                next_action,
            )
            discounts = self.gamma ** torch.arange(
                self.horizon,
                device=self.device,
                dtype=batch["rewards"].dtype,
            )
            reward = (batch["rewards"] * discounts).sum(dim=1)
            return reward + (1 - batch["done"]) * self.target_discount * torch.minimum(q1, q2)

    def update(
        self, rows: Iterable[Transition], *, online: bool = False
    ) -> dict[str, float]:
        batch = self._batch(rows)
        target = self.td_target(batch)
        q1, q2 = self.critic(batch["z"], batch["proprio"], batch["action"])
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.critic_steps += 1
        if online:
            self.online_critic_steps += 1
        q_min = torch.minimum(q1, q2)
        metrics = {
            "critic_loss": float(critic_loss.detach()),
            "q1_mean": float(q1.detach().mean()),
            "q2_mean": float(q2.detach().mean()),
            "q_min_mean": float(q_min.detach().mean()),
            "target_q_mean": float(target.mean()),
            "td_error_abs": float((q_min.detach() - target).abs().mean()),
        }
        if self.critic_steps % self.actor_update_period == 0:
            dropped, _ = reference_dropout(
                batch["reference"], self.reference_dropout_probability
            )
            action, _ = self.actor.sample(
                batch["z"], batch["proprio"], dropped, self.sigma
            )
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            q, _ = self.critic(batch["z"], batch["proprio"], action)
            loss = actor_loss(action, batch["reference"], q, self.beta)
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.actor_opt.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
            self.actor_steps += 1
            if online:
                self.online_actor_steps += 1
            metrics["actor_loss"] = float(loss.detach())
            self._soft_update()
        return metrics

    def update_for_new_transition(
        self, replay: Replay, batch_size: int = 256, *, online: bool = True
    ) -> list[dict[str, float]]:
        if len(replay) < batch_size:
            return []
        return [
            self.update(replay.sample(batch_size), online=online)
            for _ in range(self.critic_updates_per_transition)
        ]

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target, value in zip(
                self.target_actor.parameters(), self.actor.parameters(), strict=True
            ):
                target.lerp_(value, self.tau)
            for target, value in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target.lerp_(value, self.tau)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "critic_steps": self.critic_steps,
                "actor_steps": self.actor_steps,
                "online_critic_steps": self.online_critic_steps,
                "online_actor_steps": self.online_actor_steps,
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path: str | Path) -> dict:
        data = torch.load(path, map_location=self.device, weights_only=False)
        for name in ("actor", "critic", "target_actor", "target_critic"):
            getattr(self, name).load_state_dict(data[name])
        self.actor_opt.load_state_dict(data["actor_opt"])
        self.critic_opt.load_state_dict(data["critic_opt"])
        self.critic_steps, self.actor_steps = data["critic_steps"], data["actor_steps"]
        self.online_critic_steps = int(data.get("online_critic_steps", 0))
        self.online_actor_steps = int(data.get("online_actor_steps", 0))
        return data.get("metadata", {})
