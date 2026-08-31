from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlt.models import Actor, Critic, RLToken, actor_loss, reference_dropout
from rlt.online_runner import (
    _confirm_online_rollout,
    _offline_warmup,
    _read_episode_label,
    _format_sync_table,
    run_episode,
)
from rlt.rl_core import Learner, Replay, Transition, build_transitions, make_proprio
from rlt.actions import absolute_to_normalized, normalized_to_absolute, safe_command
from rlt.serve_features import FeaturePolicy
from rlt.settings import load_config
from robots.d1.online_rl_d1 import D1Hardware


def test_sync_error_uses_joint_direction_arrows():
    table = _format_sync_table(
        [1.3, 2.0, -2.3, -10.7, 2.3, 1.1],
        10,
        [0.1, 0.0, 0.1, 0.0, 0.3, 0.0],
        0.2,
    )
    plain = table.replace("\033[32m", "").replace("\033[33m", "").replace("\033[31m", "").replace("\033[0m", "")
    assert [line.rstrip() for line in plain.splitlines()] == [
        "          J1       J2       J3       J4       J5       J6    | T-pitch   T-yaw    Index    Middle    Ring    Pinky",
        "offset  +1.3°    +2.0°    -2.3°    -10.7°   +2.3°    +1.1°   |  +0.10    +0.00    +0.10    +0.00    +0.30    +0.00",
        "arrow     →        ↑        ↓        ↺        ↑        ↻     |    C        ·        C        ·        C        ·",
        "status    OK       OK       OK      NOT       OK       OK    |    OK       OK       OK       OK      NOT       OK",
    ]
    assert "\033[32m→\033[0m" in table
    assert "\033[31m↺\033[0m" in table
    assert "\033[31mC\033[0m" in table


def test_episode_label_is_collected_after_episode_ends(monkeypatch):
    labels = iter(["invalid", "s"])
    monkeypatch.setattr("builtins.input", lambda _: next(labels))
    assert _read_episode_label()


def test_episode_label_can_abort(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "q")
    assert _read_episode_label() is None


def test_online_rollout_confirmation(monkeypatch):
    answers = iter(["invalid", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert _confirm_online_rollout()


def test_rl_token_shape_causality_and_stop_gradient():
    torch.manual_seed(0)
    model = RLToken(width=16, heads=4, layers=1, mlp_ratio=2, max_tokens=5)
    tokens = torch.randn(2, 5, 16)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    z, prediction, loss = model(tokens, valid)
    assert z.shape == (2, 16) and prediction.shape == tokens.shape and loss.ndim == 0
    expected = (
        (prediction.float() - tokens.float()).square().sum(dim=-1) * valid
    ).sum(dim=1).mean()
    assert torch.allclose(loss, expected)

    changed = tokens.clone()
    changed[:, 3:] += 100
    fixed_z = torch.randn(2, 16)
    assert torch.allclose(
        model.decode(fixed_z, tokens)[:, :3],
        model.decode(fixed_z, changed)[:, :3],
        atol=1e-5,
    )
    grad_tokens = tokens.clone().requires_grad_()
    model.decode(fixed_z, grad_tokens).sum().backward()
    assert grad_tokens.grad is None


def test_reconstruction_loss_decreases():
    torch.manual_seed(1)
    model = RLToken(width=8, heads=2, layers=1, mlp_ratio=2, max_tokens=3)
    tokens = torch.randn(2, 3, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = float(model(tokens)[2])
    for _ in range(30):
        loss = model(tokens)[2]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(model(tokens)[2]) < initial * 0.5


def _transition(value: float = 0, done: bool = False) -> Transition:
    return Transition(
        np.full(8, value, np.float32),
        np.full((2, 3), value, np.float32),
        np.full((2, 3), value, np.float32),
        float(done),
        np.full(8, value + 1, np.float32),
        np.full((2, 3), value + 1, np.float32),
        np.full(4, value, np.float32),
        np.full(4, value + 1, np.float32),
        "actor",
        done,
    )


def test_reference_dropout_and_twin_c_step_target():
    torch.manual_seed(0)
    reference = torch.ones(100, 2, 3)
    dropped, keep = reference_dropout(reference)
    assert torch.equal(dropped.sum((1, 2)) > 0, keep)
    assert 25 < int(keep.sum()) < 75

    learner = Learner(Actor(8, 4, 2, 3), Critic(8, 4, 2, 3), horizon=2)
    for parameter in learner.target_critic.parameters():
        parameter.data.zero_()
    learner.target_critic.q1.net[-1].bias.data.fill_(2)
    learner.target_critic.q2.net[-1].bias.data.fill_(3)
    target = learner.td_target(learner._batch([_transition()]))
    assert torch.allclose(target, torch.tensor([0.99**2 * 2]))
    terminal = learner.td_target(learner._batch([_transition(done=True)]))
    assert torch.equal(terminal, torch.ones(1))


def test_eq5_uses_squared_l2_norm_and_c_step_rewards():
    action = torch.ones(1, 2, 3)
    reference = torch.zeros_like(action)
    loss = actor_loss(action, reference, torch.zeros(1))
    assert torch.equal(loss, torch.tensor(6.0))
    assert torch.equal(
        actor_loss(action, reference, torch.full((1,), 2.0), 0),
        torch.tensor(-2.0),
    )

    learner = Learner(Actor(8, 4, 2, 3), Critic(8, 4, 2, 3), horizon=2)
    for parameter in learner.target_critic.parameters():
        parameter.data.zero_()
    target = learner.td_target(
        learner._batch(
            [
                Transition(
                    np.zeros(8),
                    np.zeros((2, 3)),
                    np.zeros((2, 3)),
                    0.0,
                    np.zeros(8),
                    np.zeros((2, 3)),
                    np.zeros(4),
                    np.zeros(4),
                    "actor",
                    False,
                    np.array([1.0, 2.0], np.float32),
                )
            ]
        )
    )
    assert torch.allclose(target, torch.tensor([1 + 0.99 * 2]))


def test_frequency_and_task_mlp_settings_are_derived_from_yaml():
    config = load_config()
    assert (config.control_hz, config.action_horizon) == (10, 20)
    assert config.chunk_length == 10
    assert config.transition_horizon == config.chunk_length
    assert config.target_discount == 0.99**config.chunk_length
    assert "phases" not in config.online_rl and "max_hours" not in config.runtime
    assert "action_dim" not in config.online_rl and "proprio_dim" not in config.online_rl
    assert "action_dim" not in config.vla and config.action_dim == 12
    assert (config.wandb_enabled, config.project_name, config.log_interval) == (True, "openpi", 10)
    assert config.online_rl["warmup_transitions"] == 1000
    assert config.online_rl["offline_warmup_updates"] == 10000
    assert load_config(task="screw_installation").mlp.actor_hidden_dim == 512


def test_update_ratio_replay_capacity_and_restore(tmp_path):
    replay = Replay(tmp_path / "replay.sqlite", capacity=3)
    for value in range(5):
        replay.add(_transition(value))
    assert len(replay) == 3 and len(replay.sample(2)) == 2
    learner = Learner(Actor(8, 4, 2, 3), Critic(8, 4, 2, 3), horizon=2)
    for _ in range(4):
        learner.update(replay.sample(2))
    assert learner.critic_steps == 4 and learner.actor_steps == 2
    checkpoint = tmp_path / "learner.pt"
    learner.save(checkpoint, {"beta": 1})
    restored = Learner(Actor(8, 4, 2, 3), Critic(8, 4, 2, 3), horizon=2)
    assert restored.load(checkpoint) == {"beta": 1}
    assert (restored.critic_steps, restored.actor_steps) == (4, 2)
    assert (restored.online_critic_steps, restored.online_actor_steps) == (0, 0)
    replay.close()


def test_replay_metadata_rejects_incompatible_data(tmp_path):
    metadata = {"H": 20, "C": 10, "stride": 2, "kind": "online"}
    replay = Replay(tmp_path / "replay.sqlite", capacity=3, metadata=metadata)
    replay.add(_transition())
    replay.close()
    with pytest.raises(ValueError, match="metadata mismatch"):
        Replay(
            tmp_path / "replay.sqlite",
            capacity=3,
            metadata={"H": 20, "C": 20, "stride": 2, "kind": "online"},
        )


def test_transitions_do_not_cross_source_segments():
    ticks = []
    for source in ("actor", "human"):
        for index in range(4):
            value = len(ticks)
            ticks.append(
                {
                    "z": np.full(8, value),
                    "reference": np.full(3, value),
                    "action": np.full(3, 10 + value),
                    "reference_chunk": np.full((4, 3), value),
                    "proprio": np.full(4, value),
                    "source": source,
                    "done": False,
                    "reward": 0.0,
                }
            )
    rows = build_transitions(ticks, horizon=3, stride=1)
    assert [row.source for row in rows] == ["actor", "actor", "human", "human"]
    assert all(not row.done for row in rows)
    assert np.array_equal(rows[2].reference, rows[2].action)


def test_online_update_counters_are_separate():
    torch.manual_seed(3)
    learner = Learner(Actor(8, 4, 2, 3), Critic(8, 4, 2, 3), horizon=2)
    learner.update([_transition()], online=True)
    learner.update([_transition()], online=True)
    assert learner.critic_steps == 2
    assert learner.online_critic_steps == 2
    assert learner.actor_steps == 1
    assert learner.online_actor_steps == 1


def test_offline_warmup_updates_only_total_counters(tmp_path):
    replay = Replay(tmp_path / "warmup.sqlite", capacity=4)
    for index in range(4):
        replay.add(_transition(index))
    learner = Learner(Actor(8, 4, 2, 3), Critic(8, 4, 2, 3), horizon=2)
    _offline_warmup(learner, replay, updates=3, batch_size=2)
    assert learner.critic_steps == 3
    assert learner.online_critic_steps == 0
    replay.close()


def test_full_task_c_and_h_switch_between_vla_actor_and_human(monkeypatch):
    class FakeKeys:
        def __enter__(self):
            return self

        def read(self):
            return keys.pop(0)

        def __exit__(self, *_):
            return None

    class FakeHardware:
        def __init__(self):
            self.sent = []
            self.rebases = 0

        def read(self):
            return np.zeros((8, 8, 3), np.uint8), np.zeros(12, np.float32)

        def send(self, command):
            self.sent.append(command)

        def start_human_episode(self, _):
            pass

        def begin_human_sync(self):
            pass

        def human_sync_ready(self, _):
            return True

        def human_sync_target(self):
            return np.zeros(12, np.float32)

        def rebase_human(self, _):
            self.rebases += 1

        def human_target(self):
            return np.zeros(12, np.float32)

    class FakeFeature:
        def infer(self, _):
            if "convert_actions" in _:
                conversions.append(np.asarray(_["convert_actions"], np.float32))
                return {"absolute": np.asarray(_["convert_actions"], np.float32)}
            return {
                "z": np.zeros(8, np.float32),
                "reference": np.zeros((3, 12), np.float32),
                "absolute": np.zeros((3, 12), np.float32),
                "normalized_state": np.zeros(12, np.float32),
            }

    conversions = []
    args = SimpleNamespace(
        chunk_len=2,
        vla_horizon=3,
        control_hz=1000,
        sigma=0.0,
        prompt="test",
        critical_timeout_s=1.0,
        full_timeout_s=1.0,
        max_arm_delta_deg=12.0,
        arm_low=np.full(6, -10, np.float32),
        arm_high=np.full(6, 10, np.float32),
    )
    keys = [None, "c", None, "h", None, "h", None, "\n"]
    learner = Learner(Actor(8, 24, 2, 12), Critic(8, 24, 2, 12), horizon=2)
    monkeypatch.setattr("rlt.online_runner.Keys", FakeKeys)
    monkeypatch.setattr("rlt.online_runner._read_episode_label", lambda: True)
    ticks, success, _, aborted = run_episode(
        args,
        "full-task",
        FakeHardware(),
        FakeFeature(),
        learner,
        torch.no_grad(),
        np.full(12, -1, np.float32),
        np.full(12, 1, np.float32),
    )
    assert success and not aborted
    assert [tick["source"] for tick in ticks] == [
        "vla", "actor", "actor", "human", "human", "vla", "vla"
    ]
    assert [tick["critical"] for tick in ticks] == [
        False, True, True, True, True, False, False
    ]
    assert conversions and conversions[0].shape == (2, 12)

    keys = ["h", None, "h", None, "\n"]
    ticks, success, _, aborted = run_episode(
        args,
        "full-task",
        FakeHardware(),
        FakeFeature(),
        learner,
        torch.no_grad(),
        np.full(12, -1, np.float32),
        np.full(12, 1, np.float32),
    )
    assert success and not aborted
    assert [tick["source"] for tick in ticks] == ["human", "human", "vla", "vla"]
    assert [tick["critical"] for tick in ticks] == [True, True, False, False]


def test_critical_only_starts_with_teleop_and_c_starts_rl(monkeypatch, capsys):
    class FakeKeys:
        def __enter__(self):
            return self

        def read(self):
            return keys.pop(0)

        def __exit__(self, *_):
            return None

    class FakeHardware:
        def __init__(self):
            self.sent = []
            self.rebases = 0

        def read(self):
            return np.zeros((8, 8, 3), np.uint8), np.zeros(12, np.float32)

        def send(self, command):
            self.sent.append(command)

        def start_human_episode(self, _):
            pass

        def begin_human_sync(self):
            pass

        def human_sync_ready(self, _):
            return True

        def human_sync_target(self):
            return np.zeros(12, np.float32)

        def rebase_human(self, _):
            self.rebases += 1

        def human_target(self):
            return np.zeros(12, np.float32)

    class FakeFeature:
        def infer(self, _):
            if "convert_actions" in _:
                return {"absolute": np.asarray(_["convert_actions"], np.float32)}
            return {
                "z": np.zeros(8, np.float32),
                "reference": np.zeros((3, 12), np.float32),
                "absolute": np.zeros((3, 12), np.float32),
                "normalized_state": np.zeros(12, np.float32),
            }

    args = SimpleNamespace(
        chunk_len=2,
        vla_horizon=3,
        control_hz=1000,
        sigma=0.0,
        prompt="test",
        critical_timeout_s=1.0,
        full_timeout_s=1.0,
        max_arm_delta_deg=12.0,
        arm_low=np.full(6, -10, np.float32),
        arm_high=np.full(6, 10, np.float32),
    )
    hardware = FakeHardware()
    keys = [None, None, "c", None, "h", None, "h", None, "\n"]
    learner = Learner(Actor(8, 24, 2, 12), Critic(8, 24, 2, 12), horizon=2)
    monkeypatch.setattr("rlt.online_runner.Keys", FakeKeys)
    monkeypatch.setattr("rlt.online_runner._read_episode_label", lambda: True)
    ticks, success, _, aborted = run_episode(
        args,
        "critical-only",
        hardware,
        FakeFeature(),
        learner,
        torch.no_grad(),
        np.full(12, -1, np.float32),
        np.full(12, 1, np.float32),
    )
    assert success and not aborted
    assert [tick["source"] for tick in ticks] == [
        "actor", "actor", "human", "human", "actor", "actor"
    ]
    assert all(tick["critical"] for tick in ticks)
    assert len(hardware.sent) == 9
    assert hardware.rebases == 1
    assert "y=approve VLA chunk" not in capsys.readouterr().out

    hardware = FakeHardware()
    keys = ["h", None, "h", None, "\n"]
    ticks, success, _, aborted = run_episode(
        args,
        "critical-only",
        hardware,
        FakeFeature(),
        learner,
        torch.no_grad(),
        np.full(12, -1, np.float32),
        np.full(12, 1, np.float32),
    )
    assert success and not aborted
    assert [tick["source"] for tick in ticks] == ["human", "human", "actor", "actor"]
    assert all(tick["critical"] for tick in ticks)


def test_human_sync_rebases_and_prompts_without_confirmation(monkeypatch, capsys):
    class FakeKeys:
        def __enter__(self):
            return self

        def read(self):
            return keys.pop(0)

        def __exit__(self, *_):
            return None

    class FakeHardware:
        def __init__(self):
            self.ready = iter([False, False, True])
            self.rebases = 0
            self.begins = 0
            self.sent = []
            self.reads = 0
            self.sync_target = np.full(12, 2, np.float32)

        def read(self):
            self.reads += 1
            return np.zeros((8, 8, 3), np.uint8), np.full(
                12, self.reads, np.float32
            )

        def start_human_episode(self, _):
            pass

        def begin_human_sync(self):
            self.begins += 1

        def human_sync_ready(self, _):
            ready = next(self.ready)
            if not ready:
                time.sleep(0.02)
            return ready

        def human_sync_target(self):
            return self.sync_target

        def rebase_human(self, _):
            self.rebases += 1

        def human_target(self):
            return np.zeros(12, np.float32)

        def send(self, command):
            self.sent.append(command)

    class FakeFeature:
        def infer(self, request):
            if "convert_actions" in request:
                return {"absolute": np.asarray(request["convert_actions"], np.float32)}
            return {
                "z": np.zeros(8, np.float32),
                "reference": np.zeros((3, 12), np.float32),
                "absolute": np.zeros((3, 12), np.float32),
                "normalized_state": np.zeros(12, np.float32),
            }

    args = SimpleNamespace(
        chunk_len=2,
        vla_horizon=3,
        control_hz=1000,
        sigma=0.0,
        prompt="test",
        critical_timeout_s=1.0,
        full_timeout_s=0.01,
        max_arm_delta_deg=12.0,
        arm_low=np.full(6, -10, np.float32),
        arm_high=np.full(6, 10, np.float32),
    )
    hardware = FakeHardware()
    keys = [None, "h", None, None, "\n"]
    learner = Learner(Actor(8, 24, 2, 12), Critic(8, 24, 2, 12), horizon=2)
    monkeypatch.setattr("rlt.online_runner.Keys", FakeKeys)
    monkeypatch.setattr("rlt.online_runner._read_episode_label", lambda: True)
    ticks, success, _, aborted = run_episode(
        args,
        "full-task",
        hardware,
        FakeFeature(),
        learner,
        torch.no_grad(),
        np.full(12, -1, np.float32),
        np.full(12, 1, np.float32),
    )
    assert success and not aborted
    assert [tick["source"] for tick in ticks] == ["vla", "human"]
    assert all(tick["critical"] for tick in ticks[1:])
    assert hardware.begins == 1 and hardware.rebases == 1
    assert len(hardware.sent) == 5
    assert np.array_equal(hardware.sent[1], hardware.sent[2])
    assert np.array_equal(hardware.sent[2], hardware.sent[3])
    assert np.array_equal(hardware.sent[1], np.full(12, 3, np.float32))
    assert not np.array_equal(hardware.sent[1], np.ones(12, np.float32))
    output = capsys.readouterr().out
    assert "Human syncing:" in output
    assert "offset" in output
    assert "|" in output
    assert "T-pitch" in output
    assert "Ring" in output
    assert "Human control activ" in output


def test_human_target_deadband_holds_pose():
    class Read:
        ok = True
        angle_deg = 10.5

    hardware = D1Hardware.__new__(D1Hardware)
    hardware.args = SimpleNamespace(
        arm_low=np.full(6, -10, np.float32),
        arm_high=np.full(6, 10, np.float32),
        human_deadband_arm_deg=1.0,
        human_deadband_hand=0.02,
    )
    hardware.exo_zero = np.full(6, 10, np.float32)
    hardware.takeover_arm = np.zeros(6, np.float32)
    hardware.glove_zero = np.full(6, 0.5, np.float32)
    hardware.takeover_hand = np.full(6, 0.5, np.float32)
    hardware.exo = SimpleNamespace(
        read_frame=lambda: SimpleNamespace(reads=[Read() for _ in range(6)])
    )
    hardware.glove = SimpleNamespace(
        get_norm_values=lambda: np.full(6, 0.51, np.float32)
    )

    expected = np.concatenate([np.zeros(6), np.full(6, 0.5)]).astype(np.float32)
    assert np.array_equal(hardware.human_target(), expected)


def test_stride_terminal_human_reference_and_proprio():
    ticks = []
    for index in range(7):
        ticks.append(
            {
                "z": np.full(8, index),
                "reference": np.full(3, index),
                "action": np.full(3, 10 + index),
                "reference_chunk": np.full((3, 3), index),
                "proprio": np.full(4, index),
                "source": "human",
                "done": index == 5,
                "reward": 1.0 if index == 5 else 0.0,
            }
        )
    rows = build_transitions(ticks, horizon=2, stride=2)
    assert len(rows) == 3
    assert np.array_equal(rows[0].reference, rows[0].action)
    assert rows[-1].done and rows[-1].reward == 1
    proprio = make_proprio(np.ones(2), np.zeros(2), dt=0.5)
    assert np.array_equal(proprio, [1, 1, 2, 2])


def test_action_roundtrip_safety_and_fake_feature_protocol():
    rng = np.random.default_rng(2)
    state = rng.normal(size=12).astype(np.float32)
    q01, q99 = np.full(12, -2, np.float32), np.full(12, 2, np.float32)
    normalized = rng.uniform(-1, 1, size=(10, 12)).astype(np.float32)
    absolute = normalized_to_absolute(normalized, state, q01, q99)
    assert np.allclose(
        absolute_to_normalized(absolute, state, q01, q99), normalized, atol=1e-6
    )
    unsafe = np.full(12, 100, np.float32)
    safe = safe_command(unsafe, np.zeros(12), max_arm_delta_deg=12)
    assert np.all(safe[:6] <= np.deg2rad(12) + 1e-7) and np.all(safe[6:] == 1)

    class FakeAdapter:
        def infer(self, observation):
            return {
                "tokens": np.zeros((3, 8), np.float32),
                "valid": np.ones(3, bool),
                "reference": np.zeros((2, 3), np.float32),
                "absolute": np.ones((2, 3), np.float32),
                "normalized_state": np.zeros(3, np.float32),
            }

    policy = FeaturePolicy(FakeAdapter(), RLToken(8, 2, 1, 2, 3), device="cpu")
    response = policy.infer({"observation/state": np.zeros(3)})
    assert response["z"].shape == (8,) and response["reference"].shape == (2, 3)
