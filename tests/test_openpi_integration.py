"""Opt-in real-checkpoint tests: RUN_RLT_GPU_TESTS=1 pytest -q tests/test_openpi_integration.py."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RLT_GPU_TESTS") != "1",
    reason="requires a CUDA GPU and the real π0.5 checkpoint",
)
CHECKPOINT = Path(
    "/home/ai01/Downloads/imitation_learning/openpi/train/openpi/checkpoints/4999"
)


def _observation():
    import cv2
    import pyarrow.parquet as parquet

    root = Path("dataset/pick_up_the_doll_and_place_it_on_the_plate")
    table_path = root / "data/chunk-000/episode_000000.parquet"
    state = np.asarray(
        parquet.read_table(table_path, columns=["observation.state"])[0][0].as_py(),
        np.float32,
    )
    video = root / "videos/chunk-000/observation.images.d1_rgb/episode_000000.mp4"
    capture = cv2.VideoCapture(str(video))
    ok, bgr = capture.read()
    capture.release()
    assert ok
    return {
        "observation/image": np.ascontiguousarray(bgr[:, :, ::-1]),
        "observation/state": state,
        "prompt": "Pick up the doll and place it on the plate",
    }


def test_shared_prefix_matches_upstream_and_real_shapes():
    from rlt.openpi_adapter import OpenPIAdapter

    adapter = OpenPIAdapter(CHECKPOINT)
    noise = np.random.default_rng(42).standard_normal((1, 20, 32), dtype=np.float32)
    shared = adapter.infer(_observation(), noise=noise)
    upstream = adapter.policy.infer(_observation(), noise=noise[0])
    assert shared["tokens"].shape == (256, 2048)
    assert shared["reference"].shape == shared["absolute"].shape == (10, 12)
    assert np.isfinite(shared["tokens"]).all()
    assert np.allclose(
        shared["absolute"], upstream["actions"][:10, :12], atol=2e-4, rtol=2e-4
    )


def test_real_token_and_one_online_update():
    import torch

    from rlt.models import Actor, Critic, RLToken
    from rlt.online import Learner, Transition
    from rlt.openpi_adapter import OpenPIAdapter

    token_path = Path(
        os.environ.get("RLT_TOKEN_CHECKPOINT", "checkpoints/rlt_token/step_5000.pt")
    )
    if not token_path.is_file():
        pytest.skip(f"missing {token_path}")
    saved = torch.load(token_path, map_location="cuda", weights_only=False)
    token = RLToken(**saved["model_config"]).cuda().eval()
    token.load_state_dict(saved["model"])
    features = OpenPIAdapter(CHECKPOINT).infer(_observation())
    with torch.inference_mode():
        z = (
            token.encode(
                torch.as_tensor(features["tokens"][None], device="cuda"),
                torch.as_tensor(features["valid"][None], device="cuda"),
            )[0]
            .float()
            .cpu()
            .numpy()
        )
    assert z.shape == (2048,) and np.isfinite(z).all()
    row = Transition(
        z,
        features["reference"],
        features["reference"],
        0,
        z,
        features["reference"],
        np.zeros(24, np.float32),
        np.zeros(24, np.float32),
        "vla",
        False,
    )
    metrics = Learner(Actor(), Critic(), device="cuda").update([row])
    assert np.isfinite(metrics["critic_loss"])
