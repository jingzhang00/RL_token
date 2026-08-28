"""GPU websocket service for frozen VLA features, RL token and reference actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rlt.models import RLToken


class FeaturePolicy:
    def __init__(
        self,
        adapter,
        token: RLToken,
        device: str = "cuda",
        metadata: dict | None = None,
        reference_horizon: int | None = None,
    ) -> None:
        self.adapter, self.token, self.device = adapter, token.eval().to(device), device
        self._metadata = metadata or {}
        self.reference_horizon = reference_horizon

    @property
    def metadata(self):
        return self._metadata

    @torch.inference_mode()
    def infer(self, request: dict) -> dict:
        observations = request.get("observations", request)
        result = self.adapter.infer(observations)
        single = isinstance(observations, dict)
        tokens = result["tokens"][None] if single else result["tokens"]
        valid = result["valid"][None] if single else result["valid"]
        z = self.token.encode(
            torch.as_tensor(tokens, device=self.device),
            torch.as_tensor(valid, device=self.device),
        )

        def batched(value):
            return value[None] if single else value

        reference = result["reference"]
        if self.reference_horizon is not None:
            reference = reference[..., : self.reference_horizon, :]
        output = {
            "z": z.float().cpu().numpy(),
            "reference": batched(reference),
            "absolute": batched(result["absolute"]),
            "normalized_state": batched(result["normalized_state"]),
        }
        return {key: value[0] for key, value in output.items()} if single else output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--token", type=Path, required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--device")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from rlt.openpi_adapter import OpenPIAdapter
    from rlt.settings import load_config

    settings = load_config(args.config)
    host = args.host or settings.runtime["feature_host"]
    port = args.port or int(settings.runtime["feature_port"])
    device = args.device or settings.rl_token["device"]
    checkpoint = Path(args.checkpoint or settings.vla_checkpoint)
    saved = torch.load(args.token, map_location="cpu", weights_only=False)
    token = RLToken(**saved["model_config"])
    token.load_state_dict(saved["model"])
    stats_path = checkpoint / "assets" / "d1" / "norm_stats.json"
    stats = json.loads(stats_path.read_text())["norm_stats"]["actions"]
    metadata = {
        "z_dim": token.width,
        "action_horizon": settings.action_horizon,
        "chunk_len": settings.chunk_length,
        "action_dim": settings.action_dim,
        "action_q01": stats["q01"],
        "action_q99": stats["q99"],
        "token_checkpoint": str(args.token),
        "vla_checkpoint": str(checkpoint),
    }
    policy = FeaturePolicy(
        OpenPIAdapter(
            checkpoint,
            num_steps=settings.vla["inference_steps"],
            config_path=args.config,
        ),
        token,
        device,
        metadata,
        settings.action_horizon,
    )
    WebsocketPolicyServer(
        policy, host=host, port=port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
