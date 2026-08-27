#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


OPENPI_WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path("/tmp/openpi_cache")
if str(OPENPI_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(OPENPI_WORKSPACE))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", "--bind", default="127.0.0.1", dest="host")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--policy.config", "--config", dest="policy_config", required=True)
    parser.add_argument("--policy.dir", "--checkpoint-dir", dest="policy_dir", type=Path, required=True)
    parser.add_argument("--default-prompt", default="Do the task.")
    parser.add_argument("--openpi-data-home", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    os.environ.setdefault("OPENPI_DATA_HOME", str(args.openpi_data_home.expanduser().resolve()))
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    policy_dir = args.policy_dir.expanduser().resolve()
    if not (policy_dir / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {policy_dir / 'params'}")
    if not (policy_dir / "assets").is_dir():
        raise FileNotFoundError(f"Checkpoint assets directory not found: {policy_dir / 'assets'}")
    from openpi.policies import policy_config as _policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from robots.d1 import d1_policy
    from robots.d1.train import load_config

    logging.info("Loading OpenPI config=%s checkpoint=%s", args.policy_config, policy_dir)
    _config = load_config()
    train_config = _config.get_config(args.policy_config)
    policy = _policy_config.create_trained_policy(
        train_config,
        policy_dir,
        default_prompt=args.default_prompt,
    )
    metadata = {
        "policy_config": args.policy_config,
        "policy_dir": str(policy_dir),
        "action_horizon": train_config.model.action_horizon,
        # Pi05 pads D1's 12-D actions to its internal 32-D model width. The
        # D1 output transform exposes only the robot's 12-D action contract.
        "action_dim": d1_policy.ACTION_DIM,
        "model_action_dim": train_config.model.action_dim,
        "default_prompt": args.default_prompt,
        **(policy.metadata or {}),
    }
    logging.info("Starting OpenPI websocket server on %s:%d metadata=%s", args.host, args.port, metadata)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
