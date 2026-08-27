"""Run the upstream trainer with the local D1 config overlay."""

import importlib
from pathlib import Path
import sys

from rlt.finetune_vla import run


WORKSPACE = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = WORKSPACE / "openpi"


def load_config():
    for path in (WORKSPACE, UPSTREAM_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    config = importlib.import_module("openpi.training.config")
    from robots.d1.d1_config import get_d1_configs

    for item in get_d1_configs():
        if item.name not in config._CONFIGS_DICT:
            config._CONFIGS.append(item)
            config._CONFIGS_DICT[item.name] = item
    return config


def main() -> None:
    run(load_config(), UPSTREAM_ROOT)


if __name__ == "__main__":
    main()
