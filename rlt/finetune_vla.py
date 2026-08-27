"""Generic OpenPI VLA finetuning runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(config, upstream_root: str | Path) -> None:
    """Run OpenPI's trainer with a robot-provided TrainConfig registry."""
    trainer = _load(
        "openpi.training.train",
        Path(upstream_root) / "scripts" / "train.py",
    )
    trainer.main(config.cli())
