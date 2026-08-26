"""Run the upstream trainer with the local D1 config overlay."""

import importlib.util
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = WORKSPACE / "train" / "openpi"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config():
    for path in (WORKSPACE, UPSTREAM_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return _load("openpi.training.config", WORKSPACE / "d1" / "config.py")


def main() -> None:
    config = load_config()
    trainer = _load("openpi.training.train", UPSTREAM_ROOT / "scripts" / "train.py")
    trainer.main(config.cli())


if __name__ == "__main__":
    main()
