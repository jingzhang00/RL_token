"""Compute normalization statistics with the local D1 config overlay."""

import tyro

from d1.train import UPSTREAM_ROOT, _load, load_config


def main() -> None:
    load_config()
    compute = _load(
        "compute_norm_stats",
        UPSTREAM_ROOT / "scripts" / "compute_norm_stats.py",
    )
    tyro.cli(compute.main)


if __name__ == "__main__":
    main()
