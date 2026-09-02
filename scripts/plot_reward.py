#!/usr/bin/env python3
"""Plot critical success rate by episode from an online replay SQLite database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np


def read_terminal_rewards(path: Path) -> tuple[list[str], np.ndarray]:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute(
            "SELECT id, source, done, reward FROM transitions "
            "WHERE done = 1 ORDER BY id"
        ).fetchall()

    if not rows:
        raise ValueError(f"no terminal transitions found in {path}")

    # ponytail: replay has no episode_id; one contiguous terminal group is one
    # critical episode. Store episode_id if episodes can contain multiple phases.
    events = []
    previous_id = None
    for row in rows:
        if previous_id is None or row[0] != previous_id + 1:
            events.append(row)
        previous_id = row[0]

    sources = [row[1] for row in events]
    rewards = np.asarray([row[3] for row in events], dtype=np.float32)
    return sources, rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replay",
        type=Path,
        nargs="?",
        default=Path("checkpoints/online/replay.sqlite"),
    )
    parser.add_argument(
        "--source",
        choices=("actor", "human", "all"),
        default="actor",
        help="critical terminal source to plot (default: actor)",
    )
    parser.add_argument("--output", type=Path, help="save PNG instead of opening a window")
    args = parser.parse_args()

    sources, rewards = read_terminal_rewards(args.replay)
    if args.source != "all":
        mask = np.asarray([source == args.source for source in sources])
        rewards = rewards[mask]
    if not len(rewards):
        raise ValueError(f"no terminal transitions found for source={args.source}")
    import matplotlib

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(rewards) + 1)
    success_rate = np.cumsum(rewards) / episodes
    ax.plot(
        episodes,
        success_rate,
        color="tab:blue",
        linewidth=2,
        label=f"{args.source}",
    )
    ax.set(
        xlabel="critical episode",
        ylabel="critical success rate",
        title=f"{args.source} critical success rate: {len(rewards)} episodes",
        ylim=(-0.05, 1.05),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
