"""Small read-only helpers for local LeRobot v2.1 and v3.0 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_VERSIONS = ("v2.1", "v3.0")


def load_info(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset info not found: {path}")
    info = json.loads(path.read_text())
    version = info.get("codebase_version")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"expected LeRobot {' or '.join(SUPPORTED_VERSIONS)}, got {version!r}"
        )
    return info


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_episode_metadata(
    dataset_root: Path, info: dict[str, Any] | None = None
) -> pd.DataFrame:
    info = info or load_info(dataset_root)
    if info["codebase_version"] == "v2.1":
        path = dataset_root / "meta" / "episodes.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"episode metadata not found: {path}")
        episodes = pd.DataFrame(_read_jsonl(path))
    else:
        paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"episode metadata not found under: {dataset_root / 'meta' / 'episodes'}"
            )
        episodes = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    if "episode_index" not in episodes.columns:
        raise KeyError("missing episode metadata column: episode_index")
    return episodes


def list_episode_indices(
    dataset_root: Path,
    info: dict[str, Any] | None = None,
    episodes: pd.DataFrame | None = None,
) -> list[int]:
    episodes = episodes if episodes is not None else load_episode_metadata(dataset_root, info)
    indices = sorted(episodes["episode_index"].astype(int).tolist())
    if not indices:
        raise ValueError(f"no episodes found in {dataset_root}")
    return indices


def episode_row(episodes: pd.DataFrame, episode_index: int) -> pd.Series:
    rows = episodes.loc[episodes["episode_index"] == episode_index]
    if len(rows) != 1:
        raise ValueError(f"expected one metadata row for episode {episode_index}, got {len(rows)}")
    return rows.iloc[0]


def load_episode(
    dataset_root: Path,
    episode_index: int,
    info: dict[str, Any] | None = None,
    episodes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    info = info or load_info(dataset_root)
    episodes = episodes if episodes is not None else load_episode_metadata(dataset_root, info)
    row = episode_row(episodes, episode_index)
    if info["codebase_version"] == "v2.1":
        path = dataset_root / info["data_path"].format(
            episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
            episode_index=episode_index,
        )
    else:
        path = dataset_root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
    if not path.is_file():
        raise FileNotFoundError(f"episode parquet not found: {path}")
    data = pd.read_parquet(path)
    if "episode_index" not in data.columns:
        raise KeyError("missing data column: episode_index")
    if info["codebase_version"] == "v2.1":
        episode = data.reset_index(drop=True)
        if not (episode["episode_index"] == episode_index).all():
            raise ValueError(f"v2.1 episode file contains another episode: {path}")
    else:
        episode = data.loc[data["episode_index"] == episode_index].reset_index(drop=True)
    if len(episode) != int(row["length"]):
        raise ValueError(
            f"episode {episode_index} expected {int(row['length'])} frames, got {len(episode)}"
        )
    return episode, row


def load_tasks(dataset_root: Path, info: dict[str, Any] | None = None) -> dict[int, str]:
    info = info or load_info(dataset_root)
    if info["codebase_version"] == "v2.1":
        path = dataset_root / "meta" / "tasks.jsonl"
        items = _read_jsonl(path) if path.is_file() else []
        return {
            int(item.get("task_index", item.get("index"))): str(item["task"])
            for item in items
            if item.get("task_index", item.get("index")) is not None and item.get("task") is not None
        }

    path = dataset_root / "meta" / "tasks.parquet"
    if not path.is_file():
        return {}
    tasks = pd.read_parquet(path)
    names = tasks["task"] if "task" in tasks.columns else tasks.index
    indices = tasks["task_index"] if "task_index" in tasks.columns else range(len(tasks))
    return {int(index): str(task) for index, task in zip(indices, names)}


def episode_video(
    dataset_root: Path,
    episode_index: int,
    camera_key: str,
    info: dict[str, Any] | None = None,
    episodes: pd.DataFrame | None = None,
) -> tuple[Path, int, int]:
    info = info or load_info(dataset_root)
    episodes = episodes if episodes is not None else load_episode_metadata(dataset_root, info)
    row = episode_row(episodes, episode_index)
    template = info.get("video_path")
    if not template:
        raise ValueError("dataset does not contain videos")
    if info["codebase_version"] == "v2.1":
        relative = template.format(
            episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
            episode_index=episode_index,
            video_key=camera_key,
        )
        start_frame = 0
    else:
        prefix = f"videos/{camera_key}"
        relative = template.format(
            video_key=camera_key,
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
        )
        start_frame = round(float(row[f"{prefix}/from_timestamp"]) * float(info["fps"]))
    path = dataset_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"episode video not found: {path}")
    return path, start_frame, int(row["length"])
