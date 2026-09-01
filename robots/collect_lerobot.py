#!/usr/bin/env python3
"""Record robot data in local LeRobot v2.1 or v3.0 format."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import math
import os
import queue
import select
import shutil
import sys
import termios
import tempfile
import threading
import time
import tty
import warnings
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

try:
    if __package__:
        from .robot_specific import (
            ACTION_DELTA_KEY,
            ACTION_KEY,
            DEFAULT_RAW_IMAGE_SIZE,
            DEFAULT_REPO_ID_PREFIX,
            STATE_KEY,
            add_robot_arguments,
            build_collection_metadata,
            create_source,
            validate_robot_args,
            write_collection_metadata,
        )
    else:
        from robot_specific import (
            ACTION_DELTA_KEY,
            ACTION_KEY,
            DEFAULT_RAW_IMAGE_SIZE,
            DEFAULT_REPO_ID_PREFIX,
            STATE_KEY,
            add_robot_arguments,
            build_collection_metadata,
            create_source,
            validate_robot_args,
            write_collection_metadata,
        )
except ModuleNotFoundError as exc:
    if exc.name not in {"robot_specific", "robots.robot_specific"}:
        raise
    # The local writers are also reused by online RL without collector hardware.
    STATE_KEY = "observation.state"
    ACTION_KEY = "action"
    ACTION_DELTA_KEY = "action.delta"


DATA_COLLECTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = DATA_COLLECTION_ROOT / "data"
DEFAULT_VIDEO_CODEC = "h264"
DEFAULT_VIDEO_PIX_FMT = "yuv420p"
DEFAULT_VIDEO_CRF = 23
DEFAULT_VIDEO_GOP = 2
VIDEO_WINDOW_NAME = "D1 Collection"
CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_MB = 100
DEFAULT_VIDEO_FILE_SIZE_MB = 200
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V21_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"
V21_STAT_KEYS = ("min", "max", "mean", "std", "count")
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}
STAT_QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))


def build_features(
    *,
    state_names: list[str],
    action_names: list[str],
    camera_key: str,
    image_height: int,
    image_width: int,
    use_videos: bool,
) -> dict:
    return {
        STATE_KEY: {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        ACTION_KEY: {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        ACTION_DELTA_KEY: {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
        camera_key: {
            "dtype": "video" if use_videos else "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=4, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_ready(value), ensure_ascii=False) + "\n")


def _feature_shape(feature: dict) -> tuple:
    return tuple(feature.get("shape") or ())


def _hf_features(features: dict):
    import datasets

    result = {}
    for key, feature in features.items():
        dtype = feature["dtype"]
        shape = _feature_shape(feature)
        if dtype == "video":
            continue
        if dtype == "image":
            result[key] = datasets.Image()
        elif shape == (1,):
            result[key] = datasets.Value(dtype=dtype)
        elif len(shape) == 1:
            result[key] = datasets.Sequence(
                length=shape[0],
                feature=datasets.Value(dtype=dtype),
            )
        elif len(shape) == 2:
            result[key] = datasets.Array2D(shape=shape, dtype=dtype)
        elif len(shape) == 3:
            result[key] = datasets.Array3D(shape=shape, dtype=dtype)
        elif len(shape) == 4:
            result[key] = datasets.Array4D(shape=shape, dtype=dtype)
        elif len(shape) == 5:
            result[key] = datasets.Array5D(shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Unsupported parquet feature: {key} {feature}")
    return datasets.Features(result)


def _array_stats(array: np.ndarray, *, count: int, image: bool = False) -> dict:
    values = np.asarray(array)
    if image:
        values = values.astype(np.float32) / 255.0
        values = values.reshape(-1, values.shape[-1])
    elif values.ndim == 1:
        values = values.reshape(-1, 1)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    values = values.astype(np.float64, copy=False)
    stats = {
        "min": minimum,
        "max": maximum,
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([count], dtype=np.int64),
    }
    quantiles = (
        None
        if len(values) < 2
        else np.quantile(values, [quantile for _, quantile in STAT_QUANTILES], axis=0)
    )
    for index, (key, _) in enumerate(STAT_QUANTILES):
        stats[key] = stats["mean"].copy() if quantiles is None else quantiles[index]
    if image:
        for key in stats:
            if key != "count":
                stats[key] = stats[key].reshape(-1, 1, 1)
    return stats


def _image_stats(paths: list[str]) -> dict:
    samples = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        if max(height, width) >= 300:
            stride = max(1, int(max(height, width) / 150))
            image = image[::stride, ::stride]
        samples.append(image.reshape(-1, 3))
    return _array_stats(np.concatenate(samples), count=len(paths), image=True)


def _episode_stats(buffer: dict, features: dict, precomputed: dict | None = None) -> dict:
    precomputed = precomputed or {}
    stats = {}
    for key, feature in features.items():
        if key in precomputed:
            stats[key] = precomputed[key]
            continue
        data = buffer[key]
        if feature["dtype"] in ("image", "video"):
            stats[key] = _image_stats(data)
        else:
            values = np.asarray(data, dtype=feature["dtype"])
            stats[key] = _array_stats(values, count=len(values))
    return stats


def _aggregate_stats(stats_list: list[dict]) -> dict:
    result = {}
    for feature_key in stats_list[0]:
        feature_stats = [stats[feature_key] for stats in stats_list]
        means = np.stack([stats["mean"] for stats in feature_stats])
        variances = np.stack([stats["std"] ** 2 for stats in feature_stats])
        counts = np.stack([stats["count"] for stats in feature_stats])
        total_count = counts.sum(axis=0)
        weights = counts
        while weights.ndim < means.ndim:
            weights = np.expand_dims(weights, axis=-1)
        mean = (means * weights).sum(axis=0) / total_count
        result[feature_key] = {
            "min": np.min(np.stack([stats["min"] for stats in feature_stats]), axis=0),
            "max": np.max(np.stack([stats["max"] for stats in feature_stats]), axis=0),
            "mean": mean,
            "std": np.sqrt(((variances + (means - mean) ** 2) * weights).sum(axis=0) / total_count),
            "count": total_count,
        }
        for key, quantile in STAT_QUANTILES:
            values = np.stack([stats[key] for stats in feature_stats])
            result[feature_key][key] = (
                np.min(values, axis=0) if quantile <= 0.5 else np.max(values, axis=0)
            )
    return result


def _write_image(image: np.ndarray, path: Path, compress_level: int) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, compress_level=compress_level)


class AsyncImageWriter:
    """Small Python 3.8-compatible equivalent of LeRobot's thread image writer."""

    def __init__(self, threads: int) -> None:
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
        self.futures = []

    def save_image(self, image: np.ndarray, path: Path, compress_level: int) -> None:
        self.futures.append(
            self.executor.submit(_write_image, np.asarray(image).copy(), path, compress_level)
        )

    def wait_until_done(self) -> None:
        for future in self.futures:
            future.result()
        self.futures.clear()

    def stop(self) -> None:
        self.wait_until_done()
        self.executor.shutdown()


_STREAM_END = object()


def _update_rgb_histogram(histogram: np.ndarray, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    stride = max(1, int(max(height, width) / 150)) if max(height, width) >= 300 else 1
    sampled = image[::stride, ::stride]
    for channel in range(3):
        histogram[channel] += np.bincount(sampled[..., channel].reshape(-1), minlength=256)


def _rgb_histogram_stats(histogram: np.ndarray, frame_count: int) -> dict:
    totals = histogram.sum(axis=1)
    if frame_count <= 0 or np.any(totals == 0):
        raise ValueError("Cannot compute video stats without frames")

    levels = np.arange(256, dtype=np.float64) / 255.0
    mean = (histogram * levels).sum(axis=1) / totals
    variance = (histogram * (levels - mean[:, None]) ** 2).sum(axis=1) / totals
    stats = {
        "min": np.argmax(histogram > 0, axis=1) / 255.0,
        "max": (255 - np.argmax(histogram[:, ::-1] > 0, axis=1)) / 255.0,
        "mean": mean,
        "std": np.sqrt(variance),
        "count": np.array([frame_count], dtype=np.int64),
    }
    cumulative = np.cumsum(histogram, axis=1)
    for key, quantile in STAT_QUANTILES:
        values = []
        for channel in range(3):
            rank = (totals[channel] - 1) * quantile
            low_rank, high_rank = math.floor(rank), math.ceil(rank)
            low = np.searchsorted(cumulative[channel], low_rank, side="right") / 255.0
            high = np.searchsorted(cumulative[channel], high_rank, side="right") / 255.0
            values.append(low + (high - low) * (rank - low_rank))
        stats[key] = np.asarray(values)
    for key in stats:
        if key != "count":
            stats[key] = np.asarray(stats[key]).reshape(-1, 1, 1)
    return stats


def _stream_video_worker(
    frame_queue,
    stop_event,
    result: dict,
    video_path: Path,
    args: argparse.Namespace,
) -> None:
    container = None
    try:
        import av

        histogram = np.zeros((3, 256), dtype=np.int64)
        frame_count = 0
        stream = None
        while not stop_event.is_set():
            try:
                image = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if image is _STREAM_END:
                break
            image = np.ascontiguousarray(image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected HWC RGB video frame, got {image.shape}")
            if stream is None:
                height, width = image.shape[:2]
                options = {"crf": str(args.video_crf), "g": str(args.video_gop)}
                if args.video_fast_decode:
                    options["tune"] = "fastdecode"
                if args.encoder_threads is not None:
                    options["threads"] = str(args.encoder_threads)
                container = av.open(str(video_path), "w", options={"movflags": "faststart"})
                stream = container.add_stream(args.video_codec, args.fps, options=options)
                stream.width = width
                stream.height = height
                stream.pix_fmt = args.video_pix_fmt
                stream.time_base = Fraction(1, args.fps)
            video_frame = av.VideoFrame.from_image(Image.fromarray(image))
            video_frame.pts = frame_count
            video_frame.time_base = Fraction(1, args.fps)
            for packet in stream.encode(video_frame):
                container.mux(packet)
            _update_rgb_histogram(histogram, image)
            frame_count += 1

        if stop_event.is_set():
            return
        if stream is None:
            raise ValueError("Cannot encode an empty video")
        for packet in stream.encode():
            container.mux(packet)
        result["value"] = (video_path, _rgb_histogram_stats(histogram, frame_count))
    except BaseException as exc:
        result["error"] = exc
    finally:
        if container is not None:
            try:
                container.close()
            except BaseException as exc:
                result.pop("value", None)
                result["error"] = exc


class StreamingVideoEncoder:
    """Encode each camera on a background thread while frames are captured."""

    def __init__(self, root: Path, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.jobs = {}

    def start_episode(self, video_keys: list[str]) -> None:
        if self.jobs:
            raise RuntimeError("A streaming video episode is already active")
        for key in video_keys:
            temp_dir = Path(tempfile.mkdtemp(dir=str(self.root)))
            video_path = temp_dir / f"{key.replace('/', '_')}_streaming.mp4"
            frame_queue = queue.Queue(maxsize=self.args.encoder_queue_maxsize)
            stop_event = threading.Event()
            result = {}
            thread = threading.Thread(
                target=_stream_video_worker,
                args=(frame_queue, stop_event, result, video_path, self.args),
                daemon=True,
                name=f"video-encoder-{key}",
            )
            thread.start()
            self.jobs[key] = {
                "queue": frame_queue,
                "stop": stop_event,
                "result": result,
                "thread": thread,
                "path": video_path,
            }

    @staticmethod
    def _put(job: dict, value) -> None:
        while job["thread"].is_alive():
            try:
                job["queue"].put(value, timeout=0.1)
                return
            except queue.Full:
                continue
        error = job["result"].get("error")
        raise RuntimeError(f"Streaming video encoder stopped: {error or 'unknown error'}") from error

    def feed_frame(self, key: str, image: np.ndarray) -> None:
        self._put(self.jobs[key], np.asarray(image, dtype=np.uint8).copy())

    def finish_episode(self) -> dict:
        jobs = self.jobs
        for job in jobs.values():
            self._put(job, _STREAM_END)
        for job in jobs.values():
            job["thread"].join(timeout=120)

        self.jobs = {}
        results = {}
        for key, job in jobs.items():
            if job["thread"].is_alive():
                job["stop"].set()
                raise RuntimeError(f"Streaming video encoder timed out for {key}")
            error = job["result"].get("error")
            if error is not None:
                raise RuntimeError(f"Streaming video encoder failed for {key}: {error}") from error
            results[key] = job["result"]["value"]
        return results

    def cancel_episode(self) -> None:
        jobs, self.jobs = self.jobs, {}
        for job in jobs.values():
            job["stop"].set()
        for job in jobs.values():
            job["thread"].join(timeout=5)
            if not job["thread"].is_alive():
                shutil.rmtree(job["path"].parent, ignore_errors=True)

    def close(self) -> None:
        self.cancel_episode()


def _video_duration(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration / av.time_base)


def _video_info(path: Path, args: argparse.Namespace) -> dict:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = getattr(stream.codec, "canonical_name", None) or stream.codec.name
        info = {
            "video.height": stream.height,
            "video.width": stream.width,
            "video.codec": codec,
            "video.pix_fmt": stream.pix_fmt,
            "video.fps": int(stream.base_rate),
            "video.channels": 1 if "gray" in stream.pix_fmt else 3,
            "has_audio": bool(container.streams.audio),
            "video.g": args.video_gop,
            "video.crf": args.video_crf,
            "video.preset": None,
            "video.fast_decode": args.video_fast_decode,
            "video.extra_options": {},
            "video.video_backend": "pyav",
            "is_depth_map": False,
        }
    return info


def _concatenate_videos(paths: list[Path], output_path: Path) -> None:
    import av

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as concat_file:
        concat_file.write("ffconcat version 1.0\n")
        for path in paths:
            concat_file.write(f"file '{path.resolve()}'\n")
        concat_path = Path(concat_file.name)

    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=str(output_path.parent), delete=False) as file:
        temp_output = Path(file.name)
    try:
        input_container = av.open(str(concat_path), "r", format="concat", options={"safe": "0"})
        output_container = av.open(str(temp_output), "w", options={"movflags": "faststart"})
        stream_map = {}
        for input_stream in input_container.streams:
            if input_stream.type in ("video", "audio", "subtitle"):
                if hasattr(output_container, "add_stream_from_template"):
                    output_stream = output_container.add_stream_from_template(input_stream, opaque=True)
                else:
                    output_stream = output_container.add_stream(template=input_stream)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    output_stream.time_base = input_stream.time_base
                stream_map[input_stream.index] = output_stream
        for packet in input_container.demux():
            if packet.dts is not None and packet.stream.index in stream_map:
                packet.stream = stream_map[packet.stream.index]
                output_container.mux(packet)
        input_container.close()
        output_container.close()
        temp_output.replace(output_path)
    finally:
        concat_path.unlink(missing_ok=True)
        temp_output.unlink(missing_ok=True)


class LocalLeRobotV21Dataset:
    """Local v2.1 writer using the same streaming encoder as the v3 writer."""

    def __init__(self, *, args: argparse.Namespace, features: dict) -> None:
        append = bool(getattr(args, "append", False))
        if args.output_root.exists() and not append:
            raise FileExistsError(f"output root already exists: {args.output_root}")
        self.args = args
        self.repo_id = args.repo_id
        self.root = args.output_root
        self.features = {**features, **DEFAULT_FEATURES}
        self.video_keys = [key for key, value in self.features.items() if value["dtype"] == "video"]
        self.image_keys = [key for key, value in self.features.items() if value["dtype"] == "image"]
        self.camera_keys = self.video_keys + self.image_keys
        new_info = {
            "codebase_version": "v2.1",
            "robot_type": args.robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 0,
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "fps": args.fps,
            "splits": {},
            "data_path": V21_DATA_PATH,
            "video_path": V21_VIDEO_PATH if args.use_videos else None,
            "features": self.features,
        }
        if self.root.exists():
            info_path = self.root / "meta" / "info.json"
            if not info_path.is_file():
                raise FileNotFoundError(f"cannot append without dataset info: {info_path}")
            self.info = json.loads(info_path.read_text(encoding="utf-8"))
            for key in ("codebase_version", "robot_type", "fps"):
                if _json_ready(self.info.get(key)) != _json_ready(new_info[key]):
                    raise ValueError(f"cannot append: dataset {key} does not match")
            saved_features = self.info.get("features", {})
            if set(saved_features) != set(self.features):
                raise ValueError("cannot append: dataset features do not match")
            for name, feature in self.features.items():
                saved = saved_features[name]
                for key in ("dtype", "shape", "names"):
                    if _json_ready(saved.get(key)) != _json_ready(feature.get(key)):
                        raise ValueError(f"cannot append: feature {name!r} {key} does not match")
            tasks_path = self.root / "meta" / "tasks.jsonl"
            tasks = [
                json.loads(line)
                for line in tasks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ] if tasks_path.is_file() else []
            self.tasks = {str(item["task"]): int(item["task_index"]) for item in tasks}
            episode_index = int(self.info["total_episodes"])
            for name in ("episodes.jsonl", "episodes_stats.jsonl"):
                path = self.root / "meta" / name
                count = (
                    sum(
                        bool(line.strip())
                        for line in path.read_text(encoding="utf-8").splitlines()
                    )
                    if path.is_file()
                    else 0
                )
                if count != episode_index:
                    raise ValueError(
                        f"cannot append: {path} has {count} entries, expected {episode_index}"
                    )
            next_data = self.root / V21_DATA_PATH.format(
                episode_chunk=self._episode_chunk(episode_index),
                episode_index=episode_index,
            )
            if next_data.exists():
                raise FileExistsError(f"refusing to overwrite next episode: {next_data}")
            for key in self.video_keys:
                next_video = self._video_path(episode_index, key)
                if next_video.exists():
                    raise FileExistsError(f"refusing to overwrite next episode: {next_video}")
        else:
            self.info = new_info
            self.tasks = {}
            self.root.mkdir(parents=True)
            _write_json(self.root / "meta" / "info.json", self.info)
        self._finalized = False
        self.image_writer = AsyncImageWriter(args.image_writer_threads) if args.image_writer_threads else None
        self.streaming_encoder = (
            StreamingVideoEncoder(self.root, args)
            if self.video_keys
            else None
        )
        self.episode_buffer = self._new_episode_buffer()

    def _new_episode_buffer(self) -> dict:
        buffer = {"size": 0, "task": []}
        for key in self.features:
            buffer[key] = self.info["total_episodes"] if key == "episode_index" else []
        return buffer

    @staticmethod
    def _episode_chunk(episode_index: int) -> int:
        return episode_index // DEFAULT_CHUNK_SIZE

    def _image_path(self, episode_index: int, key: str, frame_index: int) -> Path:
        return self.root / V21_IMAGE_PATH.format(
            image_key=key,
            episode_index=episode_index,
            frame_index=frame_index,
        )

    def _video_path(self, episode_index: int, key: str) -> Path:
        return self.root / V21_VIDEO_PATH.format(
            episode_chunk=self._episode_chunk(episode_index),
            video_key=key,
            episode_index=episode_index,
        )

    def _wait_images(self) -> None:
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def add_frame(self, frame: dict) -> None:
        frame = dict(frame)
        if "task" not in frame:
            raise ValueError("Frame is missing task")
        expected = set(self.features) - set(DEFAULT_FEATURES)
        actual = set(frame) - {"task"}
        if actual != expected:
            raise ValueError(f"Frame features mismatch: missing={expected - actual}, extra={actual - expected}")

        frame_index = self.episode_buffer["size"]
        episode_index = self.episode_buffer["episode_index"]
        if frame_index == 0 and self.streaming_encoder is not None:
            self.streaming_encoder.start_episode(self.video_keys)
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(frame_index / float(self.args.fps))
        self.episode_buffer["task"].append(frame.pop("task"))
        for key, value in frame.items():
            feature = self.features[key]
            if feature["dtype"] == "video":
                self.streaming_encoder.feed_frame(key, value)
                self.episode_buffer[key].append(None)
            elif feature["dtype"] == "image":
                path = self._image_path(episode_index, key, frame_index)
                path.parent.mkdir(parents=True, exist_ok=True)
                if self.image_writer is None:
                    _write_image(value, path, 6)
                else:
                    self.image_writer.save_image(value, path, 6)
                self.episode_buffer[key].append(str(path))
            else:
                array = np.asarray(value, dtype=feature["dtype"])
                if array.shape != _feature_shape(feature):
                    raise ValueError(f"{key} expected shape {_feature_shape(feature)}, got {array.shape}")
                self.episode_buffer[key].append(array)
        self.episode_buffer["size"] += 1

    def _task_index(self, task: str) -> int:
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
            _append_jsonl(
                self.root / "meta" / "tasks.jsonl",
                {"task_index": self.tasks[task], "task": task},
            )
        return self.tasks[task]

    def _write_episode_parquet(self, buffer: dict, episode_index: int) -> None:
        import datasets

        parquet_features = {
            key: feature
            for key, feature in self.features.items()
            if feature["dtype"] not in ("image", "video")
        }
        data = {}
        for key, feature in parquet_features.items():
            values = buffer[key]
            if _feature_shape(feature) == (1,):
                values = np.asarray(values, dtype=feature["dtype"]).reshape(-1)
            else:
                values = np.asarray(values, dtype=feature["dtype"])
            data[key] = values
        dataset = datasets.Dataset.from_dict(data, features=_hf_features(parquet_features), split="train")
        path = self.root / V21_DATA_PATH.format(
            episode_chunk=self._episode_chunk(episode_index),
            episode_index=episode_index,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(path)

    def _save_episode_videos(self, episode_index: int, streamed_videos: dict) -> None:
        for key in self.video_keys:
            video_path = self._video_path(episode_index, key)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            temp_video = streamed_videos[key][0]
            temp_dir = temp_video.parent
            temp_video.replace(video_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            if episode_index == 0:
                info = _video_info(video_path, self.args)
                info["video.is_depth_map"] = info.pop("is_depth_map")
                self.info["features"][key]["info"] = {
                    name: info[name]
                    for name in (
                        "video.height",
                        "video.width",
                        "video.codec",
                        "video.pix_fmt",
                        "video.is_depth_map",
                        "video.fps",
                        "video.channels",
                        "has_audio",
                    )
                }
            shutil.rmtree(self._image_path(episode_index, key, 0).parent, ignore_errors=True)

    def save_episode(self, metadata: dict | None = None) -> None:
        buffer = self.episode_buffer
        episode_length = buffer["size"]
        if episode_length <= 0:
            raise ValueError("Cannot save an empty episode")
        metadata = dict(metadata or {})
        reserved = {"episode_index", "tasks", "length"}
        overlap = reserved & set(metadata)
        if overlap:
            raise ValueError(f"episode metadata uses reserved keys: {sorted(overlap)}")
        self._wait_images()
        streamed_videos = self.streaming_encoder.finish_episode() if self.streaming_encoder else {}
        episode_index = int(buffer["episode_index"])
        episode_tasks = list(dict.fromkeys(buffer["task"]))
        buffer["index"] = np.arange(
            self.info["total_frames"],
            self.info["total_frames"] + episode_length,
            dtype=np.int64,
        )
        buffer["episode_index"] = np.full(episode_length, episode_index, dtype=np.int64)
        buffer["task_index"] = np.asarray(
            [self._task_index(task) for task in buffer["task"]],
            dtype=np.int64,
        )
        for key, feature in self.features.items():
            if feature["dtype"] not in ("image", "video"):
                buffer[key] = np.asarray(buffer[key], dtype=feature["dtype"])

        video_stats = {key: value[1] for key, value in streamed_videos.items()}
        stats = _episode_stats(buffer, self.features, video_stats)
        stats = {
            feature_key: {key: values[key] for key in V21_STAT_KEYS}
            for feature_key, values in stats.items()
        }
        self._write_episode_parquet(buffer, episode_index)
        self._save_episode_videos(episode_index, streamed_videos)
        if self.video_keys and not self.image_keys:
            shutil.rmtree(self.root / "images", ignore_errors=True)

        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        self.info["total_videos"] += len(self.video_keys)
        self.info["total_chunks"] = max(
            self.info["total_chunks"],
            self._episode_chunk(episode_index) + 1,
        )
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        _write_json(self.root / "meta" / "info.json", self.info)
        episode = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        _append_jsonl(self.root / "meta" / "episodes.jsonl", episode | metadata)
        _append_jsonl(
            self.root / "meta" / "episodes_stats.jsonl",
            {"episode_index": episode_index, "stats": stats},
        )
        self.episode_buffer = self._new_episode_buffer()

    def clear_episode_buffer(self) -> None:
        if self.streaming_encoder is not None:
            self.streaming_encoder.cancel_episode()
        self._wait_images()
        episode_index = int(self.episode_buffer["episode_index"])
        for key in self.camera_keys:
            shutil.rmtree(self._image_path(episode_index, key, 0).parent, ignore_errors=True)
        self.episode_buffer = self._new_episode_buffer()

    def has_pending_frames(self) -> bool:
        return self.episode_buffer["size"] > 0

    def finalize(self) -> None:
        if self._finalized:
            return
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None
        if self.streaming_encoder is not None:
            self.streaming_encoder.close()
            self.streaming_encoder = None
        self._finalized = True


class LocalLeRobotV3Dataset:
    """Minimal local LeRobot v3 dataset writer."""

    def __init__(self, *, args: argparse.Namespace, features: dict) -> None:
        import pyarrow.parquet as pq

        if args.output_root.exists():
            raise FileExistsError(f"output root already exists: {args.output_root}")
        self.args = args
        self.repo_id = args.repo_id
        self.root = args.output_root
        self.features = {**features, **DEFAULT_FEATURES}
        self.video_keys = [key for key, value in self.features.items() if value["dtype"] == "video"]
        self.image_keys = [key for key, value in self.features.items() if value["dtype"] == "image"]
        self.camera_keys = self.video_keys + self.image_keys
        self.info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": args.robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_MB,
            "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_MB,
            "fps": args.fps,
            "splits": {},
            "data_path": DEFAULT_DATA_PATH,
            "video_path": DEFAULT_VIDEO_PATH if args.use_videos else None,
            "features": self.features,
        }
        self.tasks = {}
        self.stats = None
        self._data_writer = None
        self._data_schema = None
        self._data_chunk_index = 0
        self._data_file_index = 0
        self._data_file_start_frame = 0
        self._episode_metadata_writer = None
        self._episode_metadata_schema = None
        self._episode_metadata_buffer = []
        self._episode_metadata_chunk_index = 0
        self._episode_metadata_file_index = 0
        self._video_state = {}
        self._finalized = False
        self.image_writer = AsyncImageWriter(args.image_writer_threads) if args.image_writer_threads else None
        self.streaming_encoder = (
            StreamingVideoEncoder(self.root, args)
            if self.video_keys
            else None
        )
        self.root.mkdir(parents=True)
        _write_json(self.root / "meta" / "info.json", self.info)
        self.episode_buffer = self._new_episode_buffer()
        self._pq = pq

    def _new_episode_buffer(self) -> dict:
        buffer = {"size": 0, "task": []}
        for key in self.features:
            buffer[key] = self.info["total_episodes"] if key == "episode_index" else []
        return buffer

    def _image_path(self, episode_index: int, key: str, frame_index: int) -> Path:
        return self.root / DEFAULT_IMAGE_PATH.format(
            image_key=key,
            episode_index=episode_index,
            frame_index=frame_index,
        )

    def _wait_images(self) -> None:
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def add_frame(self, frame: dict) -> None:
        frame = dict(frame)
        if "task" not in frame:
            raise ValueError("Frame is missing task")
        expected = set(self.features) - set(DEFAULT_FEATURES)
        actual = set(frame) - {"task"}
        if actual != expected:
            raise ValueError(f"Frame features mismatch: missing={expected - actual}, extra={actual - expected}")

        frame_index = self.episode_buffer["size"]
        episode_index = self.episode_buffer["episode_index"]
        if frame_index == 0 and self.streaming_encoder is not None:
            self.streaming_encoder.start_episode(self.video_keys)
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(frame_index / float(self.args.fps))
        self.episode_buffer["task"].append(frame.pop("task"))
        for key, value in frame.items():
            feature = self.features[key]
            if feature["dtype"] == "video":
                self.streaming_encoder.feed_frame(key, value)
                self.episode_buffer[key].append(None)
            elif feature["dtype"] == "image":
                path = self._image_path(episode_index, key, frame_index)
                path.parent.mkdir(parents=True, exist_ok=True)
                if self.image_writer is None:
                    _write_image(value, path, 6)
                else:
                    self.image_writer.save_image(value, path, 6)
                self.episode_buffer[key].append(str(path))
            else:
                array = np.asarray(value, dtype=feature["dtype"])
                if array.shape != _feature_shape(feature):
                    raise ValueError(f"{key} expected shape {_feature_shape(feature)}, got {array.shape}")
                self.episode_buffer[key].append(array)
        self.episode_buffer["size"] += 1

    def _save_tasks(self, tasks: list[str]) -> None:
        import pandas as pd

        changed = False
        for task in tasks:
            if task not in self.tasks:
                self.tasks[task] = len(self.tasks)
                changed = True
        if not changed:
            return
        frame = pd.DataFrame(
            {"task_index": list(self.tasks.values())},
            index=pd.Index(list(self.tasks), name="task"),
        )
        path = self.root / "meta" / "tasks.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    def _episode_table(self, buffer: dict):
        import datasets

        data = {}
        for key, feature in self.features.items():
            if feature["dtype"] == "video":
                continue
            values = buffer[key]
            if feature["dtype"] == "image":
                values = [{"bytes": Path(path).read_bytes(), "path": None} for path in values]
            elif _feature_shape(feature) == (1,):
                values = np.asarray(values, dtype=feature["dtype"]).reshape(-1)
            else:
                values = np.asarray(values, dtype=feature["dtype"])
            data[key] = values
        dataset = datasets.Dataset.from_dict(data, features=_hf_features(self.features), split="train")
        return dataset.with_format("arrow")[:]

    def _maybe_rotate_data_file(self, episode_length: int) -> None:
        if self._data_writer is None:
            return
        path = self.root / DEFAULT_DATA_PATH.format(
            chunk_index=self._data_chunk_index,
            file_index=self._data_file_index,
        )
        frames_in_file = self.info["total_frames"] - self._data_file_start_frame
        current_size = path.stat().st_size / 1024**2 if path.exists() else 0.0
        estimated = current_size * episode_length / frames_in_file if frames_in_file else 0.0
        if current_size + estimated < DEFAULT_DATA_FILE_SIZE_MB:
            return
        self._data_writer.close()
        self._data_writer = None
        self._data_file_index += 1
        if self._data_file_index >= DEFAULT_CHUNK_SIZE:
            self._data_chunk_index += 1
            self._data_file_index = 0
        self._data_file_start_frame = self.info["total_frames"]

    def _write_episode_data(self, buffer: dict, episode_length: int) -> dict:
        table = self._episode_table(buffer)
        self._maybe_rotate_data_file(episode_length)
        path = self.root / DEFAULT_DATA_PATH.format(
            chunk_index=self._data_chunk_index,
            file_index=self._data_file_index,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._data_writer is None:
            self._data_schema = table.schema
            self._data_writer = self._pq.ParquetWriter(
                path,
                schema=table.schema,
                compression="snappy",
                use_dictionary=True,
            )
        else:
            table = table.select(self._data_schema.names)
        self._data_writer.write_table(table)
        start = self.info["total_frames"]
        return {
            "data/chunk_index": self._data_chunk_index,
            "data/file_index": self._data_file_index,
            "dataset_from_index": start,
            "dataset_to_index": start + episode_length,
        }

    def _save_episode_video(
        self,
        key: str,
        episode_index: int,
        temp_video: Path,
    ) -> dict:
        temp_dir = temp_video.parent
        duration = _video_duration(temp_video)
        state = self._video_state.get(key)
        if state is None:
            chunk_index, file_index, from_timestamp = 0, 0, 0.0
        else:
            chunk_index, file_index, from_timestamp = state
        output_path = self.root / DEFAULT_VIDEO_PATH.format(
            video_key=key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        if output_path.exists() and (
            output_path.stat().st_size + temp_video.stat().st_size
        ) / 1024**2 >= DEFAULT_VIDEO_FILE_SIZE_MB:
            file_index += 1
            if file_index >= DEFAULT_CHUNK_SIZE:
                chunk_index += 1
                file_index = 0
            from_timestamp = 0.0
            output_path = self.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            _concatenate_videos([output_path, temp_video], output_path)
        else:
            temp_video.replace(output_path)
        if episode_index == 0:
            self.info["features"][key]["info"] = _video_info(output_path, self.args)
        shutil.rmtree(temp_dir)
        to_timestamp = from_timestamp + duration
        self._video_state[key] = (chunk_index, file_index, to_timestamp)
        return {
            f"videos/{key}/chunk_index": chunk_index,
            f"videos/{key}/file_index": file_index,
            f"videos/{key}/from_timestamp": from_timestamp,
            f"videos/{key}/to_timestamp": to_timestamp,
        }

    def _episode_metadata_path(self) -> Path:
        return self.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=self._episode_metadata_chunk_index,
            file_index=self._episode_metadata_file_index,
        )

    def _flush_episode_metadata(self) -> None:
        import pyarrow as pa

        if not self._episode_metadata_buffer:
            return
        table = pa.Table.from_pylist(self._episode_metadata_buffer)
        path = self._episode_metadata_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._episode_metadata_writer is None:
            self._episode_metadata_schema = table.schema
            self._episode_metadata_writer = self._pq.ParquetWriter(
                path,
                schema=table.schema,
                compression="snappy",
                use_dictionary=True,
            )
        else:
            table = table.select(self._episode_metadata_schema.names)
        self._episode_metadata_writer.write_table(table)
        self._episode_metadata_buffer.clear()

    def _maybe_rotate_episode_metadata(self) -> None:
        if self._episode_metadata_writer is None:
            return
        path = self._episode_metadata_path()
        if not path.exists() or path.stat().st_size / 1024**2 < DEFAULT_DATA_FILE_SIZE_MB:
            return
        self._flush_episode_metadata()
        self._episode_metadata_writer.close()
        self._episode_metadata_writer = None
        self._episode_metadata_file_index += 1
        if self._episode_metadata_file_index >= DEFAULT_CHUNK_SIZE:
            self._episode_metadata_chunk_index += 1
            self._episode_metadata_file_index = 0

    def _write_episode_metadata(self, episode: dict) -> None:
        self._maybe_rotate_episode_metadata()
        episode["meta/episodes/chunk_index"] = self._episode_metadata_chunk_index
        episode["meta/episodes/file_index"] = self._episode_metadata_file_index
        self._episode_metadata_buffer.append(episode)
        if len(self._episode_metadata_buffer) >= self.args.metadata_buffer_size:
            self._flush_episode_metadata()

    def save_episode(self) -> None:
        buffer = self.episode_buffer
        episode_length = buffer["size"]
        if episode_length <= 0:
            raise ValueError("Cannot save an empty episode")
        self._wait_images()
        streamed_videos = self.streaming_encoder.finish_episode() if self.streaming_encoder else {}
        episode_index = int(buffer["episode_index"])
        episode_tasks = list(dict.fromkeys(buffer["task"]))
        self._save_tasks(episode_tasks)
        buffer["index"] = np.arange(
            self.info["total_frames"],
            self.info["total_frames"] + episode_length,
            dtype=np.int64,
        )
        buffer["episode_index"] = np.full(episode_length, episode_index, dtype=np.int64)
        buffer["task_index"] = np.asarray(
            [self.tasks[task] for task in buffer["task"]],
            dtype=np.int64,
        )
        for key, feature in self.features.items():
            if feature["dtype"] not in ("image", "video"):
                buffer[key] = np.asarray(buffer[key], dtype=feature["dtype"])

        video_stats = {key: value[1] for key, value in streamed_videos.items()}
        stats = _episode_stats(buffer, self.features, video_stats)
        metadata = self._write_episode_data(buffer, episode_length)
        for key in self.video_keys:
            metadata.update(self._save_episode_video(key, episode_index, streamed_videos[key][0]))
        for key in self.image_keys:
            shutil.rmtree(self._image_path(episode_index, key, 0).parent, ignore_errors=True)

        episode = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
            **metadata,
        }
        for feature_key, feature_stats in stats.items():
            for stat_key, value in feature_stats.items():
                episode[f"stats/{feature_key}/{stat_key}"] = _json_ready(value)
        self._write_episode_metadata(episode)
        self.stats = stats if self.stats is None else _aggregate_stats([self.stats, stats])
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        _write_json(self.root / "meta" / "info.json", self.info)
        _write_json(self.root / "meta" / "stats.json", self.stats)
        self.episode_buffer = self._new_episode_buffer()

    def clear_episode_buffer(self) -> None:
        if self.streaming_encoder is not None:
            self.streaming_encoder.cancel_episode()
        self._wait_images()
        episode_index = int(self.episode_buffer["episode_index"])
        for key in self.camera_keys:
            shutil.rmtree(self._image_path(episode_index, key, 0).parent, ignore_errors=True)
        self.episode_buffer = self._new_episode_buffer()

    def has_pending_frames(self) -> bool:
        return self.episode_buffer["size"] > 0

    def finalize(self) -> None:
        if self._finalized:
            return
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None
        if self.streaming_encoder is not None:
            self.streaming_encoder.close()
            self.streaming_encoder = None
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        self._flush_episode_metadata()
        if self._episode_metadata_writer is not None:
            self._episode_metadata_writer.close()
            self._episode_metadata_writer = None
        self._finalized = True


def parse_manual_key_sequence(value: str | None) -> list[str]:
    if not value:
        return []
    aliases = {"space": " ", "enter": "\n", "return": "\n", "tab": "\t"}
    if "," in value:
        return [
            aliases.get(token.strip().lower(), token.strip())
            for token in value.split(",")
            if token.strip()
        ]
    return [aliases.get(char.lower(), char) for char in value]


class ManualKeyReader:
    def __init__(self, key_sequence: str | None = None) -> None:
        self.keys = parse_manual_key_sequence(key_sequence)
        self.old_termios = None

    def __enter__(self) -> "ManualKeyReader":
        if not self.keys and sys.stdin.isatty():
            self.old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.old_termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_termios)

    def get_key(self) -> str | None:
        if self.keys:
            return self.keys.pop(0)
        if not sys.stdin.isatty():
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if readable else None


def discard_episode(dataset) -> None:
    dataset.clear_episode_buffer()


def save_episode_if_valid(dataset, *, frame_count: int, min_frames: int, label: str) -> bool:
    if frame_count < min_frames:
        print(f"[{label}] discard short episode frames={frame_count} min={min_frames}")
        discard_episode(dataset)
        return False
    dataset.save_episode()
    print(f"[{label}] saved episode frames={frame_count}")
    return True


def safe_next_frame(source, args, consecutive_errors: int) -> tuple[dict | None, int, bool]:
    try:
        return source.next_frame(), 0, False
    except Exception as exc:
        consecutive_errors += 1
        print(
            "[warn] skipped frame after capture/control error "
            f"({consecutive_errors}/{args.max_consecutive_frame_errors}): {exc}"
        )
        stop = consecutive_errors >= args.max_consecutive_frame_errors
        if stop:
            print("[warn] too many consecutive frame errors; stopping collection")
        return None, consecutive_errors, stop


def show_video_frame(frame: dict | None, args: argparse.Namespace) -> str | None:
    if not args.show_video:
        return None

    import cv2

    try:
        if frame is not None:
            image = np.asarray(frame[args.camera_key], dtype=np.uint8)
            cv2.imshow(VIDEO_WINDOW_NAME, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        raise RuntimeError("Could not show video; use a GUI OpenCV build with DISPLAY set") from exc
    return None if key == 0xFF else chr(key)


def close_video_window(args: argparse.Namespace) -> None:
    if not args.show_video:
        return
    import cv2

    with contextlib.suppress(cv2.error):
        cv2.destroyWindow(VIDEO_WINDOW_NAME)


def run_manual_collection(args, source, dataset) -> None:
    ready_pose = None
    recording = False
    holding_initial_pose = False
    frame_count = 0
    saved_episodes = 0
    consecutive_errors = 0
    window_key = None
    period_s = 1.0 / args.fps
    print("[manual] controls: p=record pose, r=return pose, s=start, e=end/save, x=discard, q=quit")

    with ManualKeyReader(args.manual_key_sequence) as key_reader:
        while args.episodes is None or saved_episodes < args.episodes:
            started = time.perf_counter()
            key = key_reader.get_key() or window_key
            window_key = None

            if key in ("q", "\x03"):
                if recording:
                    print(f"[manual] discard unfinished episode frames={frame_count}")
                    discard_episode(dataset)
                break
            if key == "p":
                ready_pose = source.get_robot_pose()
                holding_initial_pose = False
                print("[manual] recorded initial pose")
            elif key == "r":
                if ready_pose is None:
                    print("[manual] no initial pose recorded; press p first")
                else:
                    print("[manual] moving to initial pose")
                    source.move_to_pose(ready_pose)
                    holding_initial_pose = True
                    print("[manual] reached initial pose; press s to start recording")
            elif key == "s":
                if recording:
                    print("[manual] already recording")
                else:
                    discard_episode(dataset)
                    source.flush_camera()
                    recording = True
                    holding_initial_pose = False
                    frame_count = 0
                    print(f"[manual] start episode {saved_episodes}")
            elif key == "e":
                if recording:
                    if save_episode_if_valid(
                        dataset,
                        frame_count=frame_count,
                        min_frames=args.min_frames_per_episode,
                        label="manual",
                    ):
                        saved_episodes += 1
                    recording = False
                    holding_initial_pose = False
                    frame_count = 0
                else:
                    print("[manual] not recording")
            elif key == "x":
                discard_episode(dataset)
                message = (
                    f"[manual] discard episode frames={frame_count}"
                    if recording
                    else "[manual] cleared episode buffer"
                )
                print(message)
                recording = False
                holding_initial_pose = False
                frame_count = 0

            if holding_initial_pose and not recording:
                frame = None
            else:
                frame, consecutive_errors, stop = safe_next_frame(source, args, consecutive_errors)
                if stop:
                    if recording:
                        if save_episode_if_valid(
                            dataset,
                            frame_count=frame_count,
                            min_frames=args.min_frames_per_episode,
                            label="manual",
                        ):
                            saved_episodes += 1
                    break

            window_key = show_video_frame(frame, args)
            if recording and frame is not None:
                frame["task"] = args.task
                dataset.add_frame(frame)
                frame_count += 1

            time.sleep(max(0.0, period_s - (time.perf_counter() - started)))

    print(f"[manual] saved episodes={saved_episodes}")


def iter_episode_frames(source, args, *, frames_per_episode: int) -> Iterator[dict]:
    period_s = 1.0 / args.fps
    consecutive_errors = 0
    collected = 0
    while collected < frames_per_episode:
        started = time.perf_counter()
        frame, consecutive_errors, stop = safe_next_frame(source, args, consecutive_errors)
        if stop:
            break
        if frame is not None:
            collected += 1
            yield frame
        time.sleep(max(0.0, period_s - (time.perf_counter() - started)))


def create_dataset(args: argparse.Namespace, features: dict):
    if args.force and args.output_root.exists():
        shutil.rmtree(args.output_root)

    print(
        f"[dataset] writer=local_lerobot_{args.dataset_format} "
        f"image_threads={args.image_writer_threads}"
    )
    writer = LocalLeRobotV21Dataset if args.dataset_format == "v2.1" else LocalLeRobotV3Dataset
    return writer(args=args, features=features)


def run_collection(args: argparse.Namespace) -> None:
    image_size = (args.image_height, args.image_width)
    with create_source(args, image_size=image_size) as source:
        dataset = create_dataset(
            args,
            build_features(
                state_names=source.state_names,
                action_names=source.action_names,
                camera_key=args.camera_key,
                image_height=args.image_height,
                image_width=args.image_width,
                use_videos=args.use_videos,
            ),
        )
        try:
            collection_metadata = build_collection_metadata(args, source=source)

            frames_per_episode = (
                args.frames_per_episode
                if args.frames_per_episode is not None
                else round(args.episode_seconds * args.fps)
            )
            if frames_per_episode <= 0:
                raise ValueError("frames_per_episode must be positive")

            if args.control_mode == "manual":
                run_manual_collection(args, source, dataset)
            else:
                stop_requested = False
                for episode_index in range(args.episodes if args.episodes is not None else 1):
                    print(f"[record] episode {episode_index} frames={frames_per_episode}")
                    source.flush_camera()
                    frame_count = 0
                    for frame in iter_episode_frames(source, args, frames_per_episode=frames_per_episode):
                        frame["task"] = args.task
                        dataset.add_frame(frame)
                        frame_count += 1
                        if show_video_frame(frame, args) == "q":
                            stop_requested = True
                            break
                    save_episode_if_valid(
                        dataset,
                        frame_count=frame_count,
                        min_frames=args.min_frames_per_episode,
                        label="record",
                    )
                    if stop_requested:
                        break
        finally:
            close_video_window(args)
            with contextlib.suppress(Exception):
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
            dataset.finalize()

        write_collection_metadata(
            args.output_root,
            collection_metadata,
        )

    print(f"[done] wrote LeRobot {args.dataset_format} dataset to {args.output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_robot_arguments(parser)
    parser.add_argument("--dataset-format", choices=["v2.1", "v3.0"], default="v3.0")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--task", default="Do the task.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-seconds", type=float, default=5.0)
    parser.add_argument("--frames-per-episode", type=int)
    parser.add_argument("--control-mode", choices=["fixed", "manual"], default="fixed")
    parser.add_argument("--show-video", action="store_true", help="Show captured frames; q stops collection")
    parser.add_argument("--min-frames-per-episode", type=int, default=1)
    parser.add_argument("--max-consecutive-frame-errors", type=int, default=10)
    parser.add_argument("--manual-key-sequence", help=argparse.SUPPRESS)
    parser.add_argument("--image-height", type=int, default=DEFAULT_RAW_IMAGE_SIZE[0])
    parser.add_argument("--image-width", type=int, default=DEFAULT_RAW_IMAGE_SIZE[1])
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument("--use-videos", dest="use_videos", action="store_true", default=True)
    video_group.add_argument("--no-use-videos", dest="use_videos", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-codec", default=DEFAULT_VIDEO_CODEC)
    parser.add_argument("--video-pix-fmt", default=DEFAULT_VIDEO_PIX_FMT)
    parser.add_argument("--video-crf", type=int, default=DEFAULT_VIDEO_CRF)
    parser.add_argument("--video-gop", type=int, default=DEFAULT_VIDEO_GOP)
    parser.add_argument("--video-fast-decode", type=int, default=0)
    parser.add_argument("--encoder-queue-maxsize", type=int, default=30)
    parser.add_argument("--encoder-threads", type=int)
    parser.add_argument("--metadata-buffer-size", type=int, default=10)
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.min_frames_per_episode <= 0:
        raise ValueError("--min-frames-per-episode must be positive")
    if args.max_consecutive_frame_errors <= 0:
        raise ValueError("--max-consecutive-frame-errors must be positive")
    if args.image_writer_threads < 0:
        raise ValueError("--image-writer-threads must be non-negative")
    if args.encoder_queue_maxsize <= 0:
        raise ValueError("--encoder-queue-maxsize must be positive")
    if args.encoder_threads is not None and args.encoder_threads <= 0:
        raise ValueError("--encoder-threads must be positive")
    if args.metadata_buffer_size <= 0:
        raise ValueError("--metadata-buffer-size must be positive")
    if min(args.image_height, args.image_width) <= 0:
        raise ValueError("image dimensions must be positive")
    validate_robot_args(args)
    if args.repo_id is None:
        args.repo_id = f"{DEFAULT_REPO_ID_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_BASE / args.repo_id.replace("/", "_")
    return args


def main() -> int:
    try:
        run_collection(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
