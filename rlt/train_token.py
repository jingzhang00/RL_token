"""Cache frozen π0.5 features, then train the RL token in a separate process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from rlt.models import RLToken


DEFAULT_PROMPT = "Pick up the doll and place it on the plate"


def _episode_frames(dataset: Path, prompt: str = DEFAULT_PROMPT):
    import cv2
    import pyarrow.parquet as parquet

    for table_path in sorted((dataset / "data").glob("chunk-*/episode_*.parquet")):
        table = parquet.read_table(table_path, columns=["observation.state"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        chunk = table_path.parent.name
        video = (
            dataset
            / "videos"
            / chunk
            / "observation.images.d1_rgb"
            / f"{table_path.stem}.mp4"
        )
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {video}")
        try:
            for index, state in enumerate(states):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"video {video} ended at frame {index}/{len(states)}"
                    )
                yield {
                    "observation/image": np.ascontiguousarray(bgr[:, :, ::-1]),
                    "observation/state": state,
                    "prompt": prompt,
                }
        finally:
            capture.release()


def cache(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from rlt.openpi_adapter import OpenPIAdapter
    from rlt.settings import load_config

    settings = load_config(args.config)
    dataset = Path(args.dataset or settings.vla["dataset"]).resolve()
    checkpoint = Path(args.checkpoint or settings.vla_checkpoint)
    output = Path(args.output or settings.rl_token["cache_output"])
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else settings.rl_token.get(
            "cache_batch_size", settings.rl_token["batch_size"]
        )
    )
    seed = int(args.seed if args.seed is not None else settings.rl_token["seed"])
    total = sum(
        __import__("pyarrow.parquet", fromlist=["read_metadata"])
        .read_metadata(path)
        .num_rows
        for path in (dataset / "data").glob("chunk-*/*.parquet")
    )
    output.mkdir(parents=True, exist_ok=True)
    adapter = OpenPIAdapter(
        checkpoint,
        prompt=settings.vla["prompt"],
        seed=seed,
        config_path=args.config,
    )
    arrays = None
    offset = 0
    pending = []
    with tqdm(total=total, desc="Caching VLA features", unit="frame") as progress:
        for observation in _episode_frames(dataset, settings.vla["prompt"]):
            pending.append(observation)
            if len(pending) < batch_size:
                continue
            count = len(pending)
            offset, arrays = _write_cache_batch(
                adapter.infer(pending), output, total, offset, arrays
            )
            pending.clear()
            progress.update(count)
        if pending:
            count = len(pending)
            offset, arrays = _write_cache_batch(
                adapter.infer(pending), output, total, offset, arrays
            )
            progress.update(count)
    if offset != total:
        raise RuntimeError(f"cached {offset} frames, expected {total}")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "frames": total,
                "checkpoint": str(checkpoint.resolve()),
                "dataset": str(dataset),
                "prompt": settings.vla["prompt"],
                "seed": seed,
            },
            indent=2,
        )
    )


def _write_cache_batch(result, output: Path, total: int, offset: int, arrays):
    count, token_count, width = result["tokens"].shape
    if arrays is None:
        open_memmap = np.lib.format.open_memmap
        arrays = {
            "tokens": open_memmap(
                output / "tokens.npy",
                mode="w+",
                dtype=np.float16,
                shape=(total, token_count, width),
            ),
            "valid": open_memmap(
                output / "valid.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(total, token_count),
            ),
            "reference": open_memmap(
                output / "reference.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total, *result["reference"].shape[1:]),
            ),
            "absolute": open_memmap(
                output / "absolute.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total, *result["absolute"].shape[1:]),
            ),
            "state": open_memmap(
                output / "state.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total, *result["normalized_state"].shape[1:]),
            ),
        }
    end = offset + count
    for key in arrays:
        source = result["normalized_state"] if key == "state" else result[key]
        arrays[key][offset:end] = source
        arrays[key].flush()
    return end, arrays


def train(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from rlt.settings import load_config

    settings = load_config(args.config)
    cache_path = Path(args.cache or settings.rl_token["cache_output"])
    output_path = Path(args.output or settings.rl_token["output"])
    steps = int(args.steps if args.steps is not None else settings.rl_token["steps"])
    batch_size = int(
        args.batch_size if args.batch_size is not None else settings.rl_token["batch_size"]
    )
    lr = float(args.lr if args.lr is not None else settings.rl_token["learning_rate"])
    save_every = int(
        args.save_every if args.save_every is not None else settings.rl_token["save_every"]
    )
    log_interval = settings.log_interval
    seed = int(args.seed if args.seed is not None else settings.rl_token["seed"])
    device_name = args.device or settings.rl_token["device"]
    args.seed, args.lr, args.batch_size = seed, lr, batch_size
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokens = np.load(cache_path / "tokens.npy", mmap_mode="r")
    valid = np.load(cache_path / "valid.npy", mmap_mode="r")
    width, token_count = tokens.shape[-1], tokens.shape[-2]
    device = torch.device(device_name)
    heads = int(settings.rl_token["heads"])
    layers = int(settings.rl_token["layers"])
    mlp_ratio = int(settings.rl_token["mlp_ratio"])
    model = RLToken(
        width=width,
        heads=heads,
        layers=layers,
        mlp_ratio=mlp_ratio,
        max_tokens=token_count,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    wandb_run = None
    if settings.wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=settings.project_name,
            name=settings.rl_token.get("exp_name", "rlt_token"),
            config={
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": lr,
                "save_every": save_every,
                "seed": seed,
                "device": str(device),
                "model": {"width": width, "heads": heads, "layers": layers, "mlp_ratio": mlp_ratio},
                "vla_output": str(settings.vla["output"]),
            },
        )
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
    output_path.mkdir(parents=True, exist_ok=True)
    args.model_config = {
        "width": width,
        "heads": heads,
        "layers": layers,
        "mlp_ratio": mlp_ratio,
        "max_tokens": token_count,
    }
    with tqdm(
        total=steps,
        initial=start,
        desc="Training RL token",
        unit="step",
    ) as progress:
        for step in range(start + 1, steps + 1):
            indices = np.random.randint(0, len(tokens), size=batch_size)
            batch_tokens = torch.from_numpy(
                np.asarray(tokens[indices], dtype=np.float32)
            ).to(device)
            batch_valid = torch.from_numpy(np.asarray(valid[indices])).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                _, _, loss = model(batch_tokens, batch_valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % log_interval == 0 or step == steps:
                loss_value = float(loss)
                progress.set_postfix(reconstruction_mse=f"{loss_value:.6f}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "reconstruction_mse": loss_value,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        step=step,
                    )
            if step % save_every == 0 or step == steps:
                _save_token(
                    output_path / f"step_{step}.pt",
                    model,
                    optimizer,
                    step,
                    args,
                    token_count,
                    width,
                )
            progress.update(1)
    if wandb_run is not None:
        wandb_run.finish()


def _save_token(path, model, optimizer, step, args, token_count, width):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": {
                **args.model_config,
            },
            "metadata": {
                "seed": args.seed,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "alpha": 0,
                "vla_frozen": True,
                "vla": str(args.config or "rlt/config/config.yaml"),
            },
        },
        path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache_parser = sub.add_parser("cache")
    cache_parser.add_argument("--config", type=Path)
    cache_parser.add_argument("--dataset", type=Path)
    cache_parser.add_argument("--checkpoint", type=Path)
    cache_parser.add_argument("--output", type=Path)
    cache_parser.add_argument("--batch-size", type=int)
    cache_parser.add_argument("--seed", type=int)
    cache_parser.set_defaults(function=cache)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", type=Path)
    train_parser.add_argument("--cache", type=Path)
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument("--steps", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--save-every", type=int)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--device")
    train_parser.add_argument("--resume", type=Path)
    train_parser.set_defaults(function=train)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    arguments.function(arguments)
