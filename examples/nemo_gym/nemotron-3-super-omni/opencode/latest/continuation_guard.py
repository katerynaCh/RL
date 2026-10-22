# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refuse a continuation without a published checkpoint; skip completed targets."""

import argparse
import json
import re
from pathlib import Path


def continuation_checkpoint(
    checkpoint_dir: Path, *, target_step: int
) -> tuple[Path, bool]:
    """Use the same published step directories as CheckpointManager's restore."""
    checkpoints = [
        p
        for p in checkpoint_dir.glob("step_*")
        if p.is_dir() and re.fullmatch(r"step_\d+", p.name)
    ]
    if not checkpoints:
        raise ValueError(
            f"No published checkpoint in {checkpoint_dir}; refusing a fresh start"
        )
    latest = max(checkpoints, key=lambda p: int(p.name.split("_")[1]))
    step = int(latest.name.split("_")[1])
    info = json.loads((latest / "training_info.json").read_text())
    if info["current_step"] != step:
        raise ValueError(f"Checkpoint step mismatch in {latest}")
    for name in (
        "config.yaml",
        "train_dataloader.pt",
        "policy/weights",
        "replay_buffer_metadata.pt",
        "rollout_recovery.pt",
        "data_plane",
    ):
        if not (latest / name).exists():
            raise ValueError(f"Published checkpoint is missing {name}: {latest}")
    return latest, step >= target_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--target-step", type=int, default=200)
    args = parser.parse_args()
    checkpoint, complete = continuation_checkpoint(
        args.checkpoint_dir, target_step=args.target_step
    )
    print(f"{'complete' if complete else 'resume'} {checkpoint}")
