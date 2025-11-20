"""Utilities for enabling Weights & Biases logging with Ultralytics YOLO."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional


def attach_wandb_logging(
    model, project: str, run_name: str, offline: bool = False, *, rank: int = 0
) -> Optional["wandb.wandb_run.Run"]:
    """Route YOLO training metrics into W&B after Ultralytics enables its built-in callback.

    The official YOLO11 workflow spins up W&B internally once `yolo settings wandb=True`
    is set, so this helper focuses on configuring offline mode, naming, and the custom
    per-batch callback that feeds fine-grained losses into the same run."""

    if rank != 0:
        # Keeping only rank-0 logging avoids duplicated W&B streams under torch.distributed.
        return None

    import wandb
    from ultralytics.utils import SETTINGS

    # Toggle Ultralytics' native W&B integration rather than calling the legacy
    # wandb.integration.ultralytics hook that crashes on YOLO11 (missing RANK global).
    SETTINGS.update({"wandb": True})

    # Respect offline runs without forcing wandb.init(); Ultralytics will read these
    # env vars when it creates the run at training start.
    if offline:
        os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_NAME", run_name)

    # If Ultralytics already opened a run we reuse it, otherwise this stays None
    # until training kicks off and the callback boots wandb up.
    run = wandb.run

    def log_batch_loss(trainer):
        """Log loss terms for every batch to match Ultralytics' naming."""

        if not hasattr(trainer, "loss_items"):
            return

        # Build the per-batch loss dictionary once so both W&B and local CSV
        # can see a consistent view of training dynamics.
        loss_dict = trainer.label_loss_items(trainer.loss_items, prefix="train")
        epoch = getattr(trainer, "epoch", 0)
        batch_i = getattr(trainer, "batch_i", getattr(trainer, "ni", 0))
        nb = getattr(trainer, "nb", 1)
        step = epoch * nb + batch_i

        payload = {
            **loss_dict,
            "global_step": step,
            "epoch": epoch,
            "batch": batch_i,
        }

        # Only log when Ultralytics has brought up a W&B run; this lets dry runs
        # or offline smoke tests skip the dependency silently.
        if wandb.run is not None:
            wandb.log(payload, step=step)

        # Also persist the same payload to a local CSV so users can draw
        # curves post-hoc even without syncing to W&B.
        save_dir = Path(getattr(trainer, "save_dir", "")) if hasattr(trainer, "save_dir") else None
        if save_dir:
            csv_path = save_dir / "batch_metrics.csv"
            write_header = not csv_path.exists()
            try:
                with csv_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(payload.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(payload)
            except OSError:
                # If disk writes fail, keep training and W&B logging unaffected.
                pass

    model.add_callback("on_train_batch_end", log_batch_loss)
    return run
