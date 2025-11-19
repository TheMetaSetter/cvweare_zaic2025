"""Utilities for enabling Weights & Biases logging with Ultralytics YOLO."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


def attach_wandb_logging(
    model, project: str, run_name: str, offline: bool = False, *, rank: int = 0
) -> Optional["wandb.wandb_run.Run"]:
    """Attach W&B logging (epoch + per-step losses) to a YOLO model."""

    if rank != 0:
        # Only rank 0 creates a run so DDP launches on Kaggle do not spawn duplicates.
        return None

    import wandb
    from wandb.integration.ultralytics import add_wandb_callback

    mode = "offline" if offline else None
    run = wandb.init(project=project, name=run_name, mode=mode)
    add_wandb_callback(model, enable_model_checkpointing=True)

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

        # Send step-wise metrics to W&B exactly as before so existing
        # dashboards and offline logging workflows remain unchanged.
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
