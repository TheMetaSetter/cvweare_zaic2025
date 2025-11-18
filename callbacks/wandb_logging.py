"""Utilities for enabling Weights & Biases logging with Ultralytics YOLO."""

from __future__ import annotations

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
        loss_dict = trainer.label_loss_items(trainer.loss_items, prefix="train")
        epoch = getattr(trainer, "epoch", 0)
        batch_i = getattr(trainer, "batch_i", getattr(trainer, "ni", 0))
        nb = getattr(trainer, "nb", 1)
        step = epoch * nb + batch_i
        wandb.log({**loss_dict, "global_step": step, "epoch": epoch, "batch": batch_i}, step=step)

    model.add_callback("on_train_batch_end", log_batch_loss)
    return run
