"""Train YOLO11-n/s with small-object-friendly defaults.

This script keeps configuration in version control so experiments can reference
exact img sizes, hyperparameters, and naming conventions mentioned in
prepare.plan.md.
"""

import argparse
from pathlib import Path
from typing import List

from ultralytics import YOLO

from callbacks.wandb_logging import attach_wandb_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 on drone dataset")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["yolo11n.pt", "yolo11s.pt"],
        help="List of pretrained checkpoints to fine-tune",
    )
    parser.add_argument("--data", type=str, default="configs/yolo_data.yaml", help="YOLO data.yaml path")
    parser.add_argument("--cfg", type=str, default="configs/yolo_hyp_small_objects.yaml", help="Hyperparameter yaml")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help="CUDA device string or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train_smallobj", help="Ultralytics project dir")
    parser.add_argument("--name-suffix", type=str, default="st_iou", help="Suffix appended to run name")
    parser.add_argument("--close-mosaic", type=int, default=20, help="Epoch to disable mosaic in Ultralytics trainer")
    parser.add_argument("--log-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="Zalo AI Challenge 2025", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-prefix", type=str, default="st_iou", help="Prefix for W&B run names")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for weights in args.weights:
        model = YOLO(weights)
        base_name = Path(weights).stem
        run_name = f"{base_name}_{args.name_suffix}"
        wandb_run = None
        if args.log_wandb:
            wandb_run = attach_wandb_logging(
                model,
                project=args.wandb_project,
                run_name=f"{args.wandb_run_prefix}-{run_name}",
                offline=args.wandb_offline,
            )
        print(f"\n[TRAIN] {base_name} → imgsz={args.imgsz}, epochs={args.epochs}")
        model.train(
            data=args.data,
            cfg=args.cfg,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=run_name,
            close_mosaic=args.close_mosaic,
            val=True,
        )
        # Persist validation metrics after training to capture latest checkpoint performance.
        metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
        print(f"[VAL] {base_name}: mAP50-95={metrics.box.map:.4f} mAP50={metrics.box.map50:.4f}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
