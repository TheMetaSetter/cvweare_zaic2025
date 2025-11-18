# Execution Guidelines

This guide explains how to move from the current repo state to a fully trained
and evaluated ST-IoU pipeline. Follow each section in order; skip stages you
have already completed.

## 1. Generate Augmented Videos

1. Choose an augmentation preset (default, conservative, aggressive). For tiny
   objects, start with `default` to keep geometry stable.
2. Run `python run_pipeline.py --stages augment --train-samples data/.../samples \
   --train-annotations data/.../annotations/annotations.json --val-samples ...`
   (complete the remaining args as in the user guide). The script writes
   augmented clips and metadata into `data/augmented_train`.
3. If you have to resume, rerun with the same command; existing samples are
   skipped unless `--augment-overwrite` is added.

## 2. Export YOLO-Ready Frames + Labels

1. Ensure `export_to_yolo.py` knows about both original and augmented sources:
   ```bash
   python export_to_yolo.py \
       --source data/.../samples data/.../annotations/annotations.json \
       --source data/augmented_train/samples data/augmented_train/annotations/annotations_augmented.json \
       --data-config configs/yolo_data.yaml \
       --output-root data/yolo_dataset \
       --split train --tile-small-objects
   ```
2. Repeat for validation without augmented samples (omit the second `--source`, but keep `--data-config`).
3. Verify `data/yolo_dataset/images/{train,val}` and `labels/{train,val}` exist and contain the same number of files.

## 3. Fine-Tune YOLO11n/s

1. Edit `configs/yolo_data.yaml` if paths differ, and confirm
   `configs/yolo_hyp_small_objects.yaml` matches your hardware limits (batch
   size, mosaic schedule).
2. Launch training:
   ```bash
   python train_yolo.py --weights yolo11n.pt yolo11s.pt --imgsz 960 --epochs 200 \
       --data configs/yolo_data.yaml --cfg configs/yolo_hyp_small_objects.yaml
   ```
3. Monitor `runs/train_smallobj/<model>_st_iou/` for loss curves and ensure
   `weights/best.pt` appears for each model.
4. **Kaggle 2×T4:** Set `MASTER_ADDR/PORT`, then launch with
   `torchrun --nproc_per_node=2 train_yolo.py ...`. The scripts auto-read
   `LOCAL_RANK` so each worker trains on a dedicated GPU while only rank 0 logs
   to W&B and runs validation.

## 4. Evaluate with ST-IoU Pipeline

1. Pick the checkpoint to score (best-performing `best.pt`).
2. Run:
   ```bash
   python evaluation/st_iou_pipeline.py \
       --model runs/train_smallobj/yolo11s_st_iou/weights/best.pt \
       --videos-root data/observing_unzipped/train/samples \
       --annotations data/observing_unzipped/train/annotations/annotations.json \
       --output runs/st_iou/preds_yolo11s.json
   ```
3. Review console logs for per-video ST-IoU and check the summary mean.

## 5. Orchestrate Everything End-to-End

1. After configuring paths once, you can let `run_pipeline.py` perform all
   stages: `python run_pipeline.py --stages augment export train evaluate ...`.
2. To iterate on a single stage, pass a subset, e.g. `--stages train evaluate`
   after new augmentations are already in place.
3. Keep a log of key parameters (imgsz, thresholds, smoothing factors) alongside
   the ST-IoU score so you can compare experiments quickly.

## 6. Optional: Log Training to W&B

1. Install W&B (`pip install wandb`), set `WANDB_API_KEY`, and run `wandb login`
   (or rely on `--wandb-offline` to cache logs locally).
2. Enable logging by passing `--log-wandb --wandb-project <name>` to either
   `train_yolo.py` or `run_pipeline.py`; optionally customize
   `--wandb-run-prefix` for tidy dashboards.
3. The scripts now use `add_wandb_callback` plus a custom `on_train_batch_end`
   hook so every batch logs `train/box_loss`, `train/cls_loss`, etc., giving you
   step-level curves in the W&B UI.
