# CVWeAre2025 @ Zalo AI Challenge 2025
This repository contains implementation for the framework our team (CVWeAre) built for the Zalo AI Challenge 2025.

# Setup
- Please create a Python environment using Conda or python `venv`. Recommend using `uv` package manager for speed. Refer to `requirements.txt` for compatibility.
- Run `python download.py` to download base dataset.
- Download and place augmented reference images folder `augmented_ref_img` inside `data/observing_unzipped/train/`.
- Then, generate augmented videos using:

```bash
python run_pipeline.py \
    --stages augment \
    --train-samples data/observing_unzipped/train/samples \
    --train-annotations data/observing_unzipped/train/annotations/annotations.json \
    --val-samples data/observing_unzipped/train/samples \
    --val-annotations data/observing_unzipped/train/annotations/annotations.json \
    --augmented-dir data/augmented_train \
    --augment-preset default \
    --augment-start 0 \
    --tile-small-objects \
    --tile-scale 2.0 \
    --tile-min-area 0.02 \
    --tile-pad 8
```

- After having augmented video, please export them to `yolo_dataset` for compatibility with Ultralytics documentation using these 2 commands:

```bash
# Training split (original + augmented, with tiny-object tiles)
  python export_to_yolo.py \
    --source data/observing_unzipped/train/samples data/observing_unzipped/train/annotations/annotations.json \
    --source data/augmented_train/samples data/augmented_train/annotations/annotations_augmented.json \
    --data-config configs/yolo_data.yaml \
    --output-root data/yolo_dataset \
    --split train \
    --tile-small-objects \
    --tile-scale 2.0 \
    --tile-min-area 0.02 \
    --tile-pad 8
```

```bash
# Validation split (original only, no tiling)
  python export_to_yolo.py \
    --source data/observing_unzipped/train/samples data/observing_unzipped/train/annotations/annotations.json \
    --data-config configs/yolo_data.yaml \
    --output-root data/yolo_dataset \
    --split val
```

- Repeat for validation without augmented samples (omit the second `--source`, but keep `--data-config`).
- Verify `data/yolo_dataset/images/{train,val}` and `labels/{train,val}` exist and contain the same number of files.
- Edit `configs/yolo_data.yaml` if paths differ, and confirm `configs/yolo_hyp_small_objects.yaml` matches your hardware limits (batch size, mosaic schedule).
- Launch training using:
```bash
python train_yolo.py \
    --weights yolo11n.pt \
    --data configs/yolo_data.yaml \
    --cfg configs/yolo_hyp_small_objects.yaml \
    --imgsz 960 \
    --epochs 1 \
    --batch 32 \
    --device 0 \
    --project runs/train_smallobj_dryrun \
    --name-suffix <your_suffix>
```
Remember to replace <your_suffix> with an appropriate value.

- If you have Wandb account, you can login using your API key to save logs and metrics on Wandb platform, which is recommended.