## Plan to prepare YOLO11n/s fine-tuning with ST-IoU

### 1. Decide training & evaluation pipeline style

- Use Ultralytics YOLO (the `yolo` CLI / Python API) as the training engine for YOLO11n/s so you get mAP/precision/recall for free.
- Keep your current video augmentation code as an **offline/on-the-fly preprocessor** that creates small-object‑friendly frames, but let YOLO handle the core training loop.
- Design the *evaluation* as a **detection → tracking → ST-IoU** pipeline so all major decisions are made with ST‑IoU in mind, not just per-frame mAP.

### 2. Prepare a tiny‑object‑oriented augmented dataset (video → frames → labels)

- Use `AugmentedVideoGenerator` / `demo_augmentation.py` to generate an `augmented_train` set, starting with a **conservative or slightly-tuned default config**:
- keep rotations modest (≲10°) and scale range roughly 0.8–1.2 so tiny objects are not destroyed;
- keep pixel-level augments strong but temporally smooth (as already implemented).
- Write a conversion script that:
- reads `annotations.json` / `annotations_augmented.json` and all frames (original + augmented);
- exports one image per frame into `images/train/` and `images/val/` (optionally at higher resolution, e.g. 960–1280 on the long side);
- converts each bbox to YOLO TXT format (`class x_center y_center width height` in [0, 1]) and writes one label file per frame into `labels/train/` and `labels/val/`.
- Ensure **all frames where the GT object exists** are preserved (avoid sampling that drops near-static or very small frames), and consider:
- oversampling videos / frames where the object is extremely tiny;
- optionally generating additional “zoomed” crops/tiles around the object (SAHI-style) so YOLO also sees the object at 1–3% image size.
- Fix a single mapping from class names (Backpack, Laptop, Lifering, etc.) to integer IDs and reuse it in both the converter and YOLO `data.yaml`.

### 3. Define YOLO11 configs tuned for small UAV objects

- Create a `data.yaml` that points to `images/{train,val}` and `labels/{train,val}` and lists class names in the agreed order.
- Start from Ultralytics defaults but adjust for tiny objects:
- **Image size**: use higher `imgsz` (e.g., 800–960 for YOLO11n, 960–1280 for YOLO11s) as long as GPU memory allows; consider enabling multi-scale.
- **Loss weights**: slightly increase `box` and `dfl` loss weights so localization matters more (a common small-object trick).
- **Augment settings**: either disable the most aggressive internal augmentations (e.g. heavy mosaic) or keep them mild, since you already have a rich video augmenter.
- **Training schedule**: 150–300 epochs with early stopping; optionally use a curriculum where the last 10–20 epochs run with reduced augmentation to refine box quality.
- Prepare two nearly identical training commands (or Python calls): one fine-tunes `yolo11n.pt`, the other `yolo11s.pt`, both using the same data and hyperparameters.

### 4. Implement an ST-IoU‑oriented detection → tracking → smoothing pipeline

- Build a module (e.g., `evaluation/st_iou_pipeline.py`) that, for each validation video:
- runs YOLO11n/s on each frame to obtain detections (boxes + class + confidence);
- feeds detections into a **lightweight tracker** (SORT, ByteTrack, or a simpler single-object variant), keeping only the main track: the longest, highest-confidence track for the target class;
- applies **temporal smoothing** to the track (e.g., exponential moving average or Kalman filter on center + width/height) to reduce jitter;
- **fills short gaps** (≤3–5 frames) between two track segments by linear interpolation of box coordinates;
- **trims** low-confidence prefixes/suffixes so the final tubelet roughly matches the object’s true visibility interval.
- Implement ST‑IoU exactly as in your note:
- for each ground-truth tubelet \(B_f\) and candidate predicted tubelet \(B'_f\) of the same class, compute per-frame IoU, define `intersection` and `union` frame sets, then
\[ \text{STIoU} = \frac{1}{|\text{union}|} \sum_{f \in \text{intersection}} \text{IoU}(B_f, B'_f). \]
- in your setting (likely a single object per video), simply choose the predicted tubelet with highest ST‑IoU as the video’s prediction.
- Expose key knobs—detection confidence threshold, tracker association IoU, minimum track length, smoothing factor, interpolation window—so you can tune them **explicitly against ST‑IoU** on a validation split.

### 5. Connect everything into a repeatable ST-IoU‑driven workflow

- Define a clear loop that you can iterate on:

1. Generate / update augmented videos using a small-object‑aware config.
2. Export them to YOLO `images` + `labels` with all GT frames preserved and extra crops/tiles where needed.
3. Fine-tune YOLO11n with higher image size and adjusted loss/augment settings; log mAP and qualitative examples.
4. Run the detection → tracking → smoothing → ST‑IoU pipeline on validation videos; store ST‑IoU per video and per class.
5. Adjust (a) augmentation strength and tiling strategy, (b) YOLO hyperparameters (imgsz, loss weights, thresholds), and (c) tracker/smoothing parameters, always measuring ST‑IoU before and after.
6. Once YOLO11n reaches a satisfactory ST‑IoU / runtime balance, repeat training and evaluation with YOLO11s and compare.

- Keep the final exporter aligned with the challenge’s JSON tubelet format so the same tracked, smoothed tubelets can be used both for local ST‑IoU evaluation and for competition submission.

### Updated implementation todos

- prep-augmented-data: choose and run a small-object‑friendly augmentation config; freeze train/val video lists.
- export-yolo-format: implement the frame + YOLO TXT exporter, ensuring all GT frames are kept and optionally adding zoomed crops/tiles.
- define-yolo-config: create `data.yaml` and YOLO11n/s hyperparameters with higher `imgsz` and boosted box/dfl loss weights.
- train-yolo11n-s: run YOLO11n and YOLO11s training with the prepared data/config, saving per-frame predictions on the val set.
- implement-st-iou-metric: build the detection → tracking → smoothing → ST‑IoU module and match predicted vs GT tubelets.
- evaluate-and-iterate: tune augment, YOLO hyperparams, and tracker settings in short cycles, always watching ST‑IoU as the main metric.