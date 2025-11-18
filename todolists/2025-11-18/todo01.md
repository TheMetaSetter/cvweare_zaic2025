## Next Steps To-Do List

1. **Enforce comment standards in existing augmentation code**
   - Update `augmentation/config.py` so each `field(default_factory=...)` member has the intent-driven comments AGENTS.md demands (why we defer instantiation, how it keeps configs serializable).
   - Expand the dataset indexing/frame-loading comments in `augmentation/dataset.py` to explain how `idx` splits into `base_idx`/`aug_idx`, why only bbox-bearing frames are loaded, and what the returned dict must contain.
   - Rework `_apply_copy_paste` in `augmentation/video_augmenter.py` with the mandated header plus inline reasoning that covers PNG loading, scaling limits, jitter, alpha blending, bbox emission, and skip conditions.

2. **Implement the YOLO-format exporter**
   - Extend `AugmentedVideoGenerator` or add a new script that walks both original and generated samples, writing frames to `images/{train,val}` and YOLO TXT labels to `labels/{train,val}` while guaranteeing every GT frame is preserved.
   - Include optional zoomed crops/tiles (SAHI-style) around tiny objects per `howtopushperformance.md`, and log how oversampling decisions are made.
   - Produce a stable class→ID mapping shared by annotations and YOLO’s `data.yaml`.

3. **Author YOLO11 training configs tuned for tiny UAV targets**
   - Create `data.yaml` pointing at the exported folders with ordered class names.
   - Craft hyperparameter overrides (higher `imgsz`, boosted `box`/`dfl`, toned internal augments) and document the rationale inline so others can reproduce.
   - Provide launch scripts or CLI snippets for fine-tuning both `yolo11n.pt` and `yolo11s.pt` with identical data but separate checkpoints.

4. **Build the ST-IoU evaluation pipeline**
   - Add a module (e.g., `evaluation/st_iou_pipeline.py`) that runs YOLO detections on validation videos, tracks them via SORT/ByteTrack, smooths trajectories, fills short gaps, and trims low-confidence tails.
   - Implement the ST-IoU metric exactly as described in `howtopushperformance.md`, emitting per-video/per-class scores and writing final tubelets to the challenge JSON format.

5. **Wire an end-to-end experiment loop**
   - Script the sequence “augment → export YOLO format → train YOLO11n/s → run ST-IoU eval → log results” so a single command or notebook executes the workflow from raw videos to metrics.
   - Capture knobs for augmentation strength, YOLO hyperparameters, and tracker thresholds to enable quick what-if sweeps aimed at maximizing ST-IoU.
