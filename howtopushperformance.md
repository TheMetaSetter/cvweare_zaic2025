To push ST-IoU for YOLO11-n/s, you need to:

1. make frame-wise tiny-object detection as strong as possible, and
2. turn those detections into **smooth, correctly trimmed tubelets** (tracks) that match the ground-truth in both space *and* time.

---

### 1. Re-read the metric and what it rewards

You have ground-truth tubelet (B_f) and predicted tubelet (B'*f) over frames (f). ST-IoU is basically
[
\text{STIoU}=\frac{1}{|\text{union}|}\sum*{f\in\text{intersection}} \text{IoU}(B_f, B'_f),
]
where “intersection” is frames where both GT and prediction exist, and “union” is frames where **either** exists.

So you gain ST-IoU by:

* having **detections on as many GT frames as possible** (large intersection set),
* **not hallucinating boxes** when the object is absent (don’t enlarge union),
* and having **high per-frame IoU** where the object is visible.

Keep this picture in your head: you want one clean, stable tube per video, covering exactly the right time span.

---

### 2. Make YOLO11-n/s more “tiny-object-friendly”

From UAV / small-object YOLO variants (SOD-YOLO, ST-YOLO, etc.), the consistent tricks are: higher-res inputs, extra high-resolution feature maps, and strong data augmentation designed for tiny objects.

Given your constraints and current pipeline:

#### 2.1. Input resolution and tiling

* Train and infer at **higher image size** than default:

  * e.g. imgsz = 960 or 1280 for YOLO11-s, maybe 800–960 for YOLO11-n so Jetson NX stays real-time.
* For frames where the object is extremely small, use **tiling at train time** (and possibly at test time) ─ this is exactly what SAHI-style tiling does for YOLO on small objects.

  * At training: generate crops that keep the object inside but enlarge it in pixels.
  * At inference: optionally run sliding windows if full-frame misses objects, then merge.

#### 2.2. Label & sampler strategy

* Make sure all frames where GT exists are in training – don’t drop “boring” frames where the object is tiny or near-static (those matter a lot for ST-IoU).
* **Oversample small objects** and rare classes:

  * you already have tubelet / copy-paste augmentation; bias it to paste tiny objects into many backgrounds.
  * use your `augmented_ref_img` bank to create extra frames where the object is 1–3% of the image, not just big crops.
* In Ultralytics hyperparameters, increase weight for box/dfl loss a bit (`box`, `dfl`) so the model focuses more on localization, a known trick in small-object YOLO papers.

#### 2.3. “Soft” augmentation, not chaos

Your video augmenter is already rich. For tiny objects, too much rotation/scale can easily erase them. Borrowing from small-object UAV work where they tone down aggressive transforms:

* Use **conservative spatial ranges** for small objects: rotations ≤ 10°, scale range maybe 0.8–1.2, keep strong crops relatively rare.
* Keep **pixel-level augments strong but smooth in time** (your temporal jitter is good) – this improves robustness without destroying geometry.

---

### 3. Exploit temporal nature for better ST-IoU

YOLO11 is image-based, but ST-IoU is video-based. The big wins will come from **tracking + smoothing** on top of YOLO:

#### 3.1. Detection + lightweight tracker

* For each frame, run YOLO11-n/s → get boxes + confidences.
* Feed detections into a tiny tracker (SORT, DeepSORT without re-ID, or ByteTrack) – these are widely used for UAV tracking and very light-weight.
* Since each video has only one target object, you can simplify:

  * Keep only the **single track with highest average confidence** and reasonable length.
  * Discard short noisy tracks (< K frames).

This produces a **tubelet** (sequence of boxes) per video – exactly what the evaluator expects.

#### 3.2. Temporal smoothing & gap filling

To lift ST-IoU:

* Apply **smoothing** on the track:

  * smooth center and width/height with an exponential moving average or a small Kalman filter; this stabilizes jitter, which UAV papers show improves tube metrics.
* **Fill short gaps**: if the tracker loses the object for < 3-5 frames between two detections, linearly interpolate bounding boxes instead of leaving holes. Those frames then move from “union only” → “intersection,” raising ST-IoU.
* **Trim the tube**: don’t predict before the object first appears or long after it disappears.

  * simple rule: keep only frames belonging to the main track where confidence > τ, and cut leading/trailing sections with low confidence.

#### 3.3. Time-aware thresholds

Instead of one fixed confidence threshold:

* Start with a relatively low detection threshold (e.g. 0.1–0.2) to avoid missing tiny objects.
* After tracking, throw away very short or low-confidence tracks; it’s better to have one slightly noisy long tube than many disjoint pieces for ST-IoU.

You can tune these thresholds directly against ST-IoU on your train/val split.

---

### 4. Fine-tuning schedule for YOLO11-n/s

Pulling ideas from UAV-YOLO and SOD-YOLO training setups:

1. **Pretraining**: start from official COCO or general YOLO11-n/s weights; they contain lots of aerial-style context even if not UAV-specific.
2. **Frame extraction & labels**:

   * convert your per-frame annotations to YOLO TXT format, one object per frame (class id + normalized x/y/w/h).
   * ensure the train/val split is video-disjoint (no leakage).
3. **Hyperparameters** (example for YOLO11-s):

   * imgsz 960–1280, lr ~ 0.01 with cosine decay, 200–300 epochs with early-stop on val mAP or surrogate.
   * multi-scale training enabled; mosaic on early epochs, disabled near the end (Ultralytics `close_mosaic` trick).
   * add a small `scale` factor in augment hyperparams favoring enlargement of small boxes.
4. **Curriculum**:

   * start with **easier shots** (object larger / centered, your cropped/tiling views), then gradually mix in full-frame tiny views.
   * last 10–20 epochs: turn off the most aggressive augments, keep only light color jitter and flip, to refine localization.

After each training run, run the full **detection → tracking → ST-IoU** pipeline on validation videos and adjust:

* YOLO image size & conf / iou thresholds;
* tracker parameters (association IoU, min track length);
* smoothing & interpolation window sizes.

---

### 5. Keep the JSON format aligned with tubelets

You already know the final submission format:

```json
[
  {
    "video_id": "drone_vid001",
    "detections": [
      {
        "bboxes": [
          {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
          {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
        ]
      }
    ]
  },
  {
    "video_id": "drone_video_002",
    "detections": []
  }
]
```

In the pipeline above:

* each **final track** becomes one entry in `"detections"` (likely exactly one per video in this challenge),
* the tracker + smoothing outputs an ordered list of `{frame, x1, y1, x2, y2}` – just drop it into `"bboxes"`,
* if no confident track survives → `"detections": []` for that video.

So as long as your post-processing always produces *some* (possibly empty) tubelet per video, you’re aligned with the evaluator and ST-IoU definition.

---

### 6. Bias check + quick verification

**CF (counter-factual)**: I’m not assuming YOLO11 must be replaced by a fancy video detector. A counter-plan would be “switch to a spatio-temporal network like STDnet-ST or a transformer-based video detector.” Those do improve small UAV object metrics in the literature, but they’re heavier than your Jetson NX budget and require more engineering. That’s why I bias toward “keep YOLO11, add tracking + smart augments” as the practical optimal plan under constraints.

**AN (anchoring)**: It’s easy to over-anchor on per-frame mAP. Here I explicitly derive recommendations from the ST-IoU formula: we optimize not just IoU but also the length and alignment of the predicted tube, which is why I emphasize gap-filling, trimming, and stable thresholds rather than just cranking up mAP.

**AV (availability)**: Most published tricks I cited come from UAV-YOLO-style papers that adjust architecture. I deliberately separated **“what you can steal without surgery”** (high-res, P2-like focus on small objects via tiling, copy-paste, data weighting) from architecture changes, since not all flashy ideas are feasible inside Ultralytics YOLO11 in competition timelines.

**Quick self-check checklist for you**

When you start fine-tuning, sanity-check:

1. On a few videos, visualize: GT tubelet vs your YOLO + tracker tubelet; ST-IoU should roughly match what you see.
2. Change only one thing at a time (imgsz, thresholds, tracker params) and log ST-IoU on a fixed val set.
3. Make sure FPS on Jetson NX with YOLO11-s + tracker is still acceptable; if not, step down to YOLO11-n or smaller imgsz and re-tune.

---

**Assumptions & confidence**

*Assumptions*: (1) competition uses a tube-based ST-IoU similar to common video-tracking benchmarks; (2) you are locked into YOLO11-n/s and Jetson Xavier NX (≤50M params); (3) only one target object per video, so single-tube output is valid.

**Confidence level**: **High** that the combination of small-object-oriented training + tracking/smoothing + tuned thresholds will noticeably improve your ST-IoU, even if the exact best hyperparameters will need empirical search on your validation split.
