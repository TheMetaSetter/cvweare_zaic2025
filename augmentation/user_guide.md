# 📖 USER GUIDE - VIDEO AUGMENTATION

Augmentation system for drone videos - Zalo AI Challenge 2025

---

## 🚀 STEP 1: INSTALLATION (5 minutes)

### 1.1. Install libraries

```bash
cd /Users/thang.nguyen/Documents/projects/data
pip install -r requirements.txt
```

### 1.2. System check

```bash
python3 quick_test.py
```

**Expected output:**
```
✓ ALL TESTS PASSED!
```

✅ If you see this line → System OK, proceed to step 2!

---

## 🎨 STEP 2: VIEW AUGMENTATION (10 minutes)

### 2.1. See how augmentation works

```bash
python3 augmentation/demo_augmentation.py \
    --mode visualize \
    --data_dir data/observing_unzipped/train/samples \
    --annotations data/observing_unzipped/train/annotations/annotations.json \
    --video_id Backpack_0 \
    --num_augs 3 \
    --max_frames 50
```

**A window will appear with 4 videos:**
- 1 original video (Original)
- 3 augmented videos (Aug 1, 2, 3)

**Controls:**
- `Space` - Pause/Resume
- `n` - Next frame (when paused)
- `q` - Exit

### 2.2. Try different configs

**Conservative** (gentle):
```bash
python3 augmentation/demo_augmentation.py \
    --mode visualize \
    --video_id Person1_0 \
    --config conservative \
    --data_dir data/observing_unzipped/train/samples \
    --annotations data/observing_unzipped/train/annotations/annotations.json
```

**Aggressive** (strong):
```bash
python3 augmentation/demo_augmentation.py \
    --mode visualize \
    --video_id Laptop_0 \
    --config aggressive \
    --data_dir data/observing_unzipped/train/samples \
    --annotations data/observing_unzipped/train/annotations/annotations.json
```

---

## 💾 STEP 3A: GENERATE DATASET (If you want to save to disk)

### 3.1. Generate augmented dataset

```bash
python3 augmentation/demo_augmentation.py \
    --mode generate \
    --data_dir data/observing_unzipped/train/samples \
    --annotations data/observing_unzipped/train/annotations/annotations.json \
    --output_dir data/augmented_train \
    --config default \
    --max_frames_per_video 300
```

**Important parameters:**
- `--config` - Choose `conservative`, `default`, or `aggressive`
- `--max_frames_per_video 300` - Each video loads 300 frames (saves RAM)

**Output:**
```
data/augmented_train/
├── samples/
│   ├── Backpack_0_aug_0/
│   │   └── frames/
│   │       ├── frame_003483.jpg
│   │       └── ...
│   ├── Backpack_0_aug_1/
│   └── ...
└── annotations/
    └── annotations_augmented.json
```

### 3.2. If interrupted midway (Resume)

```bash
# Rerun the same command, system automatically skips already generated videos
python3 augmentation/demo_augmentation.py \
    --mode generate \
    --data_dir data/observing_unzipped/train/samples \
    --annotations data/observing_unzipped/train/annotations/annotations.json \
    --output_dir data/augmented_train \
    --config default \
    --max_frames_per_video 300
```

✅ System automatically detects and skips existing videos!

---

## 🎯 STEP 3B: USE DIRECTLY IN TRAINING (On-the-fly)

### 3.1. Sample Code - PyTorch Dataset

```python
from augmentation import DroneVideoDataset, get_aggressive_config
from torch.utils.data import DataLoader

# Training dataset (WITH augmentation)
train_dataset = DroneVideoDataset(
    data_dir='data/observing_unzipped/train/samples',
    annotations_path='data/observing_unzipped/train/annotations/annotations.json',
    augment=True,                      # Enable augmentation
    aug_config=get_aggressive_config(), # Choose config
    load_full_video=False,             # Only load frames with bbox
    max_frames=200                     # Limit frames
)

# Validation dataset (WITHOUT augmentation)
val_dataset = DroneVideoDataset(
    data_dir='data/public_test_unzipped/public_test/samples',
    annotations_path='data/public_test_unzipped/public_test/annotations.json',
    augment=False  # Disable augmentation
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    # Training
    for batch in train_loader:
        frames = batch['frames'][0]      # Video frames (augmented)
        bboxes = batch['bboxes'][0]      # Bounding boxes (transformed)
        
        # Your training code here
        outputs = model(frames)
        loss = criterion(outputs, bboxes)
        loss.backward()
        optimizer.step()
    
    # Validation
    for batch in val_loader:
        frames = batch['frames'][0]
        bboxes = batch['bboxes'][0]
        # Your validation code
```

---

## ⚙️ CONFIGS - WHICH ONE TO CHOOSE?

### Conservative (2x data, gentle)
```bash
--config conservative
```

**When to use:**
- ✅ **SMALL** objects (Person)
- ✅ Validation split
- ✅ Want the most accurate bboxes

**Result:** 14 videos → 28 videos

---

### Default (3x data, balanced) ⭐ RECOMMENDED
```bash
--config default
```

**When to use:**
- ✅ **Most cases**
- ✅ Average object size
- ✅ Balance between quality and quantity

**Result:** 14 videos → 42 videos

---

### Aggressive (5x data, strong)
```bash
--config aggressive
```

**When to use:**
- ✅ **LITTLE original data** (<10 videos)
- ✅ **LARGE** objects (Backpack, Laptop)
- ✅ Need a lot of variation for training

**Result:** 14 videos → 70 videos

---

## 🎨 WHAT'S IN AUGMENTATION?

### Spatial Augmentation (affects bboxes)
- **Flip** - Horizontal flip of video
- **Rotate** - Rotate video (±5° to ±15°)
- **Scale** - Zoom in/out
- **Crop** - Crop a part of the video

### Pixel Augmentation (does not affect bboxes)
- **Color Jitter** - Change color, brightness
- **Blur** - Blur
- **Noise** - Add noise
- **Fog** - Fog effect 🌫️
- **Rain** - Rain effect 🌧️

---

## ⚠️ TROUBLESHOOTING

### ❌ Out of Memory when generating

**Solution:** Reduce `--max_frames_per_video`

```bash
# Instead of 500, use 200 or 100
--max_frames_per_video 200
```

---

### ❌ Many augmentations rejected (Invalid ✗)

**Solution:** Use a lighter config

```bash
# Instead of aggressive, use default or conservative
--config conservative
```

---

### ❌ Visualization window not showing

**Solution:** Check OpenCV

```bash
pip install opencv-python
```

If it still doesn't work (server without GUI):
```bash
# Use test mode instead of visualize
python3 augmentation/demo_augmentation.py --mode test \
    --data_dir ... --annotations ...
```

---

### ❌ Bbox looks incorrect

**Solution:**

1. **Pause and check carefully** (`Space` key)
2. **Test with larger objects** (Backpack instead of Person)
3. **Use conservative config**

```bash
python3 augmentation/demo_augmentation.py \
    --mode visualize \
    --video_id Backpack_0 \
    --config conservative \
    --data_dir ... --annotations ...
```

---

## 📊 VALIDATION RATE

### What is Valid Rate?

```
Generating 5 augmented versions...
  Aug 1: Valid ✓
  Aug 2: Valid ✓
  Aug 3: Invalid ✗ (rejected)  ← Rejected
  Aug 4: Valid ✓
  Aug 5: Valid ✓

Valid rate = 4/5 = 80%
```

### What is an OK Rate?

| Valid Rate | Status | Action |
|-----------|--------|--------|
| > 85% | ✅ Good | No action needed |
| 70-85% | ⚠️ OK | Consider using conservative |
| < 70% | ❌ Low | Should use conservative |

**Reasons for rejection:**
- Bbox too small after augmentation
- Bbox outside frame
- Object cropped out

→ **This is a FEATURE, not a bug!** System automatically removes bad augmentations.

---

## 🎯 RECOMMENDED WORKFLOW

### Workflow for new users:

```
1. Install & test
   ↓
2. Visualize with 2-3 videos
   ↓  (view augmentation, choose config)
3. Decide:
   ├─ Use on-the-fly? → Code PyTorch Dataset
   └─ Generate offline? → Run generate
   ↓
4. Training
   ↓
5. If needed → tune config and regenerate
```

---

## 📝 EXAMPLES

### Example 1: Quick start
```bash
# 1. Test
python3 quick_test.py

# 2. View augmentation
python3 augmentation/demo_augmentation.py \
    --mode visualize \
    --video_id Backpack_0 \
    --data_dir data/observing_unzipped/train/samples \
    --annotations data/observing_unzipped/train/annotations/annotations.json

# 3. Train with on-the-fly augmentation
# (See sample code in Step 3B)
```

### Example 2: Generate dataset
```bash
# 1. Generate
python3 augmentation/demo_augmentation.py \
    --mode generate \
    --data_dir data/observing_unzipped/train/samples \
    --annotations data/observing_unzipped/train/annotations/annotations.json \
    --output_dir data/augmented_train \
    --config default \
    --max_frames_per_video 300

# 2. Use augmented data
# Load from data/augmented_train/
```

---

## 📞 IMPORTANT FILES

| File | Purpose |
|------|-----------|
| `quick_test.py` | System test |
| `example_usage.py` | 5 code examples |
| `augmentation/demo_augmentation.py` | Main tool (visualize, generate) |
| `augmentation/README.md` | Package details |
| `USER_GUIDE.md` | This file |

---

## ✅ CHECKLIST

### Before training:

- [ ] Ran `quick_test.py` - All tests passed
- [ ] Visualized 2-3 videos
- [ ] Chosen appropriate config
- [ ] Bboxes look OK (not too wrong)
- [ ] Valid rate > 70%

### During training:

- [ ] Training set: augment=True
- [ ] Validation set: augment=False
- [ ] Config appropriate for object size
- [ ] Monitor validation loss

---

## 🎓 TIPS & TRICKS

### Tip 1: Start small
```bash
# Test with 1 video first
--video_id Backpack_0
--max_frames 50

# Only generate full dataset when OK
```

### Tip 2: Save disk space
```bash
# On-the-fly augmentation (no disk usage)
# Instead of generating offline dataset
augment=True in PyTorch Dataset
```

### Tip 3: Quick resume
```bash
# If killed midway, rerun the same command
# System automatically skips existing videos
```

### Tip 4: Monitor memory
```bash
# If RAM is insufficient
--max_frames_per_video 200  # Reduce from 500 → 200

# Or use conservative (less augmentation)
--config conservative
```

---

## 🏁 30-SECOND SUMMARY

```bash
# 1. Test
python3 quick_test.py

# 2. Visualize
python3 augmentation/demo_augmentation.py \
    --mode visualize \
    --video_id Backpack_0 \
    --data_dir data/observing_unzipped/train/samples \
    --annotations data/observing_unzipped/train/annotations/annotations.json

# 3. Train with on-the-fly augmentation
from augmentation import DroneVideoDataset, get_default_config
dataset = DroneVideoDataset(..., augment=True, aug_config=get_default_config())

# DONE! 🎉
```
![Augmentation pipeline overview](image.png)

