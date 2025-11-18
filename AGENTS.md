**Purpose:**  
This document defines how all coding agents in this project must write **clear, concise, and human-readable code comments**. Every implementation must be accompanied by comments that help a **human reviewer** understand (1) why the code exists, (2) how it works, and (3) how it fits into the larger video-augmentation + YOLO training pipeline.

The goal is not bloated comments, but **precise guidance** that maximizes reviewer clarity.

---

# 1. Core Principles for All Agents

### **1.1. Comments must explain intent, not restate code**
Bad:
```python
x = x + 1  # increment x
````

Good:

```python
# Advance frame index to keep tubelet alignment correct.
x = x + 1
```

### **1.2. Comments must help the *human reviewer* understand reasoning**

Every block of logic should answer at least one:

* *Why is this needed?*
* *How does this interact with earlier pipeline steps?*
* *What assumptions are being made?*
* *What would break if this were removed?*

### **1.3. Prefer short, structured “micro-explanations”**

Each comment block should be:

* 1–3 lines
* Plain English
* No overly technical jargon unless necessary

---

# 2. Mandatory Comment Structure for Augmentation-Related Code

When implementing any transformation (frame augment, bbox transform, copy-paste, tubelet jitter, etc.), agents must provide:

### **2.1. Header comment (1–3 lines)**

Placed directly above a function or major block.

Includes:

* Purpose of function
* When it is called in the dataset/augmentation lifecycle
* What it returns

Example:

```python
# Applies copy-paste tubelet augmentation after all spatial transforms.
# This modifies frames in-place and produces synthetic bboxes aligned with final geometry.
# Called inside VideoAugmenter.augment_video_clip().
```

### **2.2. Inline reasoning comments**

Inside the function, agents must comment on:

* **critical decisions** (e.g., why jitter is applied)
* **constraints** (e.g., bounding box must remain within frame)
* **failure cases** (e.g., skip paste if alpha mask is empty)

Example:

```python
# Determine a stable base position for the entire tubelet.
# The jitter is added per frame to simulate realistic motion without drifting too far.
base_x1 = ...
```

### **2.3. Data-flow clarity**

Agents must explain how data moves through the system:

* Where frames came from
* What coordinate space bboxes live in
* Why pasted bboxes do **not** get passed through `_transform_bbox`

---

# 3. Mandatory Comment Structure for Dataset-Related Code

Agents must help the reviewer understand:

### **3.1. Index logic**

If implementing `__getitem__`, always explain:

* why `idx` is split into `base_idx` and `aug_idx`
* how virtual dataset length is expanded
* what each index means

### **3.2. Frame extraction**

Comments must clarify:

* why certain frames are loaded
* why frame skipping / sampling occurs
* how memory is controlled

### **3.3. Returned sample format**

Agents must restate the contract for output dictionaries whenever modifying it.

---

# 4. Specific Requirements for Copy-Paste Augmentation

Any agent implementing `_apply_copy_paste` must include:

### **4.1. High-level explanation**

Placed at top of function:

* Why copy-paste improves tiny object detection
* Why tubelet consistency is required
* Why the function runs *after* spatial transforms

### **4.2. Detailed inline comments for each step**

Agents must comment on:

* loading PNGs with alpha
* resizing logic (and why scale range is chosen)
* coordinate selection and jitter
* alpha blending details
* bbox generation for each frame
* conditions when pasting is skipped

### **4.3. Safety constraints**

Document how bounding boxes remain valid:

* enforce frame boundaries
* ensure paste is skipped rather than breaking

---

# 5. Style and Formatting Rules

### **5.1. Keep comments human-first, not machine-first**

Do not produce compiler-style documentation.
Comments must help *humans* understand reasoning.

### **5.2. No redundant comments**

Avoid restating function names.
Every comment line must add meaning.

### **5.3. Vertical spacing**

For clarity, use:

* 1 blank line to separate conceptual blocks
* No excessive spacing

### **5.4. Use active voice**

Good:

```
# Compute jitter once so the tubelet appears to move naturally.
```

Bad:

```
# Jitter is computed.
```

---

# 6. Review Checklist for Agents

Before finalizing any code, the agent must check:

* [ ] Did I explain *why* key logic exists?
* [ ] Did I comment the function’s purpose and place in pipeline?
* [ ] Are all transforms linked back to the video/tubelet/YOLO training context?
* [ ] Are error-prone parts (bbox math, alpha blending, scaling) clearly annotated?
* [ ] Will a future contributor instantly understand my reasoning?

If any answer is “no”, the agent must revise comments.

---

# 7. Summary for Agents

All code in this project must be commented with the goal of **teaching a human reviewer**, not simply documenting functionality.
The comments should illuminate reasoning behind:

* data flow
* geometry
* augmentation logic
* integration with the training pipeline

If a junior engineer can understand your code purely from comments + structure, then you have succeeded.

---