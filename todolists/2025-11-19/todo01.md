# TODO 01 - Align repo with submission guidelines

1. Containerization
   - [ ] Write a Dockerfile using `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel` as base.
   - [ ] Install system deps (`libgl1-mesa-glx`, `libglib2.0-0`, vim, python3-pip) per guidelines.
   - [ ] Copy the repo into `/code`, set `WORKDIR /code`, and install `requirements.txt`.
   - [ ] Document build/run commands (docker build + docker run --gpus '"device=0"' --network host) in README/SUBMISSION docs.

2. Prediction interface
   - [ ] Implement `predict.py` that loads the trained YOLO weights, reads `/data` inputs, writes `/result/submission.json`, and prints model load/predict times.
   - [ ] Create `predict.sh` that runs the above script; ensure it’s executable and demonstrates required stdout timing logs.
   - [ ] Update docs to describe the expected `/data` structure and output format for judges.

3. Timing notebook
   - [ ] Add a `predict_notebook.ipynb` (template ok) that executes the same inference path once with timing code for load/predict.
   - [ ] Note in README how to open/run the notebook inside the Docker container (requires `--network host`).

4. Model packaging
   - [ ] Provide/read a `saved_models/` directory path (or config option) where `best.pt` checkpoints live for inference.
   - [ ] Update docs to instruct users to copy their final weights into this folder before building the Docker image.

5. Submission documentation
   - [ ] Extend README or add `SUBMISSION.md` summarizing Docker workflow, predict.sh usage, notebook, and the requirement to commit inside the container prior to image save.
   - [ ] Reference `submissionguidelines.md` sections so future contributors know the official requirements (base image, GPU flags, output paths, etc.).

6. Helper automation
   - [ ] Optionally add a Makefile or script (e.g., `scripts/build_docker.sh`) wrapping build/run/cp commands to reduce manual errors, without touching core training logic.
