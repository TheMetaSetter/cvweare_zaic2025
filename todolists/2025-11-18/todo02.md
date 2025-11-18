## TODO 2025-11-18 #02 — W&B Step-Level Logging

- [x] Add CLI flags in `train_yolo.py` and `run_pipeline.py` for enabling Weights & Biases logging (project name, run suffix, offline toggle) and wire them into the training flow via `model.add_callback`.
- [x] Register an Ultralytics callback (`callbacks/wandb_logging.py`) that hooks `on_train_batch_end` to emit per-step loss dictionaries (`box`, `cls`, `dfl`, `loss`) to W&B via `wandb.log`.
- [x] Ensure `requirements.txt` lists `wandb` as optional, and document setup steps (login/offline mode) plus new CLI usage in `NEXT_STEPS_GUIDE.md` / README.
