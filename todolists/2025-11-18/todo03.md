# TODO 03 - Enable DDP on Kaggle 2xT4

1. Re-read PyTorch 2.9 DistributedDataParallel docs to pull required environment variables, init/destroy calls, and logging safeguards.
2. Update `train_yolo.py` to:
   - Parse rank/local-rank/world-size args or env vars (`torchrun` contract),
   - Initialize/disconnect the process group and bind each process to the correct CUDA device,
   - Wrap Ultralytics `YOLO` model with DDP if feasible or rely on built-in support, ensuring only rank 0 prints, logs, and validates.
3. Update `run_pipeline.py`'s `_train_models` stage with the same DDP wiring and guard downstream `_run_evaluation` so it executes once on rank 0.
4. Gate W&B logging in `callbacks/wandb_logging.py` so `wandb.init` and callbacks fire only when `rank == 0`, mirroring doc guidance on singleton logging.
5. Document the new workflow in `README.md` or `execution_guidelines.md`, including the `torchrun --nproc_per_node=2 ...` launch command, Kaggle notebook environment variables, and any MASTER_ADDR/PORT notes.
6. Smoke-test locally (or describe Kaggle steps) by simulating two ranks to verify both GPUs engage, metrics log only once, and the pipeline exits cleanly.
