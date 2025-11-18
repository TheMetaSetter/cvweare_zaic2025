"""End-to-end experiment runner: augment → export → train → ST-IoU eval."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml
from ultralytics import YOLO

from augmentation.config import (
    AugmentationConfig,
    get_aggressive_config,
    get_conservative_config,
    get_default_config,
)
from augmentation.dataset import AugmentedVideoGenerator
from callbacks.wandb_logging import attach_wandb_logging
from evaluation.st_iou_pipeline import STIouPipeline, TrackerConfig
from export_to_yolo import SourceSpec, TileConfig, YOLODatasetExporter


class ExperimentPipeline:
    """Coordinates data augmentation, YOLO export, training, and evaluation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.class_map = self._load_class_map(Path(args.data_config))

    def run(self) -> None:
        stages = set(self.args.stages)
        if "augment" in stages:
            self._generate_augmented_data()
        if "export" in stages:
            self._export_to_yolo()
        if "train" in stages:
            self._train_models()
        if "evaluate" in stages:
            self._run_evaluation()

    def _generate_augmented_data(self) -> None:
        config = self._resolve_aug_config(self.args.augment_preset)
        generator = AugmentedVideoGenerator(
            data_dir=str(self.args.train_samples),
            annotations_path=str(self.args.train_annotations),
            output_dir=str(self.args.augmented_dir),
            aug_config=config,
        )
        print("[STAGE] Generating augmented dataset...")
        generator.generate_augmented_dataset(
            start_from_idx=self.args.augment_start,
            skip_existing=not self.args.augment_overwrite,
        )

    def _export_to_yolo(self) -> None:
        print("[STAGE] Exporting YOLO dataset...")
        tile_cfg = TileConfig(
            enabled=self.args.tile_small_objects,
            scale=self.args.tile_scale,
            min_area_ratio=self.args.tile_min_area,
            extra_pad=self.args.tile_pad,
        )
        # Train split: combine original + augmented (if annotations exist).
        train_sources: List[SourceSpec] = [
            SourceSpec(self.args.train_samples, self.args.train_annotations)
        ]
        aug_ann = self.args.augmented_dir / "annotations" / "annotations_augmented.json"
        if aug_ann.exists():
            train_sources.append(SourceSpec(self.args.augmented_dir / "samples", aug_ann))
        exporter = YOLODatasetExporter(
            output_root=self.args.export_root,
            split="train",
            class_map=self.class_map,
            tile_cfg=tile_cfg,
        )
        exporter.export_sources(train_sources)

        # Validation split stays clean (no augmented samples).
        val_exporter = YOLODatasetExporter(
            output_root=self.args.export_root,
            split="val",
            class_map=self.class_map,
            tile_cfg=TileConfig(enabled=False),
        )
        val_exporter.export_sources([SourceSpec(self.args.val_samples, self.args.val_annotations)])

    def _train_models(self) -> None:
        print("[STAGE] Training YOLO models...")
        for weights in self.args.weights:
            model = YOLO(weights)
            run_name = f"{Path(weights).stem}_{self.args.name_suffix}"
            wandb_run = None
            if getattr(self.args, "log_wandb", False):
                wandb_run = attach_wandb_logging(
                    model,
                    project=self.args.wandb_project,
                    run_name=f"{self.args.wandb_run_prefix}-{run_name}",
                    offline=self.args.wandb_offline,
                )
            model.train(
                data=str(self.args.data_config),
                cfg=str(self.args.hyp_config),
                imgsz=self.args.imgsz,
                epochs=self.args.epochs,
                batch=self.args.batch,
                device=self.args.device,
                project=str(self.args.project),
                name=run_name,
                close_mosaic=self.args.close_mosaic,
                val=True,
            )
            metrics = model.val(data=str(self.args.data_config), imgsz=self.args.imgsz, device=self.args.device)
            print(f"[VAL] {run_name}: mAP50-95={metrics.box.map:.4f}")
            if wandb_run is not None:
                wandb_run.finish()

    def _run_evaluation(self) -> None:
        print("[STAGE] Running ST-IoU evaluation...")
        tracker_cfg = TrackerConfig(
            conf_thres=self.args.eval_conf,
            min_track_length=self.args.eval_min_track,
            smooth_factor=self.args.eval_smooth,
            max_interp_gap=self.args.eval_max_gap,
            trim_conf=self.args.eval_trim_conf,
            imgsz=self.args.imgsz,
            iou_thres=self.args.eval_iou,
            chunk_size=self.args.eval_chunk_size,
        )
        eval_model = self._resolve_eval_model()
        pipeline = STIouPipeline(
            model_path=eval_model,
            data_config=self.args.data_config,
            videos_root=self.args.val_samples,
            annotations_path=self.args.val_annotations,
            output_path=self.args.eval_output,
            tracker_cfg=tracker_cfg,
            device=self.args.device,
        )
        pipeline.run()

    def _resolve_eval_model(self) -> Path:
        if self.args.eval_model:
            return Path(self.args.eval_model)
        # Default: use best.pt from the first weight's run directory.
        run_dir = self.args.project / f"{Path(self.args.weights[0]).stem}_{self.args.name_suffix}" / "weights"
        candidate = run_dir / "best.pt"
        if not candidate.exists():
            raise FileNotFoundError("Evaluation model path not provided and best.pt not found")
        return candidate

    @staticmethod
    def _resolve_aug_config(preset: str) -> AugmentationConfig:
        preset = preset.lower()
        if preset == "conservative":
            return get_conservative_config()
        if preset == "aggressive":
            return get_aggressive_config()
        return get_default_config()

    @staticmethod
    def _load_class_map(data_yaml: Path):
        data = yaml.safe_load(data_yaml.read_text())
        names = data.get("names")
        if isinstance(names, dict):
            return {str(v): int(k) for k, v in names.items()}
        if isinstance(names, list):
            return {name: idx for idx, name in enumerate(names)}
        raise ValueError("names missing in data config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full ST-IoU training pipeline")
    parser.add_argument("--train-samples", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-samples", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--augmented-dir", type=Path, default=Path("data/augmented_train"))
    parser.add_argument("--export-root", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/yolo_data.yaml"))
    parser.add_argument("--hyp-config", type=Path, default=Path("configs/yolo_hyp_small_objects.yaml"))
    parser.add_argument("--weights", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/train_smallobj"))
    parser.add_argument("--name-suffix", type=str, default="st_iou")
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="zaic2025")
    parser.add_argument("--wandb-run-prefix", type=str, default="st_iou")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--augment-preset", type=str, default="default", choices=["default", "conservative", "aggressive"])
    parser.add_argument("--augment-start", type=int, default=0)
    parser.add_argument("--augment-overwrite", action="store_true")
    parser.add_argument("--tile-small-objects", action="store_true")
    parser.add_argument("--tile-scale", type=float, default=2.0)
    parser.add_argument("--tile-min-area", type=float, default=0.02)
    parser.add_argument("--tile-pad", type=int, default=8)
    parser.add_argument("--eval-model", type=Path, default=None)
    parser.add_argument("--eval-output", type=Path, default=Path("runs/st_iou/predictions.json"))
    parser.add_argument("--eval-conf", type=float, default=0.15)
    parser.add_argument("--eval_smooth", dest="eval_smooth", type=float, default=0.6)
    parser.add_argument("--eval-max-gap", type=int, default=5)
    parser.add_argument("--eval-trim-conf", type=float, default=0.1)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--eval-chunk-size", type=int, default=64)
    parser.add_argument("--eval-min-track", type=int, default=5)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["augment", "export", "train", "evaluate"],
        default=["augment", "export", "train", "evaluate"],
        help="Which stages to execute",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ExperimentPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
