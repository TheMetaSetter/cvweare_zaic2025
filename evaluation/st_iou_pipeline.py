"""Detection → tracking → ST-IoU evaluation pipeline.

Designed to keep the competition metric (tubelet IoU over time) at the center
of the workflow. Each video is:
1. decoded around its annotated frame span,
2. run through a YOLO11 checkpoint,
3. tracked/smoothed to form a single tubelet, and
4. scored against ground truth using the ST-IoU formula from howtopushperformance.md.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from augmentation import bbox_utils


@dataclass
class TrackerConfig:
    """Config knobs for detection filtering, smoothing, and interpolation."""

    conf_thres: float = 0.15
    min_track_length: int = 5
    smooth_factor: float = 0.6
    max_interp_gap: int = 5
    trim_conf: float = 0.1
    imgsz: int = 960
    iou_thres: float = 0.5
    chunk_size: int = 64


class STIouPipeline:
    """Runs YOLO detections, tracks tubelets, and computes ST-IoU."""

    def __init__(
        self,
        model_path: Path,
        data_config: Path,
        videos_root: Path,
        annotations_path: Path,
        output_path: Path,
        tracker_cfg: TrackerConfig,
        device: str,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.class_map = self._load_class_map(data_config)
        self.videos_root = videos_root
        self.annotations = self._load_annotations(annotations_path)
        self.output_path = output_path
        self.cfg = tracker_cfg
        self.device = device

    def run(self) -> None:
        """Evaluate every annotated video and dump both metrics + submission JSON."""

        outputs = []
        st_iou_scores: List[Tuple[str, float]] = []
        for ann in self.annotations:
            video_id = ann["video_id"]
            video_path = self.videos_root / video_id / "drone_video.mp4"
            if not video_path.exists():
                print(f"[WARN] Video not found: {video_path}")
                outputs.append({"video_id": video_id, "detections": []})
                continue
            gt_boxes = self._collect_gt_boxes(ann)
            if not gt_boxes:
                outputs.append({"video_id": video_id, "detections": []})
                continue
            pred_track = self._predict_track(video_path, gt_boxes, video_id)
            st_iou = self._compute_st_iou(gt_boxes, pred_track)
            st_iou_scores.append((video_id, st_iou))
            detections = [] if not pred_track else [{"bboxes": pred_track}]
            outputs.append({"video_id": video_id, "detections": detections})
            print(f"[INFO] {video_id}: ST-IoU={st_iou:.4f}, frames={len(pred_track)}")

        self.output_path.write_text(json.dumps(outputs, indent=2))
        if st_iou_scores:
            mean_score = sum(score for _, score in st_iou_scores) / len(st_iou_scores)
            print(f"\n[SUMMARY] Mean ST-IoU over {len(st_iou_scores)} videos: {mean_score:.4f}")

    def _predict_track(self, video_path: Path, gt_boxes: Dict[int, Dict], video_id: str) -> List[Dict]:
        """Run YOLO on the GT span, smooth detections, and return a tubelet."""

        class_name = self._infer_class_name(video_id)
        class_id = self.class_map.get(class_name)
        if class_id is None:
            raise ValueError(f"Class {class_name} missing from class map")

        frame_numbers, frames = self._load_clip(video_path, gt_boxes)
        detection_map: Dict[int, Dict] = {}
        for chunk_start in range(0, len(frames), self.cfg.chunk_size):
            chunk_frames = frames[chunk_start : chunk_start + self.cfg.chunk_size]
            chunk_numbers = frame_numbers[chunk_start : chunk_start + self.cfg.chunk_size]
            if not chunk_frames:
                continue
            results = self.model.predict(
                chunk_frames,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf_thres,
                iou=self.cfg.iou_thres,
                device=self.device,
                verbose=False,
            )
            for res, frame_no in zip(results, chunk_numbers):
                if res.boxes is None or res.boxes.shape[0] == 0:
                    continue
                best = None
                best_conf = 0.0
                for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                    if int(cls) != class_id:
                        continue
                    if float(conf) > best_conf:
                        best_conf = float(conf)
                        best = {
                            "frame": int(frame_no),
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                            "conf": float(conf),
                        }
                if best is not None:
                    detection_map[int(frame_no)] = best

        track = self._build_track(detection_map)
        return track

    def _build_track(self, detections: Dict[int, Dict]) -> List[Dict]:
        """Smooth detections across time, fill short gaps, and trim tails."""

        if not detections:
            return []
        ordered_frames = sorted(detections.keys())
        track = [detections[f] for f in ordered_frames]
        if len(track) < self.cfg.min_track_length:
            return []

        smoothed = self._smooth_track(track)
        filled = self._fill_gaps(smoothed)
        trimmed = self._trim_low_conf(filled)
        return trimmed

    def _smooth_track(self, track: List[Dict]) -> List[Dict]:
        """Apply exponential smoothing to center + size to reduce jitter."""

        if not track:
            return []
        smoothed: List[Dict] = []
        prev = track[0].copy()
        smoothed.append(prev)
        alpha = self.cfg.smooth_factor
        for bbox in track[1:]:
            prev_cx = (prev["x1"] + prev["x2"]) / 2.0
            prev_cy = (prev["y1"] + prev["y2"]) / 2.0
            prev_w = prev["x2"] - prev["x1"]
            prev_h = prev["y2"] - prev["y1"]
            cx = (bbox["x1"] + bbox["x2"]) / 2.0
            cy = (bbox["y1"] + bbox["y2"]) / 2.0
            w = bbox["x2"] - bbox["x1"]
            h = bbox["y2"] - bbox["y1"]
            cx = alpha * cx + (1 - alpha) * prev_cx
            cy = alpha * cy + (1 - alpha) * prev_cy
            w = alpha * w + (1 - alpha) * prev_w
            h = alpha * h + (1 - alpha) * prev_h
            new_bbox = {
                "frame": bbox["frame"],
                "x1": cx - w / 2.0,
                "y1": cy - h / 2.0,
                "x2": cx + w / 2.0,
                "y2": cy + h / 2.0,
                "conf": bbox.get("conf", 1.0),
            }
            smoothed.append(new_bbox)
            prev = new_bbox
        return smoothed

    def _fill_gaps(self, track: List[Dict]) -> List[Dict]:
        """Linearly interpolate boxes when the tracker drops for a few frames."""

        if not track:
            return []
        filled = [track[0]]
        for prev, nxt in zip(track, track[1:]):
            filled.append(nxt)
            gap = nxt["frame"] - prev["frame"] - 1
            if gap <= 0 or gap > self.cfg.max_interp_gap:
                continue
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                interp_bbox = {
                    "frame": prev["frame"] + step,
                    "x1": prev["x1"] + (nxt["x1"] - prev["x1"]) * t,
                    "y1": prev["y1"] + (nxt["y1"] - prev["y1"]) * t,
                    "x2": prev["x2"] + (nxt["x2"] - prev["x2"]) * t,
                    "y2": prev["y2"] + (nxt["y2"] - prev["y2"]) * t,
                    "conf": min(prev.get("conf", 1.0), nxt.get("conf", 1.0)),
                }
                filled.append(interp_bbox)
        # Sort because we appended next before inserts.
        filled.sort(key=lambda b: b["frame"])
        return filled

    def _trim_low_conf(self, track: List[Dict]) -> List[Dict]:
        """Drop leading/trailing boxes that never reach the confidence floor."""

        if not track:
            return []
        start = 0
        end = len(track)
        while start < end and track[start].get("conf", 0.0) < self.cfg.trim_conf:
            start += 1
        while end > start and track[end - 1].get("conf", 0.0) < self.cfg.trim_conf:
            end -= 1
        return track[start:end]

    def _collect_gt_boxes(self, annotation: Dict) -> Dict[int, Dict]:
        frame_map: Dict[int, Dict] = {}
        for ann_group in annotation.get("annotations", []):
            for bbox in ann_group.get("bboxes", []):
                frame_map[int(bbox["frame"])]=bbox
        return frame_map

    def _load_clip(self, video_path: Path, gt_boxes: Dict[int, Dict]) -> Tuple[List[int], List[np.ndarray]]:
        """Load the contiguous frame span covering every GT box."""

        frame_ids = sorted(gt_boxes.keys())
        start = frame_ids[0]
        end = frame_ids[-1]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: List[np.ndarray] = []
        numbers: List[int] = []
        current = start
        while current <= end:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            numbers.append(current)
            current += 1
        cap.release()
        return numbers, frames

    @staticmethod
    def _compute_st_iou(gt_boxes: Dict[int, Dict], pred_track: List[Dict]) -> float:
        if not gt_boxes:
            return 0.0
        gt_frames = set(gt_boxes.keys())
        pred_map = {bbox["frame"]: bbox for bbox in pred_track}
        union_frames = sorted(gt_frames | set(pred_map.keys()))
        if not union_frames:
            return 0.0
        intersection = gt_frames & set(pred_map.keys())
        if not intersection:
            return 0.0
        iou_sum = 0.0
        for frame in intersection:
            iou_sum += bbox_utils.calculate_iou(gt_boxes[frame], pred_map[frame])
        return iou_sum / len(union_frames)

    @staticmethod
    def _infer_class_name(video_id: str) -> str:
        """Mirror export_to_yolo's logic for mapping video_id → class name.

        This keeps GT, exported labels, and ST-IoU evaluation perfectly aligned
        even for names like "Person1_0_aug_2".
        """

        name = re.sub(r"_aug_\d+$", "", video_id)
        return re.sub(r"[_\-]*\d+$", "", name)

    @staticmethod
    def _load_annotations(path: Path) -> List[Dict]:
        return json.loads(Path(path).read_text())

    @staticmethod
    def _load_class_map(data_yaml: Path) -> Dict[str, int]:
        data = yaml.safe_load(Path(data_yaml).read_text())
        names = data.get("names")
        if isinstance(names, dict):
            return {str(v): int(k) for k, v in names.items()}
        if isinstance(names, list):
            return {name: idx for idx, name in enumerate(names)}
        raise ValueError("names entry missing in data config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST-IoU evaluation pipeline")
    parser.add_argument("--model", type=Path, required=True, help="YOLO weights (best.pt)")
    parser.add_argument("--data-config", type=Path, default=Path("configs/yolo_data.yaml"))
    parser.add_argument("--videos-root", type=Path, required=True, help="Directory containing video folders")
    parser.add_argument("--annotations", type=Path, required=True, help="annotations.json path")
    parser.add_argument("--output", type=Path, default=Path("runs/st_iou/predictions.json"))
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--smooth", type=float, default=0.6)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--trim-conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--min-track", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrackerConfig(
        conf_thres=args.conf,
        min_track_length=args.min_track,
        smooth_factor=args.smooth,
        max_interp_gap=args.max_gap,
        trim_conf=args.trim_conf,
        imgsz=args.imgsz,
        iou_thres=args.iou,
        chunk_size=args.chunk_size,
    )
    pipeline = STIouPipeline(
        model_path=args.model,
        data_config=args.data_config,
        videos_root=args.videos_root,
        annotations_path=args.annotations,
        output_path=args.output,
        tracker_cfg=cfg,
        device=args.device,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
