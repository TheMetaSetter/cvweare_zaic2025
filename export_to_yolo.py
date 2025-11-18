"""Export annotated drone videos (original + augmented) into YOLO image/txt pairs.

The script walks one or more annotation manifests, loads every frame that carries
ground-truth tubelets, and writes:

* images/<split>/VIDEOID_fXXXXXX.jpg
* labels/<split>/VIDEOID_fXXXXXX.txt (YOLO format)

Optional tiny-object tiling (SAHI-style crops) can be enabled to oversample
cases where the object occupies only a few percent of the frame.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import yaml

from augmentation import bbox_utils


@dataclass
class SourceSpec:
    """Couples a samples directory with its annotations file."""

    data_dir: Path
    annotations_path: Path


@dataclass
class TileConfig:
    """Settings for optional zoomed crops around very small objects."""

    enabled: bool = False
    scale: float = 2.0
    min_area_ratio: float = 0.02
    extra_pad: int = 8


class YOLODatasetExporter:
    """Converts annotated video frames into disk-backed YOLO datasets."""

    def __init__(
        self,
        output_root: Path,
        split: str,
        class_map: Dict[str, int],
        tile_cfg: TileConfig,
    ) -> None:
        self.output_root = output_root
        self.split = split
        self.class_map = class_map
        self.tile_cfg = tile_cfg
        self.images_dir = self.output_root / "images" / split
        self.labels_dir = self.output_root / "labels" / split
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self._tile_serial = 0

    def export_sources(self, sources: Sequence[SourceSpec]) -> None:
        """Iterate through each source manifest and export all of its videos."""

        for spec in sources:
            annotations = self._load_annotations(spec.annotations_path)
            for video_ann in annotations:
                self._export_video(spec.data_dir, video_ann)

    def _export_video(self, data_dir: Path, video_ann: Dict) -> None:
        """Write every annotated frame for a single video/augmented sample."""

        video_id = video_ann["video_id"]
        class_name = self._infer_class_name(video_id)
        if class_name not in self.class_map:
            raise ValueError(
                f"Class '{class_name}' derived from {video_id} is missing in class map"
            )
        class_id = self.class_map[class_name]

        # Flatten annotation groups into a frame → list[bbox] dict.
        frame_to_bboxes: Dict[int, List[Dict]] = defaultdict(list)
        for ann_group in video_ann.get("annotations", []):
            for bbox in ann_group.get("bboxes", []):
                frame_to_bboxes[int(bbox["frame"])].append(bbox)

        if not frame_to_bboxes:
            return

        frame_indices = sorted(frame_to_bboxes.keys())
        sample_dir = Path(data_dir) / video_id
        frames = self._load_frames(sample_dir, frame_indices)

        for frame_idx in frame_indices:
            frame = frames.get(frame_idx)
            if frame is None:
                continue
            # Persist the image before writing labels so training scripts can open it directly.
            image_name = f"{video_id}_f{frame_idx:06d}.jpg"
            image_path = self.images_dir / image_name
            cv2.imwrite(str(image_path), frame)

            label_lines: List[str] = []
            frame_h, frame_w = frame.shape[:2]
            for bbox in frame_to_bboxes[frame_idx]:
                clipped = bbox_utils.clip_bbox(bbox, frame_w, frame_h)
                if clipped is None:
                    continue
                label_lines.append(self._bbox_to_yolo_line(clipped, frame_w, frame_h, class_id))
                self._maybe_write_tile(
                    frame,
                    frame_idx,
                    clipped,
                    class_id,
                    base_name=image_name.rstrip(".jpg"),
                )

            label_path = self.labels_dir / image_name.replace(".jpg", ".txt")
            label_path.write_text("\n".join(label_lines))

    def _maybe_write_tile(
        self,
        frame,
        frame_idx: int,
        bbox: Dict,
        class_id: int,
        base_name: str,
    ) -> None:
        """Oversample tiny objects by cropping a zoomed window around them."""

        if not self.tile_cfg.enabled:
            return
        frame_h, frame_w = frame.shape[:2]
        area_ratio = bbox_utils.bbox_area(bbox) / float(frame_w * frame_h)
        if area_ratio >= self.tile_cfg.min_area_ratio:
            return

        bbox_w = bbox["x2"] - bbox["x1"]
        bbox_h = bbox["y2"] - bbox["y1"]
        cx = (bbox["x1"] + bbox["x2"]) / 2.0
        cy = (bbox["y1"] + bbox["y2"]) / 2.0
        tile_w = bbox_w * self.tile_cfg.scale
        tile_h = bbox_h * self.tile_cfg.scale
        x1 = int(max(0, cx - tile_w / 2.0 - self.tile_cfg.extra_pad))
        y1 = int(max(0, cy - tile_h / 2.0 - self.tile_cfg.extra_pad))
        x2 = int(min(frame_w, cx + tile_w / 2.0 + self.tile_cfg.extra_pad))
        y2 = int(min(frame_h, cy + tile_h / 2.0 + self.tile_cfg.extra_pad))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return

        crop = frame[y1:y2, x1:x2]
        rel_bbox = {
            "frame": frame_idx,
            "x1": bbox["x1"] - x1,
            "y1": bbox["y1"] - y1,
            "x2": bbox["x2"] - x1,
            "y2": bbox["y2"] - y1,
        }
        crop_h, crop_w = crop.shape[:2]
        clipped = bbox_utils.clip_bbox(rel_bbox, crop_w, crop_h)
        if clipped is None:
            return

        self._tile_serial += 1
        tile_name = f"{base_name}_tile{self._tile_serial:04d}.jpg"
        tile_image_path = self.images_dir / tile_name
        tile_label_path = self.labels_dir / tile_name.replace(".jpg", ".txt")
        cv2.imwrite(str(tile_image_path), crop)
        tile_label_path.write_text(
            self._bbox_to_yolo_line(clipped, crop_w, crop_h, class_id)
        )

    @staticmethod
    def _bbox_to_yolo_line(bbox: Dict, width: int, height: int, class_id: int) -> str:
        """Convert absolute bbox coordinates into YOLO TXT format."""

        x_center = ((bbox["x1"] + bbox["x2"]) / 2.0) / width
        y_center = ((bbox["y1"] + bbox["y2"]) / 2.0) / height
        box_w = (bbox["x2"] - bbox["x1"]) / width
        box_h = (bbox["y2"] - bbox["y1"]) / height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"

    @staticmethod
    def _infer_class_name(video_id: str) -> str:
        """Derive class name from folder/video naming (e.g., Backpack_0_aug_1)."""

        # Remove augmentation suffixes then trailing digits (Backpack_0_aug_1 -> Backpack).
        name = re.sub(r"_aug_\d+$", "", video_id)
        return re.sub(r"[_\-]*\d+$", "", name)

    @staticmethod
    def _load_annotations(path: Path) -> List[Dict]:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, list):
            raise ValueError(f"Annotations at {path} must be a list")
        return data

    def _load_frames(self, sample_dir: Path, frame_indices: Iterable[int]) -> Dict[int, np.ndarray]:
        """Load only the frames requested, from either frame JPGs or drone_video.mp4."""

        frames_dir = sample_dir / "frames"
        if frames_dir.exists():
            return self._load_frames_from_disk(frames_dir, frame_indices)
        video_path = sample_dir / "drone_video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Neither frames/ nor {video_path} found for {sample_dir}")
        return self._load_frames_from_video(video_path, frame_indices)

    @staticmethod
    def _load_frames_from_disk(frames_dir: Path, frame_indices: Iterable[int]) -> Dict[int, np.ndarray]:
        frames = {}
        for idx in frame_indices:
            img_path = frames_dir / f"frame_{idx:06d}.jpg"
            if not img_path.exists():
                continue
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frames[int(idx)] = frame
        return frames

    @staticmethod
    def _load_frames_from_video(video_path: Path, frame_indices: Iterable[int]) -> Dict[int, np.ndarray]:
        requested = sorted(set(int(i) for i in frame_indices))
        frames = {}
        if not requested:
            return frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        start = requested[0]
        end = requested[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        current = start
        requested_set = set(requested)
        while current <= end:
            ret, frame = cap.read()
            if not ret:
                break
            if current in requested_set:
                frames[current] = frame.copy()
            current += 1
        cap.release()
        return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export drone dataset to YOLO format")
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("DATA_DIR", "ANNOTATIONS"),
        required=True,
        help="Pair of sample directory root and annotations.json (can repeat)",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        required=True,
        help="YOLO data.yaml with names list/dict (used to build class map)",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Root folder to place images/ and labels/")
    parser.add_argument("--split", type=str, default="train", help="Split name (train/val/test)")
    parser.add_argument("--tile-small-objects", action="store_true", help="Enable zoomed crops for small objects")
    parser.add_argument("--tile-scale", type=float, default=2.0, help="Scale multiplier for tiles relative to bbox size")
    parser.add_argument("--tile-min-area", type=float, default=0.02, help="Area ratio threshold below which tiles are created")
    parser.add_argument("--tile-pad", type=int, default=8, help="Extra padding (pixels) around the tile window")
    return parser.parse_args()


def load_class_map(data_config: Path) -> Dict[str, int]:
    data = yaml.safe_load(data_config.read_text())
    names = data.get("names")
    if isinstance(names, dict):
        return {str(v): int(k) for k, v in names.items()}
    if isinstance(names, list):
        return {name: idx for idx, name in enumerate(names)}
    raise ValueError(f"`names` missing in {data_config}")


def main() -> None:
    args = parse_args()
    class_map = load_class_map(args.data_config)
    tile_cfg = TileConfig(
        enabled=args.tile_small_objects,
        scale=args.tile_scale,
        min_area_ratio=args.tile_min_area,
        extra_pad=args.tile_pad,
    )
    sources = [SourceSpec(Path(data), Path(ann)) for data, ann in args.source]
    exporter = YOLODatasetExporter(args.output_root, args.split, class_map, tile_cfg)
    exporter.export_sources(sources)


if __name__ == "__main__":
    main()
