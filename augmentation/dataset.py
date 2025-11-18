"""
PyTorch Dataset for drone videos with augmentation.
"""

import os
import gc
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from .config import AugmentationConfig, get_default_config
from .video_augmenter import VideoAugmenter
from .refbank import build_augmented_ref_bank


class DroneVideoDataset(Dataset):
    """
    Dataset for drone videos with on-the-fly augmentation.
    """
    
    def __init__(self,
                 data_dir: str,
                 annotations_path: str,
                 augment: bool = True,
                 aug_config: Optional[AugmentationConfig] = None,
                 load_full_video: bool = False,
                 sample_rate: int = 1,
                 max_frames: Optional[int] = None):
        """
        Args:
            data_dir: Directory containing (training) samples (with video folders)
            annotations_path: Path to annotations.json
            augment: True = apply augmentation
            aug_config: Augmentation config, None = use default
            load_full_video: True = load entire video, False = only load frames with bbox
            sample_rate: Load every N frames (to save memory)
            max_frames: Max number of frames to load per video
        """
        self.data_dir = Path(data_dir)
        self.augment = augment  # Augment or not?
        self.load_full_video = load_full_video  # Load full vid or only frames with bbox?
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.ref_bank = None
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Build augmented reference bank if assets exist (copy-paste stage)
        ref_bank_root = self.data_dir.parent / 'augmented_ref_img'
        if ref_bank_root.exists():
            self.ref_bank = build_augmented_ref_bank(ref_bank_root)
        
        # Setup augmenter
        if augment: # If want to augment
            self.aug_config = aug_config if aug_config is not None else get_default_config()
            self.augmenter = VideoAugmenter(self.aug_config, ref_bank=self.ref_bank)
        else:
            self.aug_config = None
            self.augmenter = None
        
        # Build index
        self._build_index()
    
    def _build_index(self):
        """
        Build index of samples.
        Each sample = 1 video with annotations.
        """
        # Khởi tạo danh sách các sample, mỗi sample là (video_i, annotations_i)
        self.samples = []
        
        for ann in self.annotations:
            video_id = ann['video_id']
            video_path = self.data_dir / video_id / 'drone_video.mp4'
            
            if not video_path.exists():
                print(f"Warning: Video not found: {video_path}")
                continue
            
            # Extract bbox info
            bboxes = []
            for ann_group in ann.get('annotations', []):
                bboxes.extend(ann_group.get('bboxes', []))
            
            self.samples.append({
                'video_id': video_id,
                'video_path': str(video_path),
                'bboxes': bboxes
            })
    
    def __len__(self) -> int:
        """
        Virtual length expands to (num videos * augment_multiplier) so DataLoader
        can iterate through each synthetic copy deterministically.
        """
        base_len = len(self.samples)
        if self.augment:
            return base_len * self.aug_config.augment_multiplier
        return base_len
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and optionally augment one video clip.
        
        Dataset contract (per AGENTS.md): callers receive raw/augmented frames,
        YOLO-space bboxes, original frame indices, and an `is_augmented` flag so
        later stages know whether copy-paste or other heavy transforms ran.
        """
        # idx is split into "which video" (base_idx) and "which augmentation
        # replica" (aug_idx). This mirrors the virtual length logic above and
        # lets us retry augmentation without touching other videos.
        if self.augment:
            base_idx = idx % len(self.samples)
            aug_idx = idx // len(self.samples)
        else:
            base_idx = idx
            aug_idx = 0
        
        sample = self.samples[base_idx]
        
        # Only pull frames we actually need downstream: either the entire clip
        # (load_full_video=True) or the tightest bbox-covered span. This keeps
        # RAM predictable so later YOLO export stages can batch many videos.
        frames, frame_indices = self._load_video_frames(
            sample['video_path'],
            sample['bboxes']
        )
        
        # Get bboxes for loaded frames
        bboxes = [bbox for bbox in sample['bboxes'] 
                 if bbox['frame'] in frame_indices]
        
        # Apply augmentation copy when aug_idx>0; aug_idx==0 always returns the
        # unmodified clip so training sees the original distribution each epoch.
        if self.augment and aug_idx > 0:
            # Augment a frame and check if the output frame and its bbox are valid or not
            frames, bboxes, is_valid = self.augmenter.augment_video_clip(
                frames,
                bboxes,
                frame_indices=frame_indices,
                video_id=sample['video_id']
            )
            
            # If augmentation is invalid, retry with another sample
            if not is_valid:
                # Returning the raw clip here keeps the sample count stable and
                # avoids poisoning YOLO training with broken boxes.
                # Fallback: return original
                frames, frame_indices = self._load_video_frames(
                    sample['video_path'],
                    sample['bboxes']
                )
                bboxes = [bbox for bbox in sample['bboxes'] 
                         if bbox['frame'] in frame_indices]
                is_augmented = False
            else:
                is_augmented = True
        else:
            is_augmented = False
        
        return {
            'video_id': sample['video_id'],
            'frames': frames,
            'bboxes': bboxes,
            'frame_indices': frame_indices,
            'is_augmented': is_augmented
        }
    
    def _load_video_frames(self, 
                          video_path: str,
                          bboxes: List[Dict]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load video frames.
        
        Returns:
            frames: List of frames
            frame_indices: List of frame indices
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_indices = []
        
        if self.load_full_video:
            # Load full video
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % self.sample_rate == 0:
                    frames.append(frame)
                    frame_indices.append(frame_idx)
                
                frame_idx += 1
                
                if self.max_frames is not None and len(frames) >= self.max_frames:
                    break
        else:
            # Only load frames with bboxes and some surrounding frames
            bbox_frames = set(bbox['frame'] for bbox in bboxes)
            
            if len(bbox_frames) == 0:
                # No bboxes, load a few initial frames
                for i in range(min(10, self.max_frames or 10)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    frame_indices.append(i)
            else:
                min_frame = min(bbox_frames)
                max_frame = max(bbox_frames)
                
                # Load frames from min to max
                frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
                
                while frame_idx <= max_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_idx % self.sample_rate == 0:
                        frames.append(frame)
                        frame_indices.append(frame_idx)
                    
                    frame_idx += 1
                    
                    if self.max_frames is not None and len(frames) >= self.max_frames:
                        break
        
        cap.release()
        
        return frames, frame_indices


class AugmentedVideoGenerator:
    """
    Generate augmented videos and save to disk.
    """
    
    def __init__(self,
                 data_dir: str,
                 annotations_path: str,
                 output_dir: str,
                 aug_config: Optional[AugmentationConfig] = None):
        """
        Args:
            data_dir: Input data directory
            annotations_path: Path to annotations.json
            output_dir: Output directory for augmented data
            aug_config: Augmentation config
        """
        self.data_dir = Path(data_dir)
        self.annotations_path = annotations_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Setup augmenter
        self.aug_config = aug_config if aug_config is not None else get_default_config()
        ref_bank_root = self.data_dir.parent / 'augmented_ref_img'
        self.ref_bank = build_augmented_ref_bank(ref_bank_root) if ref_bank_root.exists() else None
        self.augmenter = VideoAugmenter(self.aug_config, ref_bank=self.ref_bank)
    
    def generate_augmented_dataset(self, start_from_idx: int = 0, skip_existing: bool = True):
        """
        Generate augmented dataset and save to disk.
        
        Args:
            start_from_idx: Start from which video (0-indexed), default 0
            skip_existing: Automatically skip already generated videos, default True
        """
        augmented_annotations = []
        
        # Load existing annotations if any
        output_ann_path = self.output_dir / 'annotations' / 'annotations_augmented.json'
        existing_video_ids = set()
        if skip_existing:
            # Method 1: Load từ annotations file
            if output_ann_path.exists():
                with open(output_ann_path, 'r') as f:
                    existing_anns = json.load(f)
                    existing_video_ids = {ann['video_id'] for ann in existing_anns}
                    augmented_annotations = existing_anns
                print(f"Found {len(augmented_annotations)} existing augmented samples from annotations")
            
            # Method 2: Check for existing folders (fallback if annotations are lost)
            samples_dir = self.output_dir / 'samples'
            if samples_dir.exists():
                for folder in samples_dir.iterdir():
                    if folder.is_dir() and '_aug_' in folder.name:
                        existing_video_ids.add(folder.name)
                if len(existing_video_ids) > len(augmented_annotations):
                    print(f"Found {len(existing_video_ids)} existing folders (some not in annotations)")

        
        for sample_idx, ann in enumerate(self.annotations):
            # Skip if before start_from_idx
            if sample_idx < start_from_idx:
                continue
                
            video_id = ann['video_id']
            video_path = self.data_dir / video_id / 'drone_video.mp4'
            
            if not video_path.exists():
                print(f"Skipping {video_id}: video not found")
                continue
            
            print(f"Processing {sample_idx + 1}/{len(self.annotations)}: {video_id}")
            
            # Extract bboxes
            bboxes = []
            for ann_group in ann.get('annotations', []):
                bboxes.extend(ann_group.get('bboxes', []))
            
            # Generate augmented versions
            for aug_idx in range(self.aug_config.augment_multiplier):
                aug_video_id = f"{video_id}_aug_{aug_idx}"
                
                # Skip if already exists
                if skip_existing and aug_video_id in existing_video_ids:
                    print(f"  Augmentation {aug_idx + 1}/{self.aug_config.augment_multiplier} - Already exists, skipping")
                    continue
                
                print(f"  Augmentation {aug_idx + 1}/{self.aug_config.augment_multiplier}")
                
                # Load video
                frames, frame_indices = self._load_video_frames(video_path, bboxes)
                
                # Filter bboxes for loaded frames
                sample_bboxes = [bbox for bbox in bboxes if bbox['frame'] in frame_indices]
                
                # Augment
                aug_frames, aug_bboxes, is_valid = self.augmenter.augment_video_clip(
                    frames,
                    sample_bboxes,
                    frame_indices=frame_indices,
                    video_id=video_id
                )
                
                if not is_valid:
                    print(f"    Skipping invalid augmentation")
                    continue
                
                # Save augmented video
                self._save_augmented_sample(
                    aug_video_id,
                    aug_frames,
                    aug_bboxes,
                    frame_indices
                )
                
                # Add to annotations
                augmented_annotations.append({
                    'video_id': aug_video_id,
                    'annotations': [{
                        'bboxes': aug_bboxes
                    }]
                })
                existing_video_ids.add(aug_video_id)
                
                # Save annotations incrementally (after each augmentation)
                output_ann_path = self.output_dir / 'annotations' / 'annotations_augmented.json'
                output_ann_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_ann_path, 'w') as f:
                    json.dump(augmented_annotations, f, indent=2)
            
            # Clear memory after each video
            gc.collect()
        
        # Final save
        output_ann_path = self.output_dir / 'annotations' / 'annotations_augmented.json'
        output_ann_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_ann_path, 'w') as f:
            json.dump(augmented_annotations, f, indent=2)
        
        print(f"\nGenerated {len(augmented_annotations)} augmented samples")
        print(f"Saved to {self.output_dir}")
    
    def _load_video_frames(self, 
                          video_path: Path,
                          bboxes: List[Dict],
                          max_frames: int = 500) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load video frames with bboxes (with limit to save RAM).
        
        Args:
            video_path: Path to video
            bboxes: List of bboxes
            max_frames: Max number of frames to load at a time (default 500 to save RAM)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        frame_indices = []
        
        bbox_frames = set(bbox['frame'] for bbox in bboxes)
        
        if len(bbox_frames) == 0:
            cap.release()
            return frames, frame_indices
        
        min_frame = min(bbox_frames)
        max_frame_original = max(bbox_frames)
        
        # Limit max_frame to avoid loading too many frames
        max_frame = min(min_frame + max_frames, max_frame_original)
        
        # Load frames from min to max
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
        frame_idx = min_frame
        
        while frame_idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_indices.append(frame_idx)
            frame_idx += 1
            
            # Safety limit
            if len(frames) >= max_frames:
                break
        
        cap.release()
        
        return frames, frame_indices
    
    def _save_augmented_sample(self,
                              video_id: str,
                              frames: List[np.ndarray],
                              bboxes: List[Dict],
                              frame_indices: List[int]):
        """Save augmented sample."""
        # Create output directory
        output_sample_dir = self.output_dir / 'samples' / video_id
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save video
        if self.aug_config.save_augmented_video:
            output_video_path = output_sample_dir / 'drone_video.mp4'
            self._save_video(output_video_path, frames)
        else:
            # Save frames as images
            frames_dir = output_sample_dir / 'frames'
            frames_dir.mkdir(exist_ok=True)
            
            for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
                frame_path = frames_dir / f'frame_{frame_idx:06d}.jpg'
                cv2.imwrite(str(frame_path), frame)
    
    def _save_video(self, output_path: Path, frames: List[np.ndarray]):
        """Save frames as video."""
        if len(frames) == 0:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.aug_config.output_video_fps,
            (width, height)
        )
        
        for frame in frames:
            out.write(frame)
        
        out.release()
