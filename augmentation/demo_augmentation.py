"""
Demo script để test và visualize video augmentation.
"""

import os
import sys
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from augmentation.config import (
    AugmentationConfig,
    get_default_config,
    get_conservative_config,
    get_aggressive_config
)
from augmentation.video_augmenter import VideoAugmenter
from augmentation.dataset import DroneVideoDataset, AugmentedVideoGenerator


def draw_bbox(img: np.ndarray, bbox: Dict, color=(0, 255, 0), thickness=2):
    """Draw bounding box lên frame."""
    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw frame number
    frame_text = f"F:{bbox['frame']}"
    cv2.putText(img, frame_text, (x1, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img


def visualize_augmentation(video_id: str,
                          data_dir: str,
                          annotations_path: str,
                          config: AugmentationConfig,
                          num_augmentations: int = 3,
                          max_frames: int = 300):
    """
    Visualize augmentation cho một video.
    
    Args:
        video_id: Video ID
        data_dir: Data directory
        annotations_path: Path to annotations
        config: Augmentation config
        num_augmentations: Số augmented versions để visualize
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Find video annotation
    video_ann = None
    for ann in annotations:
        if ann['video_id'] == video_id:
            video_ann = ann
            break
    
    if video_ann is None:
        print(f"Video {video_id} not found in annotations")
        return
    
    # Extract bboxes
    bboxes = []
    for ann_group in video_ann.get('annotations', []):
        bboxes.extend(ann_group.get('bboxes', []))
    
    if len(bboxes) == 0:
        print(f"No bounding boxes found for {video_id}")
        return
    
    # Load video frames
    video_path = Path(data_dir) / video_id / 'drone_video.mp4'
    
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    
    print(f"Loading video: {video_path}")
    
    # Load frames có bbox (limit to max_frames)
    bbox_frames = set(bbox['frame'] for bbox in bboxes)
    min_frame = min(bbox_frames)
    max_frame_original = max(bbox_frames)
    
    # Limit max_frame for visualization
    max_frame = min(min_frame + max_frames, max_frame_original)
    
    print(f"Loading frames {min_frame} to {max_frame} (max {max_frames} frames for visualization)")
    if max_frame < max_frame_original:
        print(f"Note: Video has bboxes up to frame {max_frame_original}, showing first {max_frames} only")
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
    
    original_frames = []
    frame_indices = []
    frame_idx = min_frame
    
    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frames.append(frame.copy())
        frame_indices.append(frame_idx)
        frame_idx += 1
        
        # Safety check
        if len(original_frames) >= max_frames:
            break
    
    cap.release()
    
    print(f"Loaded {len(original_frames)} frames")
    
    # Filter bboxes
    sample_bboxes = [bbox for bbox in bboxes if bbox['frame'] in frame_indices]
    
    # Setup augmenter
    augmenter = VideoAugmenter(config)
    
    # Create visualization
    print(f"\nGenerating {num_augmentations} augmented versions...")
    
    all_versions = []
    all_bboxes = []
    
    # Original
    all_versions.append(original_frames)
    all_bboxes.append(sample_bboxes)
    
    # Augmented versions
    for i in range(num_augmentations):
        aug_frames, aug_bboxes, is_valid = augmenter.augment_video_clip(
            original_frames.copy(), 
            [bbox.copy() for bbox in sample_bboxes]
        )
        
        if is_valid:
            all_versions.append(aug_frames)
            all_bboxes.append(aug_bboxes)
            print(f"  Aug {i+1}: Valid ✓")
        else:
            print(f"  Aug {i+1}: Invalid ✗ (rejected)")
    
    # Visualize side-by-side
    print(f"\nVisualizing {len(all_versions)} versions...")
    print("Press 'q' to quit, space to pause, 'n' for next frame")
    
    frame_idx = 0
    paused = False
    
    while True:
        if not paused:
            # Get current frame from each version
            display_frames = []
            
            for version_idx, (frames, bboxes) in enumerate(zip(all_versions, all_bboxes)):
                if frame_idx < len(frames):
                    frame = frames[frame_idx].copy()
                    
                    # Draw bboxes for this frame
                    current_frame_num = frame_indices[frame_idx]
                    for bbox in bboxes:
                        if bbox['frame'] == current_frame_num:
                            color = (0, 255, 0) if version_idx == 0 else (0, 0, 255)
                            frame = draw_bbox(frame, bbox, color=color)
                    
                    # Add label
                    label = "Original" if version_idx == 0 else f"Aug {version_idx}"
                    cv2.putText(frame, label, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    display_frames.append(frame)
            
            # Arrange frames in grid
            if len(display_frames) > 0:
                grid = create_grid(display_frames)
                
                # Add progress info
                progress_text = f"Frame: {frame_idx + 1}/{len(original_frames)} ({current_frame_num})"
                cv2.putText(grid, progress_text, (10, grid.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Video Augmentation Comparison', grid)
            
            frame_idx += 1
            
            # Loop back
            if frame_idx >= len(original_frames):
                frame_idx = 0
        
        # Handle keyboard
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n') and paused:
            frame_idx = (frame_idx + 1) % len(original_frames)
            paused = False
    
    cv2.destroyAllWindows()


def create_grid(frames: List[np.ndarray], cols: int = 2) -> np.ndarray:
    """
    Arrange frames in grid.
    """
    if len(frames) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize frames to same size
    target_height = 480
    resized_frames = []
    
    for frame in frames:
        h, w = frame.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        new_h = target_height
        resized = cv2.resize(frame, (new_w, new_h))
        resized_frames.append(resized)
    
    # Find max width
    max_width = max(f.shape[1] for f in resized_frames)
    
    # Pad frames to same width
    padded_frames = []
    for frame in resized_frames:
        if frame.shape[1] < max_width:
            pad = max_width - frame.shape[1]
            frame = cv2.copyMakeBorder(frame, 0, 0, 0, pad, cv2.BORDER_CONSTANT)
        padded_frames.append(frame)
    
    # Create grid
    rows = (len(padded_frames) + cols - 1) // cols
    grid_rows = []
    
    for i in range(rows):
        row_frames = padded_frames[i*cols:(i+1)*cols]
        
        # Pad row if needed
        while len(row_frames) < cols:
            row_frames.append(np.zeros_like(row_frames[0]))
        
        row = np.hstack(row_frames)
        grid_rows.append(row)
    
    grid = np.vstack(grid_rows)
    
    return grid


def test_single_frame_augmentation(data_dir: str,
                                   annotations_path: str,
                                   config: AugmentationConfig):
    """
    Test augmentation trên một vài frames để verify bbox transforms.
    """
    # Load một sample
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    if len(annotations) == 0:
        print("No annotations found")
        return
    
    # Get first video với bboxes
    video_ann = annotations[0]
    video_id = video_ann['video_id']
    
    bboxes = []
    for ann_group in video_ann.get('annotations', []):
        bboxes.extend(ann_group.get('bboxes', []))
    
    if len(bboxes) == 0:
        print(f"No bboxes in {video_id}")
        return
    
    # Load one frame
    video_path = Path(data_dir) / video_id / 'drone_video.mp4'
    bbox = bboxes[0]
    frame_num = bbox['frame']
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to load frame")
        return
    
    print(f"Testing on {video_id}, frame {frame_num}")
    print(f"Original bbox: {bbox}")
    
    # Setup augmenter
    augmenter = VideoAugmenter(config)
    
    # Test augmentation
    aug_frames, aug_bboxes, is_valid = augmenter.augment_video_clip([frame], [bbox])
    
    print(f"Augmentation valid: {is_valid}")
    if len(aug_bboxes) > 0:
        print(f"Augmented bbox: {aug_bboxes[0]}")
    
    # Visualize
    original_vis = frame.copy()
    draw_bbox(original_vis, bbox, color=(0, 255, 0))
    
    if len(aug_frames) > 0 and len(aug_bboxes) > 0:
        aug_vis = aug_frames[0].copy()
        draw_bbox(aug_vis, aug_bboxes[0], color=(0, 0, 255))
        
        combined = np.hstack([original_vis, aug_vis])
        
        cv2.imshow('Original vs Augmented', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def generate_augmented_dataset_cli(data_dir: str,
                                   annotations_path: str,
                                   output_dir: str,
                                   config_name: str,
                                   start_from: int = 0,
                                   skip_existing: bool = True,
                                   max_frames_per_video: int = 500):
    """
    Generate augmented dataset từ command line.
    """
    # Get config
    if config_name == 'default':
        config = get_default_config()
    elif config_name == 'conservative':
        config = get_conservative_config()
    elif config_name == 'aggressive':
        config = get_aggressive_config()
    else:
        print(f"Unknown config: {config_name}")
        return
    
    print(f"Generating augmented dataset with config: {config_name}")
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentation multiplier: {config.augment_multiplier}")
    print(f"Max frames per video: {max_frames_per_video} (để tiết kiệm RAM)")
    if start_from > 0:
        print(f"Starting from video index: {start_from}")
    if skip_existing:
        print(f"Skip existing: enabled")
    
    # Create generator
    generator = AugmentedVideoGenerator(
        data_dir=data_dir,
        annotations_path=annotations_path,
        output_dir=output_dir,
        aug_config=config
    )
    
    # Override _load_video_frames với max_frames
    original_load = generator._load_video_frames
    generator._load_video_frames = lambda vp, bb: original_load(vp, bb, max_frames=max_frames_per_video)
    
    # Generate
    generator.generate_augmented_dataset(
        start_from_idx=start_from,
        skip_existing=skip_existing
    )


def main():
    parser = argparse.ArgumentParser(description='Demo video augmentation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['visualize', 'test', 'generate'],
                       help='Mode: visualize, test, hoặc generate')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory (samples/)')
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to annotations.json')
    parser.add_argument('--video_id', type=str,
                       help='Video ID to visualize (chỉ cho mode=visualize)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (chỉ cho mode=generate)')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'conservative', 'aggressive'],
                       help='Augmentation config preset')
    parser.add_argument('--num_augs', type=int, default=3,
                       help='Số augmentations để visualize')
    parser.add_argument('--max_frames', type=int, default=300,
                       help='Max số frames để load (để tránh out of memory)')
    parser.add_argument('--start_from', type=int, default=0,
                       help='Start từ video index (0-indexed), dùng để resume')
    parser.add_argument('--no_skip_existing', action='store_true',
                       help='Không skip videos đã generate (re-generate tất cả)')
    parser.add_argument('--max_frames_per_video', type=int, default=500,
                       help='Max số frames load mỗi video (default 500, giảm nếu out of memory)')
    
    args = parser.parse_args()
    
    # Get config
    if args.config == 'default':
        config = get_default_config()
    elif args.config == 'conservative':
        config = get_conservative_config()
    elif args.config == 'aggressive':
        config = get_aggressive_config()
    
    # Run mode
    if args.mode == 'visualize':
        if not args.video_id:
            print("Error: --video_id required for visualize mode")
            return
        
        visualize_augmentation(
            video_id=args.video_id,
            data_dir=args.data_dir,
            annotations_path=args.annotations,
            config=config,
            num_augmentations=args.num_augs,
            max_frames=args.max_frames
        )
    
    elif args.mode == 'test':
        test_single_frame_augmentation(
            data_dir=args.data_dir,
            annotations_path=args.annotations,
            config=config
        )
    
    elif args.mode == 'generate':
        if not args.output_dir:
            print("Error: --output_dir required for generate mode")
            return
        
        generate_augmented_dataset_cli(
            data_dir=args.data_dir,
            annotations_path=args.annotations,
            output_dir=args.output_dir,
            config_name=args.config,
            start_from=args.start_from,
            skip_existing=not args.no_skip_existing,
            max_frames_per_video=args.max_frames_per_video
        )


if __name__ == '__main__':
    main()

