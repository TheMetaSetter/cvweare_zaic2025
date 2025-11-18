"""
Example script demonstrating how to use video augmentation.
"""

import sys
sys.path.append('augmentation')

from augmentation import (
    DroneVideoDataset,
    AugmentedVideoGenerator,
    get_default_config,
    get_aggressive_config,
    VideoAugmenter
)
import cv2
import numpy as np


def example_1_pytorch_dataset():
    """
    Example 1: Using PyTorch Dataset with augmentation.
    """
    print("=" * 60)
    print("EXAMPLE 1: PyTorch Dataset with Augmentation")
    print("=" * 60)
    
    # Create dataset
    dataset = DroneVideoDataset(
        data_dir='data/observing_unzipped/train/samples',
        annotations_path='data/observing_unzipped/train/annotations/annotations.json',
        augment=True,
        aug_config=get_default_config(),
        load_full_video=False,  # Only load frames with bbox
        sample_rate=1,
        max_frames=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        
        print(f"\nSample {i}:")
        print(f"  Video ID: {sample['video_id']}")
        print(f"  Number of frames: {len(sample['frames'])}")
        print(f"  Number of bboxes: {len(sample['bboxes'])}")
        print(f"  Augmented: {sample['is_augmented']}")
        
        if len(sample['bboxes']) > 0:
            bbox = sample['bboxes'][0]
            print(f"  First bbox: frame={bbox['frame']}, "
                  f"x1={bbox['x1']}, y1={bbox['y1']}, "
                  f"x2={bbox['x2']}, y2={bbox['y2']}")


def example_2_manual_augmentation():
    """
    Example 2: Manual augmentation with VideoAugmenter.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Manual Augmentation")
    print("=" * 60)
    
    # Load a video
    video_path = 'data/observing_unzipped/train/samples/Backpack_0/drone_video.mp4'
    
    cap = cv2.VideoCapture(video_path)
    
    # Load first 50 frames
    frames = []
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    print(f"Loaded {len(frames)} frames")
    
    # Create dummy bboxes
    bboxes = [
        {'frame': i, 'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
        for i in range(0, len(frames), 10)
    ]
    
    print(f"Created {len(bboxes)} bboxes")
    
    # Setup augmenter
    config = get_aggressive_config()
    augmenter = VideoAugmenter(config)
    
    # Augment frames and bboxes
    print("\nAugmenting...")
    aug_frames, aug_bboxes, is_valid = augmenter.augment_video_clip(frames, bboxes)
    
    print(f"Augmentation valid: {is_valid}")
    print(f"Augmented frames: {len(aug_frames)}")
    print(f"Augmented bboxes: {len(aug_bboxes)}")
    
    if is_valid and len(aug_bboxes) > 0:
        print(f"\nFirst augmented bbox: {aug_bboxes[0]}")


def example_3_generate_dataset():
    """
    Example 3: Generate augmented dataset and save to disk.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Generate Augmented Dataset")
    print("=" * 60)
    
    # Create generator
    generator = AugmentedVideoGenerator(
        data_dir='data/observing_unzipped/train/samples',
        annotations_path='data/observing_unzipped/train/annotations/annotations.json',
        output_dir='data/augmented_demo',
        aug_config=get_default_config()
    )
    
    print("Generator created")
    print("Note: Uncomment the line below to generate the dataset (this may take a long time)")
    
    # Uncomment to generate
    # generator.generate_augmented_dataset()


def example_4_custom_config():
    """
    Example 4: Create custom augmentation config.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Augmentation Config")
    print("=" * 60)
    
    from augmentation import (
        AugmentationConfig,
        SpatialAugmentConfig,
        PixelAugmentConfig,
        TemporalConfig
    )
    
    # Create custom config
    custom_config = AugmentationConfig(
        spatial=SpatialAugmentConfig(
            horizontal_flip_prob=0.5,
            rotation_prob=0.4,
            rotation_angle_range=(-15, 15),
            scale_prob=0.3,
            scale_range=(0.85, 1.15),
            crop_prob=0.2,
            min_bbox_area_ratio=0.4,  # Reject if bbox is too small
        ),
        pixel=PixelAugmentConfig(
            color_jitter_prob=0.8,
            brightness_range=(0.7, 1.3),
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2),
            temporal_variation=True,  # Enable smooth variation
            blur_prob=0.3,
            noise_prob=0.2,
            fog_prob=0.15,
            rain_prob=0.1,
        ),
        temporal=TemporalConfig(
            consistency_mode="fixed",  # Fixed spatial transforms
            smooth_window_size=30,
        ),
        augment_multiplier=4,  # 4x augmentation
        seed=42  # Reproducibility
    )
    
    print("Custom config created:")
    print(f"  Spatial augmentation:")
    print(f"    - Horizontal flip prob: {custom_config.spatial.horizontal_flip_prob}")
    print(f"    - Rotation range: {custom_config.spatial.rotation_angle_range}")
    print(f"    - Scale range: {custom_config.spatial.scale_range}")
    print(f"  Pixel augmentation:")
    print(f"    - Color jitter prob: {custom_config.pixel.color_jitter_prob}")
    print(f"    - Temporal variation: {custom_config.pixel.temporal_variation}")
    print(f"    - Fog prob: {custom_config.pixel.fog_prob}")
    print(f"  Augment multiplier: {custom_config.augment_multiplier}")
    
    # Use with dataset
    dataset = DroneVideoDataset(
        data_dir='data/observing_unzipped/train/samples',
        annotations_path='data/observing_unzipped/train/annotations/annotations.json',
        augment=True,
        aug_config=custom_config,
        max_frames=50
    )
    
    print(f"\nDataset with custom config created")
    print(f"Dataset size: {len(dataset)}")


def example_5_bbox_transforms():
    """
    Example 5: Test bbox transformations.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Bbox Transformations")
    print("=" * 60)
    
    from augmentation import bbox_utils
    
    # Original bbox
    bbox = {
        'frame': 0,
        'x1': 100,
        'y1': 100,
        'x2': 200,
        'y2': 200
    }
    
    img_width = 640
    img_height = 480
    
    print(f"Original bbox: {bbox}")
    print(f"Image size: {img_width}x{img_height}")
    
    # Test horizontal flip
    flipped_bbox = bbox_utils.transform_bbox_horizontal_flip(bbox, img_width)
    print(f"\nAfter horizontal flip: {flipped_bbox}")
    
    # Test rotation
    rotated_bbox = bbox_utils.transform_bbox_rotation(
        bbox, angle=15, img_width=img_width, img_height=img_height
    )
    print(f"After 15° rotation: {rotated_bbox}")
    
    # Test scale
    scaled_bbox = bbox_utils.transform_bbox_scale(
        bbox, scale_x=1.2, scale_y=1.2, 
        img_width=int(img_width*1.2), img_height=int(img_height*1.2)
    )
    print(f"After 1.2x scale: {scaled_bbox}")
    
    # Test IoU
    iou = bbox_utils.calculate_iou(bbox, flipped_bbox)
    print(f"\nIoU between original and flipped: {iou:.3f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("VIDEO AUGMENTATION EXAMPLES")
    print("=" * 60)
    
    try:
        # Example 1
        example_1_pytorch_dataset()
        
        # Example 2
        example_2_manual_augmentation()
        
        # Example 3
        example_3_generate_dataset()
        
        # Example 4
        example_4_custom_config()
        
        # Example 5
        example_5_bbox_transforms()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have the data in the correct location:")
        print("  - data/observing_unzipped/train/samples/")
        print("  - data/observing_unzipped/train/annotations/annotations.json")


if __name__ == '__main__':
    main()