"""
Quick test script to verify the augmentation system is working.
"""

import sys
sys.path.append('augmentation')

import numpy as np
import cv2
from augmentation import (
    VideoAugmenter,
    get_default_config,
    bbox_utils
)


def test_bbox_utils():
    """Test bbox transformation utilities."""
    print("Testing bbox_utils...")
    
    bbox = {'frame': 0, 'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    img_width, img_height = 640, 480
    
    # Test horizontal flip
    flipped = bbox_utils.transform_bbox_horizontal_flip(bbox, img_width)
    assert flipped['x1'] == img_width - bbox['x2']
    assert flipped['x2'] == img_width - bbox['x1']
    print("  ✓ Horizontal flip works")
    
    # Test rotation
    rotated = bbox_utils.transform_bbox_rotation(bbox, 10, img_width, img_height)
    assert rotated is not None
    print("  ✓ Rotation works")
    
    # Test scale
    scaled = bbox_utils.transform_bbox_scale(bbox, 1.2, 1.2, 
                                             int(img_width*1.2), int(img_height*1.2))
    assert scaled is not None
    print("  ✓ Scale works")
    
    # Test IoU
    iou = bbox_utils.calculate_iou(bbox, bbox)
    assert abs(iou - 1.0) < 0.01
    print("  ✓ IoU calculation works")
    
    print("bbox_utils: OK ✓\n")


def test_augmenter():
    """Test video augmenter."""
    print("Testing VideoAugmenter...")
    
    # Create dummy frames
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
              for _ in range(10)]
    
    # Create dummy bboxes
    bboxes = [
        {'frame': i, 'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
        for i in range(0, 10, 2)
    ]
    
    # Setup augmenter
    config = get_default_config()
    augmenter = VideoAugmenter(config)
    
    # Test augmentation
    aug_frames, aug_bboxes, is_valid = augmenter.augment_video_clip(frames, bboxes)
    
    assert len(aug_frames) == len(frames), "Frame count mismatch"
    assert len(aug_bboxes) <= len(bboxes), "Too many bboxes"
    assert all(f.shape == frames[0].shape for f in aug_frames), "Frame shape mismatch"
    
    print("  ✓ Augmentation runs without errors")
    print(f"  ✓ Input: {len(frames)} frames, {len(bboxes)} bboxes")
    print(f"  ✓ Output: {len(aug_frames)} frames, {len(aug_bboxes)} bboxes")
    print(f"  ✓ Valid: {is_valid}")
    
    print("VideoAugmenter: OK ✓\n")


def test_config():
    """Test config presets."""
    print("Testing configs...")
    
    from augmentation import (
        get_default_config,
        get_conservative_config,
        get_aggressive_config
    )
    
    default = get_default_config()
    conservative = get_conservative_config()
    aggressive = get_aggressive_config()
    
    assert default.spatial.horizontal_flip_prob > 0
    assert conservative.augment_multiplier < aggressive.augment_multiplier
    
    print("  ✓ Default config loaded")
    print("  ✓ Conservative config loaded")
    print("  ✓ Aggressive config loaded")
    print(f"  ✓ Multipliers: conservative={conservative.augment_multiplier}, "
          f"default={default.augment_multiplier}, aggressive={aggressive.augment_multiplier}")
    
    print("Configs: OK ✓\n")


def test_integration():
    """Test end-to-end augmentation."""
    print("Testing end-to-end augmentation...")
    
    # Create test frames with varying content
    frames = []
    for i in range(20):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a moving rectangle
        x = 100 + i * 10
        y = 100 + i * 5
        cv2.rectangle(frame, (x, y), (x+50, y+50), (255, 255, 255), -1)
        frames.append(frame)
    
    # Bboxes following the rectangle
    bboxes = [
        {'frame': i, 'x1': 100+i*10, 'y1': 100+i*5, 
         'x2': 150+i*10, 'y2': 150+i*5}
        for i in range(20)
    ]
    
    # Test with different configs
    from augmentation import get_aggressive_config
    
    config = get_aggressive_config()
    augmenter = VideoAugmenter(config)
    
    # Multiple augmentations
    success_count = 0
    for i in range(5):
        aug_frames, aug_bboxes, is_valid = augmenter.augment_video_clip(
            [f.copy() for f in frames],
            [b.copy() for b in bboxes]
        )
        
        if is_valid:
            success_count += 1
            
            # Verify bboxes are within frame bounds
            for bbox in aug_bboxes:
                assert 0 <= bbox['x1'] < bbox['x2'] <= 640, f"Invalid x coords: {bbox}"
                assert 0 <= bbox['y1'] < bbox['y2'] <= 480, f"Invalid y coords: {bbox}"
    
    print(f"  ✓ Ran 5 augmentations, {success_count} valid")
    print(f"  ✓ All augmented bboxes are within frame bounds")
    
    print("Integration test: OK ✓\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("QUICK TEST - Video Augmentation System")
    print("=" * 60 + "\n")
    
    try:
        test_bbox_utils()
        test_config()
        test_augmenter()
        test_integration()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nAugmentation system is ready to use.")
        print("\nNext steps:")
        print("  1. See README.md for usage instructions")
        print("  2. Run example_usage.py to see examples")
        print("  3. Run demo_augmentation.py to visualize")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())


