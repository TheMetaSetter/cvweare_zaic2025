"""
Configuration cho video augmentation.
Định nghĩa các augmentation parameters và probabilities.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class SpatialAugmentConfig:
    """
    Config cho spatial transforms (cần update bbox).
    Các transforms này được apply NHẤT QUÁN cho toàn bộ frames trong clip.
    """
    
    # Horizontal Flip
    horizontal_flip_prob: float = 0.5
    
    # Rotation
    rotation_prob: float = 0.3
    rotation_angle_range: Tuple[float, float] = (-10, 10)  # degrees
    
    # Scale/Zoom
    scale_prob: float = 0.3
    scale_range: Tuple[float, float] = (0.8, 1.2)  # scale factor
    
    # Random Crop (tỷ lệ crop so với original)
    crop_prob: float = 0.2
    crop_scale_range: Tuple[float, float] = (0.8, 1.0)  # crop size / original size
    
    # Affine Transform
    affine_prob: float = 0.2
    affine_translate_percent: Tuple[float, float] = (-0.1, 0.1)  # % of image size
    affine_scale: Tuple[float, float] = (0.9, 1.1)
    affine_shear: Tuple[float, float] = (-5, 5)  # degrees
    
    # Validation - drop augmented sample nếu bbox quá nhỏ hoặc bị crop quá nhiều
    min_bbox_area_ratio: float = 0.3  # so với original bbox area
    min_iou_threshold: float = 0.5  # IoU với original bbox


@dataclass
class PixelAugmentConfig:
    """
    Config cho pixel-level transforms (không ảnh hưởng bbox).
    Có thể vary mượt mà theo thời gian để simulate lighting changes.
    """
    
    # Color Jittering
    color_jitter_prob: float = 0.7
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: Tuple[float, float] = (-0.05, 0.05)  # normalized [-0.5, 0.5]
    
    # Temporal variation - parameters change smoothly across frames
    temporal_variation: bool = True
    temporal_variation_strength: float = 0.3  # 0-1, mức độ thay đổi theo thời gian
    
    # Gaussian Blur
    blur_prob: float = 0.2
    blur_kernel_size: Tuple[int, int] = (3, 7)  # must be odd
    
    # Gaussian Noise
    noise_prob: float = 0.2
    noise_std_range: Tuple[float, float] = (0, 25)  # standard deviation
    
    # Weather Effects
    rain_prob: float = 0.1
    fog_prob: float = 0.1
    fog_intensity_range: Tuple[float, float] = (0.3, 0.7)
    
    # Motion Blur (simulate camera shake hoặc fast movement)
    motion_blur_prob: float = 0.15
    motion_blur_kernel_size_range: Tuple[int, int] = (5, 15)
    motion_blur_angle_range: Tuple[float, float] = (0, 360)


@dataclass
class TemporalConfig:
    """
    Config cho temporal consistency và sampling.
    """
    
    # Temporal consistency mode
    consistency_mode: str = "fixed"  # "fixed" hoặc "smooth"
    # - "fixed": apply cùng một transform cho tất cả frames
    # - "smooth": interpolate parameters mượt mà giữa các frames
    
    # Smooth interpolation settings (chỉ dùng khi mode = "smooth")
    smooth_window_size: int = 30  # số frames cho một interpolation window
    smooth_variation_strength: float = 0.2  # mức độ variation
    
    # Frame sampling (để giảm computation khi augment)
    sample_frames: bool = False  # True = chỉ augment một subset frames
    sample_rate: int = 5  # augment mỗi N frames
    
    # Clip settings
    min_clip_length: int = 10  # min số frames có bbox trong một clip
    max_clip_length: Optional[int] = None  # max frames, None = unlimited


@dataclass
class AugmentationConfig:
    """
    Main augmentation configuration.
    """
    
    spatial: SpatialAugmentConfig = field(default_factory=SpatialAugmentConfig) # TODO: Giải thích hàm field, tham số default_factory
    pixel: PixelAugmentConfig = field(default_factory=PixelAugmentConfig)       # TODO: Giải thích hàm field, tham số default_factory
    temporal: TemporalConfig = field(default_factory=TemporalConfig)            # TODO: Giải thích hàm field, tham số default_factory
    
    # Global settings
    seed: Optional[int] = None      # Random seed cho reproducibility
    augment_multiplier: int = 3     # số lần augment mỗi video
    
    # Output settings
    save_augmented_video: bool = False  # True = save video, False = chỉ save frames khi cần
    output_video_fps: int = 25
    output_video_quality: int = 95      # 0-100


# Preset configs
def get_conservative_config() -> AugmentationConfig:
    """
    Conservative augmentation - ít aggressive, phù hợp cho validation.
    """
    return AugmentationConfig(
        spatial=SpatialAugmentConfig(
            horizontal_flip_prob=0.3,
            rotation_prob=0.2,
            rotation_angle_range=(-5, 5),
            scale_prob=0.2,
            scale_range=(0.9, 1.1),
            crop_prob=0.1,
            affine_prob=0.1,
        ),
        pixel=PixelAugmentConfig(
            color_jitter_prob=0.5,
            brightness_range=(0.8, 1.2),
            contrast_range=(0.9, 1.1),
            blur_prob=0.1,
            noise_prob=0.1,
            rain_prob=0.05,
            fog_prob=0.05,
        ),
        temporal=TemporalConfig(
            consistency_mode="fixed"
        ),
        augment_multiplier=2
    )


def get_aggressive_config() -> AugmentationConfig:
    """
    Aggressive augmentation - nhiều variation, phù hợp cho training.
    """
    return AugmentationConfig(
        spatial=SpatialAugmentConfig(
            horizontal_flip_prob=0.5,
            rotation_prob=0.4,
            rotation_angle_range=(-15, 15),
            scale_prob=0.4,
            scale_range=(0.7, 1.3),
            crop_prob=0.3,
            affine_prob=0.3,
        ),
        pixel=PixelAugmentConfig(
            color_jitter_prob=0.8,
            brightness_range=(0.6, 1.4),
            contrast_range=(0.7, 1.3),
            saturation_range=(0.7, 1.3),
            blur_prob=0.3,
            noise_prob=0.3,
            rain_prob=0.15,
            fog_prob=0.15,
            motion_blur_prob=0.2,
        ),
        temporal=TemporalConfig(
            consistency_mode="smooth",
            smooth_variation_strength=0.3
        ),
        augment_multiplier=5
    )


def get_default_config() -> AugmentationConfig:
    """
    Default balanced augmentation.
    """
    return AugmentationConfig()


