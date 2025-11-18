"""
Video Augmentation Package cho Drone Object Detection Challenge.

Package này cung cấp augmentation cho video frames với temporal consistency
và automatic bounding box transformation.
"""

from .config import (
    AugmentationConfig,
    SpatialAugmentConfig,
    PixelAugmentConfig,
    TemporalConfig,
    get_default_config,
    get_conservative_config,
    get_aggressive_config
)

from .video_augmenter import VideoAugmenter, AugmentationState

from .dataset import DroneVideoDataset, AugmentedVideoGenerator

from . import bbox_utils

__version__ = '1.0.0'

__all__ = [
    'AugmentationConfig',
    'SpatialAugmentConfig',
    'PixelAugmentConfig',
    'TemporalConfig',
    'get_default_config',
    'get_conservative_config',
    'get_aggressive_config',
    'VideoAugmenter',
    'AugmentationState',
    'DroneVideoDataset',
    'AugmentedVideoGenerator',
    'bbox_utils',
]


