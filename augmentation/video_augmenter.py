"""
Video Frame Augmenter với temporal consistency.
Core module cho augmentation của drone video frames.
"""

import numpy as np
import cv2
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .config import AugmentationConfig, SpatialAugmentConfig, PixelAugmentConfig, TemporalConfig
from . import bbox_utils


@dataclass
class AugmentationState:
    """
    Lưu trữ augmentation parameters cho một video clip.
    Đảm bảo temporal consistency.
    """
    # Spatial transforms (fixed cho toàn bộ clip)
    apply_horizontal_flip: bool = False
    apply_rotation: bool = False
    rotation_angle: float = 0.0
    apply_scale: bool = False
    scale_x: float = 1.0
    scale_y: float = 1.0
    apply_crop: bool = False
    crop_coords: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    apply_affine: bool = False
    affine_matrix: Optional[np.ndarray] = None
    
    # Pixel transforms (có thể vary theo thời gian)
    apply_color_jitter: bool = False
    brightness_factors: Optional[np.ndarray] = None  # per frame
    contrast_factors: Optional[np.ndarray] = None
    saturation_factors: Optional[np.ndarray] = None
    hue_factors: Optional[np.ndarray] = None
    
    apply_blur: bool = False
    blur_kernel_size: int = 3
    
    apply_noise: bool = False
    noise_std: float = 0.0
    
    apply_rain: bool = False
    apply_fog: bool = False
    fog_intensity: float = 0.5
    
    apply_motion_blur: bool = False
    motion_blur_kernel_size: int = 5
    motion_blur_angle: float = 0.0


class VideoAugmenter:
    """
    Main class cho video frame augmentation với temporal consistency.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    # Các hàm _sample dùng để sample parameter cụ thể cho augmentation dựa vào config mà người dùng đưa.
    def _sample_spatial_transforms(self, 
                                   img_width: int, 
                                   img_height: int,
                                   num_frames: int) -> AugmentationState:
        """
        Sample spatial transform parameters.
        Các parameters này sẽ được apply nhất quán cho toàn bộ frames.
        """
        state = AugmentationState()
        cfg = self.config.spatial
        
        # Horizontal flip
        state.apply_horizontal_flip: bool = self.rng.rand() < cfg.horizontal_flip_prob
        
        # Rotation
        state.apply_rotation: bool = self.rng.rand() < cfg.rotation_prob
        if state.apply_rotation:
            state.rotation_angle: float = self.rng.uniform(*cfg.rotation_angle_range)
        
        # Scale
        state.apply_scale: bool = self.rng.rand() < cfg.scale_prob
        if state.apply_scale:
            scale: float = self.rng.uniform(*cfg.scale_range)
            state.scale_x: float = scale
            state.scale_y: float = scale
        
        # Crop
        state.apply_crop: bool = self.rng.rand() < cfg.crop_prob
        if state.apply_crop:
            crop_scale: float = self.rng.uniform(*cfg.crop_scale_range)
            crop_w = int(img_width * crop_scale)
            crop_h = int(img_height * crop_scale)
            
            x1 = self.rng.randint(0, img_width - crop_w + 1)
            y1 = self.rng.randint(0, img_height - crop_h + 1)
            state.crop_coords = (x1, y1, x1 + crop_w, y1 + crop_h)
        
        # Affine
        state.apply_affine = self.rng.rand() < cfg.affine_prob
        if state.apply_affine:
            # Sample affine parameters
            translate_x = self.rng.uniform(*cfg.affine_translate_percent) * img_width
            translate_y = self.rng.uniform(*cfg.affine_translate_percent) * img_height
            scale = self.rng.uniform(*cfg.affine_scale)
            shear = self.rng.uniform(*cfg.affine_shear)
            
            # Create affine matrix
            center = (img_width / 2, img_height / 2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            M[0, 2] += translate_x
            M[1, 2] += translate_y
            
            # Add shear
            shear_rad = np.deg2rad(shear)
            # Convert M to 3x3 for proper matrix multiplication
            M_3x3 = np.vstack([M, [0, 0, 1]])
            shear_matrix = np.array([
                [1, np.tan(shear_rad), 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            # Multiply and take first 2 rows
            state.affine_matrix = (M_3x3 @ shear_matrix)[:2]
        
        return state
    
    def _sample_pixel_transforms(self, num_frames: int) -> AugmentationState:
        """
        Sample pixel-level transform parameters.
        Có thể vary mượt mà theo thời gian nếu temporal_variation=True.
        """
        state = AugmentationState()
        cfg = self.config.pixel
        
        # Color jittering
        state.apply_color_jitter = self.rng.rand() < cfg.color_jitter_prob
        if state.apply_color_jitter:
            if cfg.temporal_variation:
                # Generate smooth varying factors
                state.brightness_factors = self._generate_smooth_factors(
                    num_frames, cfg.brightness_range, cfg.temporal_variation_strength
                )
                state.contrast_factors = self._generate_smooth_factors(
                    num_frames, cfg.contrast_range, cfg.temporal_variation_strength
                )
                state.saturation_factors = self._generate_smooth_factors(
                    num_frames, cfg.saturation_range, cfg.temporal_variation_strength
                )
                state.hue_factors = self._generate_smooth_factors(
                    num_frames, cfg.hue_range, cfg.temporal_variation_strength
                )
            else:
                # Fixed factors
                brightness = self.rng.uniform(*cfg.brightness_range)
                contrast = self.rng.uniform(*cfg.contrast_range)
                saturation = self.rng.uniform(*cfg.saturation_range)
                hue = self.rng.uniform(*cfg.hue_range)
                
                state.brightness_factors = np.full(num_frames, brightness)
                state.contrast_factors = np.full(num_frames, contrast)
                state.saturation_factors = np.full(num_frames, saturation)
                state.hue_factors = np.full(num_frames, hue)
        
        # Blur
        state.apply_blur = self.rng.rand() < cfg.blur_prob
        if state.apply_blur:
            kernel_size = self.rng.randint(*cfg.blur_kernel_size)
            state.blur_kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Noise
        state.apply_noise = self.rng.rand() < cfg.noise_prob
        if state.apply_noise:
            state.noise_std = self.rng.uniform(*cfg.noise_std_range)
        
        # Weather
        state.apply_rain = self.rng.rand() < cfg.rain_prob
        state.apply_fog = self.rng.rand() < cfg.fog_prob
        if state.apply_fog:
            state.fog_intensity = self.rng.uniform(*cfg.fog_intensity_range)
        
        # Motion blur
        state.apply_motion_blur = self.rng.rand() < cfg.motion_blur_prob
        if state.apply_motion_blur:
            state.motion_blur_kernel_size = self.rng.randint(*cfg.motion_blur_kernel_size_range)
            state.motion_blur_angle = self.rng.uniform(*cfg.motion_blur_angle_range)
        
        return state
    
    def _generate_smooth_factors(self, 
                                 num_frames: int, 
                                 value_range: Tuple[float, float],
                                 variation_strength: float) -> np.ndarray:
        """
        Generate smooth varying factors cho temporal variation.
        """
        # Sample key points
        window_size = self.config.temporal.smooth_window_size
        num_keypoints = max(2, num_frames // window_size + 1)
        
        keypoints = self.rng.uniform(*value_range, size=num_keypoints)
        
        # Interpolate
        x_key = np.linspace(0, num_frames - 1, num_keypoints)
        x_all = np.arange(num_frames)
        factors = np.interp(x_all, x_key, keypoints)
        
        # Add small random variation
        if variation_strength > 0:
            noise = self.rng.randn(num_frames) * variation_strength * (value_range[1] - value_range[0])
            factors += noise
            factors = np.clip(factors, *value_range)
        
        return factors
    
    def augment_video_clip(self,
                          frames: List[np.ndarray],
                          bboxes: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[Dict[str, Any]], bool]:
        """
        Augment một video clip với temporal consistency.
        
        Args:
            frames: List of frames (numpy arrays, BGR format)
            bboxes: List of bounding boxes tương ứng với frames có object
                    Format: {'frame': frame_idx, 'x1', 'y1', 'x2', 'y2'}
        
        Returns:
            augmented_frames: List of augmented frames
            augmented_bboxes: List of transformed bboxes
            is_valid: True nếu augmentation hợp lệ, False nếu cần reject
        """
        if len(frames) == 0:
            return frames, bboxes, True
        
        num_frames = len(frames)
        img_height, img_width = frames[0].shape[:2]
        
        # Sample transforms
        spatial_state = self._sample_spatial_transforms(img_width, img_height, num_frames)
        pixel_state = self._sample_pixel_transforms(num_frames)
        
        # Merge states
        state = self._merge_states(spatial_state, pixel_state)
        
        # Apply augmentation
        augmented_frames = []
        augmented_bboxes = []
        
        for i, frame in enumerate(frames):
            aug_frame = self._apply_frame_augmentation(frame, state, i)
            augmented_frames.append(aug_frame)
        
        # Transform bboxes
        for bbox in bboxes:
            aug_bbox = self._transform_bbox(bbox, state, img_width, img_height)
            if aug_bbox is not None:
                augmented_bboxes.append(aug_bbox)
        
        # Validate augmentation
        is_valid = self._validate_augmentation(bboxes, augmented_bboxes)
        
        return augmented_frames, augmented_bboxes, is_valid
    
    def _merge_states(self, spatial: AugmentationState, pixel: AugmentationState) -> AugmentationState:
        """Merge spatial và pixel states."""
        # Copy spatial state
        state = AugmentationState()
        
        # Spatial
        state.apply_horizontal_flip = spatial.apply_horizontal_flip
        state.apply_rotation = spatial.apply_rotation
        state.rotation_angle = spatial.rotation_angle
        state.apply_scale = spatial.apply_scale
        state.scale_x = spatial.scale_x
        state.scale_y = spatial.scale_y
        state.apply_crop = spatial.apply_crop
        state.crop_coords = spatial.crop_coords
        state.apply_affine = spatial.apply_affine
        state.affine_matrix = spatial.affine_matrix
        
        # Pixel
        state.apply_color_jitter = pixel.apply_color_jitter
        state.brightness_factors = pixel.brightness_factors
        state.contrast_factors = pixel.contrast_factors
        state.saturation_factors = pixel.saturation_factors
        state.hue_factors = pixel.hue_factors
        state.apply_blur = pixel.apply_blur
        state.blur_kernel_size = pixel.blur_kernel_size
        state.apply_noise = pixel.apply_noise
        state.noise_std = pixel.noise_std
        state.apply_rain = pixel.apply_rain
        state.apply_fog = pixel.apply_fog
        state.fog_intensity = pixel.fog_intensity
        state.apply_motion_blur = pixel.apply_motion_blur
        state.motion_blur_kernel_size = pixel.motion_blur_kernel_size
        state.motion_blur_angle = pixel.motion_blur_angle
        
        return state
    
    def _apply_frame_augmentation(self, 
                                  frame: np.ndarray, 
                                  state: AugmentationState,
                                  frame_idx: int) -> np.ndarray:
        """
        Apply augmentation cho một frame.
        """
        img = frame.copy()
        img_height, img_width = img.shape[:2]
        
        # ===== SPATIAL TRANSFORMS =====
        
        # Horizontal flip
        if state.apply_horizontal_flip:
            img = cv2.flip(img, 1)
        
        # Rotation
        if state.apply_rotation:
            center = (img_width / 2, img_height / 2)
            M = cv2.getRotationMatrix2D(center, state.rotation_angle, 1.0)
            img = cv2.warpAffine(img, M, (img_width, img_height), 
                                borderMode=cv2.BORDER_REFLECT)
        
        # Scale
        if state.apply_scale:
            new_width = int(img_width * state.scale_x)
            new_height = int(img_height * state.scale_y)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Pad hoặc crop về size gốc
            if new_width > img_width or new_height > img_height:
                # Crop về size gốc
                start_x = (new_width - img_width) // 2
                start_y = (new_height - img_height) // 2
                img = img[start_y:start_y+img_height, start_x:start_x+img_width]
            elif new_width < img_width or new_height < img_height:
                # Pad về size gốc
                pad_x = (img_width - new_width) // 2
                pad_y = (img_height - new_height) // 2
                img = cv2.copyMakeBorder(img, pad_y, img_height-new_height-pad_y,
                                        pad_x, img_width-new_width-pad_x,
                                        cv2.BORDER_REFLECT)
        
        # Crop
        if state.apply_crop and state.crop_coords is not None:
            x1, y1, x2, y2 = state.crop_coords
            img = img[y1:y2, x1:x2]
            # Resize về original size
            img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        
        # Affine
        if state.apply_affine and state.affine_matrix is not None:
            img = cv2.warpAffine(img, state.affine_matrix, (img_width, img_height),
                               borderMode=cv2.BORDER_REFLECT)
        
        # ===== PIXEL TRANSFORMS =====
        
        # Color jittering
        if state.apply_color_jitter:
            img = self._apply_color_jitter(
                img,
                state.brightness_factors[frame_idx],
                state.contrast_factors[frame_idx],
                state.saturation_factors[frame_idx],
                state.hue_factors[frame_idx]
            )
        
        # Blur
        if state.apply_blur:
            img = cv2.GaussianBlur(img, (state.blur_kernel_size, state.blur_kernel_size), 0)
        
        # Noise
        if state.apply_noise:
            noise = self.rng.randn(*img.shape) * state.noise_std
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Fog
        if state.apply_fog:
            img = self._apply_fog(img, state.fog_intensity)
        
        # Rain
        if state.apply_rain:
            img = self._apply_rain(img)
        
        # Motion blur
        if state.apply_motion_blur:
            img = self._apply_motion_blur(img, state.motion_blur_kernel_size, 
                                         state.motion_blur_angle)
        
        return img
    
    def _apply_color_jitter(self, img: np.ndarray, 
                           brightness: float,
                           contrast: float, 
                           saturation: float,
                           hue: float) -> np.ndarray:
        """Apply color jittering."""
        # Convert to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Brightness (V channel)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * brightness, 0, 255)
        
        # Saturation (S channel)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation, 0, 255)
        
        # Hue (H channel) - normalized to [0, 179] in OpenCV
        img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] + hue * 179, 0, 179)
        
        # Convert back to BGR
        img_bgr = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Contrast
        img_bgr = np.clip(128 + contrast * (img_bgr.astype(np.float32) - 128), 0, 255).astype(np.uint8)
        
        return img_bgr
    
    def _apply_fog(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """Simulate fog effect."""
        fog_color = np.full_like(img, 200)  # Light gray
        fogged = cv2.addWeighted(img, 1 - intensity, fog_color, intensity, 0)
        return fogged
    
    def _apply_rain(self, img: np.ndarray) -> np.ndarray:
        """Simulate rain effect."""
        rain_img = img.copy()
        height, width = img.shape[:2]
        
        # Add random rain streaks
        num_drops = self.rng.randint(100, 300)
        for _ in range(num_drops):
            x = self.rng.randint(0, width)
            y = self.rng.randint(0, height)
            length = self.rng.randint(5, 15)
            cv2.line(rain_img, (x, y), (x, min(y + length, height - 1)), 
                    (200, 200, 200), 1)
        
        return rain_img
    
    def _apply_motion_blur(self, img: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
        """Apply motion blur."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0
        kernel = kernel / kernel_size
        
        # Rotate kernel
        center = (kernel_size // 2, kernel_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        blurred = cv2.filter2D(img, -1, kernel)
        return blurred
    
    def _transform_bbox(self, 
                       bbox: Dict[str, Any],
                       state: AugmentationState,
                       img_width: int,
                       img_height: int) -> Optional[Dict[str, Any]]:
        """
        Transform bbox theo augmentation state.
        """
        transformed_bbox = bbox.copy()
        
        # Horizontal flip
        if state.apply_horizontal_flip:
            transformed_bbox = bbox_utils.transform_bbox_horizontal_flip(
                transformed_bbox, img_width
            )
        
        # Rotation
        if state.apply_rotation:
            transformed_bbox = bbox_utils.transform_bbox_rotation(
                transformed_bbox, state.rotation_angle, img_width, img_height
            )
            if transformed_bbox is None:
                return None
        
        # Scale
        if state.apply_scale:
            transformed_bbox = bbox_utils.transform_bbox_scale(
                transformed_bbox, state.scale_x, state.scale_y, img_width, img_height
            )
            if transformed_bbox is None:
                return None
        
        # Crop
        if state.apply_crop and state.crop_coords is not None:
            x1, y1, x2, y2 = state.crop_coords
            # Transform bbox về crop coordinates
            transformed_bbox = bbox_utils.transform_bbox_crop(
                transformed_bbox, x1, y1, x2, y2
            )
            if transformed_bbox is None:
                return None
            
            # Sau đó scale về original size
            crop_w = x2 - x1
            crop_h = y2 - y1
            scale_x = img_width / crop_w
            scale_y = img_height / crop_h
            transformed_bbox = bbox_utils.transform_bbox_scale(
                transformed_bbox, scale_x, scale_y, img_width, img_height
            )
            if transformed_bbox is None:
                return None
        
        # Affine
        if state.apply_affine and state.affine_matrix is not None:
            transformed_bbox = bbox_utils.transform_bbox_affine(
                transformed_bbox, state.affine_matrix, img_width, img_height
            )
            if transformed_bbox is None:
                return None
        
        return transformed_bbox
    
    def _validate_augmentation(self, 
                               original_bboxes: List[Dict[str, Any]],
                               augmented_bboxes: List[Dict[str, Any]]) -> bool:
        """
        Validate augmented bboxes.
        Reject nếu bbox bị crop quá nhiều hoặc quá nhỏ.
        """
        if len(augmented_bboxes) < len(original_bboxes) * 0.8:
            # Mất quá nhiều bboxes
            return False
        
        cfg = self.config.spatial
        
        # Check từng bbox
        for orig_bbox in original_bboxes:
            # Find corresponding augmented bbox
            aug_bbox = None
            for ab in augmented_bboxes:
                if ab['frame'] == orig_bbox['frame']:
                    aug_bbox = ab
                    break
            
            if aug_bbox is None:
                continue
            
            # Check area ratio
            orig_area = bbox_utils.bbox_area(orig_bbox)
            aug_area = bbox_utils.bbox_area(aug_bbox)
            
            if aug_area < orig_area * cfg.min_bbox_area_ratio:
                return False
            
            # Check IoU (nếu có thể so sánh)
            # Note: IoU chỉ có ý nghĩa nếu không có spatial transform
            # Ở đây ta skip check này
        
        return True

