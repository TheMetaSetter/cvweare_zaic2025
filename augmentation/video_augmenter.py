"""
Video Frame Augmenter với temporal consistency.
Core module cho augmentation của drone video frames.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .config import AugmentationConfig, SpatialAugmentConfig, PixelAugmentConfig, TemporalConfig
from . import bbox_utils
from .refbank import build_augmented_ref_bank, sample_folder_to_ref_key


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
    
    def __init__(self, 
                 config: AugmentationConfig,
                 ref_bank_root: Optional[str] = None,
                 ref_bank: Optional[Dict[str, List[Path]]] = None):
        """
        Args:
            config: Augmentation configuration
            ref_bank_root: Optional path tới augmented_ref_img để auto-build bank
            ref_bank: Optional pre-built reference bank (takes precedence over root)
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.ref_bank: Optional[Dict[str, List[Path]]] = ref_bank
        if self.ref_bank is None and ref_bank_root is not None:
            root_path = Path(ref_bank_root)
            if root_path.exists():
                self.ref_bank = build_augmented_ref_bank(root_path)
        self._current_frame_indices: Optional[List[int]] = None
    
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
                          bboxes: List[Dict[str, Any]],
                          frame_indices: Optional[List[int]] = None,
                          video_id: Optional[str] = None,
                          ref_bank: Optional[Dict[str, List[Path]]] = None
                          ) -> Tuple[List[np.ndarray], List[Dict[str, Any]], bool]:
        """
        Augment một video clip với temporal consistency.
        
        Args:
            frames: List of frames (numpy arrays, BGR format)
            bboxes: List of bounding boxes tương ứng với frames có object
                    Format: {'frame': frame_idx, 'x1', 'y1', 'x2', 'y2'}
            frame_indices: Frame numbers mapping 1-1 với frames list
            video_id: Sample/video folder name cho class_key mapping
            ref_bank: Optional override copy-paste reference bank
        
        Returns:
            augmented_frames: List of augmented frames
            augmented_bboxes: List of transformed bboxes
            is_valid: True nếu augmentation hợp lệ, False nếu cần reject
        """
        if len(frames) == 0:
            return frames, bboxes, True
        
        num_frames = len(frames)
        img_height, img_width = frames[0].shape[:2]
        if frame_indices is None or len(frame_indices) != num_frames:
            frame_indices = list(range(num_frames))
        self._current_frame_indices = frame_indices
        try:
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
            
            paste_bboxes: List[Dict[str, Any]] = []
            active_ref_bank = ref_bank if ref_bank is not None else self.ref_bank
            class_key = None
            if video_id is not None and active_ref_bank:
                class_key = sample_folder_to_ref_key(video_id, active_ref_bank)
            if class_key and class_key in active_ref_bank:
                augmented_frames, paste_bboxes, _ = self._apply_copy_paste(
                    augmented_frames,
                    bboxes,
                    class_key,
                    active_ref_bank,
                    self.config.copy_paste
                )
            
            # Transform bboxes
            for bbox in bboxes:
                aug_bbox = self._transform_bbox(bbox, state, img_width, img_height)
                if aug_bbox is not None:
                    augmented_bboxes.append(aug_bbox)
                    
            augmented_bboxes += paste_bboxes
            
            # Validate augmentation
            is_valid = self._validate_augmentation(bboxes, augmented_bboxes)
            
            return augmented_frames, augmented_bboxes, is_valid
        finally:
            self._current_frame_indices = None
    
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
    
    # Applies copy-paste tubelets once frames are in final spatial space so YOLO sees extra small objects.
    # This step explicitly targets tiny-object recall: we borrow high-quality PNGs
    # from the reference bank, paste them consistently across frames (tubelets),
    # and only do it after spatial transforms so pasted pixels match YOLO's view.
    def _apply_copy_paste(
        self,
        frames: List[np.ndarray],
        bboxes: List[dict],
        class_key: str,
        ref_bank: Dict[str, List[Path]],
        cfg
    ) -> Tuple[List[np.ndarray], List[dict], bool]:
        """
        Paste one or more reference objects (tubelets) onto the video frames.

        Called after every spatial warp so pasted pixels already live in final
        coordinates; this avoids double-transforming bboxes and keeps copy-paste
        additive. We return the mutated frames plus the extra bboxes so YOLO
        sees more tiny-object examples for the same context.

        Args and Returns follow the dataset contract so downstream exporters can
        merge paste-generated boxes with transformed GT boxes without further
        bookkeeping.
        """
        # Copy-paste injects extra objects for classes with few examples so YOLO sees varied poses.
        # We run it after spatial transforms so pasted pixels already match the final geometry,
        # and we keep tubelets (same crop across frames) to maintain temporal consistency.
        if len(frames) == 0:
            return frames, [], False
        if cfg is None or not getattr(cfg, "enabled", False):
            return frames, [], False
        if not ref_bank or class_key not in ref_bank:
            return frames, [], False
        ref_paths = ref_bank.get(class_key, [])
        if len(ref_paths) == 0:
            return frames, [], False
        if self.rng.rand() > cfg.prob:
            return frames, [], False
        max_objects = min(cfg.max_objects, len(ref_paths))
        if max_objects <= 0:
            return frames, [], False
        
        frame_h, frame_w = frames[0].shape[:2]
        # frame_numbers map each in-memory frame to its original video index for bbox bookkeeping.
        frame_numbers = getattr(self, "_current_frame_indices", None)
        if frame_numbers is None or len(frame_numbers) != len(frames):
            frame_numbers = list(range(len(frames)))
        
        # Decide how many tubelets to paste (up to max_objects) and sample the templates.
        num_objects = self.rng.randint(1, max_objects + 1)
        sample_indices = self.rng.choice(len(ref_paths), size=num_objects, replace=False)
        
        new_frames = frames
        new_bboxes: List[dict] = []
        did_paste = False
        jitter = max(0.0, float(getattr(cfg, "jitter", 0.0)))
        min_scale, max_scale = cfg.scale_range
        if max_scale < min_scale:
            min_scale, max_scale = max_scale, min_scale
        min_scale = max(min_scale, 1e-3)
        max_scale = max(max_scale, min_scale)
        
        for idx in sample_indices:
            tpl_path = ref_paths[int(idx)]
            template = cv2.imread(str(tpl_path), cv2.IMREAD_UNCHANGED)
            if template is None:
                continue
            if template.ndim == 2:
                template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGRA)
            if template.shape[2] == 3:
                # Some PNGs ship without alpha; assume fully opaque so blending still works.
                alpha_channel = np.full(template.shape[:2], 255, dtype=np.uint8)
                template = np.dstack((template, alpha_channel))
            elif template.shape[2] != 4:
                continue
            
            obj_h, obj_w = template.shape[:2]
            if obj_h == 0 or obj_w == 0:
                continue
            
            # Scale relative to object size but clamp so the crop always fits in the frame.
            # This guarantees bbox validity even when we borrow very large crops.
            scale = float(self.rng.uniform(min_scale, max_scale))
            max_scale_fit = min(frame_w / obj_w, frame_h / obj_h)
            if max_scale_fit <= 0:
                continue
            scale = min(scale, max_scale_fit)
            new_w = max(1, int(round(obj_w * scale)))
            new_h = max(1, int(round(obj_h * scale)))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            tpl_resized = cv2.resize(template, (new_w, new_h), interpolation=interpolation)
            
            max_x1 = frame_w - new_w
            max_y1 = frame_h - new_h
            if max_x1 < 0 or max_y1 < 0:
                continue
            # Determine a single base location so the tubelet appears as a coherent track.
            base_x1 = int(self.rng.randint(0, max_x1 + 1)) if max_x1 > 0 else 0
            base_y1 = int(self.rng.randint(0, max_y1 + 1)) if max_y1 > 0 else 0
            
            alpha = tpl_resized[:, :, 3].astype(np.float32) / 255.0
            if float(alpha.max()) == 0.0:
                continue
            alpha_3c = alpha[..., None]
            color = tpl_resized[:, :, :3].astype(np.float32)
            
            object_pasted = False
            # Jitter per frame creates tiny motion so the pasted tubelet feels real
            # without drifting off-screen; this mimics real drone motion for ST-IoU.
            for frame_idx, (frame, frame_number) in enumerate(zip(new_frames, frame_numbers)):
                dx = int(round(jitter * frame_w * self.rng.uniform(-1.0, 1.0))) if jitter > 0 else 0
                dy = int(round(jitter * frame_h * self.rng.uniform(-1.0, 1.0))) if jitter > 0 else 0
                paste_x1 = base_x1 + dx
                paste_y1 = base_y1 + dy
                # Clamp jittered spot so we never spill outside of the image bounds.
                paste_x1 = min(max(paste_x1, 0), max_x1)
                paste_y1 = min(max(paste_y1, 0), max_y1)
                
                paste_x2 = paste_x1 + new_w
                paste_y2 = paste_y1 + new_h
                
                # Blend with alpha so synthetic pixels respect the reference mask
                # and we never introduce hard seams that would confuse YOLO.
                roi = frame[paste_y1:paste_y2, paste_x1:paste_x2].astype(np.float32)
                blended = color * alpha_3c + roi * (1.0 - alpha_3c)
                frame[paste_y1:paste_y2, paste_x1:paste_x2] = blended.astype(np.uint8)

                # These coords already live in final image space, so we store them directly
                # without piping through _transform_bbox again. Bounding boxes stay valid
                # because we clamp the paste window above and reuse the original frame indices.
                new_bboxes.append({
                    'frame': int(frame_number),
                    'x1': float(paste_x1),
                    'y1': float(paste_y1),
                    'x2': float(paste_x2),
                    'y2': float(paste_y2),
                })
                object_pasted = True
                did_paste = True
            
            # Reaching here means the tubelet stayed in-bounds for all frames;
            # if the template never fit, we silently skip it to avoid corrupt labels.
            if not object_pasted:
                continue
        
        if not did_paste:
            return frames, [], False

        return new_frames, new_bboxes, True

    
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
