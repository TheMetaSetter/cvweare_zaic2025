"""
Utilities để transform bounding boxes khi apply spatial augmentation.
Đảm bảo bboxes vẫn hợp lệ sau khi transform.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional

def clip_bbox(bbox: Dict[str, float], img_width: int, img_height: int) -> Optional[Dict[str, float]]:
    """
    Clip bounding box để nằm trong frame và đảm bảo hợp lệ.
    
    Args:
        bbox: Dict với keys 'x1', 'y1', 'x2', 'y2', 'frame'
        img_width: Chiều rộng frame
        img_height: Chiều cao frame
    
    Returns:
        Clipped bbox hoặc None nếu bbox invalid
    """
    x1 = max(0, min(bbox['x1'], img_width))
    y1 = max(0, min(bbox['y1'], img_height))
    x2 = max(0, min(bbox['x2'], img_width))
    y2 = max(0, min(bbox['y2'], img_height))
    
    # Check validity
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Check minimum size (tránh bbox quá nhỏ)
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None
    
    return {
        'frame': bbox['frame'],
        'x1': int(x1),
        'y1': int(y1),
        'x2': int(x2),
        'y2': int(y2)
    }


def transform_bbox_horizontal_flip(bbox: Dict[str, float], img_width: int) -> Dict[str, float]:
    """
    Transform bbox khi flip horizontally.
    
    Args:
        bbox: Original bbox
        img_width: Chiều rộng frame
    
    Returns:
        Transformed bbox
    """
    return {
        'frame': bbox['frame'],
        'x1': img_width - bbox['x2'],
        'y1': bbox['y1'],
        'x2': img_width - bbox['x1'],
        'y2': bbox['y2']
    }


def transform_bbox_rotation(bbox: Dict[str, float], 
                            angle: float,
                            img_width: int, 
                            img_height: int,
                            center: Optional[Tuple[float, float]] = None) -> Optional[Dict[str, float]]:
    """
    Transform bbox khi rotate image.
    
    Args:
        bbox: Original bbox
        angle: Góc quay (degrees), dương = counter-clockwise
        img_width: Chiều rộng frame
        img_height: Chiều cao frame
        center: Tâm quay, default là center của image
    
    Returns:
        Transformed bbox hoặc None nếu invalid
    """
    if center is None:
        center = (img_width / 2, img_height / 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Transform 4 corners của bbox
    corners = np.array([
        [bbox['x1'], bbox['y1']],
        [bbox['x2'], bbox['y1']],
        [bbox['x2'], bbox['y2']],
        [bbox['x1'], bbox['y2']]
    ], dtype=np.float32)
    
    # Apply rotation
    ones = np.ones((4, 1), dtype=np.float32)
    corners_homo = np.hstack([corners, ones])
    transformed_corners = M.dot(corners_homo.T).T
    
    # Get new bounding box
    x_coords = transformed_corners[:, 0]
    y_coords = transformed_corners[:, 1]
    
    new_bbox = {
        'frame': bbox['frame'],
        'x1': float(np.min(x_coords)),
        'y1': float(np.min(y_coords)),
        'x2': float(np.max(x_coords)),
        'y2': float(np.max(y_coords))
    }
    
    return clip_bbox(new_bbox, img_width, img_height)


def transform_bbox_scale(bbox: Dict[str, float],
                         scale_x: float,
                         scale_y: float,
                         img_width: int,
                         img_height: int) -> Optional[Dict[str, float]]:
    """
    Transform bbox khi scale image.
    
    Args:
        bbox: Original bbox
        scale_x: Scale factor theo x
        scale_y: Scale factor theo y
        img_width: Chiều rộng frame SAU khi scale
        img_height: Chiều cao frame SAU khi scale
    
    Returns:
        Transformed bbox
    """
    new_bbox = {
        'frame': bbox['frame'],
        'x1': bbox['x1'] * scale_x,
        'y1': bbox['y1'] * scale_y,
        'x2': bbox['x2'] * scale_x,
        'y2': bbox['y2'] * scale_y
    }
    
    return clip_bbox(new_bbox, img_width, img_height)


def transform_bbox_crop(bbox: Dict[str, float],
                       crop_x1: int,
                       crop_y1: int,
                       crop_x2: int,
                       crop_y2: int) -> Optional[Dict[str, float]]:
    """
    Transform bbox khi crop image.
    
    Args:
        bbox: Original bbox
        crop_x1, crop_y1, crop_x2, crop_y2: Crop coordinates
    
    Returns:
        Transformed bbox hoặc None nếu bbox nằm ngoài crop region
    """
    # Translate coordinates
    new_x1 = bbox['x1'] - crop_x1
    new_y1 = bbox['y1'] - crop_y1
    new_x2 = bbox['x2'] - crop_x1
    new_y2 = bbox['y2'] - crop_y1
    
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    new_bbox = {
        'frame': bbox['frame'],
        'x1': new_x1,
        'y1': new_y1,
        'x2': new_x2,
        'y2': new_y2
    }
    
    return clip_bbox(new_bbox, crop_width, crop_height)


def transform_bbox_affine(bbox: Dict[str, float],
                          matrix: np.ndarray,
                          img_width: int,
                          img_height: int) -> Optional[Dict[str, float]]:
    """
    Transform bbox với affine transformation matrix.
    
    Args:
        bbox: Original bbox
        matrix: 2x3 affine transformation matrix
        img_width: Chiều rộng frame sau transform
        img_height: Chiều cao frame sau transform
    
    Returns:
        Transformed bbox
    """
    # Transform 4 corners của bbox
    corners = np.array([
        [bbox['x1'], bbox['y1']],
        [bbox['x2'], bbox['y1']],
        [bbox['x2'], bbox['y2']],
        [bbox['x1'], bbox['y2']]
    ], dtype=np.float32)
    
    # Apply affine transform
    ones = np.ones((4, 1), dtype=np.float32)
    corners_homo = np.hstack([corners, ones])
    transformed_corners = matrix.dot(corners_homo.T).T
    
    # Get new bounding box
    x_coords = transformed_corners[:, 0]
    y_coords = transformed_corners[:, 1]
    
    new_bbox = {
        'frame': bbox['frame'],
        'x1': float(np.min(x_coords)),
        'y1': float(np.min(y_coords)),
        'x2': float(np.max(x_coords)),
        'y2': float(np.max(y_coords))
    }
    
    return clip_bbox(new_bbox, img_width, img_height)


def calculate_iou(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
    """
    Tính IoU giữa 2 bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes
    
    Returns:
        IoU score
    """
    x1_inter = max(bbox1['x1'], bbox2['x1'])
    y1_inter = max(bbox1['y1'], bbox2['y1'])
    x2_inter = min(bbox1['x2'], bbox2['x2'])
    y2_inter = min(bbox1['y2'], bbox2['y2'])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    bbox1_area = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
    bbox2_area = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def bbox_area(bbox: Dict[str, float]) -> float:
    """Tính diện tích bbox."""
    return (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])


def bbox_center(bbox: Dict[str, float]) -> Tuple[float, float]:
    """Tính tâm của bbox."""
    cx = (bbox['x1'] + bbox['x2']) / 2
    cy = (bbox['y1'] + bbox['y2']) / 2
    return cx, cy