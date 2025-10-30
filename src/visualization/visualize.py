"""
Visualization Utilities

Provides functions for drawing detection results on frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def draw_bounding_box(frame: np.ndarray, bbox: List[int], 
                     label: Optional[str] = None,
                     confidence: Optional[float] = None,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x, y, width, height]
        label: Optional label text
        confidence: Optional confidence score
        color: Box color (BGR)
        thickness: Box thickness
    
    Returns:
        Frame with bounding box drawn
    """
    result = frame.copy()
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label and confidence
    if label or confidence is not None:
        text = []
        if label:
            text.append(label)
        if confidence is not None:
            text.append(f"{confidence:.2f}")
        
        label_text = " ".join(text)
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw background for text
        cv2.rectangle(
            result, 
            (x, y - text_height - baseline - 5),
            (x + text_width, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            result,
            label_text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return result


def draw_mask(frame: np.ndarray, mask: np.ndarray,
             color: Tuple[int, int, int] = (255, 0, 0),
             opacity: float = 0.3) -> np.ndarray:
    """
    Draw mask overlay on frame.
    
    Args:
        frame: Input frame
        mask: Binary mask array
        color: Mask color (BGR)
        opacity: Mask opacity (0.0 to 1.0)
    
    Returns:
        Frame with mask overlay
    """
    result = frame.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 0] = color
    
    # Blend with original frame
    result = cv2.addWeighted(result, 1.0, colored_mask, opacity, 0)
    
    return result


def visualize_detections(frame: np.ndarray, detections: List[Dict],
                        draw_boxes: bool = True,
                        draw_masks: bool = True,
                        draw_scores: bool = True,
                        box_color: Tuple[int, int, int] = (0, 255, 0),
                        mask_color: Tuple[int, int, int] = (255, 0, 0),
                        mask_opacity: float = 0.3) -> np.ndarray:
    """
    Visualize all detections on a frame.
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        draw_boxes: Whether to draw bounding boxes
        draw_masks: Whether to draw masks
        draw_scores: Whether to draw confidence scores
        box_color: Bounding box color (BGR)
        mask_color: Mask color (RGB)
        mask_opacity: Mask overlay opacity
    
    Returns:
        Frame with visualizations
    """
    result = frame.copy()
    
    for det in detections:
        # Draw mask
        if draw_masks and 'mask' in det:
            result = draw_mask(
                result,
                det['mask'],
                color=mask_color,
                opacity=mask_opacity
            )
        
        # Draw bounding box
        if draw_boxes and 'bbox' in det:
            label = det.get('class_name', '')
            confidence = det.get('confidence') if draw_scores else None
            
            result = draw_bounding_box(
                result,
                det['bbox'],
                label=label,
                confidence=confidence,
                color=box_color
            )
    
    return result


def display_frame(frame: np.ndarray, window_name: str = "Frame",
                 wait_time: int = 1) -> int:
    """
    Display frame in a window.
    
    Args:
        frame: Frame to display
        window_name: Window name
        wait_time: Wait time in milliseconds (0 = wait for key press)
    
    Returns:
        Key code of pressed key
    """
    cv2.imshow(window_name, frame)
    return cv2.waitKey(wait_time)

