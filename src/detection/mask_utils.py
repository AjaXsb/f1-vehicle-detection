"""
Mask Manipulation Utilities

Provides functions for manipulating pixel values based on detection masks.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def isolate_vehicle(frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """
    Isolate vehicles by masking everything except detected vehicles.
    
    Args:
        frame: Input frame
        masks: List of binary masks for each vehicle
    
    Returns:
        Frame with only vehicles visible (black background)
    """
    result = frame.copy()
    
    # Combine all masks
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Apply mask - keep only vehicle pixels
    result[combined_mask == 0] = 0
    
    return result


def blur_background(frame: np.ndarray, masks: List[np.ndarray], 
                   kernel_size: int = 15) -> np.ndarray:
    """
    Blur everything except detected vehicles.
    
    Args:
        frame: Input frame
        masks: List of binary masks for each vehicle
        kernel_size: Gaussian blur kernel size (must be odd)
    
    Returns:
        Frame with blurred background
    """
    # Combine all masks
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Create inverted mask for background
    background_mask = cv2.bitwise_not(combined_mask)
    
    # Blur the entire frame
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    # Create result
    result = frame.copy()
    
    # Apply blur to background areas only
    result[background_mask > 0] = blurred[background_mask > 0]
    
    return result


def colorize_vehicles(frame: np.ndarray, masks: List[np.ndarray],
                     color: tuple = (255, 0, 0)) -> np.ndarray:
    """
    Apply color overlay to detected vehicles.
    
    Args:
        frame: Input frame
        masks: List of binary masks for each vehicle
        color: RGB color tuple for overlay
    
    Returns:
        Frame with colorized vehicles
    """
    result = frame.copy()
    
    # Convert color to BGR for OpenCV
    bgr_color = (color[2], color[1], color[0])
    
    # Apply color overlay to each mask
    for mask in masks:
        color_overlay = np.zeros_like(frame)
        color_overlay[mask > 0] = bgr_color
        
        # Blend with original frame
        result = cv2.addWeighted(result, 1.0, color_overlay, 0.5, 0)
    
    return result


def apply_manipulation(frame: np.ndarray, masks: List[np.ndarray],
                      mode: str, **kwargs) -> np.ndarray:
    """
    Apply pixel manipulation based on mode.
    
    Args:
        frame: Input frame
        masks: List of binary masks for each vehicle
        mode: Manipulation mode ('isolate_vehicle', 'blur_background', 'colorize')
        **kwargs: Additional arguments for specific modes
    
    Returns:
        Manipulated frame
    """
    if mode == "isolate_vehicle":
        return isolate_vehicle(frame, masks)
    
    elif mode == "blur_background":
        kernel_size = kwargs.get('blur_kernel_size', 15)
        return blur_background(frame, masks, kernel_size)
    
    elif mode == "colorize":
        color = kwargs.get('mask_color', (255, 0, 0))
        return colorize_vehicles(frame, masks, color)
    
    elif mode == "track":
        # Just return original frame for tracking mode
        return frame
    
    else:
        logger.warning(f"Unknown manipulation mode: {mode}")
        return frame

