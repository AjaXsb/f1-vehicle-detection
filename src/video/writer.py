"""
Video Writer Module

Handles writing processed frames to output video files.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)


class VideoWriter:
    """
    Video writer that handles writing frames to output video files.
    
    Supports custom FPS, resolution, and codec configuration.
    """
    
    def __init__(self, output_path: str, 
                 width: int, height: int,
                 fps: Optional[float] = None,
                 fourcc: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            width: Output video width
            height: Output video height
            fps: Output FPS (optional, will try to match input)
            fourcc: Video codec fourcc string
        
        Raises:
            IOError: If output directory cannot be created
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps or 30.0
        self.fourcc = fourcc
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.writer = None
        self._init_writer()
    
    def _init_writer(self):
        """Initialize OpenCV VideoWriter."""
        logger.info(f"Creating output video: {self.output_path}")
        
        # Convert fourcc string to fourcc code
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc_code,
            self.fps,
            (self.width, self.height)
        )
        
        if not self.writer.isOpened():
            raise IOError(f"Could not create output video: {self.output_path}")
        
        logger.info(
            f"Video writer initialized: {self.width}x{self.height}, "
            f"{self.fps} FPS, codec: {self.fourcc}"
        )
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the output video.
        
        Args:
            frame: Frame to write (will be resized if dimensions don't match)
        """
        if self.writer is None:
            raise RuntimeError("Video writer not initialized")
        
        # Resize frame if dimensions don't match
        frame_h, frame_w = frame.shape[:2]
        if frame_w != self.width or frame_h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
    
    def release(self):
        """Release video writer resources."""
        if self.writer is not None:
            self.writer.release()
            logger.info("Video writer released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

