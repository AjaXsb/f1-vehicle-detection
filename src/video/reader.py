"""
Video Reader Module

Handles reading frames from video files using OpenCV.
"""

import cv2
import numpy as np
from typing import Iterator, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Video reader that provides frame-by-frame access to video files.
    
    Handles video reading with support for frame skipping,
    resolution changes, and progress tracking.
    """
    
    def __init__(self, video_path: str, frame_skip: int = 1):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
            frame_skip: Process every Nth frame (1 = all frames)
        
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        
        self._open_video()
    
    def _open_video(self):
        """Open video file and read metadata."""
        logger.info(f"Opening video: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(
            f"Video properties: {self.width}x{self.height}, "
            f"{self.fps:.2f} FPS, {self.frame_count} frames"
        )
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Iterator for video frames.
        
        Yields:
            Tuple of (frame, frame_number) for each processed frame
        """
        frame_num = 0
        processed_frame_num = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Skip frames if frame_skip > 1
            if frame_num % self.frame_skip == 0:
                yield frame, processed_frame_num
                processed_frame_num += 1
            
            frame_num += 1
    
    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Read a specific frame by number.
        
        Args:
            frame_number: Frame number to read (0-indexed)
        
        Returns:
            Frame as numpy array, or None if frame doesn't exist
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def get_total_frames(self) -> int:
        """Get total number of frames in video."""
        return self.frame_count
    
    def get_fps(self) -> float:
        """Get video frames per second."""
        return self.fps
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get video dimensions (width, height)."""
        return self.width, self.height
    
    def close(self):
        """Close video file."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video file closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def extract_sample_frames(video_path: str, output_dir: str, 
                         num_frames: int = 10) -> list:
    """
    Extract sample frames from video for analysis.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save sample frames
        num_frames: Number of frames to extract
    
    Returns:
        List of paths to saved frames
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    reader = VideoReader(video_path)
    total_frames = reader.get_total_frames()
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    saved_frames = []
    
    for idx in frame_indices:
        frame = reader.read_frame(idx)
        if frame is not None:
            output_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frames.append(output_path)
            logger.info(f"Saved frame {idx} to {output_path}")
    
    reader.close()
    
    return saved_frames

