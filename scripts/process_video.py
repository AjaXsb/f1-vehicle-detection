#!/usr/bin/env python
"""
Main Video Processing Script

Processes video files to detect and mask F1 vehicles.
"""

import sys
import os
import argparse
import yaml
import logging
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.mask_rcnn import MaskRCNNModel, COCO_CLASSES
from src.video.reader import VideoReader
from src.video.writer import VideoWriter
from src.detection.mask_utils import apply_manipulation
from src.visualization.visualize import visualize_detections

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_video(input_path: str, output_path: str, config_path: str = None):
    """
    Process video to detect and mask vehicles.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        config_path: Path to configuration file
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        logger.warning("Config file not found, using defaults")
        config = {
            'model': {'confidence_threshold': 0.7},
            'video': {'frame_skip': 1, 'fps': None},
            'filter': {'vehicle_classes': ['car', 'motorcycle', 'bus', 'truck']},
            'manipulation': {'mode': 'isolate_vehicle'},
            'visualization': {
                'draw_boxes': True,
                'draw_masks': True,
                'draw_scores': True
            }
        }
    
    # Extract config sections
    model_config = config.get('model', {})
    video_config = config.get('video', {})
    filter_config = config.get('filter', {})
    manip_config = config.get('manipulation', {})
    viz_config = config.get('visualization', {})
    
    # Initialize model
    logger.info("Initializing Mask R-CNN model...")
    model = MaskRCNNModel(
        weights_path=model_config.get('weights_path', 'data/weights/mask_rcnn_coco.h5'),
        confidence_threshold=model_config.get('confidence_threshold', 0.7)
    )
    
    # Open video reader
    logger.info(f"Opening input video: {input_path}")
    reader = VideoReader(
        input_path,
        frame_skip=video_config.get('frame_skip', 1)
    )
    
    # Get video properties
    width, height = reader.get_dimensions()
    fps = reader.get_fps()
    
    # Override with config if specified
    if 'resolution' in video_config:
        if video_config['resolution']['width']:
            width = video_config['resolution']['width']
        if video_config['resolution']['height']:
            height = video_config['resolution']['height']
    
    if video_config.get('fps') is not None:
        fps = video_config['fps']
    
    # Initialize video writer
    writer = VideoWriter(
        output_path,
        width=width,
        height=height,
        fps=fps,
        fourcc=video_config.get('fourcc', 'mp4v')
    )
    
    # Process frames
    total_frames = reader.get_total_frames() // video_config.get('frame_skip', 1)
    vehicle_classes = [c.lower() for c in filter_config.get('vehicle_classes', ['car'])]
    
    logger.info(f"Processing {total_frames} frames...")
    
    with tqdm(total=total_frames, desc="Processing") as pbar:
        for frame, frame_num in reader:
            # Run detection
            detections = model.detect(frame)
            
            # Filter for vehicles only
            vehicle_detections = model.filter_vehicles(detections, vehicle_classes)
            
            # Extract masks
            masks = [det['mask'] for det in vehicle_detections]
            
            # Apply pixel manipulation
            manip_mode = manip_config.get('mode', 'track')
            
            if manip_mode == 'track':
                # Just visualize detections
                result_frame = visualize_detections(
                    frame,
                    vehicle_detections,
                    draw_boxes=viz_config.get('draw_boxes', True),
                    draw_masks=viz_config.get('draw_masks', True),
                    draw_scores=viz_config.get('draw_scores', True)
                )
            else:
                # Apply manipulation; if no masks, keep original frame to avoid full black frames
                if len(masks) == 0:
                    result_frame = frame
                else:
                    manip_params = {k: v for k, v in manip_config.items() if k != 'mode'}
                    result_frame = apply_manipulation(
                        frame,
                        masks,
                        mode=manip_mode,
                        **manip_params
                    )
            
            # Write frame
            writer.write_frame(result_frame)
            
            pbar.update(1)
    
    # Cleanup
    reader.close()
    writer.release()
    
    logger.info(f"Processing complete! Output saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process F1 video with Mask R-CNN detection"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input video path'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output video path'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    # Check input exists
    if not os.path.exists(args.input):
        logger.error(f"Input video not found: {args.input}")
        sys.exit(1)
    
    try:
        process_video(args.input, args.output, args.config)
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

