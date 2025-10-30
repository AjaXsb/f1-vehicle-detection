#!/usr/bin/env python
"""
Frame Extraction Script

Extract sample frames from video for analysis and testing.
"""

import sys
import os
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video.reader import extract_sample_frames

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract sample frames from video"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input video path'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/samples',
        help='Output directory for frames'
    )
    parser.add_argument(
        '--frames', '-n',
        type=int,
        default=10,
        help='Number of frames to extract'
    )
    
    args = parser.parse_args()
    
    # Check input exists
    if not os.path.exists(args.input):
        logger.error(f"Input video not found: {args.input}")
        sys.exit(1)
    
    try:
        saved_frames = extract_sample_frames(
            args.input,
            args.output,
            args.frames
        )
        logger.info(f"Extracted {len(saved_frames)} frames to {args.output}")
    except Exception as e:
        logger.error(f"Error extracting frames: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

