"""
Model Weights Download Script

Downloads pre-trained Mask R-CNN weights for COCO dataset.
"""

import os
import sys
import urllib.request
import logging

logger = logging.getLogger(__name__)

# COCO pre-trained weights URLs
COCO_WEIGHTS_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_coco.h5"


def download_weights(output_dir: str = "data/weights", filename: str = "mask_rcnn_coco.h5"):
    """
    Download COCO pre-trained weights for Mask R-CNN.
    
    Args:
        output_dir: Directory to save weights
        filename: Output filename
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    # Check if already downloaded
    if os.path.exists(output_path):
        logger.info(f"Weights already exist at {output_path}")
        return output_path
    
    url = COCO_WEIGHTS_URL
    
    logger.info(f"Downloading weights from {url}...")
    logger.info("This may take several minutes (file is ~244MB)...")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if percent % 10 == 0:
                sys.stdout.write(f"\rDownload progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # New line after progress
        
        logger.info(f"Weights downloaded successfully to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error downloading weights: {e}")
        logger.error("Please download manually from:")
        logger.error(f"  {url}")
        logger.error(f"And place in: {output_path}")
        
        return None


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Download Mask R-CNN weights")
    parser.add_argument(
        '--output-dir',
        default='data/weights',
        help='Output directory for weights'
    )
    parser.add_argument(
        '--filename',
        default='mask_rcnn_coco.h5',
        help='Output filename'
    )
    
    args = parser.parse_args()
    
    download_weights(args.output_dir, args.filename)

