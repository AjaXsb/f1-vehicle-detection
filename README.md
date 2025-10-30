# F1 Vehicle Detection & Masking

Aims to detect, segment, mask, and manipulate F1 cars in video footage using OpenCV and Mask R-CNN.

## Overview

This project uses Mask R-CNN for instance segmentation to:
- Detect F1 vehicles in video footage
- Generate pixel-accurate masks around vehicles
- Manipulate pixels (blur background, isolate vehicles, colorize, etc.)

## Features

- ✅ Mask R-CNN-based instance segmentation
- ✅ Frame-by-frame detection with temporal consistency
- ✅ Configurable pixel manipulation modes
- ✅ Support for multiple vehicles per frame
- ✅ Progress tracking and visualization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd f1-vehicle-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Mask R-CNN weights**
   ```bash
   python src/model/download_weights.py
   ```

## Configuration

Edit `config/config.yaml` to customize:
- Detection confidence threshold
- Video processing settings
- Pixel manipulation mode
- Visualization options

### Key Configuration Options

- **`model.confidence_threshold`**: Minimum confidence for detections (0.0-1.0)
- **`video.frame_skip`**: Process every Nth frame (1 = all frames, 2 = every other frame)
- **`manipulation.mode`**: 
  - `"isolate_vehicle"` - Show only the vehicle, black background
  - `"background_blur"` - Blur everything except vehicles
  - `"colorize"` - Apply color filter to vehicles
  - `"track"` - Just track and visualize

## Usage

### Process a Video

```bash
python scripts/process_video.py --input footage/vid-mex-25.mp4 --output data/output/processed.mp4
```

### Extract Sample Frames

```bash
python scripts/extract_frames.py --input footage/vid-mex-25.mp4 --output data/samples --frames 10
```

### Advanced Usage

```bash
# Custom configuration
python scripts/process_video.py --input input.mp4 --output output.mp4 --config config/custom.yaml

# Debug mode with intermediate frames
python scripts/process_video.py --input input.mp4 --debug --save-intermediate
```

## Project Structure

```
f1-vehicle-detection/
├── src/                 # Source code
│   ├── model/          # Mask R-CNN model handling
│   ├── video/          # Video I/O operations
│   ├── detection/      # Detection and masking logic
│   └── visualization/  # Visualization utilities
├── config/             # Configuration files
├── data/               # Data directories
│   ├── input/         # Input videos
│   ├── output/        # Processed videos
│   └── weights/       # Model weights
├── scripts/            # Main processing scripts
├── tests/              # Unit tests
└── footage/            # Your footage folder
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint code
flake8 src/
```

## Troubleshooting

### GPU Issues
- Ensure CUDA and cuDNN are installed for GPU acceleration
- Check TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Model Download Issues
- Manual download: [COCO pre-trained weights](https://github.com/matterport/Mask_RCNN/releases)
- Place in `data/weights/`

### Memory Issues
- Increase `frame_skip` in config to process fewer frames
- Reduce video resolution in config
- Process shorter video segments

## Roadmap

- [ ] Real-time processing with GPU optimization
- [ ] Multi-threaded video processing
- [ ] Custom F1-specific model fine-tuning
- [ ] Vehicle tracking across frames
- [ ] Speed estimation from video

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please read contributing guidelines before submitting PRs.
