# Quick Start Guide

Get up and running with F1 Vehicle Detection in minutes!

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

## Step 2: Download Model Weights

```bash
python src/model/download_weights.py
```

This downloads the COCO pre-trained Mask R-CNN weights (~244MB).

## Step 3: Test with Your Video

```bash
python scripts/process_video.py \
    --input footage/vid-mex-25.mp4 \
    --output data/output/processed.mp4
```

## What You'll See

The script will:
1. Load the Mask R-CNN model
2. Process each frame to detect vehicles
3. Apply masks and pixel manipulation
4. Save the output video

Progress is shown with a progress bar.

## Try Different Modes

Edit `config/config.yaml` to change manipulation modes:

```yaml
manipulation:
  mode: "isolate_vehicle"  # or "blur_background", "colorize", "track"
```

Then run again:

```bash
python scripts/process_video.py -i footage/vid-mex-25.mp4 -o data/output/blurred.mp4
```

## Extract Sample Frames

Analyze your video before processing:

```bash
python scripts/extract_frames.py --input footage/vid-mex-25.mp4 --output data/samples --frames 20
```

This extracts 20 evenly-spaced frames to `data/samples/`.

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [DEVELOPMENT.md](DEVELOPMENT.md) for developer guide
- Customize `config/config.yaml` for your needs
- Integrate a real Mask R-CNN model (see DEVELOPMENT.md)

## Troubleshooting

**Weights won't download?** 
- Download manually from: https://github.com/matterport/Mask_RCNN/releases
- Place in `data/weights/mask_rcnn_coco.h5`

**Processing too slow?**
- Edit `config/config.yaml` to increase `video.frame_skip`
- Or reduce video resolution

**Memory errors?**
- Process shorter video segments
- Reduce resolution in config

**Need help?**
- Check the main [README.md](README.md)
- Open an issue on GitHub

## Current Limitations

⚠️ **Important**: The current implementation uses a **mock/placeholder** model for development purposes. 

To use a real Mask R-CNN model:
- See [DEVELOPMENT.md](DEVELOPMENT.md) for integration options
- Choose between Matterport, Detectron2, or TensorFlow implementations
- Update `src/model/mask_rcnn.py` with the real model loading code

The architecture is ready - you just need to connect the actual model!

