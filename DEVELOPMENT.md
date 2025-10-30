# Development Guide

This guide covers development practices, coding standards, and project architecture.

## Architecture Overview

### Module Structure

```
src/
├── model/          # Mask R-CNN model handling
├── video/          # Video I/O operations
├── detection/      # Detection and masking logic
└── visualization/  # Visualization utilities
```

### Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Configuration-Driven**: Behavior controlled via YAML config files
3. **Extensible**: Easy to add new manipulation modes or detection backends
4. **Type Hints**: All functions include type annotations for clarity

## Coding Standards

### Style Guide

- Follow PEP 8 for Python code style
- Use type hints for all function parameters and return values
- Document all public functions and classes with docstrings
- Use logging instead of print statements
- Keep functions focused and small (< 50 lines ideally)

### Documentation Standards

#### Function Docstrings

```python
def process_frame(frame: np.ndarray, model: MaskRCNNModel) -> np.ndarray:
    """
    Process a single frame through the detection pipeline.
    
    Args:
        frame: Input frame as numpy array (BGR format)
        model: Initialized Mask R-CNN model
    
    Returns:
        Processed frame with detections applied
    
    Raises:
        ValueError: If frame dimensions are invalid
    """
    pass
```

#### Class Docstrings

```python
class MyClass:
    """
    Brief description of the class purpose.
    
    Longer description explaining when and how to use this class.
    Include any important implementation details or design decisions.
    
    Attributes:
        attr1: Description of attribute
        attr2: Description of attribute
    
    Example:
        >>> obj = MyClass(param1, param2)
        >>> result = obj.method()
    """
    pass
```

## Development Workflow

### 1. Setting Up Development Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 mypy
```

### 2. Making Changes

1. Create a feature branch
2. Make your changes
3. Write/update tests
4. Run tests: `pytest tests/`
5. Format code: `black src/`
6. Lint: `flake8 src/`
7. Commit with descriptive message

### 3. Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_video.py::TestVideoReader
```

## Adding New Features

### Adding a New Manipulation Mode

1. Add function to `src/detection/mask_utils.py`
2. Update `apply_manipulation()` to handle new mode
3. Add configuration option to `config/config.yaml`
4. Update README.md with documentation
5. Add tests

Example:

```python
def new_manipulation(frame: np.ndarray, masks: List[np.ndarray], **kwargs) -> np.ndarray:
    """
    Description of the new manipulation.
    
    Args:
        frame: Input frame
        masks: List of binary masks
        **kwargs: Additional parameters
    
    Returns:
        Manipulated frame
    """
    result = frame.copy()
    # Your manipulation logic
    return result
```

### Adding a New Detection Backend

1. Create new class in `src/model/` (e.g., `mask_rcnn_torch.py`)
2. Implement the same interface as `MaskRCNNModel`
3. Add configuration option to choose backend
4. Update model loader to instantiate correct backend

## Performance Optimization

### Profiling

```bash
# Profile processing script
python -m cProfile -o profile.stats scripts/process_video.py -i input.mp4 -o output.mp4
python -m pstats profile.stats
```

### Common Optimizations

1. **Frame Skip**: Process every Nth frame
2. **Resolution**: Reduce video resolution
3. **Batch Processing**: Process multiple frames at once (if model supports)
4. **GPU**: Use GPU for model inference
5. **Threading**: Parallelize I/O operations

## Debugging

### Debug Mode

Enable debug mode in config:

```yaml
debug:
  save_intermediate_frames: true
  verbose: true
  show_preview: true
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

## Model Integration Notes

### Current Status

The Mask R-CNN implementation uses a mock/placeholder approach for development. To integrate a real model:

### Option 1: Matterport Mask R-CNN (Keras/TensorFlow)

```python
# Install: pip install mrcnn
from mrcnn import model as modellib
from mrcnn.config import Config

class InferenceConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 80 + 1  # COCO has 80 classes + background

# Load model
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="./")
model.load_weights("mask_rcnn_coco.h5", by_name=True)
```

### Option 2: Detectron2 (PyTorch)

```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
```

### Option 3: TensorFlow Object Detection API

More complex setup but highly performant.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure src is in Python path
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Model Errors**: Download weights using `python src/model/download_weights.py`
4. **Memory Issues**: Reduce resolution or use frame_skip
5. **Slow Processing**: Enable GPU or reduce quality settings

## Version Control

### Commit Messages

Use descriptive commit messages:

```
feat: Add background blur manipulation mode
fix: Correct mask application for multiple detections
docs: Update README with installation instructions
refactor: Simplify video reader frame iteration
test: Add unit tests for mask utilities
```

### Git Workflow

1. Feature branches from main
2. PR with description and tests
3. Code review before merge
4. Tag releases

## Next Steps

- [ ] Integrate real Mask R-CNN model (Matterport/Detectron2)
- [ ] Add GPU acceleration
- [ ] Implement temporal tracking
- [ ] Add unit tests for all modules
- [ ] Performance benchmarking
- [ ] Custom F1 model fine-tuning

