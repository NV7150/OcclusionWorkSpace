# Mixed Reality Occlusion System Documentation

## Overview

This document provides a comprehensive overview of the Mixed Reality (MR) occlusion system architecture, focusing on the class structure and relationships between components. The system is designed to create realistic occlusion effects in mixed reality applications by comparing real-world depth data with virtual content depth.

## Class Structure and Relationships

### Core Components

The system consists of several key components that work together:

1. **Frame** - Data container for camera frames
2. **OcclusionProvider** - Interface for occlusion mask generation
3. **ContentsDepthCal** - Calculator for MR content depth maps
4. **DataLoader** - Loader for camera frames and depth data
5. **BaseSystem** - Main coordinator for the occlusion framework
6. **Renderer** - Renderer for MR scenes

### Component Relationships

```
                  +----------------+
                  |   BaseSystem   |
                  +----------------+
                          |
                          | coordinates
                          v
    +----------+    +------------+    +-------------+
    | Renderer |<-->|ContentsDepthCal|    |DataLoader|
    +----------+    +------------+    +-------------+
         |                 |                 |
         | renders         | calculates      | loads
         v                 v                 v
    +---------+    +----------------+    +-------+
    | Models  |    | MR Depth Maps  |    | Frame |
    +---------+    +----------------+    +-------+
                          |                 |
                          |                 |
                          v                 v
                  +------------------+
                  |OcclusionProvider |
                  +------------------+
                          |
                          | generates
                          v
                  +------------------+
                  | Occlusion Masks  |
                  +------------------+
```

## Detailed Component Descriptions

### 1. Frame (Interfaces/Frame.py)

The Frame class serves as a data container for camera frames, including:
- RGB image data
- Depth image data
- IMU data (accelerometer and gyroscope)
- Timestamps

```python
class Frame:
    def __init__(self, timestamp, rgb, depth, acc, gyro, timestamp_depth=None, timestamp_imu=None):
        # Initialize frame data
```

### 2. OcclusionProvider (Interfaces/OcclusionProvider.py)

The OcclusionProvider is an abstract interface that defines the method for generating occlusion masks:

```python
class OcclusionProvider(metaclass=ABCMeta):
    @abstractmethod
    def occlusion(self, frame: Frame, mr_depth: np.ndarray) -> np.ndarray:
        # Generate occlusion mask
        pass
```

Implementations include:
- **SimpleOcclusion** - Compares real camera depth with MR content depth
- **DepthThresholdOcclusion** - Uses a threshold on depth values
- **DepthGradientOcclusion** - Uses depth gradients to detect object boundaries

### 3. ContentsDepthCal (Systems/ContentsDepthCal.py)

The ContentsDepthCal class calculates depth maps for MR contents using OpenGL rendering:

```python
class ContentsDepthCal:
    def __init__(self, renderer: Renderer):
        # Initialize with renderer
        
    def calculate_depth(self, frame: Frame, models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        # Calculate depth map for MR contents
```

Key responsibilities:
- Setting up OpenGL for depth rendering
- Rendering 3D models to a depth buffer
- Converting normalized depth values to actual depth in meters

### 4. DataLoader (Systems/DataLoader.py)

The DataLoader class is responsible for loading RGB images, depth images, and IMU data:

```python
class DataLoader:
    def __init__(self, data_dirs: List[str]):
        # Initialize with data directories
        
    def load_data(self) -> Dict[np.datetime64, Frame]:
        # Load all data and return frames
```

Key features:
- Loads depth data from CSV files
- Loads RGB images from JPG/PNG files
- Loads IMU data from CSV files
- Creates Frame objects with synchronized data

### 5. BaseSystem (Systems/BaseSystem.py)

The BaseSystem class coordinates the entire occlusion framework:

```python
class BaseSystem:
    def __init__(self, data_dirs, model_dirs, output_dir, output_prefix, occlusion_provider, log_keys=None, log_to_file=False, log_file_path=None):
        # Initialize components
        
    def process(self, scene_name=None):
        # Process all data: load, generate occlusion masks, and render
```

Key responsibilities:
- Initializing and coordinating all components
- Loading data (frames, models, scenes)
- Generating occlusion masks using the provided OcclusionProvider
- Rendering frames with occlusion masks

### 6. Renderer (Systems/Renderer.py)

The Renderer class is responsible for rendering mixed reality scenes:

```python
class Renderer:
    def __init__(self, output_dir: str):
        # Initialize renderer
        
    def render_frame(self, frame: Frame, occlusion_mask: np.ndarray, models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        # Render a frame with occlusion mask
```

Key features:
- Loading and caching 3D models
- Setting up OpenGL for rendering
- Applying transforms to models
- Rendering models with occlusion masks

## Data Flow

1. **Data Loading**:
   - DataLoader loads RGB, depth, and IMU data
   - ModelLoader loads 3D models and scene descriptions

2. **Occlusion Mask Generation**:
   - ContentsDepthCal calculates depth maps for MR contents
   - OcclusionProvider compares real depth with MR depth to generate occlusion masks

3. **Rendering**:
   - Renderer renders the MR scene with occlusion masks
   - Results are saved to the output directory

## Implementation Examples

### Creating a Custom OcclusionProvider

To create a custom occlusion provider, implement the OcclusionProvider interface:

```python
from Interfaces.OcclusionProvider import OcclusionProvider
from Interfaces.Frame import Frame
import numpy as np

class MyCustomOcclusion(OcclusionProvider):
    def __init__(self, param1, param2):
        # Initialize with parameters
        
    def occlusion(self, frame: Frame, mr_depth: np.ndarray) -> np.ndarray:
        # Custom occlusion logic
        # Compare frame.depth with mr_depth
        # Return binary mask (1 = occluded, 0 = not occluded)
```

### Using the System

```python
from Systems.BaseSystem import BaseSystem
from Occlusions.SimpleOcclusion import SimpleOcclusion

# Create occlusion provider
occlusion_provider = SimpleOcclusion(max_depth=5.0)

# Create and run the system
system = BaseSystem(
    data_dirs=["path/to/data"],
    model_dirs=["path/to/models"],
    output_dir="path/to/output",
    output_prefix="frame",
    occlusion_provider=occlusion_provider
)

# Process all data
system.process()
```

## Future Implementation Considerations

1. **Performance Optimization**:
   - Implement GPU-accelerated depth map calculation
   - Optimize OpenGL rendering for better performance

2. **Advanced Occlusion Techniques**:
   - Implement machine learning-based occlusion detection
   - Add support for semi-transparent objects

3. **Integration with Other Systems**:
   - Add support for real-time camera input
   - Integrate with AR/VR frameworks

4. **Extensibility**:
   - Create a plugin system for custom occlusion providers
   - Support for different rendering backends

## Conclusion

The Mixed Reality Occlusion System provides a flexible framework for generating realistic occlusion effects in mixed reality applications. By comparing real-world depth data with virtual content depth, the system can accurately determine which parts of virtual objects should be occluded by real-world objects.

The modular design allows for easy extension and customization, making it suitable for a wide range of mixed reality applications.