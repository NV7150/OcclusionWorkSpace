# MR Occlusion Framework

A framework for Mixed Reality occlusion handling, providing tools for detecting and rendering occlusions between real and virtual objects.

## Overview

The MR Occlusion Framework is designed to process occlusion for mixed reality applications. It provides a modular and extensible architecture that allows for the implementation of various occlusion algorithms while maintaining a consistent interface.

The framework handles:
1. Loading RGB images, depth images, and IMU data
2. Loading 3D object files and scene descriptions
3. Generating occlusion masks using pluggable occlusion providers
4. Rendering mixed reality scenes with proper occlusion
5. Visualizing the scene in 3D for debugging and analysis

## Architecture

The framework follows a modular architecture with clear separation of concerns:

### Core Components

- `/core`: Core interfaces defining the contracts for each component
  - `IFrameLoader.py`: Interface for loading frame data
  - `IOcclusionProvider.py`: Interface for occlusion detection
  - `IModel.py`: Interface for 3D models
  - `IRenderer.py`: Interface for rendering
  - `IScene.py`: Interface for scene management
  - `ITracker.py`: Interface for tracking
  - `IFrame.py`: Interface for frame data representation

### Sensor Data

- `/Data`: Sensor data components
  - `RGBData.py`: RGB image data
  - `DepthData.py`: Depth image data
  - `IMUData.py`: Inertial measurement unit data
  - `CameraData.py`: Camera parameters data
  - `OcclusionData.py`: Occlusion mask data

### Data Loading

- `/DataLoaders`: Components for loading and managing frame data
  - `Frame.py`: Data structure for sensor frames (implements IFrame)
  - `BaseFrameLoader.py`: Abstract base class for frame loaders
  - `UniformedFrameLoader.py`: Loader for uniformed data format
  - `SeparatedFrameLoader.py`: Loader for separated data format

### 3D Model Management

- `/Models`: Components for loading and managing 3D models
  - `Model.py`: Unified 3D model data structure
  - `Mesh.py`: Individual mesh component of a Model
  - `Material.py`: Material properties
  - `Texture.py`: Texture data
  - `BaseAssetLoader.py`: Abstract base class for asset loaders
  - `FbxLoader.py`: Loads FBX files using PyAssimp
  - `ObjLoader.py`: Loads OBJ files using PyWavefront
  - `BaseSceneLoader.py`: Abstract base class for scene loaders
  - `JsonSceneLoader.py`: Loads scene structure from JSON
  - `SceneManager.py`: Manages scene graph and object instances

### Rendering

- `/Rendering`: Modern OpenGL rendering pipeline
  - `Renderer.py`: Main renderer implementation
  - `ShaderManager.py`: GLSL shader management
  - `TextureManager.py`: Texture loading and management
  - `BufferManager.py`: OpenGL buffer management
  - `Framebuffer.py`: Framebuffer operations
  - `Camera.py`: Camera management
  - `Primitives.py`: Basic geometric primitives

### Occlusion

- `/OcclusionProviders`: Occlusion detection algorithms
  - `SimpleOcclusionProvider.py`: Simple depth-based occlusion
  - `DepthThresholdOcclusionProvider.py`: Threshold-based occlusion

### Tracking

- `/Trackers`: Camera pose tracking implementations
  - `ApriltagTracker.py`: AprilTag-based tracking

### System Coordination

- `/Systems`: High-level system coordinators
  - `BaseSystem.py`: Facade for OcclusionSystem and VisualizeSystem
  - `OcclusionSystem.py`: Main occlusion system
  - `OcclusionProcessor.py`: Occlusion processing logic
  - `VisualizeSystem.py`: 3D visualization system
  - `VisualizeModelRender.py`: Rendering for visualization

### Utilities

- `/Utils`: Utility functions and classes
  - `TransformUtils.py`: 3D transformation utilities
  - `Logger.py`: Logging functionality
  - `MarkerPositionLoader.py`: Marker position loading
  - `PnP_viz.py`: PnP visualization utilities

### Examples

- `/Example`: Example applications
  - `example.py`: Basic example
  - `depth_viewer.py`: Depth visualization
  - `ApriltagTest.py`: AprilTag tracking example
  - `TrackerExample.py`: Tracking example
  - `VisualizeExample.py`: Visualization example
  - `ModernVisualizeExample.py`: Modern visualization example using interfaces
  - `FrameExample.py`: Example demonstrating the new Frame structure

## Frame Structure

The framework uses a flexible component/plugin style for frame data representation:

```python
# Create a frame
frame = Frame(timestamp=np.datetime64('2025-05-08T12:00:00'))

# Create and register each sensor's data
rgb_data = RGBData(timestamp=frame.timestamp, image=rgb_image)
depth_data = DepthData(timestamp=frame.timestamp, depth=depth_image)
imu_data = IMUData(timestamp=frame.timestamp, acc=acc_array, gyro=gyro_array)

frame.add_sensor(rgb_data)
frame.add_sensor(depth_data)
frame.add_sensor(imu_data)

# Retrieve by type
if frame.has_sensor(RGBData):
    rgb = frame.get_sensor(RGBData)
    process(rgb.image)
```

This design provides several benefits:
- **Extensibility**: To support a new sensor, just define a new data class and call `frame.add_sensor()`
- **Type safety**: Each sensor lives in its own class with proper type checking
- **Stable structure**: Avoids brittle dictionary lookups with string keys
- **Simple interface**: Minimal and consistent API

## Data Flow

1. **Data Loading**:
   - `UniformedFrameLoader` loads RGB, depth, and IMU data
   - `SceneManager` loads 3D models and scene descriptions

2. **Occlusion Mask Generation**:
   - `OcclusionProcessor` uses the `Renderer` to calculate depth maps for MR contents
   - `OcclusionProvider` compares real depth with MR depth to generate occlusion masks

3. **Rendering**:
   - `Renderer` renders the MR scene with occlusion masks
   - Results are saved to the output directory

4. **Visualization**:
   - `VisualizeSystem` provides a 3D visualization of the scene
   - `ApriltagTracker` estimates camera poses for visualization

## Usage

The framework can be used in two modes:

### 1. Occlusion Mode

For generating occlusion masks and rendering mixed reality scenes:

```bash
python -m Systems.BaseSystem --system-type occlusion --data-dirs /path/to/data --model-dirs /path/to/models --output-dir /path/to/output --occlusion-provider OcclusionProviders.SimpleOcclusionProvider
```

### 2. Visualize Mode

For visualizing the scene with tracking:

```bash
python -m Systems.BaseSystem --system-type visualize --data-dirs /path/to/data --model-dirs /path/to/models --output-dir /path/to/output --scene-model /path/to/scene.obj --marker-file /path/to/markers.json
```

## Sample Code

### Creating a Custom Occlusion Provider

```python
from core.IOcclusionProvider import IOcclusionProvider
from core.IFrame import IFrame
from Data.DepthData import DepthData
import numpy as np

class MyCustomOcclusion(IOcclusionProvider):
    def __init__(self, param1, param2):
        # Initialize with parameters
        self.param1 = param1
        self.param2 = param2
        
    def occlusion(self, frame: IFrame) -> np.ndarray:
        # Simple implementation that returns an empty mask
        depth_data = frame.get_sensor(DepthData)
        if depth_data is None:
            return np.array([])
        return np.zeros_like(depth_data.depth, dtype=np.uint8)
    
    def occlusion_with_mr_depth(self, frame: IFrame, mr_depth: np.ndarray) -> np.ndarray:
        # Custom occlusion logic
        depth_data = frame.get_sensor(DepthData)
        if depth_data is None:
            return np.array([])
            
        camera_depth = depth_data.depth
        
        # Create occlusion mask based on depth comparison
        occlusion_mask = np.zeros_like(camera_depth, dtype=np.uint8)
        
        # Compare depths where both have valid values
        valid_mask = (camera_depth > 0) & (mr_depth > 0)
        
        # Real objects occlude virtual when real depth is less than virtual depth
        occlusion_mask[valid_mask] = (camera_depth[valid_mask] < mr_depth[valid_mask]).astype(np.uint8)
        
        return occlusion_mask
```

### Using the System

```python
from Systems.OcclusionSystem import OcclusionSystem
from OcclusionProviders.SimpleOcclusionProvider import SimpleOcclusionProvider

# Create occlusion provider
occlusion_provider = SimpleOcclusionProvider(max_depth=5.0)

# Create and run the system
system = OcclusionSystem.create(
    data_dirs=["path/to/data"],
    model_dirs=["path/to/models"],
    output_dir="path/to/output",
    output_prefix="frame",
    occlusion_provider=occlusion_provider
)

# Process all data
system.process()
```

### Using the Tracker

```python
import numpy as np
from Trackers.ApriltagTracker import ApriltagTracker
from DataLoaders.Frame import Frame
from Data import RGBData, DepthData, IMUData, CameraData

# Create a frame with sensor data
timestamp = np.datetime64('2025-05-08T12:00:00')
frame = Frame(timestamp)

# Add RGB data
rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
frame.add_sensor(RGBData(timestamp, rgb_image))

# Create camera matrix
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

# Add camera data to frame
frame.add_sensor(CameraData(timestamp, camera_matrix))

# Initialize tracker
tracker = ApriltagTracker(camera_matrix)
tracker.load_marker_positions("path/to/marker_positions.json")

# Use with a frame
camera_pose = tracker.track(frame)
print(f"Camera position: {camera_pose[:3, 3]}")

# Store the camera pose in the frame
frame.set_camera_pose(camera_pose)
```

## Dependencies

The framework depends on the following Python packages:
- numpy
- pandas
- Pillow (PIL)
- pyrr
- OpenGL (PyOpenGL)
- OpenCV (cv2)
- pupil-apriltags
- PyWavefront (for OBJ files)
- PyAssimp (for FBX files)

These can be installed using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Future Development

The following tasks are planned for future development:

1. **Update VisualizeSystem to use the new Frame structure**
   - Refactor VisualizeSystem.py to use the IFrame interface
   - Update VisualizeModelRender.py to work with the new sensor data components

2. **Update OcclusionProviders to use the new Frame structure**
   - Refactor SimpleOcclusionProvider and DepthThresholdOcclusionProvider to use the IFrame interface
   - Update the occlusion algorithms to work with the new sensor data components

3. **Create additional sensor data types**
   - Add support for more sensor types (e.g., LiDAR, thermal camera)
   - Implement appropriate data classes in the Data directory

4. **Improve error handling and validation**
   - Add validation for sensor data
   - Implement better error handling for missing or invalid sensor data

5. **Enhance documentation and examples**
   - Create more comprehensive examples for the new Frame structure
   - Update existing examples to use the new Frame structure

6. **Performance optimization**
   - Optimize memory usage for large datasets
   - Implement caching mechanisms for frequently accessed sensor data

## References

```
@inproceedings{walton2017accurate,
  title={Accurate real-time occlusion for mixed reality},
  author={Walton, David R and Steed, Anthony},
  booktitle={Proceedings of the 23rd ACM Symposium on Virtual Reality Software and Technology},
  pages={1--10},
  year={2017}
}
```

```
@inproceedings{hebborn2017occlusion,
  title={Occlusion matting: realistic occlusion handling for augmented reality applications},
  author={Hebborn, Anna Katharina and H{\"o}hner, Nils and M{\"u}ller, Stefan},
  booktitle={2017 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)},
  pages={62--71},
  year={2017},
  organization={IEEE}
}
```

```
@inproceedings{du2016edge,
  title={Edge snapping-based depth enhancement for dynamic occlusion handling in augmented reality},
  author={Du, Chao and Chen, Yen-Lin and Ye, Mao and Ren, Liu},
  booktitle={2016 IEEE international symposium on mixed and augmented reality (ISMAR)},
  pages={54--62},
  year={2016},
  organization={IEEE}
}