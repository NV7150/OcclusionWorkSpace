# Refactored MR Occlusion Framework

This document provides an overview of the refactored Mixed Reality Occlusion Framework, explaining the new architecture, key components, and how to use the framework.

## Overview

The MR Occlusion Framework is designed to process occlusion for mixed reality applications. It provides a modular and extensible architecture that allows for the implementation of various occlusion algorithms while maintaining a consistent interface.

The framework handles:
1. Loading RGB images, depth images, and IMU data
2. Loading 3D object files and scene descriptions
3. Generating occlusion masks using pluggable occlusion providers
4. Rendering mixed reality scenes with proper occlusion
5. Visualizing the scene for debugging and analysis

## New Architecture

The refactored architecture follows modern software design principles, with a focus on:
- **Clear Separation of Concerns**: Each component has a well-defined responsibility
- **Interface-Based Design**: Components interact through interfaces, allowing for easy substitution
- **Dependency Injection**: Components receive their dependencies through constructors
- **Centralized Resource Management**: Shared resources like OpenGL buffers are managed centrally

### Directory Structure

```
MR_Occlusion_Framework/
├── core/                      # Core interfaces (abstract base classes)
│   ├── IFrameLoader.py
│   ├── IOcclusionProvider.py
│   ├── IModel.py
│   ├── IRenderer.py
│   ├── IScene.py
│   └── ITracker.py
├── DataLoaders/               # Data loading (frames, sensor data)
│   ├── Frame.py
│   ├── BaseFrameLoader.py
│   └── UniformedFrameLoader.py
├── Models/                    # 3D Model and Scene representation/management
│   ├── Model.py
│   ├── Mesh.py
│   ├── Material.py
│   ├── SceneManager.py
│   ├── BaseAssetLoader.py
│   ├── FbxLoader.py
│   └── ObjLoader.py
├── Rendering/                 # Unified Modern OpenGL Rendering Subsystem
│   ├── Renderer.py
│   ├── ShaderManager.py
│   ├── TextureManager.py
│   ├── BufferManager.py
│   ├── Framebuffer.py
│   ├── Camera.py
│   ├── Primitives.py
│   └── VisualizationRenderer.py
├── Systems/                   # High-level system coordinators
│   ├── OcclusionSystem.py
│   ├── VisualizeSystem.py
│   └── OcclusionProcessor.py
├── Trackers/                  # Tracking implementations
│   └── ApriltagTracker.py
├── OcclusionProviders/        # Occlusion algorithm implementations
│   ├── BaseOcclusionProvider.py
│   ├── DepthThresholdOcclusionProvider.py
│   ├── DepthGradientOcclusionProvider.py
│   └── SimpleOcclusionProvider.py
└── Utils/                     # Common utilities
    ├── TransformUtils.py
    ├── Logger.py
    └── MarkerPositionLoader.py
```

## Key Components

### Core Interfaces

The `core/` directory contains interfaces that define the contract for each component type:

- **IFrameLoader**: Interface for loading frame data (RGB, depth, IMU)
- **IOcclusionProvider**: Interface for generating occlusion masks
- **IModel**: Interface for 3D model representation
- **IRenderer**: Interface for rendering
- **IScene**: Interface for scene management
- **ITracker**: Interface for tracking and camera pose estimation

### DataLoaders

The `DataLoaders/` directory contains components for loading frame data:

- **Frame**: Data structure for frame data (RGB, depth, IMU)
- **BaseFrameLoader**: Abstract base class for frame loaders
- **UniformedFrameLoader**: Concrete implementation for loading frames from a directory

### Models

The `Models/` directory contains components for 3D model and scene representation:

- **Model**: 3D model representation
- **Mesh**: Individual mesh component of a model
- **Material**: Material properties
- **SceneManager**: Manages scene graph and model instances
- **BaseAssetLoader**: Abstract base class for model loaders
- **FbxLoader**: Loads FBX files
- **ObjLoader**: Loads OBJ files

### Rendering

The `Rendering/` directory contains components for rendering:

- **Renderer**: Main rendering class
- **ShaderManager**: Manages shader programs
- **TextureManager**: Manages textures
- **BufferManager**: Manages OpenGL buffers (VBOs, VAOs, EBOs)
- **Framebuffer**: Manages framebuffers for off-screen rendering
- **Camera**: Camera representation
- **Primitives**: Utility for rendering basic shapes
- **VisualizationRenderer**: Specialized renderer for visualization

### Systems

The `Systems/` directory contains high-level system coordinators:

- **OcclusionSystem**: Main entry point for occlusion processing
- **VisualizeSystem**: Main entry point for visualization
- **OcclusionProcessor**: Handles occlusion mask generation

### OcclusionProviders

The `OcclusionProviders/` directory contains occlusion algorithm implementations:

- **BaseOcclusionProvider**: Abstract base class for occlusion providers
- **DepthThresholdOcclusionProvider**: Uses depth thresholding for occlusion
- **DepthGradientOcclusionProvider**: Uses depth gradients for occlusion
- **SimpleOcclusionProvider**: Simple occlusion provider for testing

### Utils

The `Utils/` directory contains utility classes:

- **TransformUtils**: Utilities for 3D transformations
- **Logger**: Logging utilities
- **MarkerPositionLoader**: Loads marker positions from JSON files

## How to Use the Framework

### Basic Usage

Here's a simple example of how to use the framework for occlusion processing:

```python
from DataLoaders.UniformedFrameLoader import UniformedFrameLoader
from OcclusionProviders.SimpleOcclusionProvider import SimpleOcclusionProvider
from Models.SceneManager import SceneManager
from Rendering.Renderer import Renderer
from Systems.OcclusionSystem import OcclusionSystem
from Logger.Logger import Logger

# Create logger
logger = Logger(['SYSTEM', 'ERROR'])

# Create components
frame_loader = UniformedFrameLoader(['data/frames'], logger=logger)
scene_manager = SceneManager(['data/models'], logger=logger)
renderer = Renderer(logger=logger)
occlusion_provider = SimpleOcclusionProvider(logger=logger)

# Create occlusion system
system = OcclusionSystem(
    frame_loader=frame_loader,
    scene_manager=scene_manager,
    renderer=renderer,
    occlusion_provider=occlusion_provider,
    output_dir='output',
    output_prefix='frame',
    logger=logger
)

# Process data
system.process()
```

### Visualization

To visualize the scene:

```python
from Systems.VisualizeSystem import create_visualization_system

# Create visualization system
system = create_visualization_system(
    scene_model_path='data/scene.fbx',
    frames_dir='data/frames',
    marker_file='data/markers.json',
    camera_matrix_file='data/camera_matrix.csv',
    render_obj_dir='data/models',
    log_keys=['SYSTEM', 'ERROR']
)

# Load data and run visualization
system.load_data()
system.run()
```

### Custom Occlusion Provider

To create a custom occlusion provider:

```python
from OcclusionProviders.BaseOcclusionProvider import BaseOcclusionProvider
from DataLoaders.Frame import Frame
import numpy as np

class MyOcclusionProvider(BaseOcclusionProvider):
    def __init__(self, my_param: float = 0.5, logger=None):
        super().__init__(name="MyOcclusionProvider")
        self.set_parameter("my_param", my_param)
        self._logger = logger
    
    def generate_occlusion_mask(self, frame: Frame, virtual_depth: Optional[np.ndarray] = None) -> np.ndarray:
        # Implement your occlusion algorithm here
        # Use frame.rgb, frame.depth, etc.
        
        # Example: simple threshold on depth
        my_param = self.get_parameter("my_param")
        mask = (frame.depth < my_param).astype(np.uint8)
        
        # Store the mask for later retrieval
        self._last_mask = mask
        
        return mask
```

## Command Line Interface

The framework provides command line interfaces for both occlusion processing and visualization:

### Occlusion Processing

```bash
python -m Systems.OcclusionSystem \
    --data-dirs data/frames \
    --model-dirs data/models \
    --output-dir output \
    --output-prefix frame \
    --occlusion-provider OcclusionProviders.SimpleOcclusionProvider.SimpleOcclusionProvider \
    --log-keys SYSTEM ERROR
```

### Visualization

```bash
python -m Systems.VisualizeSystem \
    --scene-model data/scene.fbx \
    --frames-dir data/frames \
    --marker-file data/markers.json \
    --camera-matrix data/camera_matrix.csv \
    --render-obj-dir data/models \
    --log-keys SYSTEM ERROR
```

## Example

A complete example is provided in `Example/refactored_example.py`:

```bash
python Example/refactored_example.py \
    --data-dirs data/frames \
    --model-dirs data/models \
    --output-dir output \
    --scene-model data/scene.fbx \
    --marker-file data/markers.json \
    --camera-matrix data/camera_matrix.csv \
    --visualize
```

## Benefits of the Refactored Architecture

1. **Modularity**: Components can be easily replaced or extended
2. **Testability**: Components can be tested in isolation
3. **Maintainability**: Clear separation of concerns makes the code easier to understand and modify
4. **Reusability**: Components can be reused in different contexts
5. **Performance**: Centralized resource management improves performance
6. **Extensibility**: New occlusion algorithms, renderers, etc. can be easily added

## Future Improvements

1. **Performance Optimization**: Further optimize rendering and occlusion mask generation
2. **Additional Occlusion Algorithms**: Implement more sophisticated occlusion algorithms
3. **Real-time Processing**: Support for real-time occlusion processing
4. **Mobile Support**: Adapt the framework for mobile platforms
5. **Integration with Game Engines**: Provide integrations with Unity, Unreal, etc.