# MR Occlusion Framework Examples

This directory contains examples demonstrating how to use the MR Occlusion Framework.

## Primary Example Runner

The main example runner is `run_system.py`, which provides a unified interface to run both the Occlusion System and Visualization System.

### Using run_system.py

```bash
# Run the occlusion system
python run_system.py occlusion \
  --data-dirs data/frames \
  --model-dirs models/scene1 \
  --output-dir output/occlusion \
  --occlusion-provider OcclusionProviders.SimpleOcclusionProvider

# Run the visualization system
python run_system.py visualize \
  --scene-model models/scene.fbx \
  --frames-dir data/frames \
  --marker-file markers.json \
  --camera-matrix camera.csv \
  --render-obj-dir models/scene1

# Run using a JSON configuration file
python run_system.py --config config_example.json
```

### JSON Configuration Files

You can now use JSON configuration files to specify all parameters instead of command-line arguments. This is especially useful for complex configurations or when debugging with VS Code.

```bash
python run_system.py --config config_example.json
```

The JSON file must include a `command` field specifying the system type (`visualize` or `occlusion`). Other fields should match the command-line argument names with camelCase formatting.

Example configuration file for the Visualization System (`config_example.json`):

```json
{
  "command": "visualize",
  "sceneModel": "../LocalData/DepthIMUData2/Env_3DModels/on_the_desk.fbx",
  "framesDir": "../LocalData/DepthIMUData2/slow",
  "markerFile": "../LocalData/DepthIMUData2/Env_3DModels/marker_poses_opengl.json",
  "cameraMatrix": "../LocalData/camera_ipadpro.csv",
  "renderObjDir": "../LocalData/Models",
  "tagSize": 0.086,
  "tagFamily": "tagStandard41h12",
  "log_keys": ["error", "system", "debug"],
  "log_to_file": true,
  "log_file": "visualization_run.log"
}
```

Field mapping for Visualization System:
- `command`: System type (`visualize`)
- `sceneModel`: Path to the 3D scan model (.fbx)
- `framesDir`: Directory containing frame images
- `markerFile`: Path to marker positions JSON file
- `cameraMatrix`: Path to camera matrix CSV file
- `renderObjDir`: Directory containing MR content models
- `tagSize`: Size of AprilTag markers in meters
- `tagFamily`: AprilTag family to use for detection
- `log_keys`: Logging keys to enable
- `log_to_file`: Enable logging to file
- `log_file`: Path to log file

Field mapping for Occlusion System:
- `command`: System type (`occlusion`)
- `dataDirs`: Directories containing RGB, depth, and IMU data
- `modelDirs`: Directories containing 3D models and scene descriptions
- `outputDir`: Directory where rendered images will be saved
- `outputPrefix`: Prefix for output filenames
- `occlusionProvider`: Python module and class for occlusion provider
- `cameraMatrix`: Path to camera matrix file
- `sceneName`: Name of the scene to render
- `log_keys`: Logging keys to enable
- `log_to_file`: Enable logging to file
- `log_file`: Path to log file

### Common Options

Both systems support the following common options:

- `--log-keys`: Logging keys to enable (e.g., system, debug, error)
- `--log-to-file`: Enable logging to file
- `--log-file`: Path to log file (default: auto-generated based on system type)

### Occlusion System Options

```bash
python run_system.py occlusion --help
```

Key options:
- `--data-dirs`: Directories containing RGB, depth, and IMU data
- `--model-dirs`: Directories containing 3D models and scene descriptions
- `--output-dir`: Directory where rendered images will be saved
- `--output-prefix`: Prefix for output filenames
- `--occlusion-provider`: Python module and class for occlusion provider
- `--camera-matrix`: Path to camera matrix file
- `--scene-name`: Name of the scene to render

### Visualization System Options

```bash
python run_system.py visualize --help
```

Key options:
- `--scene-model`: Path to the 3D scan model (.fbx) of the scene
- `--frames-dir`: Directory containing frame images for camera pose estimation
- `--marker-file`: Path to the marker positions JSON file
- `--camera-matrix`: Path to the camera matrix CSV file
- `--render-obj-dir`: Directory containing MR content models and scene description
- `--tag-size`: Size of AprilTag markers in meters
- `--tag-family`: AprilTag family to use for detection

## Other Example Files

The following examples are provided for reference and educational purposes:

- `ModernVisualizeExample.py`: Demonstrates how to use the VisualizeSystem with proper dependency injection (deprecated in favor of run_system.py)
- `FrameExample.py`: Example demonstrating the Frame data structure
- `example.py`: Basic example showing core functionality
- `depth_viewer.py`: Utility for visualizing depth images
- `ApriltagTest.py`: Demonstrates AprilTag tracking
- `TrackerExample.py`: Example showing the tracking functionality
- `VisualizeExample.py`: Original visualization example

## Creating Your Own Examples

To create your own examples using the framework:

1. Import the necessary interfaces from the `core` module
2. Create concrete implementations or use existing ones
3. Wire them together using dependency injection
4. Use the systems to process or visualize data

Example of creating a custom script:

```python
import sys
import os
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core interfaces
from core.IFrameLoader import IFrameLoader
from core.IScene import IScene
from core.IOcclusionProvider import IOcclusionProvider
from core.IRenderer import IRenderer

# Import concrete implementations
from DataLoaders.UniformedFrameLoader import UniformedFrameLoader
from Models.SceneManager import SceneManager
from OcclusionProviders.SimpleOcclusionProvider import SimpleOcclusionProvider
from Rendering.Renderer import Renderer
from Systems.OcclusionSystem import OcclusionSystem

# Create components
frame_loader = UniformedFrameLoader(["data/frames"])
scene_manager = SceneManager()
scene_manager.load_models_from_directory("models/scene1")
renderer = Renderer("output/custom")
renderer.initialize()
occlusion_provider = SimpleOcclusionProvider()

# Create and run system
system = OcclusionSystem(
    frame_loader=frame_loader,
    scene_manager=scene_manager,
    renderer=renderer,
    occlusion_provider=occlusion_provider,
    output_dir="output/custom",
    output_prefix="frame"
)

system.process() 