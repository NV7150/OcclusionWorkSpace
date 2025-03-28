# Occlusion Framework for Mixed Reality

This document describes the API and architecture of the Occlusion Framework for Mixed Reality.

## Overview

The Occlusion Framework is designed to process occlusion for mixed reality applications. It provides a modular and extensible architecture that allows for the implementation of various occlusion algorithms while maintaining a consistent interface.

The framework handles:
1. Loading RGB images, depth images, and IMU data
2. Loading 3D object files and scene descriptions
3. Generating occlusion masks using pluggable occlusion providers
4. Rendering mixed reality scenes with proper occlusion
5. Saving the rendered results

## Architecture

The framework consists of the following main components:

### BaseSystem

The main entry point and coordinator for the framework. It orchestrates the data loading, occlusion mask generation, and rendering processes.

### DataLoader

Responsible for loading RGB images, depth images, and IMU data from specified directories. It creates Frame objects that encapsulate all the sensor data for a specific timestamp.

### ModelLoader

Handles loading 3D object files (.obj, .fbx) and scene description files (.json) that define the virtual content to be rendered.

### Renderer

Renders the mixed reality scenes by combining real-world images with virtual content, applying occlusion masks to ensure proper depth ordering.

### OcclusionProvider (Interface)

An abstract interface that must be implemented by occlusion algorithms. It provides the method to generate occlusion masks from Frame objects.

### Frame (Interface)

A data structure that encapsulates all sensor data (RGB, depth, IMU) for a specific timestamp.

## API Reference

### BaseSystem

```python
class BaseSystem:
    def __init__(self, data_dirs: List[str], model_dirs: List[str], 
                 output_dir: str, occlusion_provider: OcclusionProvider):
        """
        Initialize the BaseSystem with directories and an occlusion provider.
        
        Args:
            data_dirs: List of directories containing RGB, depth, and IMU data
            model_dirs: List of directories containing 3D models and scene descriptions
            output_dir: Directory where rendered images will be saved
            occlusion_provider: Implementation of OcclusionProvider to generate occlusion masks
        """
        
    def load_data(self):
        """
        Load all data (frames, models, scenes).
        """
        
    def generate_occlusion_masks(self):
        """
        Generate occlusion masks for all frames using the provided occlusion provider.
        """
        
    def render_frames(self, scene_name: Optional[str] = None):
        """
        Render all frames with occlusion masks and save the results.
        
        Args:
            scene_name: Name of the scene to render. If None, the first scene will be used.
        """
        
    def process(self, scene_name: Optional[str] = None):
        """
        Process all data: load, generate occlusion masks, and render.
        
        Args:
            scene_name: Name of the scene to render. If None, the first scene will be used.
        """
```

### DataLoader

```python
class DataLoader:
    def __init__(self, data_dirs: List[str]):
        """
        Initialize the DataLoader with a list of data directories.
        
        Args:
            data_dirs: List of directory paths containing the data
        """
        
    def load_data(self) -> Dict[np.datetime64, Frame]:
        """
        Load all RGB images, depth images, and IMU data from the specified directories.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        
    def get_frame_by_timestamp(self, timestamp: np.datetime64) -> Optional[Frame]:
        """
        Get a frame by its timestamp.
        
        Args:
            timestamp: Timestamp of the frame to retrieve
            
        Returns:
            Frame object or None if not found
        """
        
    def get_frames_sorted(self) -> List[Frame]:
        """
        Get all frames sorted by timestamp.
        
        Returns:
            List of Frame objects sorted by timestamp
        """
```

### ModelLoader

```python
class ModelLoader:
    def __init__(self, model_dirs: List[str]):
        """
        Initialize the ModelLoader with a list of model directories.
        
        Args:
            model_dirs: List of directory paths containing the models and scene descriptions
        """
        
    def load_models_and_scenes(self) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
        """
        Load all 3D models and scene descriptions from the specified directories.
        
        Returns:
            Tuple containing:
            - Dictionary mapping model IDs to model objects
            - Dictionary mapping directory names to scene descriptions
        """
        
    def get_model_by_id(self, model_id: str) -> Optional[Any]:
        """
        Get a model by its ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Model object or None if not found
        """
        
    def get_scene_by_directory(self, directory: str) -> Optional[Dict]:
        """
        Get a scene description by its directory name.
        
        Args:
            directory: Name of the directory containing the scene
            
        Returns:
            Scene description dictionary or None if not found
        """
        
    def get_object_transform(self, scene_name: str, object_id: str) -> Optional[Dict]:
        """
        Get the transform (position, rotation) of an object in a scene.
        
        Args:
            scene_name: Name of the scene
            object_id: ID of the object
            
        Returns:
            Dictionary containing position and rotation or None if not found
        """
```

### Renderer

```python
class Renderer:
    def __init__(self, output_dir: str):
        """
        Initialize the Renderer with an output directory.
        
        Args:
            output_dir: Directory path where rendered images will be saved
        """
        
    def render_frame(self, frame: Frame, occlusion_mask: np.ndarray, 
                     models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        """
        Render a mixed reality frame based on the occlusion mask and scene description.
        
        Args:
            frame: Frame object containing RGB and depth images
            occlusion_mask: Binary mask indicating occluded areas
            models: Dictionary of 3D models
            scene_data: Scene description dictionary
            
        Returns:
            Rendered image as a numpy array
        """
        
    def save_rendered_image(self, image: np.ndarray, timestamp: np.datetime64) -> str:
        """
        Save a rendered image to the output directory.
        
        Args:
            image: Rendered image as a numpy array
            timestamp: Timestamp of the frame
            
        Returns:
            Path to the saved image file
        """
        
    def render_and_save_batch(self, frames: List[Frame], 
                             occlusion_masks: Dict[np.datetime64, np.ndarray],
                             models: Dict[str, Any], 
                             scene_data: Dict) -> List[str]:
        """
        Render and save a batch of frames.
        
        Args:
            frames: List of Frame objects
            occlusion_masks: Dictionary mapping timestamps to occlusion masks
            models: Dictionary of 3D models
            scene_data: Scene description dictionary
            
        Returns:
            List of paths to the saved image files
        """
```

### OcclusionProvider (Interface)

```python
class OcclusionProvider(metaclass=ABCMeta):
    @abstractmethod
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Generate an occlusion mask for a frame.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        pass
```

### Frame (Interface)

```python
class Frame:
    def __init__(self, timestamp: np.datetime64, rgb: np.ndarray, depth: np.ndarray, 
                 acc: np.ndarray, gyro: np.ndarray, 
                 timestamp_depth: np.datetime64 = None, 
                 timestamp_imu: np.datetime64 = None):
        """
        Initialize a Frame with sensor data.
        
        Args:
            timestamp: Timestamp of the frame
            rgb: RGB image as a numpy array
            depth: Depth image as a numpy array
            acc: Acceleration data as a numpy array [x, y, z]
            gyro: Gyroscope data as a numpy array [x, y, z]
            timestamp_depth: Timestamp of the depth image (if different from frame timestamp)
            timestamp_imu: Timestamp of the IMU data (if different from frame timestamp)
        """
```

## Usage

### Command Line Interface

The framework can be run from the command line using the following syntax:

```bash
python BaseSystem.py --data-dirs <data_dir1> <data_dir2> ... --model-dirs <model_dir1> <model_dir2> ... --output-dir <output_dir> --occlusion-provider <module.ClassName> [--scene-name <scene_name>]
```

Arguments:
- `--data-dirs`: One or more directories containing RGB, depth, and IMU data
- `--model-dirs`: One or more directories containing 3D models and scene descriptions
- `--output-dir`: Directory where rendered images will be saved
- `--occlusion-provider`: Python module and class for occlusion provider (e.g., "my_module.MyOcclusionProvider")
- `--scene-name` (optional): Name of the scene to render

### Implementing a Custom Occlusion Provider

To implement a custom occlusion algorithm, create a class that inherits from `OcclusionProvider` and implements the `occlusion` method:

```python
from Interfaces.OcclusionProvider import OcclusionProvider
from Interfaces.Frame import Frame
import numpy as np

class MyOcclusionProvider(OcclusionProvider):
    def occlusion(self, frame: Frame) -> np.ndarray:
        # Implement your occlusion algorithm here
        # Use frame.rgb, frame.depth, frame.acc, frame.gyro as needed
        
        # Return a binary mask where 1 = occluded, 0 = not occluded
        height, width = frame.height, frame.width
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill in the mask based on your algorithm
        
        return mask
```

## Data Format

### RGB and Depth Images

- RGB images are expected to be in the format `rgb_{timestamp}.png` or `rgb_{timestamp}.jpg`
- Depth images are expected to be in the format `depth_{timestamp}.png`

### IMU Data

IMU data is expected to be in a CSV file named `imu.csv` with the following columns:
- `timestamp`: Timestamp in seconds
- `accel_x`, `accel_y`, `accel_z`: Acceleration data
- `gyro_x`, `gyro_y`, `gyro_z`: Gyroscope data

### 3D Models

3D models are expected to be in OBJ or FBX format, named as `{id}.obj` or `{id}.fbx`. The framework automatically detects the file type from the file extension and uses the appropriate loader:

- OBJ files are loaded using PyWavefront
- FBX files are loaded using PyAssimp

### Scene Descriptions

Scene descriptions are expected to be in JSON format, with one file per directory. The format is:

```json
{
    "(id)": {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"x": x, "y": y, "z": z, "w": w},
    },
    ...
}
```

Where `(id)` corresponds to the ID of a 3D model file.

## Dependencies

The framework depends on the following Python packages:
- numpy
- pandas
- Pillow (PIL)
- pyrr
- PyOpenGL
- PyWavefront (for OBJ files)
- PyAssimp (for FBX files)

These can be installed using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt