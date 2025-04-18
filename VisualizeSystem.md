# Visualization System for MR Occlusion

This document describes the Visualization System for Mixed Reality Occlusion, which provides a 3D visualization of the MR scene for debugging and analysis purposes.

## Overview

The Visualization System is designed to help debug and analyze Mixed Reality occlusion by providing a 3D visualization of the scene. It renders the following elements:

1. **Scene Model**: The 3D scan of the physical environment
2. **Reference Markers**: The markers used for tracking, with their position, normal, and tangent vectors
3. **Camera Positions**: The estimated camera positions and view frustums for each frame
4. **MR Contents**: The virtual content placed in the scene

The system provides interactive navigation with mouse and keyboard controls, allowing users to explore the scene from different viewpoints.

## Architecture

The Visualization System consists of the following main components:

### VisualizeSystem

The main entry point and coordinator for the visualization system. It orchestrates the data loading, camera pose estimation, and rendering processes.

```python
class VisualizeSystem:
    def __init__(self,
                 scene_model_path: str,
                 frames_dir: str,
                 marker_file: str,
                 camera_matrix_file: str,
                 render_obj_dir: str,
                 tag_size: float = 0.05,
                 tag_family: str = 'tag36h11',
                 log_keys: Optional[List[str]] = None):
        # Initialize components
        
    def load_data(self):
        # Load frames, models, scenes, markers
        
    def _estimate_camera_poses(self):
        # Estimate camera poses for all frames
        
    def initialize_visualization(self):
        # Initialize the visualization
        
    def run(self):
        # Run the visualization loop
        
    def set_view_mode(self, mode: str, index: Optional[int] = None):
        # Set the view mode (free, camera, marker)
```

### VisualizeModelRender

Responsible for rendering the 3D scene using OpenGL. It handles the rendering of the scene model, markers, camera positions, and MR contents.

```python
class VisualizeModelRender:
    def __init__(self):
        # Initialize renderer
        
    def initialize(self, scene_model_path: str):
        # Initialize OpenGL and load scene model
        
    def setup_scene(self, marker_positions: Dict, camera_poses: Dict, models: Dict, scenes: Dict):
        # Set up the scene with data
        
    def start_render_loop(self):
        # Start the rendering loop
        
    def set_camera_view(self, timestamp):
        # Set view to a specific camera pose
        
    def set_marker_view(self, marker_id):
        # Set view to a specific marker
        
    def set_free_view(self):
        # Set view to free navigation mode
```

## Usage

### Command Line Interface

The visualization system can be run from the command line using the following syntax:

```bash
python VisualizeExample.py
```

Or with custom parameters:

```bash
python VisualizeSystem.py --scene-model <scene_model_path> --frames-dir <frames_dir> --marker-file <marker_file> --camera-matrix <camera_matrix_file> --render-obj-dir <render_obj_dir> [--tag-size <tag_size>] [--tag-family <tag_family>] [--log-keys <log_keys>]
```

### Interactive Controls

The visualization system provides the following interactive controls:

- **Mouse Left Button**: Rotate the scene
- **Mouse Right Button**: Zoom in/out
- **Arrow Keys**: Rotate the scene
- **m**: Toggle marker visibility
- **c**: Toggle camera visibility
- **o**: Toggle MR contents visibility
- **s**: Toggle scene model visibility
- **g**: Toggle grid visibility
- **a**: Toggle axes visibility
- **f**: Reset to free view mode
- **r**: Reset view
- **q/ESC**: Exit

### View Modes

The visualization system supports three view modes:

1. **Free View**: Free navigation with mouse and keyboard
2. **Camera View**: View from a specific camera's perspective
3. **Marker View**: View from a specific marker's perspective

## Implementation Details

### Scene Model Loading

The scene model is loaded from an FBX file using PyAssimp. The model is rendered using OpenGL.

```python
def _load_scene_model(self, model_path: str):
    # Load the scene model from a file
    with pyassimp.load(model_path, processing=processing_flags) as scene:
        # Process and store the scene
```

### Marker Visualization

Markers are visualized as small cubes with arrows indicating their normal and tangent vectors.

```python
def _draw_markers(self):
    # Draw each marker as a cube with normal and tangent vectors
```

### Camera Pose Estimation

Camera poses are estimated using AprilTag tracking. The poses are represented as 4x4 transformation matrices.

```python
def _estimate_camera_poses(self):
    # Estimate camera poses for all frames using AprilTag tracking
```

### MR Content Rendering

MR contents are loaded from FBX files and rendered using OpenGL. The contents are positioned according to the scene description.

```python
def _draw_contents(self):
    # Draw MR contents based on scene description
```

## Future Improvements

The visualization system could be improved in the following ways:

1. **Performance Optimization**: Optimize the rendering for better performance with large scenes and many frames.
2. **Additional Visualization Options**: Add more visualization options, such as depth maps, occlusion masks, etc.
3. **Timeline Scrubbing**: Add a timeline scrubber to navigate through frames.
4. **Recording**: Add the ability to record the visualization as a video.
5. **Multiple View Windows**: Add support for multiple view windows to see different perspectives simultaneously.
6. **Annotation**: Add the ability to annotate the scene with text, measurements, etc.
7. **Comparison View**: Add a comparison view to compare different occlusion algorithms.
8. **Real-time Visualization**: Add support for real-time visualization of live data.
9. **VR/AR Visualization**: Add support for visualizing the scene in VR or AR.
10. **Export/Import**: Add the ability to export and import visualization settings.

## Dependencies

The visualization system depends on the following Python packages:

- numpy
- OpenGL (PyOpenGL)
- PyAssimp
- GLUT (PyOpenGL-accelerate)
- pyrr
- PIL (Pillow)
- cv2 (OpenCV)

## Conclusion

The Visualization System for MR Occlusion provides a powerful tool for debugging and analyzing Mixed Reality occlusion. It allows users to visualize the scene in 3D, explore camera poses, and understand the spatial relationships between markers, cameras, and MR contents.