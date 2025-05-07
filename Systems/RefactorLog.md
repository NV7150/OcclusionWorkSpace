# Refactoring Proposal for Systems Directory

## Current Code Architecture

The Systems directory currently contains the following classes:

1.  **BaseSystem** - Main execution framework, coordinates data loading, occlusion mask generation, and rendering
2.  **DataLoader** - Responsible for loading RGB images, depth images, and IMU data
3.  **ModelLoader** - Responsible for loading 3D models and scene descriptions
4.  **Renderer** - Renders MR scenes based on occlusion masks and scene descriptions
5.  **ContentsDepthCal** - Calculates depth maps of MR content
6.  **VisualizeSystem** - System for visualizing MR scenes in 3D
7.  **VisualizeModelRender** - Modern OpenGL-based rendering system for visualization

### Dependencies

```
BaseSystem
 ├── DataLoader
 ├── ModelLoader
 ├── Renderer
 └── ContentsDepthCal
      └── Renderer

VisualizeSystem
 ├── DataLoader
 ├── ModelLoader
 └── VisualizeModelRender
```

### Issues

1.  **Duplicate OpenGL Implementation** - Redundant initialization and usage of OpenGL between `Renderer` and `VisualizeModelRender`
2.  **Mixed Rendering Code** - Similar rendering code exists in multiple places: `Renderer`, `ContentsDepthCal`, and `VisualizeModelRender`
3.  **Duplicate Transform Processing** - Similar transformation code (position, rotation, scale) in multiple classes
4.  **Different OpenGL Approaches** - Mix of legacy OpenGL (immediate mode) and modern OpenGL (shader-based)
5.  **Duplicate Model Loading** - Model loading logic implemented in multiple places
6.  **Unclear Module Responsibilities** - Some classes have ambiguous responsibility boundaries

## Refactoring Proposal

### 1. Integrated Rendering Subsystem

Reconstruct `Renderer` and `VisualizeModelRender` into a single integrated rendering system.

```
Rendering/
 ├── OpenGLRenderer.py         - OpenGL implementation (modern approach), extends Renderer
 ├── ShaderManager.py          - Shader program management
 ├── GeometryPrimitives.py     - Basic geometric shapes (cube, grid, arrow, etc.)
 ├── TextureManager.py         - Texture loading and management
 └── ModelRenderer.py          - 3D model rendering logic
```

### 2. Improved Model Management

Integrate model loading and management into a consistent interface.

```
Models/
 ├── ModelLoader.py            - Integrated model loader interface
 ├── ObjModelLoader.py         - Loader for OBJ files, extends ModelLoader
 ├── FbxModelLoader.py         - Loader for FBX files, extends ModelLoader
 ├── Model.py                  - Unified model representation
 ├── SceneManager.py           - Scene composition management
 └── TransformUtils.py         - Common transformation utilities
```

### 3. Improved System Architecture

```
Systems/
 ├── BaseSystem.py             - Facade for Occlusion System and VisualizeSystem
 ├── OcclusionSystem.py        - Current Base System
 ├── VisualizeSystem.py        - Visualization system (using new rendering system)
 └── OcclusionProcessor.py     - Occlusion processing (including ContentsDepthCal responsibilities)
```

### 4. DataLoader Architecture

```
DataLoader/
 ├── Frame.py                  - Current Frame.py in Interfaces
 ├── UniformedFrameLoader.py   - Current Frame loader (adopt before DepthIMUData2)
 ├── SeparatedFrameLoader.py   - New filesystem Frame loader (adopt after DepthIMUData3)
 └── JsonBasedSceneLoader.py   - Current Scene Loader
```

### 5. Interface Clarification (core/)

```
core/
 ├── IRenderer.py              - Abstract interface for rendering system
 ├── IModel.py                 - Abstract interface for model management
 ├── IFrameLoader.py           - The Frame loader framework
 ├── ISceneLoader.py           - Scene loader framework
 ├── ITracker.py               - Abstraction of Tracker (current Tracker.py)
 ├── IOcclusionProvider.py     - Abstraction of OcclusionProvider (current OcclusionProvider.py)
 └── IScene.py                 - Abstraction of MR Scene
```

## Specific Improvements

### 1. OpenGL Initialization Consolidation

Consolidate OpenGL initialization in the `OpenGLRenderer` class, supporting both legacy and modern rendering paths:

```python
class OpenGLRenderer:
    def __init__(self, use_modern=True):
        self.use_modern = use_modern
        self._initialize_opengl()
        
    def _initialize_opengl(self):
        # Common initialization
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        
        if self.use_modern:
            # Modern OpenGL initialization (shader-based)
            # ...
        else:
            # Legacy OpenGL initialization
            # ...

```

### 2. Transform Utility Integration

Consolidate position, rotation, and scale transformation processing:

```python
class TransformUtils:
    @staticmethod
    def create_transform_matrix(position, rotation, scale=None):
        """
        Create a combined transformation matrix from position, rotation and scale.
        
        Args:
            position: Dict with x, y, z position values
            rotation: Dict with x, y, z, w quaternion or x, y, z Euler angles
            scale: Optional Dict with x, y, z scale values
            
        Returns:
            4x4 transformation matrix as numpy array
        """
        # Implementation...
        
    @staticmethod
    def quaternion_from_euler(euler_angles):
        """Convert Euler angles to quaternion"""
        # Implementation...
        
    @staticmethod
    def euler_from_quaternion(quaternion):
        """Convert quaternion to Euler angles"""
        # Implementation...
```

### 3. Model Loading Integration

Integrate model loading logic:

```python
class ModelLoader:
    def __init__(self):
        self.loaders = {
            '.obj': ObjModelLoader(),
            '.fbx': FbxModelLoader()
        }
        self.model_cache = {}
        
    def load_model(self, file_path):
        """Load a 3D model with appropriate loader based on extension"""
        if file_path in self.model_cache:
            return self.model_cache[file_path]
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.loaders:
            raise ValueError(f"Unsupported model format: {ext}")
            
        model = self.loaders[ext].load(file_path)
        self.model_cache[file_path] = model
        return model
```

### 4. Data Pipeline Improvement

Improve parallel processing and memory efficiency:

```python
class DataPipeline: # This will replace the current DataLoader
    def __init__(self, data_dirs, batch_size=1, preload=False):
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.preload = preload
        self._frames = {} # Consider using a more memory-efficient structure if preloading large datasets
        
    def load_data(self, on_progress=None):
        """Load data with optional parallel processing and progress reporting"""
        # Implementation with parallel processing option (e.g., using concurrent.futures)
        # Call on_progress callback if provided
        
    def get_frame_batch(self, start_idx, count=None):
        """Get a batch of frames for processing.
           Potentially load on demand if not preloaded."""
        # Implementation...

    def __iter__(self):
        # Allow iteration over frames or batches
        pass

    def __len__(self):
        # Return total number of frames
        pass
```

### 5. Shader Management Improvement

```python
class ShaderManager:
    def __init__(self):
        self.shaders = {} # Cache for compiled shader programs
        
    def load_shader(self, name, vertex_source_path, fragment_source_path):
        """Compile and store a shader program from file paths"""
        # Read shader source from files
        # Implementation...
        
    def use_shader(self, name):
        """Activate a specific shader program"""
        # Implementation...
        
    def set_uniform(self, shader_name, uniform_name, value):
        """Set a uniform value for a specific shader"""
        # Implementation...
```

## Implementation Steps

1.  Create common interfaces (`core/`) and utility classes (`TransformUtils`, etc.) first.
2.  Build the new rendering subsystem (`Rendering/` directory), focusing on `OpenGLRenderer` and `ShaderManager`.
3.  Implement the model management system (`Models/` directory), including `ModelLoader` and `Model` representation.
4.  Refactor `DataLoader` into the new `DataPipeline` structure with improved loading strategies.
5.  Update `OcclusionSystem` (formerly `BaseSystem`) and `VisualizeSystem` to use the new components and interfaces.
6.  Develop comprehensive unit tests for each module to ensure the robustness of the new architecture.

## Expected Benefits

1.  **Code Duplication Reduction** - Achieved through common utilities, interfaces, and consolidated rendering/model loading.
2.  **Improved Maintainability** - Clear separation of responsibilities and modular design.
3.  **Enhanced Extensibility** - Easier addition of new renderers, model formats, or data sources.
4.  **Potential Performance Improvement** - Unified and optimized use of modern OpenGL; efficient data handling.
5.  **Improved Testability** - Interface-based design and modularity facilitate unit testing.

## Summary

This refactoring proposal aims to significantly improve the current `Systems` directory by reducing code duplication, enhancing separation of concerns, and establishing a more modular and extensible architecture. By adhering to modern software design principles and clearly defined interfaces, this refactoring will create a more robust foundation for future development and feature enhancements within the MR occlusion framework.