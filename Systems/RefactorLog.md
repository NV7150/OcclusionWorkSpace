# Refactoring Log

This document summarizes the refactoring work done on the MR Occlusion Framework.

## Refactoring Goals

1. Improve modularity and separation of concerns
2. Create clear interfaces for each component
3. Make the system more extensible and maintainable
4. Modernize the rendering pipeline
5. Improve code organization and readability

## Refactoring Steps

### 1. Core Interfaces

Created abstract base classes in the `core/` directory:
- `IFrameLoader.py`: Interface for loading frame data
- `IOcclusionProvider.py`: Interface for occlusion detection
- `IModel.py`: Interface for 3D models
- `IRenderer.py`: Interface for rendering
- `IScene.py`: Interface for scene management
- `ITracker.py`: Interface for tracking

### 2. Utilities

- Developed `Utils/TransformUtils.py` for centralized transformation operations
- Moved and updated `Utils/Logger.py` from the original Logger module

### 3. Model Subsystem

- Implemented Model, Mesh, Material, and Texture classes
- Created asset loaders (BaseAssetLoader, FbxLoader, ObjLoader)
- Implemented scene management (BaseSceneLoader, JsonSceneLoader, SceneManager)

### 4. DataLoaders Subsystem

- Moved and enhanced Frame class
- Created BaseFrameLoader abstract class
- Implemented UniformedFrameLoader and SeparatedFrameLoader

### 5. Rendering Subsystem

- Implemented modern OpenGL rendering with ShaderManager, TextureManager, BufferManager
- Created Framebuffer, Camera, and Primitives classes
- Implemented the main Renderer class

### 6. Trackers Subsystem

- Moved and updated ApriltagTracker to implement the ITracker interface

### 7. OcclusionProviders Subsystem

- Moved and updated occlusion providers to implement the IOcclusionProvider interface
- Added new HybridOcclusionProvider class

### 8. Systems Subsystem

- Implemented OcclusionProcessor to handle occlusion logic
- Refactored BaseSystem into OcclusionSystem using the new architecture
- Updated BaseSystem to be a facade for OcclusionSystem and VisualizeSystem

### 9. Legacy Code Management

- Moved legacy code to the `legacy/` directory for reference
- Created documentation explaining the migration path

## Architecture Changes

### Before

```
- Systems/
  - BaseSystem.py
  - DataLoader.py
  - ModelLoader.py
  - Renderer.py
  - ContentsDepthCal.py
- Interfaces/
  - Frame.py
  - OcclusionProvider.py
  - Tracker.py
- Occlusions/
  - SimpleOcclusion.py
  - DepthThresholdOcclusion.py
- Tracker/
  - ApriltagTracker.py
- Logger/
  - Logger.py
```

### After

```
- core/
  - IFrameLoader.py
  - IOcclusionProvider.py
  - IModel.py
  - IRenderer.py
  - IScene.py
  - ITracker.py
- DataLoaders/
  - Frame.py
  - BaseFrameLoader.py
  - UniformedFrameLoader.py
  - SeparatedFrameLoader.py
- Models/
  - Model.py
  - Mesh.py
  - Material.py
  - Texture.py
  - BaseAssetLoader.py
  - FbxLoader.py
  - ObjLoader.py
  - BaseSceneLoader.py
  - JsonSceneLoader.py
  - SceneManager.py
- Rendering/
  - Renderer.py
  - ShaderManager.py
  - TextureManager.py
  - BufferManager.py
  - Framebuffer.py
  - Camera.py
  - Primitives.py
- OcclusionProviders/
  - SimpleOcclusionProvider.py
  - DepthThresholdOcclusionProvider.py
- Trackers/
  - ApriltagTracker.py
- Systems/
  - BaseSystem.py (facade)
  - OcclusionSystem.py
  - OcclusionProcessor.py
- Utils/
  - TransformUtils.py
  - Logger.py
  - MarkerPositionLoader.py
- legacy/
  - (Legacy code for reference)
```

## Benefits of the New Architecture

1. **Modularity**: Each component has a well-defined responsibility and interface
2. **Extensibility**: Easy to add new implementations of each interface
3. **Maintainability**: Clear separation of concerns makes the code easier to understand and maintain
4. **Testability**: Interfaces make it easier to test components in isolation
5. **Flexibility**: Different implementations can be swapped out without affecting the rest of the system

## Future Work

1. Implement VisualizeSystem.py
2. Add more occlusion providers
3. Improve rendering performance
4. Add more model formats
5. Enhance documentation