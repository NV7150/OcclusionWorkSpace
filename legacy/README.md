# Legacy Code

This directory contains legacy code from the original implementation of the MR Occlusion Framework. These files are kept for reference purposes but are no longer actively used in the refactored architecture.

## Directory Structure

- `Systems/`: Legacy system components
  - `ContentsDepthCal.py`: Original depth calculation for MR contents
  - `DataLoader.py`: Original data loading functionality
  - `ModelLoader.py`: Original model loading functionality
  - `Renderer.py`: Original rendering implementation

- `Interfaces/`: Legacy interfaces
  - `Frame.py`: Original frame data structure
  - `OcclusionProvider.py`: Original occlusion provider interface
  - `Tracker.py`: Original tracker interface

- `Occlusions/`: Legacy occlusion implementations
  - `SimpleOcclusion.py`: Original simple occlusion implementation
  - `DepthThresholdOcclusion.py`: Original depth threshold occlusion implementation

- `Tracker/`: Legacy tracker implementations
  - `ApriltagTracker.py`: Original AprilTag tracker implementation
  - `__init__.py`: Original tracker module initialization

- `Logger/`: Legacy logging functionality
  - `Logger.py`: Original logger implementation
  - `__init__.py`: Original logger module initialization

## Migration

These components have been refactored and moved to the following locations in the new architecture:

- Frame → DataLoaders/Frame.py
- DataLoader → DataLoaders/UniformedFrameLoader.py and DataLoaders/SeparatedFrameLoader.py
- OcclusionProvider → core/IOcclusionProvider.py
- Occlusions → OcclusionProviders/
- Tracker → core/ITracker.py and Trackers/
- Renderer → Rendering/Renderer.py
- Logger → Utils/Logger.py
- BaseSystem → Systems/OcclusionSystem.py and Systems/BaseSystem.py (facade)

The new architecture follows a more modular and extensible design with clear separation of concerns.