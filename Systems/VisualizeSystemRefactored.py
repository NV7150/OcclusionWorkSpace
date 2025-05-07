import os
import sys
import numpy as np
import argparse
from typing import Dict, List, Any, Optional, Tuple
import json
import pyrr

from ..core.IFrameLoader import IFrameLoader
from ..core.IScene import IScene
from ..core.IRenderer import IRenderer
from ..core.ITracker import ITracker
from ..DataLoaders.Frame import Frame
from ..Rendering.Renderer import Renderer
from ..Rendering.Camera import Camera
from ..Rendering.ShaderManager import ShaderManager
from ..Rendering.BufferManager import BufferManager
from ..Rendering.TextureManager import TextureManager
from ..Rendering.Primitives import Primitives
from ..Trackers.ApriltagTracker import ApriltagTracker
from ..Utils.MarkerPositionLoader import MarkerPositionLoader
from ..Utils.TransformUtils import TransformUtils
from ..Logger.Logger import Logger


class VisualizeSystem:
    """
    VisualizeSystem is a class for visualizing MR scenes in 3D.
    It renders the scene model, reference markers, camera positions, and MR contents.
    
    This is a refactored version that uses the new architecture with cleaner
    separation of concerns and better modularity.
    """
    
    def __init__(self,
                 frame_loader: IFrameLoader,
                 scene_manager: IScene,
                 renderer: IRenderer,
                 tracker: ITracker,
                 scene_model_path: str,
                 marker_file: str,
                 logger: Optional[Logger] = None):
        """
        Initialize the VisualizeSystem with required components.
        
        Args:
            frame_loader: Component for loading frame data
            scene_manager: Component for managing scene and models
            renderer: Component for rendering
            tracker: Component for tracking and camera pose estimation
            scene_model_path: Path to the 3D scan model (.fbx) of the scene
            marker_file: Path to the marker positions JSON file
            logger: Optional logger instance
        """
        # Store components
        self._frame_loader = frame_loader
        self._scene_manager = scene_manager
        self._renderer = renderer
        self._tracker = tracker
        self._scene_model_path = scene_model_path
        self._marker_file = marker_file
        self._logger = logger
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, "Initializing VisualizeSystem")
        
        # Data storage
        self._frames = {}
        self._marker_positions = {}
        self._camera_poses = {}
        
        # Visualization state
        self._is_running = False
        self._current_view = "free"  # "free", "camera", "marker"
        self._selected_camera = None
        self._selected_marker = None
        
        # Load marker positions
        self._load_marker_positions()
    
    def _load_marker_positions(self) -> None:
        """
        Load marker positions from the marker file.
        """
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Loading marker positions from {self._marker_file}")
        
        try:
            self._marker_positions = MarkerPositionLoader.load_marker_positions(self._marker_file)
            
            if self._logger:
                self._logger.log(Logger.SYSTEM, f"Loaded {len(self._marker_positions)} marker positions")
            
            # Initialize the tracker with marker positions
            self._tracker.set_reference_markers(self._marker_positions)
        except Exception as e:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Failed to load marker positions: {e}")
            raise
    
    def load_data(self) -> None:
        """
        Load all data (frames, models, scenes) and estimate camera poses.
        """
        # Load frames
        if self._logger:
            self._logger.log(Logger.SYSTEM, "Loading frames...")
        
        self._frames = self._frame_loader.load_data()
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Loaded {len(self._frames)} frames")
        
        # Estimate camera poses for all frames
        if self._logger:
            self._logger.log(Logger.SYSTEM, "Estimating camera poses...")
        
        self._estimate_camera_poses()
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Estimated {len(self._camera_poses)} camera poses")
    
    def _estimate_camera_poses(self) -> None:
        """
        Estimate camera poses for all frames using the tracker.
        """
        frames_sorted = self._frame_loader.get_frames_sorted()
        
        for frame in frames_sorted:
            # Process the frame to get camera pose
            camera_pose = self._tracker.process_frame(frame)
            
            if camera_pose is not None:
                self._camera_poses[frame.timestamp] = camera_pose
                
                if self._logger:
                    self._logger.log(Logger.DEBUG, f"Estimated camera pose for timestamp {frame.timestamp}")
            else:
                if self._logger:
                    self._logger.log(Logger.WARNING, f"Failed to estimate camera pose for timestamp {frame.timestamp}")
    
    def initialize_visualization(self) -> None:
        """
        Initialize the visualization system.
        """
        if self._logger:
            self._logger.log(Logger.SYSTEM, "Initializing visualization...")
        
        # Initialize the renderer
        self._renderer.initialize()
        
        # Set up the scene
        models = self._scene_manager.get_all_models()
        scenes = self._scene_manager.get_all_scenes()
        
        # Setup the visualization scene
        self._setup_visualization_scene(models, scenes)
        
        self._is_running = True
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, "Visualization initialized")
    
    def _setup_visualization_scene(self, models: Dict[str, Any], scenes: Dict[str, Dict]) -> None:
        """
        Set up the visualization scene with models, scenes, markers, and camera poses.
        
        Args:
            models: Dictionary of models
            scenes: Dictionary of scenes
        """
        # This method would be implemented in a concrete renderer
        # For now, we'll just pass the data to the renderer
        if hasattr(self._renderer, 'setup_visualization_scene'):
            self._renderer.setup_visualization_scene(
                self._scene_model_path,
                self._marker_positions,
                self._camera_poses,
                models,
                scenes
            )
        else:
            if self._logger:
                self._logger.log(Logger.WARNING, "Renderer does not support visualization scene setup")
    
    def run(self) -> None:
        """
        Run the visualization loop.
        """
        if not self._is_running:
            self.initialize_visualization()
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, "Starting visualization loop...")
        
        # Start the rendering loop
        if hasattr(self._renderer, 'start_visualization_loop'):
            self._renderer.start_visualization_loop()
        else:
            if self._logger:
                self._logger.log(Logger.WARNING, "Renderer does not support visualization loop")
    
    def set_view_mode(self, mode: str, index: Optional[int] = None) -> None:
        """
        Set the view mode for the visualization.
        
        Args:
            mode: View mode ("free", "camera", "marker")
            index: Index of the camera or marker to view from (if applicable)
        """
        self._current_view = mode
        
        if not hasattr(self._renderer, 'set_view_mode'):
            if self._logger:
                self._logger.log(Logger.WARNING, "Renderer does not support view mode setting")
            return
        
        if mode == "camera" and index is not None:
            self._selected_camera = index
            frames_sorted = self._frame_loader.get_frames_sorted()
            if 0 <= index < len(frames_sorted):
                timestamp = frames_sorted[index].timestamp
                self._renderer.set_view_mode("camera", timestamp)
        
        elif mode == "marker" and index is not None:
            self._selected_marker = index
            marker_ids = list(self._marker_positions.keys())
            if 0 <= index < len(marker_ids):
                marker_id = marker_ids[index]
                self._renderer.set_view_mode("marker", marker_id)
        
        elif mode == "free":
            self._renderer.set_view_mode("free")


def create_visualization_system(
    scene_model_path: str,
    frames_dir: str,
    marker_file: str,
    camera_matrix_file: str,
    render_obj_dir: str,
    tag_size: float = 0.05,
    tag_family: str = 'tag36h11',
    log_keys: Optional[List[str]] = None
) -> VisualizeSystem:
    """
    Create a VisualizeSystem instance with all required components.
    
    Args:
        scene_model_path: Path to the 3D scan model (.fbx) of the scene
        frames_dir: Directory containing frame images for camera pose estimation
        marker_file: Path to the marker positions JSON file
        camera_matrix_file: Path to the camera matrix CSV file
        render_obj_dir: Directory containing MR content models and scene description
        tag_size: Size of AprilTag markers in meters
        tag_family: AprilTag family to use for detection
        log_keys: List of logging keys to enable
        
    Returns:
        Configured VisualizeSystem instance
    """
    # Create logger
    logger = Logger(log_keys)
    
    # Load camera matrix
    camera_matrix = _load_camera_matrix(camera_matrix_file, logger)
    
    # Create components
    from ..DataLoaders.UniformedFrameLoader import UniformedFrameLoader
    from ..Models.SceneManager import SceneManager
    
    # Create frame loader
    frame_loader = UniformedFrameLoader([frames_dir], logger=logger)
    
    # Create scene manager
    scene_manager = SceneManager([render_obj_dir], logger=logger)
    
    # Create tracker
    tracker = ApriltagTracker(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),  # No distortion for simplicity
        tag_size=tag_size,
        tag_family=tag_family,
        logger=logger
    )
    
    # Create renderer components
    shader_manager = ShaderManager(logger=logger)
    buffer_manager = BufferManager(logger=logger)
    texture_manager = TextureManager(logger=logger)
    primitives = Primitives(buffer_manager, shader_manager, logger=logger)
    
    # Create visualization renderer
    from ..Rendering.VisualizationRenderer import VisualizationRenderer
    renderer = VisualizationRenderer(
        shader_manager=shader_manager,
        buffer_manager=buffer_manager,
        texture_manager=texture_manager,
        primitives=primitives,
        logger=logger
    )
    
    # Create and return the visualization system
    return VisualizeSystem(
        frame_loader=frame_loader,
        scene_manager=scene_manager,
        renderer=renderer,
        tracker=tracker,
        scene_model_path=scene_model_path,
        marker_file=marker_file,
        logger=logger
    )


def _load_camera_matrix(file_path: str, logger: Optional[Logger] = None) -> np.ndarray:
    """
    Load camera matrix from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing the camera matrix
        logger: Optional logger instance
        
    Returns:
        3x3 camera intrinsic matrix
    """
    if not os.path.exists(file_path):
        if logger:
            logger.log(Logger.ERROR, f"Camera matrix file not found: {file_path}")
        raise FileNotFoundError(f"Camera matrix file not found: {file_path}")
    
    try:
        # Load the camera matrix from the CSV file
        matrix = np.loadtxt(file_path, delimiter=',')
        
        # Check if the loaded matrix has the correct shape
        if matrix.shape != (3, 3):
            if logger:
                logger.log(Logger.ERROR, f"Invalid camera matrix shape in {file_path}. Expected (3, 3), got {matrix.shape}")
            raise ValueError(f"Invalid camera matrix shape in {file_path}. Expected (3, 3), got {matrix.shape}")
        
        if logger:
            logger.log(Logger.SYSTEM, f"Loaded camera matrix from {file_path}")
        return matrix.astype(np.float32)
    except Exception as e:
        if logger:
            logger.log(Logger.ERROR, f"Failed to load camera matrix from {file_path}: {e}")
        raise


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='MR Scene Visualization')
    
    parser.add_argument('--scene-model', required=True,
                        help='Path to the 3D scan model (.fbx) of the scene')
    parser.add_argument('--frames-dir', required=True,
                        help='Directory containing frame images for camera pose estimation')
    parser.add_argument('--marker-file', required=True,
                        help='Path to the marker positions JSON file')
    parser.add_argument('--camera-matrix', required=True,
                        help='Path to the camera matrix CSV file')
    parser.add_argument('--render-obj-dir', required=True,
                        help='Directory containing MR content models and scene description')
    parser.add_argument('--tag-size', type=float, default=0.05,
                        help='Size of AprilTag markers in meters (default: 0.05)')
    parser.add_argument('--tag-family', default='tag36h11',
                        help='AprilTag family to use for detection (default: tag36h11)')
    
    # Logging options
    parser.add_argument('--log-keys', nargs='+', default=['ERROR', 'SYSTEM'],
                        help='Logging keys to enable (e.g., SYSTEM, DEBUG, ERROR)')
    
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # Create the visualization system
    system = create_visualization_system(
        scene_model_path=args.scene_model,
        frames_dir=args.frames_dir,
        marker_file=args.marker_file,
        camera_matrix_file=args.camera_matrix,
        render_obj_dir=args.render_obj_dir,
        tag_size=args.tag_size,
        tag_family=args.tag_family,
        log_keys=args.log_keys
    )
    
    # Load data and run the visualization
    system.load_data()
    system.run()


if __name__ == '__main__':
    main()