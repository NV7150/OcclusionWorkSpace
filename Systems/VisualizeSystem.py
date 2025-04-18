import os
import sys
import numpy as np
import argparse
from typing import Dict, List, Any, Optional, Tuple
import json
import pyrr
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2

from Systems.DataLoader import DataLoader
from Systems.ModelLoader import ModelLoader
from Systems.VisualizeModelRender import VisualizeModelRender
from Tracker.ApriltagTracker import ApriltagTracker
from Utils.MarkerPositionLoader import MarkerPositionLoader
from Logger import Logger, logger
from Interfaces.Frame import Frame


class VisualizeSystem:
    """
    VisualizeSystem is a class for visualizing MR scenes in 3D.
    It renders the scene model, reference markers, camera positions, and MR contents.
    """
    
    def __init__(self,
                 scene_model_path: str,
                 frames_dir: str,
                 marker_file: str,
                 camera_matrix_file: str,
                 render_obj_dir: str,
                 tag_size: float = 0.05,
                 tag_family: str = 'tag36h11',
                 log_keys: Optional[List[str]] = None):
        """
        Initialize the VisualizeSystem with paths to required files.
        
        Args:
            scene_model_path: Path to the 3D scan model (.fbx) of the scene
            frames_dir: Directory containing frame images for camera pose estimation
            marker_file: Path to the marker positions JSON file
            camera_matrix_file: Path to the camera matrix CSV file
            render_obj_dir: Directory containing MR content models and scene description
            tag_size: Size of AprilTag markers in meters
            tag_family: AprilTag family to use for detection
            log_keys: List of logging keys to enable
        """
        # Store parameters
        self.scene_model_path = scene_model_path
        self.frames_dir = frames_dir
        self.marker_file = marker_file
        self.camera_matrix_file = camera_matrix_file
        self.render_obj_dir = render_obj_dir
        self.tag_size = tag_size
        self.tag_family = tag_family
        
        # Configure logger
        self.logger = Logger(log_keys)
        self.logger.log(Logger.SYSTEM, "Initializing VisualizeSystem")
        
        # Initialize components
        self.data_loader = DataLoader([frames_dir])
        self.model_loader = ModelLoader([render_obj_dir])
        
        # Load camera matrix
        self.camera_matrix = self._load_camera_matrix(camera_matrix_file)
        
        # Initialize tracker
        self.tracker = ApriltagTracker(
            camera_matrix=self.camera_matrix,
            dist_coeffs=np.zeros(5),  # No distortion for simplicity
            tag_size=tag_size,
            tag_family=tag_family
        )
        
        # Initialize the tracker with marker positions
        if not self.tracker.initialize(marker_file):
            self.logger.log(Logger.ERROR, f"Failed to initialize tracker with marker positions from {marker_file}")
            raise RuntimeError(f"Failed to initialize tracker with marker positions from {marker_file}")
        
        # Initialize renderer
        self.renderer = VisualizeModelRender()
        
        # Data storage
        self.frames = {}
        self.models = {}
        self.scenes = {}
        self.marker_positions = {}
        self.camera_poses = {}
        
        # Visualization state
        self.is_running = False
        self.current_view = "free"  # "free", "camera", "marker"
        self.selected_camera = None
        self.selected_marker = None
        
    def _load_camera_matrix(self, file_path: str) -> np.ndarray:
        """
        Load camera matrix from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing the camera matrix
            
        Returns:
            3x3 camera intrinsic matrix
        """
        if not os.path.exists(file_path):
            self.logger.log(Logger.ERROR, f"Camera matrix file not found: {file_path}")
            raise FileNotFoundError(f"Camera matrix file not found: {file_path}")
        
        try:
            # Load the camera matrix from the CSV file
            matrix = np.loadtxt(file_path, delimiter=',')
            
            # Check if the loaded matrix has the correct shape
            if matrix.shape != (3, 3):
                self.logger.log(Logger.ERROR, f"Invalid camera matrix shape in {file_path}. Expected (3, 3), got {matrix.shape}")
                raise ValueError(f"Invalid camera matrix shape in {file_path}. Expected (3, 3), got {matrix.shape}")
            
            self.logger.log(Logger.SYSTEM, f"Loaded camera matrix from {file_path}")
            return matrix.astype(np.float32)
        except Exception as e:
            self.logger.log(Logger.ERROR, f"Failed to load camera matrix from {file_path}: {e}")
            raise
    
    def load_data(self):
        """
        Load all data (frames, models, scenes, markers).
        """
        # Load frames
        self.logger.log(Logger.SYSTEM, "Loading frames...")
        self.frames = self.data_loader.load_data()
        self.logger.log(Logger.SYSTEM, f"Loaded {len(self.frames)} frames")
        
        # Load models and scenes
        self.logger.log(Logger.SYSTEM, "Loading models and scenes...")
        self.models, self.scenes = self.model_loader.load_models_and_scenes()
        self.logger.log(Logger.SYSTEM, f"Loaded {len(self.models)} models and {len(self.scenes)} scenes")
        
        # Load marker positions
        self.logger.log(Logger.SYSTEM, "Loading marker positions...")
        self.marker_positions = MarkerPositionLoader.load_marker_positions(self.marker_file)
        self.logger.log(Logger.SYSTEM, f"Loaded {len(self.marker_positions)} marker positions")
        
        # Estimate camera poses for all frames
        self.logger.log(Logger.SYSTEM, "Estimating camera poses...")
        self._estimate_camera_poses()
        self.logger.log(Logger.SYSTEM, f"Estimated {len(self.camera_poses)} camera poses")
    
    def _estimate_camera_poses(self):
        """
        Estimate camera poses for all frames using AprilTag tracking.
        """
        frames_sorted = self.data_loader.get_frames_sorted()
        
        for frame in frames_sorted:
            # Track the frame to get camera pose
            camera_pose = self.tracker.track(frame)
            self.camera_poses[frame.timestamp] = camera_pose
            self.logger.log(Logger.DEBUG, f"Estimated camera pose for timestamp {frame.timestamp}")
    
    def initialize_visualization(self):
        """
        Initialize the visualization system.
        """
        self.logger.log(Logger.SYSTEM, "Initializing visualization...")
        
        # Initialize the renderer
        self.renderer.initialize(self.scene_model_path)
        
        # Set up the scene
        self.renderer.setup_scene(
            self.marker_positions,
            self.camera_poses,
            self.models,
            self.scenes
        )
        
        self.is_running = True
        self.logger.log(Logger.SYSTEM, "Visualization initialized")
    
    def run(self):
        """
        Run the visualization loop.
        """
        if not self.is_running:
            self.initialize_visualization()
        
        self.logger.log(Logger.SYSTEM, "Starting visualization loop...")
        self.renderer.start_render_loop()
    
    def set_view_mode(self, mode: str, index: Optional[int] = None):
        """
        Set the view mode for the visualization.
        
        Args:
            mode: View mode ("free", "camera", "marker")
            index: Index of the camera or marker to view from (if applicable)
        """
        self.current_view = mode
        
        if mode == "camera" and index is not None:
            self.selected_camera = index
            frames_sorted = self.data_loader.get_frames_sorted()
            if 0 <= index < len(frames_sorted):
                timestamp = frames_sorted[index].timestamp
                self.renderer.set_camera_view(timestamp)
        
        elif mode == "marker" and index is not None:
            self.selected_marker = index
            marker_ids = list(self.marker_positions.keys())
            if 0 <= index < len(marker_ids):
                marker_id = marker_ids[index]
                self.renderer.set_marker_view(marker_id)
        
        elif mode == "free":
            self.renderer.set_free_view()


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
    parser.add_argument('--log-keys', nargs='+', default=[Logger.ERROR, Logger.SYSTEM],
                        help=f'Logging keys to enable (e.g., {Logger.SYSTEM}, {Logger.DEBUG}, {Logger.ERROR})')
    
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # Configure global logger
    logger.configure(
        enabled_log_keys=args.log_keys
    )
    
    logger.log(Logger.SYSTEM, "Starting MR Scene Visualization")
    
    # Create and run the visualization system
    system = VisualizeSystem(
        scene_model_path=args.scene_model,
        frames_dir=args.frames_dir,
        marker_file=args.marker_file,
        camera_matrix_file=args.camera_matrix,
        render_obj_dir=args.render_obj_dir,
        tag_size=args.tag_size,
        tag_family=args.tag_family,
        log_keys=args.log_keys
    )
    
    system.load_data()
    system.run()
    
    logger.log(Logger.SYSTEM, "Visualization complete")


if __name__ == '__main__':
    main()