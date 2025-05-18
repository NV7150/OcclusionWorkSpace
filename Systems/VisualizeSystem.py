import os
import sys
import numpy as np
import argparse
from typing import Dict, List, Any, Optional, Tuple
import json
import pyrr
import cv2

# Import only core interfaces
from core.IFrameLoader import IFrameLoader
from core.IScene import IScene
from core.ITracker import ITracker
from core.IRenderer import IRenderer
from core.IFrame import IFrame

# Import utilities
from Utils.Logger import logger


class VisualizeSystem:
    """
    VisualizeSystem is a class for visualizing MR scenes in 3D.
    It renders the scene model, reference markers, camera positions, and MR contents.
    """
    
    def __init__(self,
                 frame_loader: IFrameLoader,
                 scene_manager: IScene,
                 tracker: ITracker,
                 renderer: IRenderer,
                 scene_model_path: str,
                 log_keys: Optional[List[str]] = None):
        """
        Initialize the VisualizeSystem with required components.
        
        Args:
            frame_loader: Frame loader for loading sensor data
            scene_manager: Scene manager for managing 3D models and scenes
            tracker: Tracker for camera pose estimation
            renderer: Renderer for visualization
            scene_model_path: Path to the 3D scan model of the scene
            log_keys: List of logging keys to enable
        """
        # Store parameters
        self.scene_model_path = scene_model_path
        
        # Configure logger
        logger.configure(enabled_log_keys=log_keys)
        logger.log(logger.SYSTEM, "Initializing VisualizeSystem")
        
        # Store injected components
        self.data_loader = frame_loader
        self.scene_manager = scene_manager
        self.tracker = tracker
        self.renderer = renderer
        
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
        
    
    def load_data(self):
        """
        Load all data (frames, models, scenes, markers).
        """
        # Load frames
        logger.log(logger.SYSTEM, "Loading frames...")
        self.frames = self.data_loader.load_data()
        logger.log(logger.SYSTEM, f"Loaded {len(self.frames)} frames")
        
        # Load models and scenes
        logger.log(logger.SYSTEM, "Loading models and scenes...")
        self.models = self.scene_manager.models
        self.scenes = self.scene_manager.get_scene_data()
        logger.log(logger.SYSTEM, f"Loaded {len(self.models)} models and {len(self.scenes.get('objects', {}))} scene objects")
        
        # Estimate camera poses for all frames
        logger.log(logger.SYSTEM, "Estimating camera poses...")
        self._estimate_camera_poses()
        logger.log(logger.SYSTEM, f"Estimated {len(self.camera_poses)} camera poses")
    
    def _estimate_camera_poses(self):
        """
        Estimate camera poses for all frames using the tracker.
        """
        frames_sorted = self.data_loader.get_frames_sorted()
        
        logger.log(logger.SYSTEM, f"Estimating camera poses for {len(frames_sorted)} frames")
        
        for frame in frames_sorted:
            # Track the frame to get camera pose
            camera_pose = self.tracker.track(frame)
            if camera_pose is not None:
                self.camera_poses[frame.timestamp] = camera_pose
                logger.log(logger.DEBUG, f"Estimated camera pose for timestamp {frame.timestamp}")
            else:
                logger.log(logger.WARNING, f"Could not estimate camera pose for timestamp {frame.timestamp}")
    
    def initialize_visualization(self):
        """
        Initialize the visualization system.
        """
        logger.log(logger.SYSTEM, "Initializing visualization...")
        
        # Initialize the renderer
        self.renderer.initialize()
        if self.scene_model_path and os.path.exists(self.scene_model_path):
            self.renderer.load_scene(self.scene_model_path)
        elif self.scene_model_path:
            logger.log(logger.ERROR, f"VisualizeSystem: scene_model_path provided but file not found: {self.scene_model_path}")
        else:
            logger.log(logger.WARNING, "VisualizeSystem: No scene_model_path provided. Main scene will be empty.")
        
        # Set up the scene
        self.renderer.setup_scene(
            self.tracker.marker_positions,
            self.camera_poses,
            self.models,
            self.scenes
        )
        
        self.is_running = True
        logger.log(logger.SYSTEM, "Visualization initialized")
    
    def run(self):
        """
        Run the visualization loop.
        """
        if not self.is_running:
            self.initialize_visualization()
        
        logger.log(logger.SYSTEM, "Starting visualization loop...")
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

