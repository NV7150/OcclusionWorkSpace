import os
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

from ..core.IFrameLoader import IFrameLoader
from ..core.IModel import IModel
from ..core.IOcclusionProvider import IOcclusionProvider
from ..core.IRenderer import IRenderer
from ..core.ITracker import ITracker
from ..core.IScene import IScene

from ..DataLoaders.Frame import Frame
from ..DataLoaders.UniformedFrameLoader import UniformedFrameLoader
from ..OcclusionProviders.DepthThresholdOcclusionProvider import DepthThresholdOcclusionProvider
from ..Models.SceneManager import SceneManager
from ..Rendering.OpenGLRenderer import OpenGLRenderer
from ..Logger.Logger import Logger


class MROcclusionSystem:
    """
    Main Mixed Reality Occlusion System class.
    
    This class coordinates all components of the MR occlusion system,
    including frame loading, tracking, rendering, and occlusion detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MR Occlusion System.
        
        Args:
            config: Optional configuration dictionary
        """
        # Initialize logger
        self._logger = Logger()
        self._logger.info("Initializing MR Occlusion System...")
        
        # Initialize default components
        self._frame_loader = None
        self._tracker = None
        self._scene_manager = None
        self._renderer = None
        self._occlusion_provider = None
        
        # Initialize properties
        self._width = 640
        self._height = 480
        self._output_directory = "Output"
        self._initialized = False
        self._running = False
        self._current_frame = None
        self._config = config or {}
        
        # Apply configuration
        self._apply_config()
    
    def _apply_config(self) -> None:
        """Apply the configuration settings."""
        if not self._config:
            return
            
        # Apply basic settings
        self._width = self._config.get("width", self._width)
        self._height = self._config.get("height", self._height)
        self._output_directory = self._config.get("output_directory", self._output_directory)
        
        # Ensure output directory exists
        os.makedirs(self._output_directory, exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Initialize the system and all its components.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self._initialized:
            self._logger.warning("System is already initialized")
            return True
            
        try:
            # Initialize renderer
            if self._renderer is None:
                self._renderer = OpenGLRenderer()
                
            init_success = self._renderer.initialize(self._width, self._height)
            if not init_success:
                self._logger.error("Failed to initialize renderer")
                return False
                
            # Initialize scene manager
            if self._scene_manager is None:
                self._scene_manager = SceneManager()
                
            # Initialize occlusion provider
            if self._occlusion_provider is None:
                self._occlusion_provider = DepthThresholdOcclusionProvider()
            
            self._initialized = True
            self._logger.info("MR Occlusion System initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Error initializing MR Occlusion System: {e}")
            return False
    
    def set_frame_loader(self, frame_loader: IFrameLoader) -> None:
        """
        Set the frame loader component.
        
        Args:
            frame_loader: Frame loader implementation
        """
        self._frame_loader = frame_loader
        self._logger.info(f"Frame loader set: {frame_loader.name}")
    
    def set_tracker(self, tracker: ITracker) -> None:
        """
        Set the tracker component.
        
        Args:
            tracker: Tracker implementation
        """
        self._tracker = tracker
        self._logger.info(f"Tracker set: {tracker.name}")
    
    def set_renderer(self, renderer: IRenderer) -> None:
        """
        Set the renderer component.
        
        Args:
            renderer: Renderer implementation
        """
        self._renderer = renderer
        self._logger.info(f"Renderer set: {renderer.__class__.__name__}")
    
    def set_occlusion_provider(self, occlusion_provider: IOcclusionProvider) -> None:
        """
        Set the occlusion provider component.
        
        Args:
            occlusion_provider: Occlusion provider implementation
        """
        self._occlusion_provider = occlusion_provider
        self._logger.info(f"Occlusion provider set: {occlusion_provider.name}")
    
    def set_scene_manager(self, scene_manager: IScene) -> None:
        """
        Set the scene manager component.
        
        Args:
            scene_manager: Scene manager implementation
        """
        self._scene_manager = scene_manager
        self._logger.info(f"Scene manager set: {scene_manager.__class__.__name__}")
    
    def load_model(self, model_path: str, model_name: Optional[str] = None) -> Optional[IModel]:
        """
        Load a 3D model and add it to the scene.
        
        Args:
            model_path: Path to the model file
            model_name: Optional name for the model
            
        Returns:
            The loaded model, or None if loading failed
        """
        if not self._scene_manager:
            self._logger.error("Cannot load model: Scene manager not initialized")
            return None
            
        try:
            # Determine model name if not provided
            if model_name is None:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # Load the model
            model = self._scene_manager.load_model(model_path, model_name)
            if model:
                self._logger.info(f"Model '{model_name}' loaded successfully from {model_path}")
                return model
            else:
                self._logger.error(f"Failed to load model from {model_path}")
                return None
                
        except Exception as e:
            self._logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def get_models(self) -> List[IModel]:
        """
        Get all models in the scene.
        
        Returns:
            List of models
        """
        if not self._scene_manager:
            return []
            
        return self._scene_manager.get_models()
    
    def process_frame(self, frame: Optional[Frame] = None) -> Optional[np.ndarray]:
        """
        Process a single frame.
        
        Args:
            frame: Optional frame to process. If None, the next frame will be loaded from the frame loader.
            
        Returns:
            Processed image or None if processing failed
        """
        if not self._initialized:
            self._logger.error("System not initialized")
            return None
            
        try:
            # Get frame from loader if not provided
            if frame is None:
                if not self._frame_loader:
                    self._logger.error("No frame loader set and no frame provided")
                    return None
                    
                frame = self._frame_loader.get_next_frame()
                
            if frame is None:
                self._logger.error("Failed to get frame")
                return None
                
            # Store current frame
            self._current_frame = frame
            
            # Update tracker if available
            if self._tracker:
                self._tracker.update(frame)
                
                # Update model poses based on tracking
                if self._scene_manager:
                    for model in self._scene_manager.get_models():
                        tracked_name = model.get_name()
                        if self._tracker.is_target_visible(tracked_name):
                            pose = self._tracker.get_target_pose(tracked_name)
                            model.set_transform(pose)
            
            # Generate occlusion mask
            occlusion_mask = np.zeros((frame.height, frame.width), dtype=bool)
            if self._occlusion_provider:
                # Get the depth buffer from renderer if available
                virtual_depth = None
                if self._renderer:
                    # TODO: Implement depth buffer retrieval in realistic scenario
                    # For now, we'll use the occlusion provider without virtual depth
                    pass
                    
                occlusion_mask = self._occlusion_provider.generate_occlusion_mask(frame, virtual_depth)
            
            # Render the scene
            if self._renderer and self._scene_manager:
                # Get models to render
                models = self._scene_manager.get_models()
                
                # Set camera parameters from frame if available
                if hasattr(frame, 'camera_matrix') and hasattr(frame, 'projection_matrix'):
                    self._renderer.set_camera(frame.camera_matrix, frame.projection_matrix)
                
                # Render the frame
                result = self._renderer.render_frame(
                    frame.rgb_image,
                    frame.depth_image,
                    models,
                    occlusion_mask
                )
                
                return result
            
            # If no renderer or no models, return the original RGB image
            return frame.rgb_image
            
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return None
    
    def start_processing(self) -> bool:
        """
        Start processing frames continuously.
        
        Returns:
            True if processing started successfully, False otherwise
        """
        if not self._initialized:
            self._logger.error("System not initialized")
            return False
            
        if not self._frame_loader:
            self._logger.error("No frame loader set")
            return False
            
        self._running = True
        self._logger.info("Starting continuous frame processing...")
        
        try:
            frame_count = 0
            start_time = time.time()
            
            while self._running:
                # Process next frame
                frame = self._frame_loader.get_next_frame()
                if frame is None:
                    self._logger.info("End of frames reached")
                    break
                    
                result = self.process_frame(frame)
                if result is None:
                    self._logger.error("Failed to process frame")
                    continue
                    
                # Save result if output directory is set
                if self._output_directory:
                    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
                    filename = os.path.join(
                        self._output_directory, 
                        f"frame_{frame_count}_{timestamp}.png"
                    )
                    self._renderer.save_frame_to_image(result, filename)
                
                frame_count += 1
                
                # Print progress
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    self._logger.info(f"Processed {frame_count} frames ({fps:.2f} FPS)")
            
            self._logger.info(f"Finished processing {frame_count} frames")
            return True
            
        except Exception as e:
            self._logger.error(f"Error in continuous processing: {e}")
            return False
        finally:
            self._running = False
    
    def stop_processing(self) -> None:
        """Stop continuous frame processing."""
        self._running = False
        self._logger.info("Stopping frame processing")
    
    def cleanup(self) -> None:
        """Clean up resources used by the system."""
        self._logger.info("Cleaning up MR Occlusion System...")
        
        if self._renderer:
            self._renderer.cleanup()
            
        self._initialized = False
        self._running = False
        
        self._logger.info("Cleanup complete")