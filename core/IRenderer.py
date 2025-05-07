from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional

from ..DataLoaders.Frame import Frame


class IRenderer(ABC):
    """
    Interface for renderers.
    
    Defines methods that all renderers must implement to provide a consistent
    way to render scenes across different rendering implementations.
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the renderer.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the renderer and release resources.
        """
        pass
    
    @abstractmethod
    def render_frame(self, 
                    frame: Frame, 
                    occlusion_mask: np.ndarray, 
                    models: Dict[str, Any], 
                    scene_data: Dict) -> np.ndarray:
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
        pass
    
    @abstractmethod
    def render_depth_only(self, 
                         frame: Frame, 
                         models: Dict[str, Any], 
                         scene_data: Dict) -> np.ndarray:
        """
        Render only the depth buffer of virtual objects.
        
        This is used for occlusion calculations.
        
        Args:
            frame: Frame object containing RGB and depth images
            models: Dictionary of 3D models
            scene_data: Scene description dictionary
            
        Returns:
            Depth buffer as a numpy array
        """
        pass
    
    @abstractmethod
    def save_rendered_image(self, 
                           image: np.ndarray, 
                           output_path: str) -> str:
        """
        Save a rendered image to a file.
        
        Args:
            image: Rendered image as a numpy array
            output_path: Path where the image should be saved
            
        Returns:
            Path to the saved image file
        """
        pass
    
    @abstractmethod
    def render_and_save_batch(self, 
                             frames: List[Frame], 
                             occlusion_masks: Dict[np.datetime64, np.ndarray],
                             models: Dict[str, Any], 
                             scene_data: Dict,
                             output_dir: str,
                             output_prefix: str = "frame") -> List[str]:
        """
        Render and save a batch of frames.
        
        Args:
            frames: List of Frame objects
            occlusion_masks: Dictionary mapping timestamps to occlusion masks
            models: Dictionary of 3D models
            scene_data: Scene description dictionary
            output_dir: Directory where rendered images will be saved
            output_prefix: Prefix for output filenames
            
        Returns:
            List of paths to the saved image files
        """
        pass
    
    @abstractmethod
    def set_camera_parameters(self, 
                             fov: float, 
                             aspect_ratio: float, 
                             near_plane: float, 
                             far_plane: float) -> None:
        """
        Set camera parameters for rendering.
        
        Args:
            fov: Field of view in degrees
            aspect_ratio: Aspect ratio (width / height)
            near_plane: Near clipping plane distance
            far_plane: Far clipping plane distance
        """
        pass
    
    @abstractmethod
    def set_lighting_parameters(self, 
                               ambient: List[float], 
                               diffuse: List[float], 
                               specular: List[float], 
                               light_position: List[float]) -> None:
        """
        Set lighting parameters for rendering.
        
        Args:
            ambient: Ambient light color [r, g, b]
            diffuse: Diffuse light color [r, g, b]
            specular: Specular light color [r, g, b]
            light_position: Light position [x, y, z]
        """
        pass