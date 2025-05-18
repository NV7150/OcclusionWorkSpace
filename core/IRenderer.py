from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

class IRenderer(ABC):
    """
    Interface for renderers.
    
    This interface defines the contract for classes that render 3D scenes,
    including models, cameras, and lighting.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the renderer.
        
        This method should set up any necessary resources, such as OpenGL contexts,
        shaders, and buffers.
        """
        pass
    
    @abstractmethod
    def render_frame(self, frame, models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        """
        Render a frame with the given models and scene data.
        
        Args:
            frame: Frame object containing RGB and depth images
            models: Dictionary mapping model IDs to model objects
            scene_data: Dictionary containing scene description
            
        Returns:
            Rendered image as a numpy array
        """
        pass
    
    @abstractmethod
    def render_depth(self, models: Dict[str, Any], scene_data: Dict, camera_matrix: np.ndarray) -> np.ndarray:
        """
        Render a depth map for the given models and scene data.
        
        Args:
            models: Dictionary mapping model IDs to model objects
            scene_data: Dictionary containing scene description
            camera_matrix: Camera matrix for the view
            
        Returns:
            Depth map as a numpy array
        """
        pass
    
    @abstractmethod
    def set_camera(self, position: Tuple[float, float, float], 
                  target: Tuple[float, float, float], 
                  up: Tuple[float, float, float]) -> None:
        """
        Set the camera position, target, and up vector.
        
        Args:
            position: Camera position as (x, y, z)
            target: Camera target as (x, y, z)
            up: Camera up vector as (x, y, z)
        """
        pass
    
    @abstractmethod
    def set_projection(self, fov: float, aspect: float, near: float, far: float) -> None:
        """
        Set the projection parameters.
        
        Args:
            fov: Field of view in degrees
            aspect: Aspect ratio (width / height)
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources used by the renderer.
        
        This method should release any resources, such as OpenGL contexts,
        shaders, and buffers.
        """
        pass