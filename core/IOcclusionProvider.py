from abc import ABC, abstractmethod
import numpy as np

# Using relative import that will work after full refactoring is complete
from ..DataLoaders.Frame import Frame


class IOcclusionProvider(ABC):
    """
    Interface for occlusion providers.
    
    Defines the method that all occlusion algorithms must implement to generate
    occlusion masks from frame data.
    """
    
    @abstractmethod
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Generate an occlusion mask for a frame.
        
        Args:
            frame: Frame object containing RGB and depth images and other sensor data
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        pass