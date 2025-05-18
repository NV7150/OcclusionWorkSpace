from abc import ABC, abstractmethod
import numpy as np

class IOcclusionProvider(ABC):
    """
    Interface for occlusion providers.
    
    This interface defines the contract for classes that generate occlusion masks
    from frame data. Occlusion providers implement different algorithms for
    determining which parts of the virtual content should be occluded by real-world
    objects.
    """
    
    @abstractmethod
    def occlusion(self, frame) -> np.ndarray:
        """
        Generate an occlusion mask for a frame.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        pass