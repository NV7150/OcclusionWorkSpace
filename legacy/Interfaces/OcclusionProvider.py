import numpy as np
from abc import ABCMeta, abstractmethod
from . import Frame

## Provides occlusion mask (interface for accurate AI code generation)
class OcclusionProvider(metaclass=ABCMeta):
    @abstractmethod
    def occlusion(self, frame: Frame, mr_depth: np.ndarray) -> np.ndarray:
        """ 
        Get a oclusion mask of the mr scene using the current camera frame and the mr scene depth map.
        
        Args:
            frame(Frame): Frame object containing the current camera frame data
            mr_depth(np.ndarray): numpy array representing the depth map of the mr scene trying to render
        Returns:
            np.ndarray: a numpy array representing the occlusion mask of mr scene
        """ 
        pass
