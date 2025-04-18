import numpy as np
from abc import ABCMeta, abstractmethod
from . import Frame

## Provides occlusion mask (interface for accurate AI code generation)
class Tracker(metaclass=ABCMeta):
    @abstractmethod
    def track(self, frame: Frame) -> np.ndarray:
        """ 
        Track the camera frame and return the tracking data.
        The Tracker can be a fusion of multiple Tracker classes, using Kalman filter, Particle filter, etc.
        
        Args:
            frame(Frame): Frame object containing the current camera frame data
        Returns:
            np.ndarray: a numpy array representing the tracking data of the camera frame
            It is a 4 * 4 matrix representing the camera pose in the world coordinate system.
            T = [R|t] where R is the rotation matrix and t is the translation vector.
            The rotation matrix R is a 3 * 3 matrix and the translation vector t is a 3 * 1 vector.
            Last row is [0, 0, 0, 1] to make it a 4 * 4 matrix.
        """ 
        pass
    