import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

# Using relative import that will work after full refactoring is complete
from ..core.IFrameLoader import IFrameLoader
from .Frame import Frame


class BaseFrameLoader(IFrameLoader):
    """
    Abstract base class for frame data loaders.
    
    Implements common functionality for all frame loaders and provides
    a template for concrete frame loader implementations.
    """
    
    def __init__(self):
        """Initialize the BaseFrameLoader."""
        self._frames: Dict[np.datetime64, Frame] = {}
    
    def get_frame_by_timestamp(self, timestamp: np.datetime64) -> Optional[Frame]:
        """
        Get a frame by its timestamp.
        
        Args:
            timestamp: Timestamp of the frame to retrieve
            
        Returns:
            Frame object or None if not found
        """
        return self._frames.get(timestamp, None)
    
    def get_frames_sorted(self) -> List[Frame]:
        """
        Get all frames sorted by timestamp.
        
        Returns:
            List of Frame objects sorted by timestamp
        """
        # Sort timestamps and return frames in order
        sorted_timestamps = sorted(self._frames.keys())
        return [self._frames[ts] for ts in sorted_timestamps]
    
    @abstractmethod
    def load_data(self) -> Dict[np.datetime64, Frame]:
        """
        Load all RGB images, depth images, and IMU data from the specified directories.
        
        To be implemented by concrete subclasses.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        pass
    
    @abstractmethod
    def _load_rgb_images(self) -> Dict[np.datetime64, np.ndarray]:
        """
        Load RGB images from the data directory.
        
        To be implemented by concrete subclasses.
        
        Returns:
            Dictionary mapping timestamps to RGB images
        """
        pass
    
    @abstractmethod
    def _load_depth_images(self) -> Dict[np.datetime64, np.ndarray]:
        """
        Load depth images from the data directory.
        
        To be implemented by concrete subclasses.
        
        Returns:
            Dictionary mapping timestamps to depth images
        """
        pass
    
    @abstractmethod
    def _load_imu_data(self) -> Dict[np.datetime64, Dict[str, np.ndarray]]:
        """
        Load IMU data from the data directory.
        
        To be implemented by concrete subclasses.
        
        Returns:
            Dictionary mapping timestamps to dictionaries with 'acc' and 'gyro' keys
        """
        pass