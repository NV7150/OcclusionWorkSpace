from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from core.IFrameLoader import IFrameLoader
from .Frame import Frame

class BaseFrameLoader(IFrameLoader, ABC):
    """
    Abstract base class for frame loaders.
    
    This class implements common functionality for frame loaders and defines
    abstract methods that must be implemented by concrete subclasses.
    """
    
    def __init__(self):
        """
        Initialize the BaseFrameLoader.
        """
        self.frames = {}  # Dictionary mapping timestamps to Frame objects
        self.camera_matrix: Optional[np.ndarray] = None
    
    @abstractmethod
    def load_data(self) -> Dict[np.datetime64, Frame]:
        """
        Load all data from the specified source.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        pass
    
    def get_frame_by_timestamp(self, timestamp: np.datetime64) -> Optional[Frame]:
        """
        Get a frame by its timestamp.
        
        Args:
            timestamp: Timestamp of the frame to retrieve
            
        Returns:
            Frame object or None if not found
        """
        return self.frames.get(timestamp)
    
    def get_frames_sorted(self) -> List[Frame]:
        """
        Get all frames sorted by timestamp.
        
        Returns:
            List of Frame objects sorted by timestamp
        """
        return [self.frames[ts] for ts in sorted(self.frames.keys())]
    
    def get_frames(self) -> Dict[np.datetime64, Frame]:
        """
        Get all frames.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        return self.frames
    
    def get_camera_matrix(self) -> Optional[np.ndarray]:
        """
        Get the camera intrinsic matrix.
        
        Returns:
            Camera intrinsic matrix (3x3) or None if not set
        """
        return self.camera_matrix
    
    def set_camera_matrix(self, camera_matrix: np.ndarray) -> None:
        """
        Set the camera intrinsic matrix.
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
        """
        self.camera_matrix = camera_matrix
        
        # Update camera matrix for all frames
        for frame in self.frames.values():
            if frame.get_camera_matrix() is None:
                frame.set_camera_matrix(camera_matrix)
    
    def add_frame(self, frame: Frame) -> None:
        """
        Add a frame to the loader.
        
        Args:
            frame: Frame object to add
        """
        try:
            key = frame.timestamp
            self.frames[key] = frame
        except AttributeError as e:
            print(f"!!! AttributeError accessing frame.timestamp property inside add_frame: {e}")
            raise e
        except Exception as e_add:
            key_repr = repr(key) if 'key' in locals() else 'undefined'
            print(f"!!! Error adding frame with key {key_repr} inside add_frame: {e_add}")
            raise e_add
    
    def clear_frames(self) -> None:
        """
        Clear all frames.
        """
        self.frames = {}
    
    def get_frame_count(self) -> int:
        """
        Get the number of frames.
        
        Returns:
            Number of frames
        """
        return len(self.frames)
    
    def get_timestamp_range(self) -> Optional[tuple[np.datetime64, np.datetime64]]:
        """
        Get the range of timestamps.
        
        Returns:
            Tuple of (min_timestamp, max_timestamp) or None if no frames
        """
        if not self.frames:
            return None
        
        timestamps = list(self.frames.keys())
        return (min(timestamps), max(timestamps))