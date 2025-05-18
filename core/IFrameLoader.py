from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np

class IFrameLoader(ABC):
    """
    Interface for frame data loaders.
    
    This interface defines the contract for classes that load frame data
    (RGB images, depth images, IMU data) from various sources.
    """
    
    @abstractmethod
    def load_data(self) -> Dict:
        """
        Load all data from the specified source.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        pass
    
    @abstractmethod
    def get_frame_by_timestamp(self, timestamp) -> Optional:
        """
        Get a frame by its timestamp.
        
        Args:
            timestamp: Timestamp of the frame to retrieve
            
        Returns:
            Frame object or None if not found
        """
        pass
    
    @abstractmethod
    def get_frames_sorted(self) -> List:
        """
        Get all frames sorted by timestamp.
        
        Returns:
            List of Frame objects sorted by timestamp
        """
        pass