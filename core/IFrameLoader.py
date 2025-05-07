from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional

# Using relative import that will work after full refactoring is complete
from ..DataLoaders.Frame import Frame


class IFrameLoader(ABC):
    """
    Interface for frame data loaders.
    
    Defines methods that all frame loaders must implement to provide a consistent
    way to access frame data across different data sources.
    """
    
    @abstractmethod
    def load_data(self) -> Dict[np.datetime64, Frame]:
        """
        Load all RGB images, depth images, and IMU data from the specified directories.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        pass
    
    @abstractmethod
    def get_frame_by_timestamp(self, timestamp: np.datetime64) -> Optional[Frame]:
        """
        Get a frame by its timestamp.
        
        Args:
            timestamp: Timestamp of the frame to retrieve
            
        Returns:
            Frame object or None if not found
        """
        pass
    
    @abstractmethod
    def get_frames_sorted(self) -> List[Frame]:
        """
        Get all frames sorted by timestamp.
        
        Returns:
            List of Frame objects sorted by timestamp
        """
        pass