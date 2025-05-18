from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

class ITracker(ABC):
    """
    Interface for trackers.
    
    This interface defines the contract for classes that track markers or features
    in images to estimate camera poses.
    """
    
    @abstractmethod
    def initialize(self, camera_matrix: np.ndarray, tag_size: float = 0.05, tag_family: str = 'tag36h11') -> None:
        """
        Initialize the tracker with camera parameters.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            tag_size: Size of the tags in meters
            tag_family: Family of tags to detect
        """
        pass
    
    @abstractmethod
    def detect_markers(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect markers in an image.
        
        Args:
            image: RGB or grayscale image
            
        Returns:
            Dictionary mapping marker IDs to marker data
        """
        pass
    
    @abstractmethod
    def estimate_pose(self, image: np.ndarray, marker_positions: Dict[str, Dict[str, List[float]]]) -> Optional[np.ndarray]:
        """
        Estimate camera pose from an image and known marker positions.
        
        Args:
            image: RGB or grayscale image
            marker_positions: Dictionary mapping marker IDs to position data
            
        Returns:
            4x4 transformation matrix representing camera pose, or None if pose cannot be estimated
        """
        pass
    
    @abstractmethod
    def track_features(self, prev_image: np.ndarray, curr_image: np.ndarray, prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track features from one image to another.
        
        Args:
            prev_image: Previous image
            curr_image: Current image
            prev_points: Points to track from previous image
            
        Returns:
            Tuple of (tracked_points, status), where status indicates which points were successfully tracked
        """
        pass
    
    @abstractmethod
    def get_camera_matrix(self) -> np.ndarray:
        """
        Get the camera intrinsic matrix.
        
        Returns:
            Camera intrinsic matrix
        """
        pass