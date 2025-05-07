from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from ..DataLoaders.Frame import Frame


class ITracker(ABC):
    """
    Interface for tracking systems.
    
    Defines methods that all tracker implementations must provide to ensure
    consistent camera pose estimation across different tracking algorithms.
    """
    
    @abstractmethod
    def initialize(self, camera_matrix: np.ndarray, dist_coeffs: Optional[np.ndarray] = None) -> bool:
        """
        Initialize the tracker with camera parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients (optional)
            
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process_frame(self, frame: Frame) -> Optional[np.ndarray]:
        """
        Process a frame to estimate camera pose.
        
        Args:
            frame: Frame object containing RGB image and other sensor data
            
        Returns:
            4x4 camera pose matrix (world to camera transform) or None if tracking failed
        """
        pass
    
    @abstractmethod
    def get_last_pose(self) -> Optional[np.ndarray]:
        """
        Get the last estimated camera pose.
        
        Returns:
            4x4 camera pose matrix (world to camera transform) or None if no pose has been estimated
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the tracker state.
        """
        pass
    
    @abstractmethod
    def set_reference_markers(self, markers: Dict[str, Dict[str, Any]]) -> None:
        """
        Set reference markers for tracking.
        
        Args:
            markers: Dictionary mapping marker IDs to marker data
                    Each marker data should contain at least 'position', 'normal', and 'tangent'
        """
        pass
    
    @abstractmethod
    def get_detected_markers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get markers detected in the last processed frame.
        
        Returns:
            Dictionary mapping marker IDs to marker data
        """
        pass
    
    @abstractmethod
    def get_tracking_quality(self) -> float:
        """
        Get the quality of the current tracking.
        
        Returns:
            Tracking quality value between 0.0 (poor) and 1.0 (excellent)
        """
        pass
    
    @abstractmethod
    def get_tracking_status(self) -> str:
        """
        Get the current tracking status as a string.
        
        Returns:
            Status string (e.g., "TRACKING", "LOST", "INITIALIZING")
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set tracker parameters.
        
        Args:
            parameters: Dictionary of parameter name-value pairs
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current tracker parameters.
        
        Returns:
            Dictionary of parameter name-value pairs
        """
        pass