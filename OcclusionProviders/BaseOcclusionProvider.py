import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple

from ..core.IOcclusionProvider import IOcclusionProvider
from ..DataLoaders.Frame import Frame


class BaseOcclusionProvider(IOcclusionProvider, ABC):
    """
    Abstract base class for occlusion providers.
    
    Implements the IOcclusionProvider interface and provides common functionality
    for derived occlusion classes. This class handles the base operations needed
    for generating occlusion masks from depth and RGB data.
    """
    
    def __init__(self, name: str = "BaseOcclusionProvider"):
        """
        Initialize the base occlusion provider.
        
        Args:
            name: Name of the occlusion provider
        """
        self._name = name
        self._parameters: Dict[str, Any] = {}
        self._enabled = True
        self._last_mask = None
    
    @property
    def name(self) -> str:
        """Get the name of this occlusion provider."""
        return self._name
    
    def is_enabled(self) -> bool:
        """
        Check if this occlusion provider is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self._enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable this occlusion provider.
        
        Args:
            enabled: Boolean flag indicating whether to enable the provider
        """
        self._enabled = enabled
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set a parameter for this occlusion provider.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self._parameters[name] = value
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value.
        
        Args:
            name: Parameter name
            default: Default value to return if parameter doesn't exist
            
        Returns:
            Parameter value or default if not found
        """
        return self._parameters.get(name, default)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters.
        
        Returns:
            Dictionary of all parameters
        """
        return self._parameters.copy()
    
    def get_last_occlusion_mask(self) -> Optional[np.ndarray]:
        """
        Get the last generated occlusion mask.
        
        Returns:
            The last occlusion mask, or None if no mask has been generated yet
        """
        return self._last_mask
    
    @abstractmethod
    def generate_occlusion_mask(self, frame: Frame, virtual_depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate an occlusion mask from the given frame.
        
        This method must be implemented by subclasses to provide specific
        occlusion detection algorithms.
        
        Args:
            frame: Frame containing RGB and depth data
            virtual_depth: Optional depth buffer from rendered virtual objects
            
        Returns:
            Binary occlusion mask where True/1 indicates occlusion areas
        """
        pass
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize a depth image to range [0, 1].
        
        Args:
            depth: Depth image
            
        Returns:
            Normalized depth image
        """
        if depth is None or depth.size == 0:
            return np.zeros((1, 1), dtype=np.float32)
            
        # Handle NaN and inf values
        valid_mask = np.isfinite(depth)
        if not np.any(valid_mask):
            return np.zeros_like(depth)
            
        # Get valid depth range
        valid_depth = depth[valid_mask]
        min_val = np.min(valid_depth)
        max_val = np.max(valid_depth)
        
        if min_val == max_val:
            return np.zeros_like(depth)
            
        # Create normalized depth
        normalized = np.zeros_like(depth)
        normalized[valid_mask] = (depth[valid_mask] - min_val) / (max_val - min_val)
        
        return normalized
    
    def _post_process_mask(self, mask: np.ndarray, 
                          morphology_kernel_size: int = 3, 
                          blur_kernel_size: int = 5) -> np.ndarray:
        """
        Apply post-processing to the occlusion mask.
        
        This method applies morphological operations and blurring to
        refine the occlusion mask and remove noise.
        
        Args:
            mask: Raw occlusion mask
            morphology_kernel_size: Size of kernel for morphological operations
            blur_kernel_size: Size of kernel for blurring
            
        Returns:
            Processed occlusion mask
        """
        try:
            import cv2
            
            # Ensure the mask is binary and of the right type
            binary_mask = mask.astype(np.uint8) * 255
            
            # Create kernel for morphological operations
            kernel = np.ones((morphology_kernel_size, morphology_kernel_size), np.uint8)
            
            # Apply closing to fill small holes
            processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply opening to remove small noise
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
            
            # Apply Gaussian blur to smooth the edges
            if blur_kernel_size > 0:
                processed_mask = cv2.GaussianBlur(processed_mask, 
                                                 (blur_kernel_size, blur_kernel_size), 
                                                 0)
            
            # Convert back to binary mask
            processed_mask = processed_mask > 127
            
            return processed_mask
            
        except ImportError:
            # If OpenCV is not available, return the original mask
            print("Warning: OpenCV not available for mask post-processing")
            return mask > 0