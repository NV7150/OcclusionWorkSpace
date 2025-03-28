import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

from Interfaces.OcclusionProvider import OcclusionProvider
from Interfaces.Frame import Frame


class DepthThresholdOcclusion(OcclusionProvider):
    """
    A simple occlusion provider that uses depth thresholding to generate occlusion masks.
    """
    
    def __init__(self, threshold: float = 0.5, max_depth: float = 5.0):
        """
        Initialize the DepthThresholdOcclusion provider.
        
        Args:
            threshold: Depth threshold value (normalized between 0 and 1)
            max_depth: Maximum depth value in meters
        """
        self.threshold = threshold
        self.max_depth = max_depth
        
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Generate an occlusion mask based on depth thresholding.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        # Get depth image
        depth_image = frame.depth
        
        # Normalize depth values to [0, 1]
        normalized_depth = depth_image.astype(float) / 255.0
        
        # Create occlusion mask based on threshold
        # Pixels with depth less than threshold are considered occluded (foreground objects)
        occlusion_mask = (normalized_depth < self.threshold).astype(np.uint8)
        
        return occlusion_mask


class DepthGradientOcclusion(OcclusionProvider):
    """
    An occlusion provider that uses depth gradients to detect object boundaries.
    """
    
    def __init__(self, gradient_threshold: float = 0.05, blur_size: int = 3):
        """
        Initialize the DepthGradientOcclusion provider.
        
        Args:
            gradient_threshold: Threshold for depth gradients
            blur_size: Size of the blur kernel for preprocessing
        """
        self.gradient_threshold = gradient_threshold
        self.blur_size = blur_size
        
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Generate an occlusion mask based on depth gradients.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        # Get depth image
        depth_image = frame.depth
        
        # Normalize depth values to [0, 1]
        normalized_depth = depth_image.astype(float) / 255.0
        
        # Apply Gaussian blur to reduce noise

        blurred_depth = gaussian_filter(normalized_depth, sigma=self.blur_size/3)
        
        # Compute gradients
        gradient_x = np.abs(np.diff(blurred_depth, axis=1, prepend=blurred_depth[:, :1]))
        gradient_y = np.abs(np.diff(blurred_depth, axis=0, prepend=blurred_depth[:1, :]))
        
        # Combine gradients
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Create occlusion mask based on gradient threshold
        occlusion_mask = (gradient_magnitude > self.gradient_threshold).astype(np.uint8)
        
        # Dilate the mask to ensure complete coverage of object boundaries
        occlusion_mask = binary_dilation(occlusion_mask, iterations=2)
        
        return occlusion_mask.astype(np.uint8)