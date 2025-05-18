import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

from core.IOcclusionProvider import IOcclusionProvider
from DataLoaders.Frame import Frame
from Utils.Logger import logger

class DepthThresholdOcclusionProvider(IOcclusionProvider):
    """
    An occlusion provider that uses depth thresholding to generate occlusion masks.
    """
    
    def __init__(self, threshold: float = 0.5, max_depth: float = 5.0):
        """
        Initialize the DepthThresholdOcclusionProvider.
        
        Args:
            threshold: Depth threshold value (normalized between 0 and 1)
            max_depth: Maximum depth value in meters
        """
        self.threshold = threshold
        self.max_depth = max_depth
        logger.log(logger.RENDER, f"Initialized DepthThresholdOcclusionProvider with threshold={threshold}, max_depth={max_depth}")
    
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Generate an occlusion mask based on depth thresholding.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        logger.log(logger.DEBUG, f"Generating occlusion mask for frame with timestamp {frame.timestamp}")
        
        # Get depth image
        depth_image = frame.depth
        
        # Normalize depth values to [0, 1]
        normalized_depth = depth_image.astype(float) / 255.0
        
        # Create occlusion mask based on threshold
        # Pixels with depth less than threshold are considered occluded (foreground objects)
        occlusion_mask = (normalized_depth < self.threshold).astype(np.uint8)
        
        # Log statistics about the mask
        occluded_pixels = np.sum(occlusion_mask)
        total_pixels = occlusion_mask.size
        occluded_percentage = (occluded_pixels / total_pixels) * 100
        logger.log(logger.DEBUG, f"Occlusion mask generated: {occluded_percentage:.2f}% of pixels occluded")
        
        return occlusion_mask


class DepthGradientOcclusionProvider(IOcclusionProvider):
    """
    An occlusion provider that uses depth gradients to detect object boundaries.
    """
    
    def __init__(self, gradient_threshold: float = 0.05, blur_size: int = 3):
        """
        Initialize the DepthGradientOcclusionProvider.
        
        Args:
            gradient_threshold: Threshold for depth gradients
            blur_size: Size of the blur kernel for preprocessing
        """
        self.gradient_threshold = gradient_threshold
        self.blur_size = blur_size
        logger.log(logger.RENDER, f"Initialized DepthGradientOcclusionProvider with gradient_threshold={gradient_threshold}, blur_size={blur_size}")
    
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Generate an occlusion mask based on depth gradients.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        logger.log(logger.DEBUG, f"Generating gradient-based occlusion mask for frame with timestamp {frame.timestamp}")
        
        # Get depth image
        depth_image = frame.depth
        
        # Normalize depth values to [0, 1]
        normalized_depth = depth_image.astype(float) / 255.0
        
        # Apply Gaussian blur to reduce noise
        logger.log(logger.DEBUG, f"Applying Gaussian blur with sigma={self.blur_size/3}")
        blurred_depth = gaussian_filter(normalized_depth, sigma=self.blur_size/3)
        
        # Compute gradients
        gradient_x = np.abs(np.diff(blurred_depth, axis=1, prepend=blurred_depth[:, :1]))
        gradient_y = np.abs(np.diff(blurred_depth, axis=0, prepend=blurred_depth[:1, :]))
        
        # Combine gradients
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Create occlusion mask based on gradient threshold
        occlusion_mask = (gradient_magnitude > self.gradient_threshold).astype(np.uint8)
        
        # Dilate the mask to ensure complete coverage of object boundaries
        logger.log(logger.DEBUG, "Dilating occlusion mask")
        occlusion_mask = binary_dilation(occlusion_mask, iterations=2)
        
        # Log statistics about the mask
        occluded_pixels = np.sum(occlusion_mask)
        total_pixels = occlusion_mask.size
        occluded_percentage = (occluded_pixels / total_pixels) * 100
        logger.log(logger.DEBUG, f"Gradient occlusion mask generated: {occluded_percentage:.2f}% of pixels occluded")
        
        return occlusion_mask.astype(np.uint8)


class HybridOcclusionProvider(IOcclusionProvider):
    """
    A hybrid occlusion provider that combines multiple occlusion techniques.
    
    This provider combines depth thresholding and gradient-based occlusion
    to create a more robust occlusion mask.
    """
    
    def __init__(self, threshold: float = 0.5, gradient_threshold: float = 0.05, 
                 blur_size: int = 3, max_depth: float = 5.0):
        """
        Initialize the HybridOcclusionProvider.
        
        Args:
            threshold: Depth threshold value (normalized between 0 and 1)
            gradient_threshold: Threshold for depth gradients
            blur_size: Size of the blur kernel for preprocessing
            max_depth: Maximum depth value in meters
        """
        self.threshold = threshold
        self.gradient_threshold = gradient_threshold
        self.blur_size = blur_size
        self.max_depth = max_depth
        
        # Create individual occlusion providers
        self.threshold_provider = DepthThresholdOcclusionProvider(threshold, max_depth)
        self.gradient_provider = DepthGradientOcclusionProvider(gradient_threshold, blur_size)
        
        logger.log(logger.RENDER, f"Initialized HybridOcclusionProvider with threshold={threshold}, "
                  f"gradient_threshold={gradient_threshold}, blur_size={blur_size}, max_depth={max_depth}")
    
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Generate a hybrid occlusion mask combining multiple techniques.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        logger.log(logger.DEBUG, f"Generating hybrid occlusion mask for frame with timestamp {frame.timestamp}")
        
        # Get individual occlusion masks
        threshold_mask = self.threshold_provider.occlusion(frame)
        gradient_mask = self.gradient_provider.occlusion(frame)
        
        # Combine masks (union)
        combined_mask = np.maximum(threshold_mask, gradient_mask)
        
        # Log statistics about the mask
        occluded_pixels = np.sum(combined_mask)
        total_pixels = combined_mask.size
        occluded_percentage = (occluded_pixels / total_pixels) * 100
        logger.log(logger.DEBUG, f"Hybrid occlusion mask generated: {occluded_percentage:.2f}% of pixels occluded")
        
        return combined_mask