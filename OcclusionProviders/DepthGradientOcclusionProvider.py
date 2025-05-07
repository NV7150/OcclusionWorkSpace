import numpy as np
from typing import Optional
from scipy.ndimage import gaussian_filter, binary_dilation

from ..core.IOcclusionProvider import IOcclusionProvider
from ..DataLoaders.Frame import Frame
from .BaseOcclusionProvider import BaseOcclusionProvider
from ..Logger.Logger import Logger


class DepthGradientOcclusionProvider(BaseOcclusionProvider):
    """
    An occlusion provider that uses depth gradients to detect object boundaries.
    
    This provider analyzes the gradients in the depth image to identify
    discontinuities that typically occur at object boundaries.
    """
    
    def __init__(self, 
                 gradient_threshold: float = 0.05, 
                 blur_size: int = 3,
                 dilation_iterations: int = 2,
                 apply_post_processing: bool = True,
                 morphology_kernel_size: int = 3,
                 blur_kernel_size: int = 5,
                 logger: Optional[Logger] = None):
        """
        Initialize the DepthGradientOcclusionProvider.
        
        Args:
            gradient_threshold: Threshold for depth gradients
            blur_size: Size of the blur kernel for preprocessing
            dilation_iterations: Number of iterations for binary dilation
            apply_post_processing: Whether to apply additional morphological post-processing
            morphology_kernel_size: Size of kernel for morphological operations
            blur_kernel_size: Size of kernel for blurring
            logger: Optional logger instance
        """
        super().__init__(name="DepthGradientOcclusionProvider")
        
        # Set parameters
        self.set_parameter("gradient_threshold", gradient_threshold)
        self.set_parameter("blur_size", blur_size)
        self.set_parameter("dilation_iterations", dilation_iterations)
        self.set_parameter("apply_post_processing", apply_post_processing)
        self.set_parameter("morphology_kernel_size", morphology_kernel_size)
        self.set_parameter("blur_kernel_size", blur_kernel_size)
        
        self._logger = logger
        
        if self._logger:
            self._logger.log(Logger.OCCLUSION, 
                           f"Initialized DepthGradientOcclusionProvider with gradient_threshold={gradient_threshold}, blur_size={blur_size}")
    
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Implementation of the IOcclusionProvider interface method.
        
        This method is kept for backward compatibility with the old interface.
        It calls generate_occlusion_mask with None for virtual_depth.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        return self.generate_occlusion_mask(frame)
    
    def generate_occlusion_mask(self, frame: Frame, virtual_depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate an occlusion mask based on depth gradients.
        
        Args:
            frame: Frame object containing RGB and depth images
            virtual_depth: Optional depth buffer from rendered virtual objects
                          (not used in this implementation, but required by interface)
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        if self._logger:
            self._logger.log(Logger.DEBUG, f"Generating gradient-based occlusion mask for frame with timestamp {frame.timestamp}")
        
        # Get parameters
        gradient_threshold = self.get_parameter("gradient_threshold")
        blur_size = self.get_parameter("blur_size")
        dilation_iterations = self.get_parameter("dilation_iterations")
        apply_post_processing = self.get_parameter("apply_post_processing")
        morphology_kernel_size = self.get_parameter("morphology_kernel_size")
        blur_kernel_size = self.get_parameter("blur_kernel_size")
        
        # Get depth image
        depth_image = frame.depth
        
        # Normalize depth values to [0, 1]
        normalized_depth = self._normalize_depth(depth_image)
        
        # Apply Gaussian blur to reduce noise
        if self._logger:
            self._logger.log(Logger.DEBUG, f"Applying Gaussian blur with sigma={blur_size/3}")
        
        blurred_depth = gaussian_filter(normalized_depth, sigma=blur_size/3)
        
        # Compute gradients
        gradient_x = np.abs(np.diff(blurred_depth, axis=1, prepend=blurred_depth[:, :1]))
        gradient_y = np.abs(np.diff(blurred_depth, axis=0, prepend=blurred_depth[:1, :]))
        
        # Combine gradients
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Create occlusion mask based on gradient threshold
        occlusion_mask = (gradient_magnitude > gradient_threshold).astype(np.uint8)
        
        # Dilate the mask to ensure complete coverage of object boundaries
        if self._logger:
            self._logger.log(Logger.DEBUG, f"Dilating occlusion mask with {dilation_iterations} iterations")
        
        occlusion_mask = binary_dilation(occlusion_mask, iterations=dilation_iterations)
        
        # Apply additional post-processing if enabled
        if apply_post_processing:
            occlusion_mask = self._post_process_mask(
                occlusion_mask, 
                morphology_kernel_size=morphology_kernel_size,
                blur_kernel_size=blur_kernel_size
            )
        
        # Store the mask for later retrieval
        self._last_mask = occlusion_mask.astype(np.uint8)
        
        # Log statistics about the mask
        if self._logger:
            occluded_pixels = np.sum(occlusion_mask)
            total_pixels = occlusion_mask.size
            occluded_percentage = (occluded_pixels / total_pixels) * 100
            self._logger.log(Logger.DEBUG, f"Gradient occlusion mask generated: {occluded_percentage:.2f}% of pixels occluded")
        
        return occlusion_mask.astype(np.uint8)