import numpy as np
from typing import Optional

from ..core.IOcclusionProvider import IOcclusionProvider
from ..DataLoaders.Frame import Frame
from .BaseOcclusionProvider import BaseOcclusionProvider
from ..Logger.Logger import Logger


class DepthThresholdOcclusionProvider(BaseOcclusionProvider):
    """
    An occlusion provider that uses depth thresholding to generate occlusion masks.
    
    This provider compares depth values against a threshold to determine which areas
    should be considered as occluders.
    """
    
    def __init__(self, 
                 threshold: float = 0.5, 
                 max_depth: float = 5.0,
                 apply_post_processing: bool = True,
                 morphology_kernel_size: int = 3,
                 blur_kernel_size: int = 5,
                 logger: Optional[Logger] = None):
        """
        Initialize the DepthThresholdOcclusionProvider.
        
        Args:
            threshold: Depth threshold value (normalized between 0 and 1)
            max_depth: Maximum depth value in meters
            apply_post_processing: Whether to apply morphological post-processing
            morphology_kernel_size: Size of kernel for morphological operations
            blur_kernel_size: Size of kernel for blurring
            logger: Optional logger instance
        """
        super().__init__(name="DepthThresholdOcclusionProvider")
        
        # Set parameters
        self.set_parameter("threshold", threshold)
        self.set_parameter("max_depth", max_depth)
        self.set_parameter("apply_post_processing", apply_post_processing)
        self.set_parameter("morphology_kernel_size", morphology_kernel_size)
        self.set_parameter("blur_kernel_size", blur_kernel_size)
        
        self._logger = logger
        
        if self._logger:
            self._logger.log(Logger.OCCLUSION, 
                           f"Initialized DepthThresholdOcclusionProvider with threshold={threshold}, max_depth={max_depth}")
    
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
        Generate an occlusion mask based on depth thresholding.
        
        Args:
            frame: Frame object containing RGB and depth images
            virtual_depth: Optional depth buffer from rendered virtual objects
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        if self._logger:
            self._logger.log(Logger.DEBUG, f"Generating occlusion mask for frame with timestamp {frame.timestamp}")
        
        # Get parameters
        threshold = self.get_parameter("threshold")
        max_depth = self.get_parameter("max_depth")
        apply_post_processing = self.get_parameter("apply_post_processing")
        morphology_kernel_size = self.get_parameter("morphology_kernel_size")
        blur_kernel_size = self.get_parameter("blur_kernel_size")
        
        # Get depth image
        depth_image = frame.depth
        
        # If virtual depth is provided, use it for comparison
        if virtual_depth is not None:
            # Ensure both depth maps have the same dimensions
            if depth_image.shape != virtual_depth.shape:
                if self._logger:
                    self._logger.log(Logger.WARNING, 
                                   f"Depth map dimensions don't match: camera={depth_image.shape}, virtual={virtual_depth.shape}")
                
                # Resize virtual_depth to match depth_image if needed
                try:
                    from PIL import Image
                    
                    height, width = depth_image.shape[:2]
                    virt_height, virt_width = virtual_depth.shape[:2]
                    
                    if (height, width) != (virt_height, virt_width):
                        if self._logger:
                            self._logger.log(Logger.DEBUG, 
                                           f"Resizing virtual depth from {virtual_depth.shape[:2]} to {depth_image.shape[:2]}")
                        # Convert to PIL Image for resizing
                        virt_depth_img = Image.fromarray(virtual_depth)
                        virt_depth_resized = virt_depth_img.resize((width, height), Image.NEAREST)
                        virtual_depth = np.array(virt_depth_resized)
                except ImportError:
                    if self._logger:
                        self._logger.log(Logger.ERROR, "PIL not available for depth map resizing")
                    virtual_depth = None
            
            # Compare real depth with virtual depth
            if virtual_depth is not None:
                # Create occlusion mask based on depth comparison
                # Real objects occlude virtual when real depth is less than virtual depth
                valid_mask = (depth_image > 0) & (virtual_depth > 0)
                occlusion_mask = np.zeros_like(depth_image, dtype=np.uint8)
                occlusion_mask[valid_mask] = (depth_image[valid_mask] < virtual_depth[valid_mask]).astype(np.uint8)
                
                if self._logger:
                    self._logger.log(Logger.DEBUG, "Generated occlusion mask using depth comparison")
                
                # Apply post-processing if enabled
                if apply_post_processing:
                    occlusion_mask = self._post_process_mask(
                        occlusion_mask, 
                        morphology_kernel_size=morphology_kernel_size,
                        blur_kernel_size=blur_kernel_size
                    )
                
                # Store the mask for later retrieval
                self._last_mask = occlusion_mask
                
                # Log statistics about the mask
                if self._logger:
                    occluded_pixels = np.sum(occlusion_mask)
                    total_pixels = occlusion_mask.size
                    occluded_percentage = (occluded_pixels / total_pixels) * 100
                    self._logger.log(Logger.DEBUG, 
                                   f"Occlusion mask generated: {occluded_percentage:.2f}% of pixels occluded")
                
                return occlusion_mask
        
        # If no virtual depth is provided or comparison failed, use simple thresholding
        # Normalize depth values to [0, 1]
        normalized_depth = self._normalize_depth(depth_image)
        
        # Create occlusion mask based on threshold
        # Pixels with depth less than threshold are considered occluded (foreground objects)
        occlusion_mask = (normalized_depth < threshold).astype(np.uint8)
        
        # Apply post-processing if enabled
        if apply_post_processing:
            occlusion_mask = self._post_process_mask(
                occlusion_mask, 
                morphology_kernel_size=morphology_kernel_size,
                blur_kernel_size=blur_kernel_size
            )
        
        # Store the mask for later retrieval
        self._last_mask = occlusion_mask
        
        # Log statistics about the mask
        if self._logger:
            occluded_pixels = np.sum(occlusion_mask)
            total_pixels = occlusion_mask.size
            occluded_percentage = (occluded_pixels / total_pixels) * 100
            self._logger.log(Logger.DEBUG, f"Occlusion mask generated: {occluded_percentage:.2f}% of pixels occluded")
        
        return occlusion_mask