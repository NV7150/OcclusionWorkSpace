import numpy as np
from typing import Optional

from ..core.IOcclusionProvider import IOcclusionProvider
from ..DataLoaders.Frame import Frame
from .BaseOcclusionProvider import BaseOcclusionProvider
from ..Logger.Logger import Logger


class SimpleOcclusionProvider(BaseOcclusionProvider):
    """
    A simple occlusion provider that compares real camera depth with MR content depth
    to determine occlusion. If the real camera depth is less than the MR content depth,
    the real object is in front and should occlude the virtual content.
    """
    
    def __init__(self, max_depth: float = 5.0, logger: Optional[Logger] = None):
        """
        Initialize the SimpleOcclusionProvider.
        
        Args:
            max_depth: Maximum depth value in meters
            logger: Optional logger instance
        """
        super().__init__(name="SimpleOcclusionProvider")
        
        # Set parameters
        self.set_parameter("max_depth", max_depth)
        self._logger = logger
        
        if self._logger:
            self._logger.log(Logger.OCCLUSION, f"Initialized SimpleOcclusionProvider with max_depth={max_depth}")
    
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
        Generate an occlusion mask by comparing real camera depth with MR content depth.
        
        Args:
            frame: Frame object containing RGB and depth images
            virtual_depth: Optional depth buffer from rendered virtual objects
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        if self._logger:
            self._logger.log(Logger.DEBUG, f"Generating simple occlusion mask for frame with timestamp {frame.timestamp}")
        
        # Get camera depth image
        camera_depth = frame.depth
        
        # If no virtual depth is provided, we can't do a comparison
        if virtual_depth is None:
            # Create a mask where all real objects with valid depth are considered occluders
            occlusion_mask = (camera_depth > 0).astype(np.uint8)
            self._last_mask = occlusion_mask
            return occlusion_mask
        
        # Ensure both depth maps have the same dimensions
        if camera_depth.shape != virtual_depth.shape:
            if self._logger:
                self._logger.log(Logger.WARNING, 
                                f"Depth map dimensions don't match: camera={camera_depth.shape}, virtual={virtual_depth.shape}")
            
            # Resize virtual_depth to match camera_depth if needed
            try:
                from PIL import Image
                
                height, width = camera_depth.shape[:2]
                mr_height, mr_width = virtual_depth.shape[:2]
                
                if (height, width) != (mr_height, mr_width):
                    if self._logger:
                        self._logger.log(Logger.DEBUG, 
                                        f"Resizing virtual depth from {virtual_depth.shape[:2]} to {camera_depth.shape[:2]}")
                    # Convert to PIL Image for resizing
                    mr_depth_img = Image.fromarray(virtual_depth)
                    mr_depth_resized = mr_depth_img.resize((width, height), Image.NEAREST)
                    virtual_depth = np.array(mr_depth_resized)
            except ImportError:
                if self._logger:
                    self._logger.log(Logger.ERROR, "PIL not available for depth map resizing")
                # Create a default mask if we can't resize
                occlusion_mask = (camera_depth > 0).astype(np.uint8)
                self._last_mask = occlusion_mask
                return occlusion_mask
        
        # Create occlusion mask based on depth comparison
        occlusion_mask = np.zeros_like(camera_depth, dtype=np.uint8)
        
        # Compare depths where both have valid values
        valid_mask = (camera_depth > 0) & (virtual_depth > 0)
        
        # Create the occlusion mask: 1 where real objects occlude virtual objects
        # Real objects occlude virtual when real depth is less than virtual depth
        occlusion_mask[valid_mask] = (camera_depth[valid_mask] < virtual_depth[valid_mask]).astype(np.uint8)
        
        # Apply post-processing if enabled
        apply_post_processing = self.get_parameter("apply_post_processing", False)
        if apply_post_processing:
            morphology_kernel_size = self.get_parameter("morphology_kernel_size", 3)
            blur_kernel_size = self.get_parameter("blur_kernel_size", 5)
            occlusion_mask = self._post_process_mask(
                occlusion_mask, 
                morphology_kernel_size=morphology_kernel_size,
                blur_kernel_size=blur_kernel_size
            )
        
        # Log statistics about the mask
        if self._logger:
            occluded_pixels = np.sum(occlusion_mask)
            total_pixels = occlusion_mask.size
            occluded_percentage = (occluded_pixels / total_pixels) * 100
            self._logger.log(Logger.DEBUG, f"Simple occlusion mask generated: {occluded_percentage:.2f}% of pixels occluded")
        
        # Store the mask for later retrieval
        self._last_mask = occlusion_mask
        
        return occlusion_mask