import numpy as np
from core.IOcclusionProvider import IOcclusionProvider
from DataLoaders.Frame import Frame
from Utils.Logger import logger

class SimpleOcclusionProvider(IOcclusionProvider):
    """
    A simple occlusion provider that compares real camera depth with MR content depth
    to determine occlusion. If the real camera depth is less than the MR content depth,
    the real object is in front and should occlude the virtual content.
    """
    
    def __init__(self, max_depth: float = 5.0):
        """
        Initialize the SimpleOcclusionProvider.
        
        Args:
            max_depth: Maximum depth value in meters
        """
        self.max_depth = max_depth
        logger.log(logger.RENDER, f"Initialized SimpleOcclusionProvider with max_depth={max_depth}")
    
    def occlusion(self, frame: Frame) -> np.ndarray:
        """
        Generate an occlusion mask for a frame.
        
        Args:
            frame: Frame object containing RGB and depth images
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        logger.log(logger.DEBUG, f"Generating simple occlusion mask for frame with timestamp {frame.timestamp}")
        
        # Get camera depth image
        camera_depth = frame.depth
        
        # If there's no MR depth available, we can't determine occlusion
        # So we'll return an empty mask (no occlusion)
        return np.zeros_like(camera_depth, dtype=np.uint8)
    
    def occlusion_with_mr_depth(self, frame: Frame, mr_depth: np.ndarray) -> np.ndarray:
        """
        Generate an occlusion mask by comparing real camera depth with MR content depth.
        
        Args:
            frame: Frame object containing RGB and depth images
            mr_depth: Depth map of the MR contents trying to render
            
        Returns:
            Binary mask indicating occluded areas (1 = occluded, 0 = not occluded)
        """
        logger.log(logger.DEBUG, f"Generating simple occlusion mask for frame with timestamp {frame.timestamp}")
        
        # Get camera depth image
        camera_depth = frame.depth
        
        # Ensure both depth maps have the same dimensions
        if camera_depth.shape != mr_depth.shape:
            logger.log(logger.WARNING, f"Depth map dimensions don't match: camera={camera_depth.shape}, mr={mr_depth.shape}")
            
            # Resize mr_depth to match camera_depth if needed
            from PIL import Image
            
            height, width = camera_depth.shape[:2]
            mr_height, mr_width = mr_depth.shape[:2]
            
            if (height, width) != (mr_height, mr_width):
                logger.log(logger.DEBUG, f"Resizing MR depth from {mr_depth.shape[:2]} to {camera_depth.shape[:2]}")
                # Convert to PIL Image for resizing
                mr_depth_img = Image.fromarray(mr_depth)
                mr_depth_resized = mr_depth_img.resize((width, height), Image.NEAREST)
                mr_depth = np.array(mr_depth_resized)
        
        # Create occlusion mask based on depth comparison
        occlusion_mask = np.zeros_like(camera_depth, dtype=np.uint8)
        
        # Compare depths where both have valid values
        valid_mask = (camera_depth > 0) & (mr_depth > 0)
        
        # Create the occlusion mask: 1 where real objects occlude virtual objects
        # Real objects occlude virtual when real depth is less than virtual depth
        # Simple front/behind check without threshold
        occlusion_mask[valid_mask] = (camera_depth[valid_mask] < mr_depth[valid_mask]).astype(np.uint8)
        
        # Log statistics about the mask
        occluded_pixels = np.sum(occlusion_mask)
        total_pixels = occlusion_mask.size
        occluded_percentage = (occluded_pixels / total_pixels) * 100
        logger.log(logger.DEBUG, f"Simple occlusion mask generated: {occluded_percentage:.2f}% of pixels occluded")
        
        return occlusion_mask