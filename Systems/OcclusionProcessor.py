import numpy as np
from typing import Dict, Any, Optional
import PIL.Image

# Import core interfaces
from core.IFrame import IFrame
from core.IOcclusionProvider import IOcclusionProvider
from core.IRenderer import IRenderer
from core.IModel import IModel

# Import utilities
from Utils.Logger import logger
from Utils.TransformUtils import look_at

# Import data types
from Data.DepthData import DepthData

class OcclusionProcessor:
    """
    OcclusionProcessor is responsible for generating occlusion masks by combining
    real-world depth data with virtual content depth data.
    
    It uses the Renderer to generate depth maps of virtual content and then
    combines them with real-world depth data using an occlusion provider.
    """
    
    def __init__(self, renderer: IRenderer, occlusion_provider: IOcclusionProvider):
        """
        Initialize the OcclusionProcessor.
        
        Args:
            renderer: Renderer instance for generating virtual content depth maps
            occlusion_provider: OcclusionProvider instance for generating occlusion masks
        """
        self.renderer = renderer
        self.occlusion_provider = occlusion_provider
        logger.log(logger.SYSTEM, f"Initialized OcclusionProcessor with {type(occlusion_provider).__name__}")
    
    def process_frame(self, frame: IFrame, models: Dict[str, IModel], scene_data: Dict) -> np.ndarray:
        """
        Process a frame to generate an occlusion mask.
        
        Args:
            frame: Frame object containing RGB and depth images
            models: Dictionary of models to render
            scene_data: Scene description with model positions and orientations
            
        Returns:
            Binary occlusion mask (1 = occluded, 0 = not occluded)
        """
        logger.log(logger.DEBUG, f"Processing frame with timestamp {frame.timestamp}")
        
        # Check if the frame already has an occlusion mask
        occlusion_mask = frame.get_metadata('occlusion_mask')
        if occlusion_mask is not None:
            logger.log(logger.DEBUG, "Using pre-computed occlusion mask")
            return occlusion_mask
        
        # Get camera pose from frame if available
        camera_matrix = None
        camera_pose = frame.get_metadata('camera_pose')
        if camera_pose is not None:
            camera_matrix = camera_pose
        elif frame.get_metadata('camera_matrix') is not None:
            # Create a simple view matrix if only intrinsic matrix is available
            camera_matrix = look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))
        
        # Generate virtual content depth map
        mr_depth = self.calculate_content_depth(frame, models, scene_data, camera_matrix)
        
        # Generate occlusion mask
        if hasattr(self.occlusion_provider, 'occlusion_with_mr_depth'):
            # Use specialized method if available
            occlusion_mask = self.occlusion_provider.occlusion_with_mr_depth(frame, mr_depth)
        else:
            # Fall back to standard method
            # Note: This might not work correctly with all occlusion providers
            occlusion_mask = self.occlusion_provider.occlusion(frame)
        
        # Store the occlusion mask in the frame for future use
        frame.set_metadata('occlusion_mask', occlusion_mask)
        
        return occlusion_mask
    
    def calculate_content_depth(self, frame: IFrame, models: Dict[str, IModel], 
                               scene_data: Dict, camera_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the depth map of virtual content.
        
        Args:
            frame: Frame object containing RGB and depth images
            models: Dictionary of models to render
            scene_data: Scene description with model positions and orientations
            camera_matrix: Optional camera matrix to use for rendering
            
        Returns:
            Depth map of virtual content
        """
        logger.log(logger.DEBUG, "Calculating virtual content depth map")
        
        # Use the renderer to generate a depth map
        depth_map = self.renderer.render_depth(models, scene_data, camera_matrix)
        
        # Get depth data from the frame
        depth_data = frame.get_sensor(DepthData)
        if depth_data is None:
            logger.log(logger.WARNING, "No depth data in frame")
            return depth_map
        
        # Ensure the depth map has the same dimensions as the frame's depth
        if depth_map.shape != depth_data.depth.shape:
            logger.log(logger.WARNING, 
                      f"Depth map dimensions don't match: {depth_map.shape} vs {depth_data.depth.shape}")
            
            # Resize depth map if needed
            height, width = depth_data.depth.shape[:2]
            depth_img = PIL.Image.fromarray(depth_map)
            depth_resized = depth_img.resize((width, height), PIL.Image.NEAREST)
            depth_map = np.array(depth_resized)
        
        return depth_map
    
    def process_batch(self, frames: Dict[np.datetime64, IFrame], 
                     models: Dict[str, IModel], scene_data: Dict) -> Dict[np.datetime64, np.ndarray]:
        """
        Process a batch of frames to generate occlusion masks.
        
        Args:
            frames: Dictionary mapping timestamps to Frame objects
            models: Dictionary of models to render
            scene_data: Scene description with model positions and orientations
            
        Returns:
            Dictionary mapping timestamps to occlusion masks
        """
        logger.log(logger.SYSTEM, f"Processing {len(frames)} frames for occlusion masks")
        
        occlusion_masks = {}
        
        for timestamp, frame in frames.items():
            occlusion_mask = self.process_frame(frame, models, scene_data)
            occlusion_masks[timestamp] = occlusion_mask
            logger.log(logger.DEBUG, f"Generated mask for timestamp {timestamp}")
        
        logger.log(logger.SYSTEM, f"Generated {len(occlusion_masks)} occlusion masks")
        return occlusion_masks