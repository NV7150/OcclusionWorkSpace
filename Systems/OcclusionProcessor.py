import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..core.IOcclusionProvider import IOcclusionProvider
from ..core.IRenderer import IRenderer
from ..DataLoaders.Frame import Frame
from ..Logger.Logger import Logger


class OcclusionProcessor:
    """
    OcclusionProcessor is responsible for generating occlusion masks by coordinating
    between the renderer (for virtual content depth) and occlusion providers.
    
    This class replaces the functionality of the original ContentsDepthCal class
    but with a cleaner separation of concerns and better integration with the
    new architecture.
    """
    
    def __init__(self, 
                 renderer: IRenderer,
                 occlusion_providers: List[IOcclusionProvider],
                 logger: Optional[Logger] = None):
        """
        Initialize the OcclusionProcessor.
        
        Args:
            renderer: Renderer instance used to render virtual content depth
            occlusion_providers: List of occlusion providers to use
            logger: Optional logger instance
        """
        self._renderer = renderer
        self._occlusion_providers = occlusion_providers
        self._logger = logger
        
        # Keep track of the last generated masks
        self._last_masks: Dict[str, np.ndarray] = {}
    
    def add_occlusion_provider(self, provider: IOcclusionProvider) -> None:
        """
        Add an occlusion provider to the processor.
        
        Args:
            provider: Occlusion provider to add
        """
        self._occlusion_providers.append(provider)
    
    def remove_occlusion_provider(self, provider_name: str) -> bool:
        """
        Remove an occlusion provider by name.
        
        Args:
            provider_name: Name of the provider to remove
            
        Returns:
            True if provider was found and removed, False otherwise
        """
        for i, provider in enumerate(self._occlusion_providers):
            if provider.name == provider_name:
                self._occlusion_providers.pop(i)
                return True
        return False
    
    def get_occlusion_providers(self) -> List[IOcclusionProvider]:
        """
        Get all occlusion providers.
        
        Returns:
            List of occlusion providers
        """
        return self._occlusion_providers.copy()
    
    def get_last_masks(self) -> Dict[str, np.ndarray]:
        """
        Get the last generated masks.
        
        Returns:
            Dictionary mapping provider names to masks
        """
        return self._last_masks.copy()
    
    def process_frame(self, 
                      frame: Frame, 
                      models: Dict[str, Any], 
                      scene_data: Dict,
                      combine_method: str = "first") -> np.ndarray:
        """
        Process a frame to generate an occlusion mask.
        
        Args:
            frame: Frame to process
            models: Dictionary of models
            scene_data: Scene description
            combine_method: Method to combine multiple masks ('first', 'union', 'intersection')
                - 'first': Use only the first enabled provider
                - 'union': Combine masks with logical OR
                - 'intersection': Combine masks with logical AND
            
        Returns:
            Occlusion mask as a numpy array
        """
        if self._logger:
            self._logger.log(Logger.DEBUG, f"Processing frame for occlusion mask generation")
        
        # Render virtual content depth
        virtual_depth = self._renderer.render_depth_only(frame, models, scene_data)
        
        # Clear previous masks
        self._last_masks.clear()
        
        # Find enabled providers
        enabled_providers = [p for p in self._occlusion_providers if p.is_enabled()]
        
        if not enabled_providers:
            if self._logger:
                self._logger.log(Logger.WARNING, "No enabled occlusion providers found")
            # Return empty mask
            return np.zeros((frame.height, frame.width), dtype=np.uint8)
        
        # Generate masks from all enabled providers
        masks = []
        for provider in enabled_providers:
            try:
                mask = provider.generate_occlusion_mask(frame, virtual_depth)
                self._last_masks[provider.name] = mask
                masks.append(mask)
                
                if self._logger:
                    self._logger.log(Logger.DEBUG, f"Generated mask using {provider.name}")
                
                # If using 'first' method, we only need the first mask
                if combine_method == "first":
                    break
                    
            except Exception as e:
                if self._logger:
                    self._logger.log(Logger.ERROR, f"Error generating mask with {provider.name}: {e}")
        
        # Combine masks based on the specified method
        if not masks:
            if self._logger:
                self._logger.log(Logger.WARNING, "No masks were successfully generated")
            # Return empty mask
            return np.zeros((frame.height, frame.width), dtype=np.uint8)
        
        if combine_method == "first" or len(masks) == 1:
            # Use the first mask
            final_mask = masks[0]
        elif combine_method == "union":
            # Combine masks with logical OR
            final_mask = np.zeros_like(masks[0], dtype=bool)
            for mask in masks:
                final_mask = final_mask | (mask > 0)
            final_mask = final_mask.astype(np.uint8)
        elif combine_method == "intersection":
            # Combine masks with logical AND
            final_mask = np.ones_like(masks[0], dtype=bool)
            for mask in masks:
                final_mask = final_mask & (mask > 0)
            final_mask = final_mask.astype(np.uint8)
        else:
            if self._logger:
                self._logger.log(Logger.WARNING, f"Unknown combine method: {combine_method}, using 'first'")
            final_mask = masks[0]
        
        if self._logger:
            occluded_pixels = np.sum(final_mask)
            total_pixels = final_mask.size
            occluded_percentage = (occluded_pixels / total_pixels) * 100
            self._logger.log(Logger.DEBUG, 
                           f"Final occlusion mask: {occluded_percentage:.2f}% of pixels occluded")
        
        return final_mask
    
    def process_frames(self, 
                       frames: List[Frame], 
                       models: Dict[str, Any], 
                       scene_data: Dict,
                       combine_method: str = "first") -> Dict[np.datetime64, np.ndarray]:
        """
        Process multiple frames to generate occlusion masks.
        
        Args:
            frames: List of frames to process
            models: Dictionary of models
            scene_data: Scene description
            combine_method: Method to combine multiple masks ('first', 'union', 'intersection')
            
        Returns:
            Dictionary mapping timestamps to occlusion masks
        """
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Processing {len(frames)} frames for occlusion masks")
        
        masks = {}
        for frame in frames:
            mask = self.process_frame(frame, models, scene_data, combine_method)
            masks[frame.timestamp] = mask
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Generated {len(masks)} occlusion masks")
        
        return masks