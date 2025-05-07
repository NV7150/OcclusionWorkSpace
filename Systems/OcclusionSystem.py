import os
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import glob

from ..core.IFrameLoader import IFrameLoader
from ..core.IOcclusionProvider import IOcclusionProvider
from ..core.IRenderer import IRenderer
from ..core.IScene import IScene
from ..core.IModel import IModel
from ..DataLoaders.Frame import Frame
from ..Logger.Logger import Logger
from ..Utils.TransformUtils import TransformUtils


class OcclusionSystem:
    """
    OcclusionSystem is the main class for the occlusion framework.
    It coordinates data loading, occlusion mask generation, and rendering.
    
    This is a refactored version of the original BaseSystem class that follows
    the new architecture with cleaner separation of concerns and better modularity.
    """
    
    def __init__(self,
                 frame_loader: IFrameLoader,
                 scene_manager: IScene,
                 renderer: IRenderer,
                 occlusion_provider: IOcclusionProvider,
                 output_dir: str,
                 output_prefix: str = "frame",
                 logger: Optional[Logger] = None):
        """
        Initialize the OcclusionSystem with components and configuration.
        
        Args:
            frame_loader: Component for loading frame data
            scene_manager: Component for managing scene and models
            renderer: Component for rendering
            occlusion_provider: Component for generating occlusion masks
            output_dir: Directory where rendered images will be saved
            output_prefix: Prefix for output filenames
            logger: Optional logger instance
        """
        self._frame_loader = frame_loader
        self._scene_manager = scene_manager
        self._renderer = renderer
        self._occlusion_provider = occlusion_provider
        self._output_dir = output_dir
        self._output_prefix = output_prefix
        self._logger = logger
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Initialized OcclusionSystem")
            self._logger.log(Logger.SYSTEM, f"Output directory: {output_dir}")
        
        # Data storage
        self._frames = {}
        self._occlusion_masks = {}
    
    def load_data(self) -> Tuple[Dict[np.datetime64, Frame], Dict[str, Any], Dict[str, Dict]]:
        """
        Load all data (frames, models, scenes).
        
        Returns:
            Tuple containing:
            - Dictionary mapping timestamps to Frame objects
            - Dictionary of models
            - Dictionary of scenes
        """
        if self._logger:
            self._logger.log(Logger.SYSTEM, "Loading frames...")
        
        # Load frames
        self._frames = self._frame_loader.load_data()
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Loaded {len(self._frames)} frames")
            self._logger.log(Logger.SYSTEM, "Loading models and scenes...")
        
        # Load models and scenes
        models = self._scene_manager.get_all_models()
        scenes = self._scene_manager.get_all_scenes()
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Loaded {len(models)} models and {len(scenes)} scenes")
        
        return self._frames, models, scenes
    
    def generate_occlusion_masks(self, scene_name: Optional[str] = None) -> Dict[np.datetime64, np.ndarray]:
        """
        Generate occlusion masks for all frames using the provided occlusion provider.
        
        Args:
            scene_name: Name of the scene to use for depth calculation. If None, the first scene will be used.
            
        Returns:
            Dictionary mapping timestamps to occlusion masks
        """
        if self._logger:
            self._logger.log(Logger.SYSTEM, "Generating occlusion masks...")
        
        # Get all scenes
        scenes = self._scene_manager.get_all_scenes()
        
        if not scenes:
            if self._logger:
                self._logger.log(Logger.ERROR, "No scenes loaded")
            return {}
        
        # Use the specified scene or the first one
        if scene_name is None:
            scene_name = next(iter(scenes))
            if self._logger:
                self._logger.log(Logger.SYSTEM, f"No scene specified, using first scene: '{scene_name}'")
        elif scene_name not in scenes:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Scene '{scene_name}' not found")
            return {}
        
        scene_data = scenes[scene_name]
        models = self._scene_manager.get_all_models()
        
        # Process each frame
        for timestamp, frame in self._frames.items():
            # Calculate MR content depth using the renderer
            virtual_depth = self._renderer.render_depth_only(frame, models, scene_data)
            
            # Generate occlusion mask
            occlusion_mask = self._occlusion_provider.generate_occlusion_mask(frame, virtual_depth)
            self._occlusion_masks[timestamp] = occlusion_mask
            
            if self._logger:
                self._logger.log(Logger.DEBUG, f"Generated mask for timestamp {timestamp}")
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Generated {len(self._occlusion_masks)} occlusion masks")
        
        return self._occlusion_masks
    
    def render_frames(self, scene_name: Optional[str] = None) -> List[str]:
        """
        Render all frames with occlusion masks and save the results.
        
        Args:
            scene_name: Name of the scene to render. If None, the first scene will be used.
            
        Returns:
            List of paths to the rendered output files
        """
        # Get all scenes
        scenes = self._scene_manager.get_all_scenes()
        
        if not scenes:
            if self._logger:
                self._logger.log(Logger.ERROR, "No scenes loaded")
            return []
        
        # Use the specified scene or the first one
        if scene_name is None:
            scene_name = next(iter(scenes))
            if self._logger:
                self._logger.log(Logger.SYSTEM, f"No scene specified, using first scene: '{scene_name}'")
        elif scene_name not in scenes:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Scene '{scene_name}' not found")
            return []
        
        scene_data = scenes[scene_name]
        models = self._scene_manager.get_all_models()
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Rendering frames with scene '{scene_name}'...")
        
        # Get sorted frames
        frames_list = self._frame_loader.get_frames_sorted()
        
        if self._logger:
            self._logger.log(Logger.DEBUG, f"Rendering {len(frames_list)} frames")
        
        # Render and save frames
        output_paths = self._renderer.render_and_save_batch(
            frames_list,
            self._occlusion_masks,
            models,
            scene_data,
            self._output_dir,
            self._output_prefix
        )
        
        if self._logger:
            self._logger.log(Logger.SYSTEM, f"Rendered {len(output_paths)} frames")
            self._logger.log(Logger.SYSTEM, f"Output saved to {self._output_dir}")
        
        return output_paths
    
    def process(self, scene_name: Optional[str] = None) -> List[str]:
        """
        Process all data: load, generate occlusion masks, and render.
        
        Args:
            scene_name: Name of the scene to render. If None, the first scene will be used.
            
        Returns:
            List of paths to the rendered output files
        """
        self.load_data()
        self.generate_occlusion_masks(scene_name)
        return self.render_frames(scene_name)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Mixed Reality Occlusion Framework')
    
    parser.add_argument('--data-dirs', nargs='+', required=True,
                        help='Directories containing RGB, depth, and IMU data')
    parser.add_argument('--model-dirs', nargs='+', required=True,
                        help='Directories containing 3D models and scene descriptions')
    parser.add_argument('--output-dir', required=True,
                        help='Directory where rendered images will be saved')
    parser.add_argument('--output-prefix', default='frame',
                        help='Prefix for output filenames')
    parser.add_argument('--occlusion-provider', required=True,
                        help='Python module and class for occlusion provider (e.g., "my_module.MyOcclusionProvider")')
    parser.add_argument('--scene-name', default=None,
                        help='Name of the scene to render')
    
    # Logging options
    parser.add_argument('--log-keys', nargs='+', default=[Logger.ERROR],
                        help=f'Logging keys to enable (e.g., {Logger.SYSTEM}, {Logger.MODEL}, {Logger.RENDER})')
    parser.add_argument('--log-to-file', action='store_true',
                        help='Enable logging to file')
    parser.add_argument('--log-file', default=None,
                        help='Path to log file (default: occlusion_framework.log)')
    
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # Create logger
    logger = Logger(
        enabled_log_keys=args.log_keys,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file
    )
    
    logger.log(Logger.SYSTEM, "Starting Occlusion Framework")
    logger.log(Logger.SYSTEM, f"Enabled log keys: {args.log_keys}")
    
    # Import and instantiate the occlusion provider
    module_name, class_name = args.occlusion_provider.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    occlusion_provider_class = getattr(module, class_name)
    occlusion_provider = occlusion_provider_class()
    logger.log(Logger.SYSTEM, f"Initialized occlusion provider: {args.occlusion_provider}")
    
    # Import and instantiate other components
    # Note: In a real implementation, these would be properly instantiated with their dependencies
    # This is just a placeholder for the command-line interface
    from ..DataLoaders.UniformedFrameLoader import UniformedFrameLoader
    from ..Models.SceneManager import SceneManager
    from ..Rendering.Renderer import Renderer
    
    # Create components
    frame_loader = UniformedFrameLoader(args.data_dirs)
    scene_manager = SceneManager(args.model_dirs)
    renderer = Renderer(logger=logger)
    
    # Create and run the system
    system = OcclusionSystem(
        frame_loader=frame_loader,
        scene_manager=scene_manager,
        renderer=renderer,
        occlusion_provider=occlusion_provider,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        logger=logger
    )
    
    system.process(args.scene_name)
    logger.log(Logger.SYSTEM, "Processing complete")


if __name__ == '__main__':
    main()