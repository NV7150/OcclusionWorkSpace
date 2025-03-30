import os
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
import glob
from Systems.DataLoader import DataLoader
from Systems.ModelLoader import ModelLoader
from Systems.Renderer import Renderer
from Systems.Logger import Logger, logger
from Interfaces.OcclusionProvider import OcclusionProvider
from Interfaces.Frame import Frame


class BaseSystem:
    """
    BaseSystem is the main class for the occlusion framework.
    It coordinates data loading, occlusion mask generation, and rendering.
    """
    def __init__(self,
                 data_dirs: List[str],
                 model_dirs: List[str],
                 output_dir: str,
                 output_prefix: str,
                 occlusion_provider: OcclusionProvider,
                 log_options: Optional[List[str]] = None,
                 log_to_file: bool = False,
                 log_file_path: Optional[str] = None
                 ):
        """
        Initialize the BaseSystem with directories and an occlusion provider.
        
        Args:
            data_dirs: List of directories containing RGB, depth, and IMU data
            model_dirs: List of directories containing 3D models and scene descriptions
            output_dir: Directory where rendered images will be saved
            output_prefix: Prefix for output filenames
            occlusion_provider: Implementation of OcclusionProvider to generate occlusion masks
            log_options: List of logging categories to enable (e.g., ["system-logs", "model-logs"])
            log_to_file: Whether to log to a file in addition to console
            log_file_path: Path to the log file. If None, logs are written to 'occlusion_framework.log'
        """
        self.data_dirs = data_dirs
        self.model_dirs = model_dirs
        self.output_dir = output_dir
        self.occlusion_provider = occlusion_provider
        self.output_prefix = output_prefix
        
        # Configure logger
        self.logger = Logger(log_options, log_to_file, log_file_path)
        self.logger.system(f"Initializing BaseSystem with {len(data_dirs)} data directories and {len(model_dirs)} model directories")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.logger.system(f"Output directory: {output_dir}")
        
        # Initialize components
        self.data_loader = DataLoader(data_dirs)
        self.model_loader = ModelLoader(model_dirs)
        self.renderer = Renderer(output_dir)
        
        # Data storage
        self.frames = {}
        self.models = {}
        self.scenes = {}
        self.occlusion_masks = {}
        self.occlusion_masks = {}
        
    def load_data(self):
        """
        Load all data (frames, models, scenes).
        """
        self.logger.system("Loading frames...")
        self.frames = self.data_loader.load_data()
        self.logger.system(f"Loaded {len(self.frames)} frames")
        
        self.logger.system("Loading models and scenes...")
        self.models, self.scenes = self.model_loader.load_models_and_scenes()
        self.logger.system(f"Loaded {len(self.models)} models and {len(self.scenes)} scenes")
        
    def generate_occlusion_masks(self):
        """
        Generate occlusion masks for all frames using the provided occlusion provider.
        """
        self.logger.system("Generating occlusion masks...")
        for timestamp, frame in self.frames.items():
            occlusion_mask = self.occlusion_provider.occlusion(frame)
            self.occlusion_masks[timestamp] = occlusion_mask
            self.logger.debug(f"Generated mask for timestamp {timestamp}")
        self.logger.system(f"Generated {len(self.occlusion_masks)} occlusion masks")
        
    def render_frames(self, scene_name: Optional[str] = None):
        """
        Render all frames with occlusion masks and save the results.
        
        Args:
            scene_name: Name of the scene to render. If None, the first scene will be used.
        """
        if not self.scenes:
            self.logger.error("No scenes loaded")
            return
            
        # Use the specified scene or the first one
        if scene_name is None:
            scene_name = next(iter(self.scenes))
            self.logger.system(f"No scene specified, using first scene: '{scene_name}'")
        elif scene_name not in self.scenes:
            self.logger.error(f"Scene '{scene_name}' not found")
            return
            
        scene_data = self.scenes[scene_name]
        
        self.logger.system(f"Rendering frames with scene '{scene_name}'...")
        frames_list = self.data_loader.get_frames_sorted()
        self.logger.debug(f"Rendering {len(frames_list)} frames")
        
        output_paths = self.renderer.render_and_save_batch(
            frames_list,
            self.occlusion_masks,
            self.models,
            scene_data,
            self.output_prefix
        )
        self.logger.system(f"Rendered {len(output_paths)} frames")
        self.logger.system(f"Output saved to {self.output_dir}")
        
    def process(self, scene_name: Optional[str] = None):
        """
        Process all data: load, generate occlusion masks, and render.
        
        Args:
            scene_name: Name of the scene to render. If None, the first scene will be used.
        """
        self.load_data()
        self.generate_occlusion_masks()
        self.render_frames(scene_name)


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
    parser.add_argument('--log-options', nargs='+', default=['error-logs'],
                        help='Logging categories to enable (e.g., system-logs, model-logs, render-logs)')
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
    
    # Configure global logger
    logger.configure(
        enabled_categories=args.log_options,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file
    )
    
    logger.system("Starting Occlusion Framework")
    logger.system(f"Enabled log categories: {args.log_options}")
    
    # Import and instantiate the occlusion provider
    module_name, class_name = args.occlusion_provider.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    occlusion_provider_class = getattr(module, class_name)
    occlusion_provider = occlusion_provider_class()
    logger.system(f"Initialized occlusion provider: {args.occlusion_provider}")
    
    # Create and run the system
    system = BaseSystem(
        data_dirs=args.data_dirs,
        model_dirs=args.model_dirs,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        occlusion_provider=occlusion_provider,
        log_options=args.log_options,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file
    )
    
    system.process(args.scene_name)
    logger.system("Processing complete")


if __name__ == '__main__':
    main()