import os
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
import glob
from DataLoader import DataLoader
from ModelLoader import ModelLoader
from Renderer import Renderer
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
                 occlusion_provider: OcclusionProvider):
        """
        Initialize the BaseSystem with directories and an occlusion provider.
        
        Args:
            data_dirs: List of directories containing RGB, depth, and IMU data
            model_dirs: List of directories containing 3D models and scene descriptions
            output_dir: Directory where rendered images will be saved
            occlusion_provider: Implementation of OcclusionProvider to generate occlusion masks
        """
        self.data_dirs = data_dirs
        self.model_dirs = model_dirs
        self.output_dir = output_dir
        self.occlusion_provider = occlusion_provider
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_dirs)
        self.model_loader = ModelLoader(model_dirs)
        self.renderer = Renderer(output_dir)
        
        # Data storage
        self.frames = {}
        self.models = {}
        self.scenes = {}
        self.occlusion_masks = {}
        
    def load_data(self):
        """
        Load all data (frames, models, scenes).
        """
        print("Loading frames...")
        self.frames = self.data_loader.load_data()
        print(f"Loaded {len(self.frames)} frames")
        
        print("Loading models and scenes...")
        self.models, self.scenes = self.model_loader.load_models_and_scenes()
        print(f"Loaded {len(self.models)} models and {len(self.scenes)} scenes")
        
    def generate_occlusion_masks(self):
        """
        Generate occlusion masks for all frames using the provided occlusion provider.
        """
        print("Generating occlusion masks...")
        for timestamp, frame in self.frames.items():
            occlusion_mask = self.occlusion_provider.occlusion(frame)
            self.occlusion_masks[timestamp] = occlusion_mask
        print(f"Generated {len(self.occlusion_masks)} occlusion masks")
        
    def render_frames(self, scene_name: Optional[str] = None):
        """
        Render all frames with occlusion masks and save the results.
        
        Args:
            scene_name: Name of the scene to render. If None, the first scene will be used.
        """
        if not self.scenes:
            print("Error: No scenes loaded")
            return
            
        # Use the specified scene or the first one
        if scene_name is None:
            scene_name = next(iter(self.scenes))
        elif scene_name not in self.scenes:
            print(f"Error: Scene '{scene_name}' not found")
            return
            
        scene_data = self.scenes[scene_name]
        
        print(f"Rendering frames with scene '{scene_name}'...")
        frames_list = self.data_loader.get_frames_sorted()
        output_paths = self.renderer.render_and_save_batch(
            frames_list, 
            self.occlusion_masks, 
            self.models, 
            scene_data
        )
        print(f"Rendered {len(output_paths)} frames")
        print(f"Output saved to {self.output_dir}")
        
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
    parser.add_argument('--occlusion-provider', required=True,
                        help='Python module and class for occlusion provider (e.g., "my_module.MyOcclusionProvider")')
    parser.add_argument('--scene-name', default=None,
                        help='Name of the scene to render')
    
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # Import and instantiate the occlusion provider
    module_name, class_name = args.occlusion_provider.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    occlusion_provider_class = getattr(module, class_name)
    occlusion_provider = occlusion_provider_class()
    
    # Create and run the system
    system = BaseSystem(
        data_dirs=args.data_dirs,
        model_dirs=args.model_dirs,
        output_dir=args.output_dir,
        occlusion_provider=occlusion_provider
    )
    
    system.process(args.scene_name)


if __name__ == '__main__':
    main()