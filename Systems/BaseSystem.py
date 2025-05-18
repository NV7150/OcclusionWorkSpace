import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib
from typing import Dict, List, Any, Optional, Union, Literal
from Utils.Logger import logger

class BaseSystem:
    """
    BaseSystem is a facade for the OcclusionSystem and VisualizeSystem.
    
    This class provides a unified interface to both systems, allowing users
    to choose which system to use based on their needs.
    """
    
    def __init__(self, system_type: Literal["occlusion", "visualize"]):
        """
        Initialize the BaseSystem with the specified system type.
        
        Args:
            system_type: Type of system to use ("occlusion" or "visualize")
        """
        self.system_type = system_type
        self.system = None
        logger.log(logger.SYSTEM, f"Initialized BaseSystem with system_type={system_type}")
    
    def create(self, **kwargs):
        """
        Create the underlying system with the specified parameters.
        
        Args:
            **kwar  gs: Parameters to pass to the system's create method
        """
        if self.system_type == "occlusion":
            from Systems.OcclusionSystem import OcclusionSystem
            self.system = OcclusionSystem.create(**kwargs)
            logger.log(logger.SYSTEM, "Created OcclusionSystem")
        elif self.system_type == "visualize":
            from Systems.VisualizeSystem import VisualizeSystem
            self.system = VisualizeSystem(**kwargs)
            logger.log(logger.SYSTEM, "Created VisualizeSystem")
        else:
            raise ValueError(f"Unknown system type: {self.system_type}")
    
    def process(self, **kwargs):
        """
        Process data using the underlying system.
        
        Args:
            **kwargs: Parameters to pass to the system's process method
        """
        if self.system is None:
            raise ValueError("System not created. Call create() first.")
        
        self.system.process(**kwargs)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Mixed Reality Framework')
    
    # System type
    parser.add_argument('--system-type', choices=['occlusion', 'visualize'], required=True,
                        help='Type of system to use (occlusion or visualize)')
    
    # Common arguments
    parser.add_argument('--data-dirs', nargs='+', required=True,
                        help='Directories containing RGB, depth, and IMU data')
    parser.add_argument('--model-dirs', nargs='+', required=True,
                        help='Directories containing 3D models and scene descriptions')
    parser.add_argument('--output-dir', 
                        help='Directory where output will be saved')
    
    # Occlusion system arguments
    parser.add_argument('--output-prefix', default='frame',
                        help='Prefix for output filenames (for occlusion system)')
    parser.add_argument('--occlusion-provider', 
                        help='Python module and class for occlusion provider (e.g., "OcclusionProviders.SimpleOcclusionProvider")')
    parser.add_argument('--camera-matrix', default=None,
                        help='Path to camera matrix file')
    
    # Visualize system arguments
    parser.add_argument('--scene-model', 
                        help='Path to the scene model file (for visualize system)')
    parser.add_argument('--marker-file', 
                        help='Path to the marker positions file (for visualize system)')
    parser.add_argument('--tag-size', type=float, default=0.05,
                        help='Size of AprilTags in meters (for visualize system)')
    parser.add_argument('--tag-family', default='tag36h11',
                        help='AprilTag family (for visualize system)')
    
    # Common optional arguments
    parser.add_argument('--scene-name', default=None,
                        help='Name of the scene to render')
    
    # Logging options
    parser.add_argument('--log-keys', nargs='+', default=['error'],
                        help='Logging keys to enable (e.g., system, model, render)')
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
    
    # Configure logger
    logger.configure(
        enabled_log_keys=args.log_keys,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file
    )
    
    logger.log(logger.SYSTEM, f"Starting {args.system_type.capitalize()} Framework")
    logger.log(logger.SYSTEM, f"Enabled log keys: {args.log_keys}")
    
    # Create the appropriate system
    system = BaseSystem(args.system_type)
    
    if args.system_type == "occlusion":
        # Create occlusion system
        system.create(
            data_dirs=args.data_dirs,
            model_dirs=args.model_dirs,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            occlusion_provider=args.occlusion_provider,
            camera_matrix_file=args.camera_matrix
        )
        
        # Process with occlusion system
        system.process(scene_name=args.scene_name)
        
    elif args.system_type == "visualize":
        # Create visualize system
        system.create(
            scene_model_path=args.scene_model,
            frames_dir=args.data_dirs[0] if args.data_dirs else None,
            marker_file=args.marker_file,
            camera_matrix_file=args.camera_matrix,
            render_obj_dir=args.model_dirs[0] if args.model_dirs else None,
            tag_size=args.tag_size,
            tag_family=args.tag_family,
            log_keys=args.log_keys
        )
        
        # Process with visualize system
        system.process()
    
    logger.log(logger.SYSTEM, "Processing complete")


if __name__ == '__main__':
    main()