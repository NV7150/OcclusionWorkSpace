#!/usr/bin/env python3
"""
Example demonstrating the usage of the refactored MR Occlusion Framework.

This example shows how to:
1. Set up the necessary components
2. Load data
3. Generate occlusion masks
4. Render frames with occlusion
5. Visualize the scene

Usage:
    python refactored_example.py --data-dirs <data_dir> --model-dirs <model_dir> --output-dir <output_dir>
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core interfaces
from core.IFrameLoader import IFrameLoader
from core.IOcclusionProvider import IOcclusionProvider
from core.IRenderer import IRenderer
from core.IScene import IScene
from core.ITracker import ITracker

# Import implementations
from DataLoaders.UniformedFrameLoader import UniformedFrameLoader
from OcclusionProviders.SimpleOcclusionProvider import SimpleOcclusionProvider
from OcclusionProviders.DepthThresholdOcclusionProvider import DepthThresholdOcclusionProvider
from Rendering.Renderer import Renderer
from Rendering.ShaderManager import ShaderManager
from Rendering.BufferManager import BufferManager
from Rendering.TextureManager import TextureManager
from Rendering.Primitives import Primitives
from Models.SceneManager import SceneManager
from Systems.OcclusionSystem import OcclusionSystem
from Systems.OcclusionProcessor import OcclusionProcessor
from Systems.VisualizeSystemRefactored import VisualizeSystem, create_visualization_system
from Logger.Logger import Logger


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='MR Occlusion Framework Example')
    
    parser.add_argument('--data-dirs', nargs='+', required=True,
                        help='Directories containing RGB, depth, and IMU data')
    parser.add_argument('--model-dirs', nargs='+', required=True,
                        help='Directories containing 3D models and scene descriptions')
    parser.add_argument('--output-dir', required=True,
                        help='Directory where rendered images will be saved')
    parser.add_argument('--output-prefix', default='frame',
                        help='Prefix for output filenames')
    parser.add_argument('--scene-name', default=None,
                        help='Name of the scene to render')
    parser.add_argument('--scene-model', default=None,
                        help='Path to the 3D scan model (.fbx) of the scene for visualization')
    parser.add_argument('--marker-file', default=None,
                        help='Path to the marker positions JSON file for visualization')
    parser.add_argument('--camera-matrix', default=None,
                        help='Path to the camera matrix CSV file')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    
    # Logging options
    parser.add_argument('--log-keys', nargs='+', default=['ERROR', 'SYSTEM'],
                        help='Logging keys to enable (e.g., SYSTEM, DEBUG, ERROR)')
    parser.add_argument('--log-to-file', action='store_true',
                        help='Enable logging to file')
    parser.add_argument('--log-file', default=None,
                        help='Path to log file (default: occlusion_framework.log)')
    
    return parser.parse_args()


def run_occlusion_system(args):
    """
    Run the occlusion system to process frames and generate output.
    
    Args:
        args: Command line arguments
    """
    # Create logger
    logger = Logger(
        enabled_log_keys=args.log_keys,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file
    )
    
    logger.log(Logger.SYSTEM, "Starting MR Occlusion Framework Example")
    
    # Create components
    frame_loader = UniformedFrameLoader(args.data_dirs, logger=logger)
    scene_manager = SceneManager(args.model_dirs, logger=logger)
    
    # Create renderer components
    shader_manager = ShaderManager(logger=logger)
    buffer_manager = BufferManager(logger=logger)
    texture_manager = TextureManager(logger=logger)
    primitives = Primitives(buffer_manager, shader_manager, logger=logger)
    
    # Create renderer
    renderer = Renderer(
        shader_manager=shader_manager,
        buffer_manager=buffer_manager,
        texture_manager=texture_manager,
        primitives=primitives,
        logger=logger
    )
    
    # Create occlusion providers
    simple_provider = SimpleOcclusionProvider(logger=logger)
    threshold_provider = DepthThresholdOcclusionProvider(
        threshold=0.5,
        max_depth=5.0,
        apply_post_processing=True,
        logger=logger
    )
    
    # Create occlusion processor with multiple providers
    occlusion_processor = OcclusionProcessor(
        renderer=renderer,
        occlusion_providers=[simple_provider, threshold_provider],
        logger=logger
    )
    
    # Create occlusion system
    system = OcclusionSystem(
        frame_loader=frame_loader,
        scene_manager=scene_manager,
        renderer=renderer,
        occlusion_provider=simple_provider,  # Primary provider
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        logger=logger
    )
    
    # Process data
    logger.log(Logger.SYSTEM, "Processing data...")
    output_paths = system.process(args.scene_name)
    
    logger.log(Logger.SYSTEM, f"Processing complete. Generated {len(output_paths)} output files.")
    logger.log(Logger.SYSTEM, f"Output saved to {args.output_dir}")


def run_visualization(args):
    """
    Run the visualization system to visualize the scene.
    
    Args:
        args: Command line arguments
    """
    if not args.scene_model or not args.marker_file or not args.camera_matrix:
        print("Error: --scene-model, --marker-file, and --camera-matrix are required for visualization")
        return
    
    # Create and run the visualization system
    system = create_visualization_system(
        scene_model_path=args.scene_model,
        frames_dir=args.data_dirs[0],  # Use the first data directory
        marker_file=args.marker_file,
        camera_matrix_file=args.camera_matrix,
        render_obj_dir=args.model_dirs[0],  # Use the first model directory
        tag_size=0.05,
        tag_family='tag36h11',
        log_keys=args.log_keys
    )
    
    # Load data and run the visualization
    system.load_data()
    system.run()


def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    if args.visualize:
        run_visualization(args)
    else:
        run_occlusion_system(args)


if __name__ == '__main__':
    main()