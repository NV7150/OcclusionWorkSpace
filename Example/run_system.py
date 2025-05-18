#!/usr/bin/env python3
"""
MR Occlusion Framework Runner

This script provides a unified command-line interface for running the various systems
in the MR Occlusion Framework, including OcclusionSystem and VisualizeSystem.

Usage:
    python run_system.py occlusion --data-dirs /path/to/data --model-dirs /path/to/models --output-dir /path/to/output
    python run_system.py visualize --scene-model /path/to/scene.fbx --frames-dir /path/to/frames --marker-file /path/to/markers.json
    python run_system.py --config /path/to/config.json
"""

import os
import sys
import argparse
import numpy as np
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core interfaces
from core.IFrameLoader import IFrameLoader
from core.IScene import IScene
from core.ITracker import ITracker
from core.IRenderer import IRenderer
from core.IOcclusionProvider import IOcclusionProvider

# Import utility components
from Utils.Logger import logger

# Import loader implementations
from DataLoaders.UniformedFrameLoader import UniformedFrameLoader
from DataLoaders.SeparatedFrameLoader import SeparatedFrameLoader

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='MR Occlusion Framework Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Run occlusion system:
    python run_system.py occlusion --data-dirs data/rgb_depth --model-dirs models/scene1 --output-dir output/occlusion --occlusion-provider OcclusionProviders.SimpleOcclusionProvider
        
  Run visualization system:
    python run_system.py visualize --scene-model models/scene.fbx --frames-dir data/frames --marker-file markers.json --camera-matrix camera.csv --render-obj-dir models/scene1 --scene-json models/scene1/scene.json
        
  Run using configuration file:
    python run_system.py --config config.json
        '''
    )
    
    # Add configuration file option
    parser.add_argument('--config', help='Path to configuration JSON file')
    
    subparsers = parser.add_subparsers(dest='system_type', help='System to run')
    
    # Occlusion System arguments
    occlusion_parser = subparsers.add_parser('occlusion', help='Run the Occlusion System')
    occlusion_parser.add_argument('--data-dirs', nargs='+',
                        help='Directories containing data. For \'uniformed\' loader, list all. For \'separated\' loader, provide the single root dataset path.')
    occlusion_parser.add_argument('--model-dirs', nargs='+',
                        help='Directories containing 3D models and scene descriptions')
    occlusion_parser.add_argument('--output-dir',
                        help='Directory where rendered images will be saved')
    occlusion_parser.add_argument('--output-prefix', default='frame',
                        help='Prefix for output filenames')
    occlusion_parser.add_argument('--occlusion-provider',
                        help='Python module and class for occlusion provider (e.g., "OcclusionProviders.SimpleOcclusionProvider")')
    occlusion_parser.add_argument('--camera-matrix', default=None,
                        help='Path to camera matrix file')
    occlusion_parser.add_argument('--scene-name', default=None,
                        help='Name of the scene to render')
    occlusion_parser.add_argument('--loader-type', choices=['uniformed', 'separated'],
                        help='Specify the data loader format type')
    
    # Visualization System arguments
    viz_parser = subparsers.add_parser('visualize', help='Run the Visualization System')
    viz_parser.add_argument('--scene-model',
                        help='Path to the 3D scan model (.fbx) of the scene')
    viz_parser.add_argument('--frames-dir',
                        help='Directory containing frame data. For \'uniformed\', this is the dir with rgb/depth/imu.csv. For \'separated\', this is the root dataset path.')
    viz_parser.add_argument('--marker-file',
                        help='Path to the marker positions JSON file')
    viz_parser.add_argument('--camera-matrix',
                        help='Path to the camera matrix CSV file')
    viz_parser.add_argument('--render-obj-dir',
                        help='Directory containing MR content models and scene description')
    viz_parser.add_argument('--scene-json',
                        help='Path to the scene description JSON file')
    viz_parser.add_argument('--tag-size', type=float, default=0.05,
                        help='Size of AprilTag markers in meters (default: 0.05)')
    viz_parser.add_argument('--tag-family', default='tag36h11',
                        help='AprilTag family to use for detection (default: tag36h11)')
    viz_parser.add_argument('--loader-type', choices=['uniformed', 'separated'],
                        help='Specify the data loader format type')
    
    # Common options
    parser.add_argument('--log-keys', nargs='+', default=['error', 'system'],
                        help='Logging keys to enable (e.g., system, debug, error)')
    parser.add_argument('--log-to-file', action='store_true',
                        help='Enable logging to file')
    parser.add_argument('--log-file', default=None,
                        help='Path to log file (default: auto-generated based on system type)')
    
    return parser.parse_args()

def load_config_from_json(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Namespace object with configuration settings
    """
    logger.log(logger.SYSTEM, f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create an argparse.Namespace to hold the configuration
        args = argparse.Namespace()
        
        # Check if the configuration specifies a system type
        if "command" in config:
            args.system_type = config["command"]
        else:
            logger.log(logger.ERROR, "Configuration file must specify a 'command' field with value 'occlusion' or 'visualize'")
            sys.exit(1)
        
        # Set default log settings
        args.log_keys = config.get("log_keys", ["error", "system"])
        args.log_to_file = config.get("log_to_file", False)
        args.log_file = config.get("log_file", None)
        
        # ---- Get Loader Type ----
        args.loader_type = config.get("loader-type") # Read from JSON, default to uniformed
        allowed_loaders = ['uniformed', 'separated']
        if args.loader_type not in allowed_loaders:
            logger.log(logger.ERROR, f"Invalid 'loaderType' in config: {args.loader_type}. Must be one of {allowed_loaders}")           
            sys.exit(1)
        # -------------------------
        
        # Map JSON configuration fields to their corresponding argument names
        # For visualization system
        if args.system_type == "visualize":
            args.scene_model = config.get("sceneModel")
            args.frames_dir = config.get("framesDir")
            args.marker_file = config.get("markerFile")
            args.camera_matrix = config.get("cameraMatrix")
            args.render_obj_dir = config.get("renderObjDir")
            args.scene_json = config.get("sceneJson")
            args.tag_size = float(config.get("tagSize", 0.05))
            args.tag_family = config.get("tagFamily", "tag36h11")
        
        # For occlusion system
        elif args.system_type == "occlusion":
            args.data_dirs = config.get("dataDirs", [])
            args.model_dirs = config.get("modelDirs", [])
            args.output_dir = config.get("outputDir")
            args.output_prefix = config.get("outputPrefix", "frame")
            args.occlusion_provider = config.get("occlusionProvider")
            args.camera_matrix = config.get("cameraMatrix")
            args.scene_name = config.get("sceneName")
        
        return args
        
    except FileNotFoundError:
        logger.log(logger.ERROR, f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.log(logger.ERROR, f"Error parsing JSON configuration: {e}")
        sys.exit(1)

def run_occlusion_system(args):
    """
    Run the Occlusion System with the specified arguments.
    
    Args:
        args: Command line arguments
    """
    from Systems.OcclusionSystem import OcclusionSystem
    
    # Configure logger
    logger.configure(
        enabled_log_keys=args.log_keys,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file or "occlusion_system.log"
    )
    
    logger.log(logger.SYSTEM, "Starting Occlusion Framework")
    logger.log(logger.SYSTEM, f"Enabled log keys: {args.log_keys}")
    
    # Create and run the system
    system = OcclusionSystem.create(
        data_dirs=args.data_dirs,
        model_dirs=args.model_dirs,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        occlusion_provider=args.occlusion_provider,
        camera_matrix_file=args.camera_matrix
    )
    
    system.process(args.scene_name)
    logger.log(logger.SYSTEM, "Occlusion System processing complete")

def create_visualization_components(args):
    """
    Create and configure components for the Visualization System.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple containing (frame_loader, scene_manager, tracker, renderer)
    """
    # Import concrete implementations
    from Models.SceneManager import SceneManager
    from Trackers.ApriltagTracker import ApriltagTracker
    from Systems.VisualizeModelRender import VisualizeModelRender
    
    # Load camera matrix
    camera_matrix = np.loadtxt(args.camera_matrix, delimiter=',')
    
    # --- Create frame loader based on type --- 
    if args.loader_type == 'separated':
        logger.log(logger.SYSTEM, f"Using SeparatedFrameLoader for dataset path: {args.frames_dir}")
        frame_loader = SeparatedFrameLoader(dataset_path=args.frames_dir)
    elif args.loader_type == 'uniformed':
        logger.log(logger.SYSTEM, f"Using UniformedFrameLoader for data directory: {args.frames_dir}")
        frame_loader = UniformedFrameLoader(data_dirs=[args.frames_dir])
    else:
        # Should not happen due to argparse choices, but handle defensively
        logger.log(logger.ERROR, f"Invalid loader_type specified: {args.loader_type}")
        sys.exit(1)
    # -------------------------------------------
    
    # Create scene manager (implements IScene)
    scene_manager = SceneManager()
    
    # If there's a scene JSON file specified, load it
    if hasattr(args, 'scene_json') and args.scene_json and os.path.isfile(args.scene_json):
        logger.log(logger.SYSTEM, f"Loading scene from {args.scene_json}")
        scene_manager.load_scene_from_file(args.scene_json)
    else:
        logger.log(logger.WARNING, "No scene JSON file specified or file does not exist")
    
    # Create tracker (implements ITracker)
    tracker = ApriltagTracker(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        tag_size=args.tag_size,
        tag_family=args.tag_family
    )
    tracker.load_marker_positions(args.marker_file)
    
    # Create renderer (implements IRenderer)
    renderer = VisualizeModelRender()
    
    return frame_loader, scene_manager, tracker, renderer

def run_visualization_system(args):
    """
    Run the Visualization System with the specified arguments.
    
    Args:
        args: Command line arguments
    """
    from Systems.VisualizeSystem import VisualizeSystem
    
    # Configure logger
    logger.configure(
        enabled_log_keys=args.log_keys,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file or "visualize_system.log"
    )
    
    logger.log(logger.SYSTEM, "Starting Visualization System")
    
    # Create components
    frame_loader, scene_manager, tracker, renderer = create_visualization_components(args)
    
    # Create visualization system
    # VisualizeSystem will be responsible for calling renderer.initialize(), 
    # renderer.load_scene(), renderer.setup_scene(), and renderer.start_render_loop()
    system = VisualizeSystem(
        frame_loader=frame_loader,
        scene_manager=scene_manager,
        tracker=tracker,
        renderer=renderer,
        scene_model_path=args.scene_model, # Pass scene_model_path for VisualizeSystem to use
        log_keys=args.log_keys
    )
    
    # Load data and run visualization (run() should contain the main loop)
    system.load_data() # This method in VisualizeSystem should prepare data for renderer.setup_scene
    system.run()       # This method in VisualizeSystem should call renderer.start_render_loop
    
    logger.log(logger.SYSTEM, "Visualization System complete")

def main():
    """
    Main entry point for the MR Occlusion Framework Runner.
    """
    args = parse_args()
    
    # If a configuration file is provided, load it
    if args.config:
        args = load_config_from_json(args.config)
    
    # Validate required arguments are present
    if args.system_type == 'occlusion':
        run_occlusion_system(args)
    elif args.system_type == 'visualize':
        run_visualization_system(args)
    else:
        print(f"Unknown system type: {args.system_type}")
        sys.exit(1)

if __name__ == '__main__':
    main() 