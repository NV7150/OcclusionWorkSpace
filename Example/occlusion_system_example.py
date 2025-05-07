#!/usr/bin/env python3
"""
MR Occlusion System Example

This script demonstrates how to use the MR Occlusion System with all its components.
It loads data from recorded datasets and performs occlusion-aware rendering of 3D models.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Systems.MROcclusionSystem import MROcclusionSystem
from DataLoaders.UniformedFrameLoader import UniformedFrameLoader
from Trackers.AprilTagTracker import AprilTagTracker
from Models.SceneManager import SceneManager
from OcclusionProviders.DepthThresholdOcclusionProvider import DepthThresholdOcclusionProvider
from Logger.Logger import Logger


def main():
    """Main function to demonstrate the MR Occlusion System."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MR Occlusion System Example")
    parser.add_argument(
        "--data", 
        type=str, 
        default="../LocalData/DepthIMUData1",
        help="Path to recorded data directory"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="../LocalData/Models/cube.obj",
        help="Path to 3D model file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="../Output",
        help="Output directory for processed frames"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=640,
        help="Width of output images"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=480,
        help="Height of output images"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.05,
        help="Depth threshold for occlusion detection (meters)"
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger()
    logger.info("Starting MR Occlusion System example...")
    
    # Make paths absolute
    data_path = os.path.abspath(args.data)
    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)
    
    # Check if directories exist
    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {data_path}")
        return
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create config for occlusion system
    config = {
        "width": args.width,
        "height": args.height,
        "output_directory": output_path
    }
    
    # Create MR Occlusion System
    system = MROcclusionSystem(config)
    
    # Create frame loader
    frame_loader = UniformedFrameLoader(data_path, "rgb", "depth", "imu")
    frame_loader.load_directory()
    system.set_frame_loader(frame_loader)
    
    # Create tracker (if available)
    try:
        tracker = AprilTagTracker()
        system.set_tracker(tracker)
        logger.info("AprilTag tracker initialized")
    except Exception as e:
        logger.warning(f"Could not initialize tracker: {e}")
    
    # Create occlusion provider with configured threshold
    occlusion_provider = DepthThresholdOcclusionProvider(depth_threshold=args.threshold)
    system.set_occlusion_provider(occlusion_provider)
    
    # Initialize system
    if not system.initialize():
        logger.error("Failed to initialize MR Occlusion System")
        return
    
    # Load 3D model
    model = system.load_model(model_path)
    if model is None:
        logger.error(f"Failed to load model from {model_path}")
        return
    
    # Set initial model position (this would normally be updated by tracking)
    # This positions the model in front of the camera
    model_transform = np.eye(4, dtype=np.float32)
    model_transform[0, 3] = 0.0  # x position
    model_transform[1, 3] = 0.0  # y position
    model_transform[2, 3] = -0.5  # z position (negative is in front of the camera)
    model.set_transform(model_transform)
    
    logger.info("Starting frame processing...")
    
    # Process frames
    system.start_processing()
    
    # Clean up
    system.cleanup()
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()