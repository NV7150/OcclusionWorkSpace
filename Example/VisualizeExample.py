import os
import sys
import numpy as np
import cv2

# Add parent directory to path so we can import from Systems
sys.path.append('..')

from Systems.VisualizeSystem import VisualizeSystem
from Logger import Logger, logger

# Parameters
# Path to the 3D scan model of the scene
scene_model_path = os.path.join('..', 'LocalData', 'DepthIMUData2', 'Env_3DModels', 'on_the_desk.fbx')

# Directory containing frame and depth images
frames_dir = os.path.join('..', 'LocalData', 'DepthIMUData2', 'slow')

# IMU data file (CSV)
imu_file = os.path.join(frames_dir, 'imu.csv')

# Camera matrix file (CSV)
camera_matrix_file = os.path.join('..', 'LocalData', 'camera_ipadpro.csv')

# Marker positions file (JSON)
marker_file = os.path.join('..', 'LocalData', 'DepthIMUData2', 'Env_3DModels', 'marker_poses_opengl.json')

# Directory containing 3D models for rendering
render_obj_dir = os.path.join('..', 'LocalData', 'Models', 'Scene1')

# AprilTag parameters
tag_size = 0.086  # 8.6cm
tag_family = 'tagStandard41h12'  # This needs to match the actual tags you're using

# Logging parameters
log_keys = [Logger.SYSTEM, Logger.DEBUG, Logger.ERROR]


def main():
    """
    Example usage of the VisualizeSystem for MR scene visualization.
    """
    # Allow user to override default parameters
    global scene_model_path, frames_dir, marker_file, camera_matrix_file, render_obj_dir, log_keys, tag_family, tag_size
    
    # Configure logging
    log_options = input("Enter log options (comma-separated, leave empty for default) > ").strip()
    if log_options:
        # Parse log options
        log_keys = [option.strip() for option in log_options.split(',')]
        logger.configure(enabled_log_keys=log_keys)
        logger.log(Logger.SYSTEM, f"Enabled log keys: {log_keys}")
    else:
        # Default to system logs, debug, and errors
        logger.configure(enabled_log_keys=log_keys)
        logger.log(Logger.SYSTEM, f"Using default log keys: {', '.join(log_keys)}")
    
    # Check if scene model file exists
    if not os.path.exists(scene_model_path):
        logger.log(Logger.WARNING, f"Scene model file not found: {scene_model_path}")
        user_scene_model_path = input(f"Enter scene model file path (default: {scene_model_path}): ").strip()
        if user_scene_model_path:
            scene_model_path = user_scene_model_path
    
    # Create and run the visualization system
    try:
        logger.log(Logger.SYSTEM, "Creating VisualizeSystem...")
        system = VisualizeSystem(
            scene_model_path=scene_model_path,
            frames_dir=frames_dir,
            marker_file=marker_file,
            camera_matrix_file=camera_matrix_file,
            render_obj_dir=render_obj_dir,
            tag_size=tag_size,
            tag_family=tag_family,
            log_keys=log_keys
        )
        
        logger.log(Logger.SYSTEM, "Loading data...")
        system.load_data()
        
        logger.log(Logger.SYSTEM, "Starting visualization...")
        logger.log(Logger.SYSTEM, "Controls:")
        logger.log(Logger.SYSTEM, "  Mouse Left Button: Rotate the scene")
        logger.log(Logger.SYSTEM, "  Mouse Right Button: Zoom in/out")
        logger.log(Logger.SYSTEM, "  Arrow Keys: Rotate the scene")
        logger.log(Logger.SYSTEM, "  m: Toggle marker visibility")
        logger.log(Logger.SYSTEM, "  c: Toggle camera visibility")
        logger.log(Logger.SYSTEM, "  o: Toggle MR contents visibility")
        logger.log(Logger.SYSTEM, "  s: Toggle scene model visibility")
        logger.log(Logger.SYSTEM, "  g: Toggle grid visibility")
        logger.log(Logger.SYSTEM, "  a: Toggle axes visibility")
        logger.log(Logger.SYSTEM, "  f: Reset to free view mode")
        logger.log(Logger.SYSTEM, "  r: Reset view")
        logger.log(Logger.SYSTEM, "  q/ESC: Exit")
        
        system.run()
        
    except Exception as e:
        logger.log(Logger.ERROR, f"Error in VisualizeExample: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()