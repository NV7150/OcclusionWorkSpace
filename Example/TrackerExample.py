import os
import sys
import numpy as np
import cv2
import pupil_apriltags as apriltags
# Add parent directory to path so we can import from Systems and Occlusions
sys.path.append('..')
from Systems.BaseSystem import BaseSystem
from Logger import Logger, logger
from Occlusions.DepthThresholdOcclusion import DepthThresholdOcclusion
from Occlusions.SimpleOcclusion import SimpleOcclusion
from Tracker.ApriltagTracker import ApriltagTracker
from Interfaces.Frame import Frame

# Parameters
# Directory containing frame and depth images
frames_dir = os.path.join('..', 'LocalData', 'DepthIMUData2', 'fastslow')

# IMU data file (CSV)
imu_file = os.path.join(frames_dir, 'imu.csv')

# Camera matrix file (CSV)
camera_matrix_file = os.path.join('..', 'LocalData', 'camera_ipadpro.csv')

# Marker positions file (JSON)
marker_file = os.path.join('..', 'LocalData', 'DepthIMUData2', 'Env_3DModels', 'marker_poses_opengl.json')

# Directory containing 3D models for rendering
render_obj_dir = os.path.join('..', 'LocalData', 'Models', 'Scene1')

# Output directory for rendered results
output_dir = os.path.join('..', 'Output', 'tracker_example6')

# Output file prefix
output_prefix = 'tracked_frame'

# Camera distortion coefficients [k1, k2, p1, p2, k3]
dist_coeffs = np.zeros(5, dtype=np.float32)

# Size of AprilTag in meters
tag_size = 0.086  # 8.6cm

# AprilTag family to use
tag_family = 'tagStandard41h12'  # This needs to match the actual tags you're using

def load_camera_matrix(file_path):
    """
    Load camera matrix from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the camera matrix
        
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix
        
    Raises:
        FileNotFoundError: If the camera matrix file does not exist
        ValueError: If the camera matrix has an invalid format or shape
    """
    if not os.path.exists(file_path):
        logger.log(Logger.ERROR, f"Camera matrix file not found: {file_path}")
        raise FileNotFoundError(f"Camera matrix file not found: {file_path}")
    
    try:
        # Load the camera matrix from the CSV file
        matrix = np.loadtxt(file_path, delimiter=',')
        
        # Check if the loaded matrix has the correct shape
        if matrix.shape != (3, 3):
            logger.log(Logger.ERROR, f"Invalid camera matrix shape in {file_path}. Expected (3, 3), got {matrix.shape}")
            raise ValueError(f"Invalid camera matrix shape in {file_path}. Expected (3, 3), got {matrix.shape}")
        
        logger.log(Logger.SYSTEM, f"Loaded camera matrix from {file_path}")
        return matrix.astype(np.float32)
    except Exception as e:
        logger.log(Logger.ERROR, f"Failed to load camera matrix from {file_path}: {e}")
        raise

# Occlusion parameters
occlusion_threshold = 0.3  # For DepthThresholdOcclusion
max_depth = 5.0  # For SimpleOcclusion

# Logging parameters
log_keys = [Logger.SYSTEM, Logger.DEBUG, Logger.ERROR]
log_to_file = False
log_file_path = None


def process_frame_with_tracker(frame, tracker):
    """
    Process a single frame with the AprilTag tracker.
    
    Args:
        frame (Frame): The frame to process
        tracker (ApriltagTracker): The initialized tracker
        
    Returns:
        np.ndarray: The camera pose matrix (4x4)
    """
    # Track the frame to get camera pose
    camera_pose = tracker.track(frame)
    
    # Log the camera pose
    logger.log(Logger.DEBUG, f"Camera pose:\n{camera_pose}")
    
    return camera_pose


def main():
    """
    Example usage of the AprilTag Tracker with the Occlusion Framework.
    """
    
    # Allow user to override default parameters
    global frames_dir, marker_file, camera_matrix_file, render_obj_dir, output_dir, output_prefix, log_keys, tag_family, tag_size
    
    # user_frames_dir = input(f"Enter frames directory path (default: {frames_dir}): ").strip()
    # if user_frames_dir:
    #     frames_dir = user_frames_dir
    
    # user_marker_file = input(f"Enter marker positions file path (default: {marker_file}): ").strip()
    # if user_marker_file:
    #     marker_file = user_marker_file
    
    # user_camera_matrix_file = input(f"Enter camera matrix file path (default: {camera_matrix_file}): ").strip()
    # if user_camera_matrix_file:
    #     camera_matrix_file = user_camera_matrix_file
    
    # user_render_obj_dir = input(f"Enter render objects directory path (default: {render_obj_dir}): ").strip()
    # if user_render_obj_dir:
    #     render_obj_dir = user_render_obj_dir
    
    # user_output_dir = input(f"Enter output directory path (default: {output_dir}): ").strip()
    # if user_output_dir:
    #     output_dir = user_output_dir
    
    # user_output_prefix = input(f"Enter output file prefix (default: {output_prefix}): ").strip()
    # if user_output_prefix:
    #     output_prefix = user_output_prefix
    
    # AprilTag parameters
    # user_tag_family = input(f"Enter AprilTag family (e.g., tag36h11, tag25h9, tag16h5): ").strip()
    # if user_tag_family:
    #     tag_family = user_tag_family
    # logger.log(Logger.SYSTEM, f"Using AprilTag family: {tag_family}")
    
    # user_tag_size = input(f"Enter AprilTag size in meters (default: {tag_size}): ").strip()
    # if user_tag_size:
    #     try:
    #         tag_size = float(user_tag_size)
    #         logger.log(Logger.SYSTEM, f"Using AprilTag size: {tag_size} meters")
    #     except ValueError:
    #         logger.log(Logger.ERROR, f"Invalid tag size: {user_tag_size}. Using default: {tag_size}")
    
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create occlusion provider
    # You can choose between different occlusion providers:
    
    # 1. SimpleOcclusion - New provider that compares real camera depth with MR content depth
    occlusion_provider = SimpleOcclusion(max_depth=max_depth)
    
    # 2. DepthThresholdOcclusion - Original provider that uses a threshold on depth
    # occlusion_provider = DepthThresholdOcclusion(threshold=occlusion_threshold)
    
    # Load camera matrix from file
    try:
        camera_matrix = load_camera_matrix(camera_matrix_file)
    except Exception as e:
        logger.log(Logger.ERROR, f"Failed to load camera matrix: {e}")
        return
    
    # Initialize the AprilTag tracker with custom tag family
    # We need to modify the ApriltagTracker to accept tag_family parameter
    # For now, we'll directly modify the detector after initialization
    tracker = ApriltagTracker(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        tag_size=tag_size
    )
    
    # Override the default tag family in the detector
    tracker.detector = apriltags.Detector(
        families=tag_family,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    logger.log(Logger.SYSTEM, f"Initialized AprilTag detector with family: {tag_family}")
    
    # Initialize the tracker with marker positions
    if not tracker.initialize(marker_file):
        logger.log(Logger.ERROR, f"Failed to initialize tracker with marker positions from {marker_file}")
        return
    
    logger.log(Logger.SYSTEM, f"Successfully initialized AprilTag tracker with marker positions from {marker_file}")
    
    # Create and run the system with the tracker
    system = BaseSystem(
        data_dirs=[frames_dir],
        model_dirs=[render_obj_dir],
        output_dir=output_dir,
        output_prefix=output_prefix,
        occlusion_provider=occlusion_provider,
        log_keys=log_keys,
        log_to_file=log_to_file,
        log_file_path=log_file_path,
        tracker=tracker  # Pass the tracker to the BaseSystem
    )
    
    # Load data
    system.load_data()
    
    # Process each frame with the tracker
    logger.log(Logger.SYSTEM, "Processing frames with AprilTag tracker...")
    
    # Get frames in sorted order
    frames = system.data_loader.get_frames_sorted()
    
    for frame in frames:
        # Process the frame with the tracker to get camera pose
        # This is now redundant since the Renderer will use the tracker directly,
        # but we'll keep it for debugging purposes
        camera_pose = process_frame_with_tracker(frame, tracker)
    
    # Generate occlusion masks and render frames as in the original example
    system.generate_occlusion_masks()
    system.render_frames()
    
    logger.log(Logger.SYSTEM, "Processing complete. Results saved to: " + output_dir)
    logger.log(Logger.SYSTEM, "Note: This example demonstrates the use of AprilTag tracking")
    logger.log(Logger.SYSTEM, "      for camera pose estimation in the occlusion framework.")


if __name__ == '__main__':
    main()