import numpy as np
import cv2
from typing import Dict, Any, Optional
from Utils.MarkerPositionLoader import MarkerPositionLoader
import pupil_apriltags as apriltags
from Interfaces.Tracker import Tracker
from Interfaces.Frame import Frame
from Logger import logger, Logger

class ApriltagTracker(Tracker):
    """
    ApriltagTracker uses the pupil-apriltags library to detect AprilTags in camera frames
    and calculate the camera pose based on the detected tags and their known positions.
    """
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray = None, tag_size: float = 0.05, tag_family: str = 'tag36h11'):
        """
        Initialize the AprilTag tracker.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients (optional)
            tag_size: Size of the AprilTag in meters (default: 5cm)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.tag_size = tag_size
        
        # Initialize the AprilTag detector
        self.detector = apriltags.Detector(
            families=tag_family,  # Default tag family
            nthreads=1,           # Number of threads
            quad_decimate=1.0,    # Image decimation factor
            quad_sigma=0.0,       # Gaussian blur sigma
            refine_edges=1,       # Refine edges
            decode_sharpening=0.25,  # Sharpening factor
            debug=0               # Debug level
        )
        
        # Dictionary to store marker positions
        self.marker_positions = {}
        
        # Last valid pose for fallback
        self.last_valid_pose = np.eye(4)
    
    def initialize(self, marker_positions_file: str) -> bool:
        """
        Initialize the tracker with marker positions from a JSON file.
        
        Args:
            marker_positions_file: Path to the JSON file containing marker positions
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.marker_positions = self.load_marker_positions(marker_positions_file)
            logger.log(Logger.DEBUG, f"Loaded {len(self.marker_positions)} marker positions from {marker_positions_file}")
            return True
        except Exception as e:
            logger.log(Logger.ERROR, f"Failed to initialize ApriltagTracker: {e}")
            return False
    
    def load_marker_positions(self, json_file_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load marker positions from a JSON file using the MarkerPositionLoader.
        
        The JSON format should be:
        {
            "{id}": {
                "pos": [x, y, z],
                "norm": [x, y, z],
                "tangent": [x, y, z]
            }
        }
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            Dictionary mapping marker IDs to their positions and normals
        """
        return MarkerPositionLoader.load_marker_positions(json_file_path)
    
    def track(self, frame: Frame) -> np.ndarray:
        """
        Track AprilTags in the frame and calculate the camera pose.
        
        Args:
            frame: Frame object containing the current camera frame data
            
        Returns:
            np.ndarray: 4x4 transformation matrix representing the camera pose
        """
        # Convert RGB image to grayscale for tag detection
        if frame.rgb.ndim == 3:
            gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.rgb
        
        # Detect AprilTags in the image
        detections = self.detector.detect(gray)
        
        if not detections:
            logger.log(Logger.DEBUG, "No AprilTags detected in frame")
            return self.last_valid_pose
        
        # Collect object points (3D points in world coordinate) and 
        # image points (2D points in image plane) for PnP solver
        object_points = []
        image_points = []
        
        for detection in detections:
            tag_id = detection.tag_id
            
            # Skip if we don't have the position for this tag
            if tag_id not in self.marker_positions:
                continue
            
            # Get the marker position in world coordinates
            marker_pos = self.marker_positions[tag_id]["pos"]
            marker_norm = self.marker_positions[tag_id]["norm"]
            marker_tangent = self.marker_positions[tag_id]["tangent"]
            
            # Calculate the corners of the tag in 3D space
            # We need to define the tag corners relative to the tag center
            # Assuming the tag is in the XY plane with Z aligned with the normal
            
            # First, create a coordinate system where Z is aligned with the normal
            z_axis = marker_norm / np.linalg.norm(marker_norm)
            
            # Use the provided tangent for x-axis
            # Make sure tangent is not parallel to z_axis
            if abs(np.dot(z_axis, marker_tangent)) > 0.99:
                logger.log(Logger.WARNING, f"Tangent for tag {tag_id} is nearly parallel to normal, using arbitrary vector")
                # Fall back to arbitrary vector method
                if abs(np.dot(z_axis, [1, 0, 0])) < 0.9:
                    temp = np.array([1, 0, 0])
                else:
                    temp = np.array([0, 1, 0])
                x_axis = np.cross(temp, z_axis)
            else:
                # Project tangent onto the plane perpendicular to z_axis
                x_axis = marker_tangent - np.dot(marker_tangent, z_axis) * z_axis
            
            # Normalize x_axis and create y_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Calculate the corners of the tag in 3D space
            half_size = self.tag_size / 2
            corners_3d = [
                marker_pos + (-half_size * x_axis - half_size * y_axis),  # Top-left
                marker_pos + (half_size * x_axis - half_size * y_axis),   # Top-right
                marker_pos + (half_size * x_axis + half_size * y_axis),   # Bottom-right
                marker_pos + (-half_size * x_axis + half_size * y_axis)   # Bottom-left
            ]
            
            # Get the corners of the tag in the image
            corners_2d = detection.corners
            
            # Add to our collection of points
            object_points.extend(corners_3d)
            image_points.extend(corners_2d)
        
        if not object_points:
            logger.log(Logger.DEBUG, "No known AprilTags found in frame")
            return self.last_valid_pose
        
        # Convert to numpy arrays
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Use solvePnP to get the rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            object_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            logger.log(Logger.WARNING, "Failed to solve PnP")
            return self.last_valid_pose
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create the 4x4 transformation matrix
        # The camera pose in world coordinates is the inverse of the object pose in camera coordinates
        # T_world_camera = inv(T_camera_world)
        T_camera_world = np.eye(4)
        T_camera_world[:3, :3] = R
        T_camera_world[:3, 3] = tvec.flatten()
        
        # Invert to get camera pose in world coordinates
        T_world_camera = np.linalg.inv(T_camera_world)
        
        # Store this pose as the last valid pose
        self.last_valid_pose = T_world_camera
        
        return T_world_camera