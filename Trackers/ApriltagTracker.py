import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from Utils.MarkerPositionLoader import MarkerPositionLoader
import pupil_apriltags as apriltags
from core.ITracker import ITracker
from DataLoaders.Frame import Frame
from Utils.Logger import logger, Logger
from Utils.PnP_viz import visualize_pnp_result

class ApriltagTracker(ITracker):
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
            tag_family: AprilTag family to detect (default: tag36h11)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.tag_size = tag_size
        
        # Initialize the AprilTag detector
        self.detector = apriltags.Detector(
            families=tag_family,
            nthreads=4,
            quad_decimate=0.5,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        # Dictionary to store marker positions
        self.marker_positions = {}
        
        # Last valid pose for fallback
        self.last_valid_pose = np.eye(4)
    
    def initialize(self, camera_matrix: np.ndarray, tag_size: float = 0.05, tag_family: str = 'tag36h11') -> None:
        """
        Initialize the tracker with camera parameters.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            tag_size: Size of the tags in meters
            tag_family: Family of tags to detect
        """
        self.camera_matrix = camera_matrix
        self.tag_size = tag_size
        
        # Initialize the AprilTag detector
        self.detector = apriltags.Detector(
            families=tag_family,
            nthreads=4,
            quad_decimate=0.5,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        logger.log(logger.SYSTEM, f"ApriltagTracker initialized with tag family {tag_family}, tag size {tag_size}m")
    
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
        self.marker_positions = MarkerPositionLoader.load_marker_positions(json_file_path)
        logger.log(logger.SYSTEM, f"Loaded {len(self.marker_positions)} marker positions from {json_file_path}")
        return self.marker_positions
    
    def detect_markers(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect markers in an image.
        
        Args:
            image: RGB or grayscale image
            
        Returns:
            Dictionary mapping marker IDs to marker data
        """
        # Convert RGB image to grayscale for tag detection
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect AprilTags in the image
        detections = self.detector.detect(gray)
        
        # Convert detections to dictionary
        markers = {}
        for detection in detections:
            tag_id = str(detection.tag_id)
            markers[tag_id] = {
                'corners': detection.corners,
                'center': detection.center,
                'decision_margin': detection.decision_margin,
                'hamming': detection.hamming
            }
        
        return markers
    
    def estimate_pose(self, image: np.ndarray, marker_positions: Dict[str, Dict[str, List[float]]]) -> Optional[np.ndarray]:
        """
        Estimate camera pose from an image and known marker positions.
        
        Args:
            image: RGB or grayscale image
            marker_positions: Dictionary mapping marker IDs to position data
            
        Returns:
            4x4 transformation matrix representing camera pose, or None if pose cannot be estimated
        """
        # Store marker positions
        self.marker_positions = marker_positions
        
        # Convert RGB image to grayscale for tag detection
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect AprilTags in the image
        detections = self.detector.detect(gray)
        
        if not detections:
            logger.log(logger.DEBUG, "No AprilTags detected in image")
            return None
        
        # Collect object points (3D points in world coordinate) and 
        # image points (2D points in image plane) for PnP solver
        object_points = []
        image_points = []
        
        for detection in detections:
            tag_id = int(detection.tag_id)
            
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
            
            # Use the provided tangent for y-axis
            # Make sure tangent is not parallel to z_axis
            if abs(np.dot(z_axis, marker_tangent)) > 0.99:
                logger.log(logger.WARNING, f"Tangent for tag {tag_id} is nearly parallel to normal, using arbitrary vector")
                # Fall back to arbitrary vector method
                if abs(np.dot(z_axis, [1, 0, 0])) < 0.9:
                    temp = np.array([1, 0, 0])
                else:
                    temp = np.array([0, 1, 0])
                y_axis = np.cross(z_axis, temp)  # Changed cross product order to get y-axis
            else:
                # Project tangent onto the plane perpendicular to z_axis
                y_axis = marker_tangent - np.dot(marker_tangent, z_axis) * z_axis
            
            # Normalize y_axis and create x_axis for a right-handed coordinate system
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)  # x = y × z for right-handed system
            
            # Calculate the corners of the tag in 3D space
            half_size = self.tag_size / 2
            corners_3d = [
                marker_pos + (-half_size * x_axis - half_size * y_axis),  # Bottom-left
                marker_pos + (half_size * x_axis - half_size * y_axis),   # Bottom-right
                marker_pos + (half_size * x_axis + half_size * y_axis),   # Top-right
                marker_pos + (-half_size * x_axis + half_size * y_axis)   # Top-left
            ]
            
            # Get the corners of the tag in the image
            corners_2d = detection.corners
            
            # Add to our collection of points
            object_points.extend(corners_3d)
            image_points.extend(corners_2d)
        
        if not object_points:
            logger.log(logger.DEBUG, "No known AprilTags found in image")
            return None
        
        # Convert each 3D point to OpenCV coordinate system
        object_points_opencv = []
        for point in object_points:
            # Apply the transformation to each point
            point_opencv = np.array([
                point[0],
                -point[1],  # Flip Y axis
                -point[2]   # Flip Z axis
            ])
            object_points_opencv.append(point_opencv)
        
        # Convert to numpy arrays
        object_points = np.array(object_points_opencv, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Replace solvePnP with solvePnPRansac for better robustness against outliers
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs,
            iterationsCount=200,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            logger.log(logger.WARNING, "Failed to solve PnP")
            return None
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = -R.T @ tvec
        
        opencv_to_opengl = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        R = opencv_to_opengl @ R
        t = opencv_to_opengl @ t
    
        T = np.eye(4)
        T[:3, :3] = R.T  # Transpose rotation
        T[:3, 3] = t.flatten()  # Translation
        
        ## for debugging
        if Logger.VIS_M in logger.enabled_log_keys:
            visualize_pnp_result(object_points, np.array([rvec, tvec]).flatten(), self.marker_positions)
        
        
        self.last_valid_pose = T
        
        return T
    
    def track(self, frame: Frame) -> np.ndarray:
        """
        Track AprilTags in the frame and calculate the camera pose.
        
        Args:
            frame: Frame object containing sensor data
            
        Returns:
            np.ndarray: 4x4 transformation matrix representing the camera pose
        """
        # Get RGB image from frame
        rgb_image = frame.get_rgb()
        
        if rgb_image is None:
            logger.log(logger.WARNING, f"No RGB image found in frame {frame.timestamp}")
            return self.last_valid_pose
        
        # Estimate pose using the RGB image
        pose = self.estimate_pose(rgb_image, self.marker_positions)
        
        if pose is None:
            logger.log(logger.WARNING, "Failed to estimate pose")
            return self.last_valid_pose
        
        return pose
    
    def track_features(self, prev_image: np.ndarray, curr_image: np.ndarray, prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track features from one image to another.
        
        Args:
            prev_image: Previous image
            curr_image: Current image
            prev_points: Points to track from previous image
            
        Returns:
            Tuple of (tracked_points, status), where status indicates which points were successfully tracked
        """
        # Convert images to grayscale if needed
        if prev_image.ndim == 3:
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev_image
            
        if curr_image.ndim == 3:
            curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = curr_image
        
        # Use Lucas-Kanade optical flow to track points
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        return next_points, status
    
    def get_camera_matrix(self) -> np.ndarray:
        """
        Get the camera intrinsic matrix.
        
        Returns:
            Camera intrinsic matrix
        """
        return self.camera_matrix