import numpy as np
import cv2
from typing import Dict, Any, Optional

import pandas as pd
from Utils.MarkerPositionLoader import MarkerPositionLoader
import pupil_apriltags as apriltags
from Interfaces.Tracker import Tracker
from Interfaces.Frame import Frame
from Logger import logger, Logger
from Utils.PnP_viz import visualize_pnp_result

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
            z_axis =  marker_norm / np.linalg.norm(marker_norm)
            
            # Use the provided tangent for y-axis
            # Make sure tangent is not parallel to z_axis
            if abs(np.dot(z_axis, marker_tangent)) > 0.99:
                logger.log(Logger.WARNING, f"Tangent for tag {tag_id} is nearly parallel to normal, using arbitrary vector")
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
            
            # It is from bottom-left(?) (https://github.com/pupil-labs/apriltags/issues/33)
            half_size = self.tag_size / 2
            corners_3d = [
                marker_pos + (-half_size * x_axis - half_size * y_axis),     # Bottom-left
                marker_pos + (half_size * x_axis - half_size * y_axis),     # Bottom-right
                marker_pos + (half_size * x_axis + half_size * y_axis),     # Top-right
                marker_pos + (-half_size * x_axis + half_size * y_axis)    # Top-left
            ]
            # Get the corners of the tag in the image
            corners_2d = detection.corners
            logger.log(Logger.DEBUG, f"Tag ID: {tag_id}, 2D corners: {corners_2d}")
            
            # Add to our collection of points
            object_points.extend(corners_3d)
            image_points.extend(corners_2d)
        
        if not object_points:
            logger.log(Logger.DEBUG, "No known AprilTags found in frame")
            return self.last_valid_pose

        # Convert each 3D point to OpenCV coordinate system
        object_points_opencv = []
        for point in object_points:
            # Apply the transformation to each point
            point_opencv = np.array([
                point[0],
                -point[1],  # Flip Y axis
                -point[2]   # Flip Z axis
            ])
            logger.log(Logger.DEBUG, f"Point in OpenGL: {point}, converted to OpenCV: {point_opencv}")
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
            iterationsCount=100,           # RANSAC iterations
            reprojectionError=8.0,         # Maximum allowed reprojection error (pixels)
            confidence=0.99,               # Confidence probability
            flags=cv2.SOLVEPNP_ITERATIVE  # Same flag as before
        )
        
        # Log how many inliers were found (useful for debugging)
        if success:
            inlier_count = len(inliers) if inliers is not None else 0
            logger.log(Logger.DEBUG, f"PnP solved with {inlier_count}/{len(object_points)} inliers")

        ##### for debgug: visualize_points #####
        
        # 画像のコピーを作成してデバッグ用の可視化を行う
        debug_image = frame.rgb.copy()
        
        # グレースケールの場合はカラー画像に変換
        if debug_image.ndim == 2:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            
        # 検出されたポイントを描画
        for i, point in enumerate(image_points):
            # 点を赤い円で表示
            cv2.circle(debug_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            # ポイント番号を表示
            cv2.putText(debug_image, str(i), (int(point[0]) + 5, int(point[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # AprilTagごとに4つの点を線で結ぶ
        for i in range(0, len(image_points), 4):
            if i + 3 < len(image_points):
                quad_points = image_points[i:i+4]
                
                # 四角形の輪郭を青い線で描画
                for j in range(4):
                    pt1 = (int(quad_points[j][0]), int(quad_points[j][1]))
                    pt2 = (int(quad_points[(j+1)%4][0]), int(quad_points[(j+1)%4][1]))
                    cv2.line(debug_image, pt1, pt2, (255, 0, 0), 2)
                
                # タグの中心を計算
                center_x = int(sum(p[0] for p in quad_points) / 4)
                center_y = int(sum(p[1] for p in quad_points) / 4)
                tag_index = i // 4
                
                # 検出されたタグ番号を表示
                cv2.putText(debug_image, f"Tag {tag_index}", (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # デバッグ画像を表示
        cv2.imshow("AprilTag Detection", debug_image)
        cv2.waitKey(1)  # 1ms待機（画像を表示するために必要）
        
        # Visualize the PnP result
        unix_time = str(int(pd.Timestamp(frame.timestamp).timestamp()))
        print(unix_time[-3:])
        if unix_time[-3:] == "463" or unix_time[-3:] == "476":
            visualize_pnp_result(object_points, np.vstack([rvec, tvec]))
        
        ####### end debug ########
        
        if not success:
            logger.log(Logger.WARNING, "Failed to solve PnP")
            return self.last_valid_pose
        
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        opencv_to_opengl = np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],  # Flip Y axis
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0]# Flip Z axis
        ])

        # Create the camera-to-world transformation in OpenCV coordinates
        # T_camera_world_opencv = np.eye(4)
        # T_camera_world_opencv[:3, :3] = R.T @ opencv_to_opengl[:3, :3]
        # tvec_opengl = (-R.T @ tvec).flatten()
        # # tvec_opengl[1:] *= -1
        # T_camera_world_opencv[:3, 3] = tvec_opengl
        
        T_camera_world_opencv = np.eye(4)
        T_camera_world_opencv[:3, :3] = R.T 
        tvec_opengl = (-R.T @ tvec).flatten()
        T_camera_world_opencv[:3, 3] = tvec_opengl
        
        # Apply the coordinate transformation to the rotation matrix
        # T_camera_opengl = T_camera_world_opencv @ opencv_to_opengl

        # Store this pose as the last valid pose
        self.last_valid_pose = T_camera_world_opencv
        
        return T_camera_world_opencv
        
        # self.last_valid_pose = T_camera_world_opencv
        
        # return T_camera_world_opencv