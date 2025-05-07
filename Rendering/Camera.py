import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import cv2

# Make sure we can import from parent directory
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Logger.Logger import Logger


class Camera:
    """
    Camera class for managing view and projection transformations.
    
    This class handles view and projection matrices for rendering, supporting
    both intrinsic parameters (focal length, principal point) for projection
    and extrinsic parameters (position, rotation) for the view transformation.
    """
    
    def __init__(self):
        """Initialize a new camera with default parameters."""
        self._logger = Logger()
        self._logger.debug("Creating Camera instance")
        
        # Default camera intrinsics (projection)
        self._fx = 500.0  # Focal length x
        self._fy = 500.0  # Focal length y
        self._cx = 320.0  # Principal point x
        self._cy = 240.0  # Principal point y
        self._width = 640  # Image width
        self._height = 480  # Image height
        self._near = 0.01  # Near clipping plane
        self._far = 100.0  # Far clipping plane
        
        # Default camera extrinsics (view)
        self._position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._rotation = np.eye(3, dtype=np.float32)  # Rotation matrix (identity = looking down -Z axis)
        
        # Cached matrices
        self._view_matrix = np.eye(4, dtype=np.float32)
        self._projection_matrix = np.eye(4, dtype=np.float32)
        self._need_update_view = True
        self._need_update_projection = True
    
    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float, 
                      width: int, height: int, near: float = 0.01, far: float = 100.0) -> None:
        """
        Set camera intrinsic parameters.
        
        Args:
            fx: Focal length in x direction (pixels)
            fy: Focal length in y direction (pixels)
            cx: Principal point x coordinate (pixels)
            cy: Principal point y coordinate (pixels)
            width: Image width in pixels
            height: Image height in pixels
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._width = width
        self._height = height
        self._near = near
        self._far = far
        self._need_update_projection = True
    
    def set_intrinsics_from_matrix(self, intrinsic_matrix: np.ndarray, width: int, height: int,
                                  near: float = 0.01, far: float = 100.0) -> None:
        """
        Set camera intrinsics from a standard 3x3 camera intrinsic matrix.
        
        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix
                [fx  0  cx]
                [0  fy  cy]
                [0   0   1]
            width: Image width in pixels
            height: Image height in pixels
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        if intrinsic_matrix.shape != (3, 3):
            self._logger.error(f"Invalid intrinsic matrix shape: {intrinsic_matrix.shape}, expected (3, 3)")
            return
        
        self._fx = intrinsic_matrix[0, 0]
        self._fy = intrinsic_matrix[1, 1]
        self._cx = intrinsic_matrix[0, 2]
        self._cy = intrinsic_matrix[1, 2]
        self._width = width
        self._height = height
        self._near = near
        self._far = far
        self._need_update_projection = True
    
    def set_position(self, position: np.ndarray) -> None:
        """
        Set camera position in world space.
        
        Args:
            position: 3D position vector [x, y, z]
        """
        if position.shape != (3,):
            self._logger.error(f"Invalid position shape: {position.shape}, expected (3,)")
            return
        
        self._position = position.copy()
        self._need_update_view = True
    
    def set_rotation(self, rotation: np.ndarray) -> None:
        """
        Set camera rotation as a 3x3 rotation matrix.
        
        Args:
            rotation: 3x3 rotation matrix
        """
        if rotation.shape != (3, 3):
            self._logger.error(f"Invalid rotation matrix shape: {rotation.shape}, expected (3, 3)")
            return
        
        self._rotation = rotation.copy()
        self._need_update_view = True
    
    def set_rotation_from_euler(self, euler_angles: np.ndarray) -> None:
        """
        Set camera rotation from Euler angles (in radians).
        
        Args:
            euler_angles: Euler angles [roll, pitch, yaw] in radians
        """
        if euler_angles.shape != (3,):
            self._logger.error(f"Invalid euler angles shape: {euler_angles.shape}, expected (3,)")
            return
        
        # Convert Euler angles to rotation matrix (using OpenCV)
        rotation_mat = cv2.Rodrigues(euler_angles)[0]
        self._rotation = rotation_mat
        self._need_update_view = True
    
    def set_extrinsics(self, rotation: np.ndarray, position: np.ndarray) -> None:
        """
        Set camera extrinsic parameters.
        
        Args:
            rotation: 3x3 rotation matrix
            position: 3D position vector [x, y, z]
        """
        self.set_rotation(rotation)
        self.set_position(position)
    
    def set_extrinsics_from_matrix(self, extrinsic_matrix: np.ndarray) -> None:
        """
        Set camera extrinsics from a 4x4 extrinsic matrix.
        
        Args:
            extrinsic_matrix: 4x4 extrinsic matrix
                [R  t]
                [0  1]
            where R is a 3x3 rotation matrix and t is a 3x1 translation vector
        """
        if extrinsic_matrix.shape != (4, 4):
            self._logger.error(f"Invalid extrinsic matrix shape: {extrinsic_matrix.shape}, expected (4, 4)")
            return
        
        self._rotation = extrinsic_matrix[:3, :3].copy()
        self._position = extrinsic_matrix[:3, 3].copy()
        self._need_update_view = True
    
    def get_view_matrix(self) -> np.ndarray:
        """
        Get the current view matrix (world-to-camera transformation).
        
        Returns:
            4x4 view matrix
        """
        if self._need_update_view:
            self._update_view_matrix()
        
        return self._view_matrix.copy()
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the current projection matrix.
        
        Returns:
            4x4 projection matrix
        """
        if self._need_update_projection:
            self._update_projection_matrix()
        
        return self._projection_matrix.copy()
    
    def _update_view_matrix(self) -> None:
        """Update the view matrix based on current position and rotation."""
        # Create view matrix (transform from world space to camera space)
        # View matrix = [R  -R*t]
        #               [0     1]
        # where R is the camera rotation and t is the camera position
        
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, :3] = self._rotation
        view_matrix[:3, 3] = -np.dot(self._rotation, self._position)
        
        self._view_matrix = view_matrix
        self._need_update_view = False
    
    def _update_projection_matrix(self) -> None:
        """Update the projection matrix based on current intrinsic parameters."""
        # Create OpenGL-style perspective projection matrix from camera intrinsics
        # Convert from OpenCV camera model (right-handed) to OpenGL (right-handed, but with -Z as forward)
        
        # Scale factors for normalized device coordinates
        ndc_x = 2.0 * self._fx / self._width
        ndc_y = 2.0 * self._fy / self._height
        
        # Principal point offset
        offset_x = (self._cx - self._width / 2.0) / self._width
        offset_y = (self._cy - self._height / 2.0) / self._height
        
        # Near and far planes
        n = self._near
        f = self._far
        
        # Build the projection matrix
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = ndc_x
        proj[1, 1] = ndc_y
        proj[2, 0] = 2.0 * offset_x
        proj[2, 1] = 2.0 * offset_y
        proj[2, 2] = -(f + n) / (f - n)
        proj[2, 3] = -2.0 * f * n / (f - n)
        proj[3, 2] = -1.0
        
        self._projection_matrix = proj
        self._need_update_projection = False
    
    def get_position(self) -> np.ndarray:
        """
        Get the camera position in world space.
        
        Returns:
            3D position vector [x, y, z]
        """
        return self._position.copy()
    
    def get_rotation(self) -> np.ndarray:
        """
        Get the camera rotation matrix.
        
        Returns:
            3x3 rotation matrix
        """
        return self._rotation.copy()
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Get the camera intrinsic matrix.
        
        Returns:
            3x3 intrinsic matrix
        """
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = self._fx
        intrinsic[1, 1] = self._fy
        intrinsic[0, 2] = self._cx
        intrinsic[1, 2] = self._cy
        return intrinsic
    
    def get_extrinsic_matrix(self) -> np.ndarray:
        """
        Get the camera extrinsic matrix.
        
        Returns:
            4x4 extrinsic matrix
        """
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = self._rotation
        extrinsic[:3, 3] = self._position
        return extrinsic