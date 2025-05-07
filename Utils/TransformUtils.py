import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional
import pyrr


class TransformUtils:
    """
    Utility class for 3D transformations.
    
    This class provides static methods for common 3D transformation operations,
    including conversions between different rotation representations, matrix operations,
    and coordinate system transformations.
    """
    
    @staticmethod
    def euler_to_quaternion(euler: Union[Dict[str, float], List[float], Tuple[float, float, float]]) -> pyrr.Quaternion:
        """
        Convert Euler angles to quaternion.
        
        Args:
            euler: Euler angles in degrees, can be:
                  - Dictionary with 'x', 'y', 'z' keys
                  - List/tuple of [x, y, z] values
                  
        Returns:
            Quaternion object
        """
        # Extract Euler angles in degrees
        if isinstance(euler, dict):
            euler_x = math.radians(euler.get('x', 0))
            euler_y = math.radians(euler.get('y', 0))
            euler_z = math.radians(euler.get('z', 0))
        else:
            euler_x = math.radians(euler[0])
            euler_y = math.radians(euler[1])
            euler_z = math.radians(euler[2])
        
        # Calculate quaternion components
        # This uses the ZYX rotation order (roll, pitch, yaw)
        cy = math.cos(euler_z * 0.5)
        sy = math.sin(euler_z * 0.5)
        cp = math.cos(euler_y * 0.5)
        sp = math.sin(euler_y * 0.5)
        cr = math.cos(euler_x * 0.5)
        sr = math.sin(euler_x * 0.5)
        
        # Create quaternion
        quat = pyrr.Quaternion([
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
            cr * cp * cy + sr * sp * sy   # w
        ])
        
        return quat
    
    @staticmethod
    def quaternion_to_euler(quat: Union[Dict[str, float], List[float], Tuple[float, float, float, float], pyrr.Quaternion]) -> Dict[str, float]:
        """
        Convert quaternion to Euler angles.
        
        Args:
            quat: Quaternion, can be:
                 - Dictionary with 'x', 'y', 'z', 'w' keys
                 - List/tuple of [x, y, z, w] values
                 - pyrr.Quaternion object
                 
        Returns:
            Dictionary with 'x', 'y', 'z' keys containing Euler angles in degrees
        """
        # Convert input to pyrr.Quaternion
        if isinstance(quat, dict):
            q = pyrr.Quaternion([quat['x'], quat['y'], quat['z'], quat['w']])
        elif isinstance(quat, pyrr.Quaternion):
            q = quat
        else:
            q = pyrr.Quaternion(quat)
        
        # Extract quaternion components
        x, y, z, w = q.x, q.y, q.z, q.w
        
        # Convert to Euler angles (roll, pitch, yaw) in ZYX order
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return {
            'x': math.degrees(roll),
            'y': math.degrees(pitch),
            'z': math.degrees(yaw)
        }
    
    @staticmethod
    def create_transform_matrix(position: Union[Dict[str, float], List[float], Tuple[float, float, float]],
                               rotation: Union[Dict[str, float], List[float], Tuple[float, float, float], pyrr.Quaternion] = None,
                               scale: Union[Dict[str, float], List[float], Tuple[float, float, float], float] = None) -> np.ndarray:
        """
        Create a 4x4 transformation matrix from position, rotation, and scale.
        
        Args:
            position: Position vector, can be:
                     - Dictionary with 'x', 'y', 'z' keys
                     - List/tuple of [x, y, z] values
            rotation: Rotation, can be:
                     - Dictionary with 'x', 'y', 'z' keys (Euler angles in degrees)
                     - Dictionary with 'x', 'y', 'z', 'w' keys (Quaternion)
                     - List/tuple of [x, y, z] values (Euler angles in degrees)
                     - List/tuple of [x, y, z, w] values (Quaternion)
                     - pyrr.Quaternion object
                     - None for no rotation
            scale: Scale, can be:
                  - Dictionary with 'x', 'y', 'z' keys
                  - List/tuple of [x, y, z] values
                  - Single float for uniform scale
                  - None for no scaling
                  
        Returns:
            4x4 transformation matrix as numpy array
        """
        # Create translation matrix
        if isinstance(position, dict):
            translation = pyrr.matrix44.create_from_translation([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
        else:
            translation = pyrr.matrix44.create_from_translation(position)
        
        # Create rotation matrix
        if rotation is None:
            rotation_matrix = pyrr.matrix44.create_identity()
        elif isinstance(rotation, dict):
            if 'w' in rotation:  # It's a quaternion
                quat = pyrr.Quaternion([rotation['x'], rotation['y'], rotation['z'], rotation['w']])
                rotation_matrix = pyrr.matrix44.create_from_quaternion(quat)
            else:  # It's Euler angles
                quat = TransformUtils.euler_to_quaternion(rotation)
                rotation_matrix = pyrr.matrix44.create_from_quaternion(quat)
        elif isinstance(rotation, pyrr.Quaternion):
            rotation_matrix = pyrr.matrix44.create_from_quaternion(rotation)
        elif len(rotation) == 3:  # It's Euler angles
            quat = TransformUtils.euler_to_quaternion(rotation)
            rotation_matrix = pyrr.matrix44.create_from_quaternion(quat)
        else:  # It's a quaternion as a list/tuple
            quat = pyrr.Quaternion(rotation)
            rotation_matrix = pyrr.matrix44.create_from_quaternion(quat)
        
        # Create scale matrix
        if scale is None:
            scale_matrix = pyrr.matrix44.create_identity()
        elif isinstance(scale, (int, float)):
            scale_matrix = pyrr.matrix44.create_from_scale([scale, scale, scale])
        elif isinstance(scale, dict):
            scale_matrix = pyrr.matrix44.create_from_scale([scale.get('x', 1), scale.get('y', 1), scale.get('z', 1)])
        else:
            scale_matrix = pyrr.matrix44.create_from_scale(scale)
        
        # Combine matrices: translation * rotation * scale
        return pyrr.matrix44.multiply(translation, pyrr.matrix44.multiply(rotation_matrix, scale_matrix))
    
    @staticmethod
    def decompose_transform_matrix(matrix: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Decompose a 4x4 transformation matrix into position, rotation, and scale.
        
        Args:
            matrix: 4x4 transformation matrix
            
        Returns:
            Tuple of (position, rotation, scale) dictionaries
        """
        # Extract position
        position = {
            'x': matrix[3, 0],
            'y': matrix[3, 1],
            'z': matrix[3, 2]
        }
        
        # Extract scale
        scale_x = np.linalg.norm(matrix[0:3, 0])
        scale_y = np.linalg.norm(matrix[0:3, 1])
        scale_z = np.linalg.norm(matrix[0:3, 2])
        
        scale = {
            'x': scale_x,
            'y': scale_y,
            'z': scale_z
        }
        
        # Remove scale from rotation part
        rotation_matrix = matrix.copy()
        rotation_matrix[0:3, 0] /= scale_x
        rotation_matrix[0:3, 1] /= scale_y
        rotation_matrix[0:3, 2] /= scale_z
        
        # Convert to quaternion
        quat = pyrr.Quaternion.from_matrix(rotation_matrix[0:3, 0:3])
        
        # Convert quaternion to Euler angles
        rotation = TransformUtils.quaternion_to_euler(quat)
        
        return position, rotation, scale
    
    @staticmethod
    def apply_transform(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply a transformation matrix to a set of points.
        
        Args:
            points: Nx3 array of points
            transform_matrix: 4x4 transformation matrix
            
        Returns:
            Transformed points as Nx3 array
        """
        # Convert to homogeneous coordinates
        homogeneous_points = np.ones((points.shape[0], 4))
        homogeneous_points[:, 0:3] = points
        
        # Apply transformation
        transformed_points = np.dot(homogeneous_points, transform_matrix.T)
        
        # Convert back to 3D coordinates
        return transformed_points[:, 0:3]
    
    @staticmethod
    def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
        """
        Create a view matrix for a camera looking at a target.
        
        Args:
            eye: Camera position as [x, y, z]
            target: Target position as [x, y, z]
            up: Up vector as [x, y, z], defaults to [0, 1, 0]
            
        Returns:
            4x4 view matrix
        """
        return pyrr.matrix44.create_look_at(eye, target, up)
    
    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
        """
        Create a perspective projection matrix.
        
        Args:
            fov_y: Vertical field of view in degrees
            aspect: Aspect ratio (width / height)
            near: Near clipping plane distance
            far: Far clipping plane distance
            
        Returns:
            4x4 perspective projection matrix
        """
        return pyrr.matrix44.create_perspective_projection(fov_y, aspect, near, far)
    
    @staticmethod
    def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
        """
        Create an orthographic projection matrix.
        
        Args:
            left: Left clipping plane
            right: Right clipping plane
            bottom: Bottom clipping plane
            top: Top clipping plane
            near: Near clipping plane
            far: Far clipping plane
            
        Returns:
            4x4 orthographic projection matrix
        """
        return pyrr.matrix44.create_orthogonal_projection(left, right, bottom, top, near, far)
    
    @staticmethod
    def interpolate_quaternions(q1: pyrr.Quaternion, q2: pyrr.Quaternion, t: float) -> pyrr.Quaternion:
        """
        Interpolate between two quaternions using spherical linear interpolation (SLERP).
        
        Args:
            q1: First quaternion
            q2: Second quaternion
            t: Interpolation parameter in range [0, 1]
            
        Returns:
            Interpolated quaternion
        """
        return pyrr.quaternion.slerp(q1, q2, t)