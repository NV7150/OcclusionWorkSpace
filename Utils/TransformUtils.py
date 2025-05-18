import numpy as np
import math
from typing import Tuple, List, Dict, Union, Optional

def euler_to_quaternion(euler: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles in radians (roll, pitch, yaw)
        
    Returns:
        Quaternion as (x, y, z, w)
    """
    roll, pitch, yaw = euler
    
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (x, y, z, w)

def quaternion_to_euler(quaternion: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles.
    
    Args:
        quaternion: Quaternion as (x, y, z, w)
        
    Returns:
        Euler angles in radians (roll, pitch, yaw)
    """
    x, y, z, w = quaternion
    
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
    
    return (roll, pitch, yaw)

def create_rotation_matrix(quaternion: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Create a 3x3 rotation matrix from a quaternion.
    
    Args:
        quaternion: Quaternion as (x, y, z, w)
        
    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = quaternion
    
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    
    yy = y * y
    yz = y * z
    yw = y * w
    
    zz = z * z
    zw = z * w
    
    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
    ])
    
    return rotation_matrix

def create_transformation_matrix(position: Tuple[float, float, float], 
                                quaternion: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Create a 4x4 transformation matrix from position and quaternion.
    
    Args:
        position: Position as (x, y, z)
        quaternion: Quaternion as (x, y, z, w)
        
    Returns:
        4x4 transformation matrix
    """
    rotation_matrix = create_rotation_matrix(quaternion)
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    
    return transformation_matrix

def decompose_transformation_matrix(matrix: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Decompose a 4x4 transformation matrix into position and quaternion.
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        Tuple of (position, quaternion), where position is (x, y, z) and quaternion is (x, y, z, w)
    """
    position = tuple(matrix[:3, 3])
    
    rotation_matrix = matrix[:3, :3]
    
    trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s
    
    return (position, (x, y, z, w))

def look_at(eye: Tuple[float, float, float], 
           target: Tuple[float, float, float], 
           up: Tuple[float, float, float] = (0, 1, 0)) -> np.ndarray:
    """
    Create a view matrix for a camera looking at a target.
    
    Args:
        eye: Camera position as (x, y, z)
        target: Target position as (x, y, z)
        up: Up vector as (x, y, z), default is (0, 1, 0)
        
    Returns:
        4x4 view matrix
    """
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)
    
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    
    view_matrix = np.eye(4)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = new_up
    view_matrix[:3, 2] = -forward
    view_matrix[:3, 3] = eye
    
    return np.linalg.inv(view_matrix)

def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Create a perspective projection matrix.
    
    Args:
        fov: Field of view in degrees
        aspect: Aspect ratio (width / height)
        near: Near clipping plane distance
        far: Far clipping plane distance
        
    Returns:
        4x4 perspective projection matrix
    """
    fov_rad = math.radians(fov)
    f = 1.0 / math.tan(fov_rad / 2.0)
    
    projection_matrix = np.zeros((4, 4))
    projection_matrix[0, 0] = f / aspect
    projection_matrix[1, 1] = f
    projection_matrix[2, 2] = (far + near) / (near - far)
    projection_matrix[2, 3] = (2.0 * far * near) / (near - far)
    projection_matrix[3, 2] = -1.0
    
    return projection_matrix

def transform_point(point: Tuple[float, float, float], matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Transform a point using a 4x4 transformation matrix.
    
    Args:
        point: Point as (x, y, z)
        matrix: 4x4 transformation matrix
        
    Returns:
        Transformed point as (x, y, z)
    """
    point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
    transformed_point = matrix @ point_homogeneous
    
    if transformed_point[3] != 0:
        transformed_point = transformed_point / transformed_point[3]
    
    return (transformed_point[0], transformed_point[1], transformed_point[2])

def interpolate_quaternions(q1: Tuple[float, float, float, float], 
                           q2: Tuple[float, float, float, float], 
                           t: float) -> Tuple[float, float, float, float]:
    """
    Interpolate between two quaternions using spherical linear interpolation (SLERP).
    
    Args:
        q1: First quaternion as (x, y, z, w)
        q2: Second quaternion as (x, y, z, w)
        t: Interpolation parameter in [0, 1]
        
    Returns:
        Interpolated quaternion as (x, y, z, w)
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # Calculate cosine of angle between quaternions
    dot = np.sum(q1 * q2)
    
    # If dot product is negative, negate one of the quaternions to take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot product to [-1, 1] range
    dot = min(max(dot, -1.0), 1.0)
    
    # Calculate interpolation parameters
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    
    if sin_theta < 1e-6:
        # If quaternions are very close, use linear interpolation
        return tuple((1.0 - t) * q1 + t * q2)
    
    # Use spherical linear interpolation
    s1 = math.sin((1.0 - t) * theta) / sin_theta
    s2 = math.sin(t * theta) / sin_theta
    
    result = s1 * q1 + s2 * q2
    return tuple(result)