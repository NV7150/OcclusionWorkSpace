import numpy as np
from typing import Tuple, Optional, Dict, Any
import math
from Utils.TransformUtils import look_at, perspective
from Utils.Logger import logger

class Camera:
    """
    Class for managing camera parameters and transformations.
    
    This class handles camera position, orientation, and projection,
    as well as generating view and projection matrices.
    """
    
    def __init__(self, position: Tuple[float, float, float] = (0, 0, 5),
                target: Tuple[float, float, float] = (0, 0, 0),
                up: Tuple[float, float, float] = (0, 1, 0),
                fov: float = 60.0,
                aspect: float = 1.0,
                near: float = 0.1,
                far: float = 100.0):
        """
        Initialize the Camera.
        
        Args:
            position: Camera position as (x, y, z)
            target: Camera target (look-at point) as (x, y, z)
            up: Camera up vector as (x, y, z)
            fov: Field of view in degrees
            aspect: Aspect ratio (width / height)
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        
        # Derived vectors
        self.forward = None
        self.right = None
        self.world_up = np.array(up, dtype=np.float32)
        
        # Matrices
        self.view_matrix = None
        self.projection_matrix = None
        
        # Camera parameters
        self.yaw = -90.0  # Yaw is initialized to -90 degrees since a yaw of 0 results in a direction vector pointing to the right
        self.pitch = 0.0
        self.movement_speed = 2.5
        self.mouse_sensitivity = 0.1
        
        # Update vectors and matrices
        self._update_camera_vectors()
        self._update_view_matrix()
        self._update_projection_matrix()
    
    def _update_camera_vectors(self) -> None:
        """
        Update the camera's coordinate vectors (forward, right, up) based on yaw and pitch.
        """
        # Calculate the new forward vector
        forward = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ], dtype=np.float32)
        
        # Normalize the vectors
        self.forward = forward / np.linalg.norm(forward)
        
        # Re-calculate the right and up vector
        self.right = np.cross(self.forward, self.world_up)
        self.right = self.right / np.linalg.norm(self.right)
        
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)
        
        # Update target
        self.target = self.position + self.forward
    
    def _update_view_matrix(self) -> None:
        """
        Update the view matrix based on the camera's position, target, and up vector.
        """
        self.view_matrix = look_at(
            tuple(self.position),
            tuple(self.target),
            tuple(self.up)
        )
    
    def _update_projection_matrix(self) -> None:
        """
        Update the projection matrix based on the camera's projection parameters.
        """
        self.projection_matrix = perspective(
            self.fov,
            self.aspect,
            self.near,
            self.far
        )
    
    def set_position(self, position: Tuple[float, float, float]) -> None:
        """
        Set the camera position.
        
        Args:
            position: Camera position as (x, y, z)
        """
        self.position = np.array(position, dtype=np.float32)
        self.target = self.position + self.forward
        self._update_view_matrix()
    
    def set_target(self, target: Tuple[float, float, float]) -> None:
        """
        Set the camera target (look-at point).
        
        Args:
            target: Camera target as (x, y, z)
        """
        self.target = np.array(target, dtype=np.float32)
        self.forward = self.target - self.position
        self.forward = self.forward / np.linalg.norm(self.forward)
        
        # Re-calculate the right and up vector
        self.right = np.cross(self.forward, self.world_up)
        self.right = self.right / np.linalg.norm(self.right)
        
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)
        
        # Update yaw and pitch
        self.yaw = math.degrees(math.atan2(self.forward[2], self.forward[0]))
        self.pitch = math.degrees(math.asin(self.forward[1]))
        
        self._update_view_matrix()
    
    def set_orientation(self, yaw: float, pitch: float) -> None:
        """
        Set the camera orientation using yaw and pitch angles.
        
        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
        """
        self.yaw = yaw
        self.pitch = pitch
        
        # Constrain pitch
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0
        
        self._update_camera_vectors()
        self._update_view_matrix()
    
    def set_projection(self, fov: float, aspect: float, near: float, far: float) -> None:
        """
        Set the camera projection parameters.
        
        Args:
            fov: Field of view in degrees
            aspect: Aspect ratio (width / height)
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self._update_projection_matrix()
    
    def get_view_matrix(self) -> np.ndarray:
        """
        Get the view matrix.
        
        Returns:
            4x4 view matrix
        """
        return self.view_matrix
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the projection matrix.
        
        Returns:
            4x4 projection matrix
        """
        return self.projection_matrix
    
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get the camera position.
        
        Returns:
            Camera position as (x, y, z)
        """
        return tuple(self.position)
    
    def get_target(self) -> Tuple[float, float, float]:
        """
        Get the camera target (look-at point).
        
        Returns:
            Camera target as (x, y, z)
        """
        return tuple(self.target)
    
    def get_up(self) -> Tuple[float, float, float]:
        """
        Get the camera up vector.
        
        Returns:
            Camera up vector as (x, y, z)
        """
        return tuple(self.up)
    
    def get_forward(self) -> Tuple[float, float, float]:
        """
        Get the camera forward vector.
        
        Returns:
            Camera forward vector as (x, y, z)
        """
        return tuple(self.forward)
    
    def get_right(self) -> Tuple[float, float, float]:
        """
        Get the camera right vector.
        
        Returns:
            Camera right vector as (x, y, z)
        """
        return tuple(self.right)
    
    def move_forward(self, distance: float) -> None:
        """
        Move the camera forward.
        
        Args:
            distance: Distance to move
        """
        self.position += self.forward * distance
        self.target = self.position + self.forward
        self._update_view_matrix()
    
    def move_backward(self, distance: float) -> None:
        """
        Move the camera backward.
        
        Args:
            distance: Distance to move
        """
        self.position -= self.forward * distance
        self.target = self.position + self.forward
        self._update_view_matrix()
    
    def move_right(self, distance: float) -> None:
        """
        Move the camera right.
        
        Args:
            distance: Distance to move
        """
        self.position += self.right * distance
        self.target = self.position + self.forward
        self._update_view_matrix()
    
    def move_left(self, distance: float) -> None:
        """
        Move the camera left.
        
        Args:
            distance: Distance to move
        """
        self.position -= self.right * distance
        self.target = self.position + self.forward
        self._update_view_matrix()
    
    def move_up(self, distance: float) -> None:
        """
        Move the camera up.
        
        Args:
            distance: Distance to move
        """
        self.position += self.up * distance
        self.target = self.position + self.forward
        self._update_view_matrix()
    
    def move_down(self, distance: float) -> None:
        """
        Move the camera down.
        
        Args:
            distance: Distance to move
        """
        self.position -= self.up * distance
        self.target = self.position + self.forward
        self._update_view_matrix()
    
    def process_keyboard_input(self, direction: str, delta_time: float) -> None:
        """
        Process keyboard input for camera movement.
        
        Args:
            direction: Direction to move ('forward', 'backward', 'left', 'right', 'up', 'down')
            delta_time: Time elapsed since last frame
        """
        velocity = self.movement_speed * delta_time
        
        if direction == 'forward':
            self.move_forward(velocity)
        elif direction == 'backward':
            self.move_backward(velocity)
        elif direction == 'left':
            self.move_left(velocity)
        elif direction == 'right':
            self.move_right(velocity)
        elif direction == 'up':
            self.move_up(velocity)
        elif direction == 'down':
            self.move_down(velocity)
    
    def process_mouse_movement(self, x_offset: float, y_offset: float, constrain_pitch: bool = True) -> None:
        """
        Process mouse movement for camera rotation.
        
        Args:
            x_offset: Mouse x offset
            y_offset: Mouse y offset
            constrain_pitch: Whether to constrain the pitch angle
        """
        x_offset *= self.mouse_sensitivity
        y_offset *= self.mouse_sensitivity
        
        self.yaw += x_offset
        self.pitch += y_offset
        
        # Constrain pitch
        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0
        
        self._update_camera_vectors()
        self._update_view_matrix()
    
    def process_mouse_scroll(self, y_offset: float) -> None:
        """
        Process mouse scroll for camera zoom.
        
        Args:
            y_offset: Mouse scroll y offset
        """
        if self.fov >= 1.0 and self.fov <= 45.0:
            self.fov -= y_offset
        
        if self.fov < 1.0:
            self.fov = 1.0
        if self.fov > 45.0:
            self.fov = 45.0
        
        self._update_projection_matrix()
    
    def set_from_matrix(self, view_matrix: np.ndarray) -> None:
        """
        Set camera parameters from a view matrix.
        
        Args:
            view_matrix: 4x4 view matrix
        """
        # Extract camera position
        inv_view = np.linalg.inv(view_matrix)
        self.position = inv_view[:3, 3]
        
        # Extract camera orientation
        self.forward = -inv_view[:3, 2]
        self.right = inv_view[:3, 0]
        self.up = inv_view[:3, 1]
        
        # Normalize vectors
        self.forward = self.forward / np.linalg.norm(self.forward)
        self.right = self.right / np.linalg.norm(self.right)
        self.up = self.up / np.linalg.norm(self.up)
        
        # Update target
        self.target = self.position + self.forward
        
        # Update yaw and pitch
        self.yaw = math.degrees(math.atan2(self.forward[2], self.forward[0]))
        self.pitch = math.degrees(math.asin(self.forward[1]))
        
        # Store the view matrix
        self.view_matrix = view_matrix
    
    def set_default(self) -> None:
        """
        Reset the camera to default position and orientation.
        """
        self.position = np.array([0, 0, 5], dtype=np.float32)
        self.target = np.array([0, 0, 0], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.world_up = np.array([0, 1, 0], dtype=np.float32)
        
        self.yaw = -90.0
        self.pitch = 0.0
        
        self._update_camera_vectors()
        self._update_view_matrix()
        logger.log(logger.DEBUG, "Camera reset to default position and orientation")