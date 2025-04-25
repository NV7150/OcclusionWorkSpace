import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import pyrr
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pyassimp
import pyassimp.postprocess
from PIL import Image

from Logger import logger, Logger


class VisualizeModelRender:
    """
    VisualizeModelRender is responsible for rendering the MR scene visualization.
    It handles rendering the scene model, reference markers, camera positions, and MR contents.
    """
    
    def __init__(self):
        """
        Initialize the VisualizeModelRender.
        """
        self.scene_model = None
        self.marker_positions = {}
        self.camera_poses = {}
        self.models = {}
        self.scenes = {}
        
        # Window properties
        self.window_width = 1024
        self.window_height = 768
        self.window_title = b"MR Scene Visualization"
        self.window_id = None
        
        # Camera properties
        self.camera_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Mouse interaction
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_button = -1
        self.mouse_state = GLUT_UP
        
        # Rotation and zoom
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 1.0
        
        # Keyboard state
        self.keys = {}
        
        # Visualization options
        self.show_markers = True
        self.show_cameras = True
        self.show_contents = True
        self.show_scene = True
        self.show_grid = True
        self.show_axes = True
        
        # Current view mode
        self.view_mode = "free"  # "free", "camera", "marker"
        self.current_camera_timestamp = None
        self.current_marker_id = None
        
        # Model cache
        self.model_cache = {}
    
    def initialize(self, scene_model_path: str):
        """
        Initialize the renderer with the scene model.
        
        Args:
            scene_model_path: Path to the 3D scan model (.fbx) of the scene
        """
        # Initialize GLUT
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.window_width, self.window_height)
        self.window_id = glutCreateWindow(self.window_title)
        
        # Set up OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        
        # Set up light
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Set up callbacks
        glutDisplayFunc(self._display_callback)
        glutReshapeFunc(self._reshape_callback)
        glutMouseFunc(self._mouse_callback)
        glutMotionFunc(self._motion_callback)
        glutKeyboardFunc(self._keyboard_callback)
        glutSpecialFunc(self._special_key_callback)
        
        # Load scene model
        self._load_scene_model(scene_model_path)
        
        logger.log(Logger.SYSTEM, "Renderer initialized")
    
    def _load_scene_model(self, model_path: str):
        """
        Load the scene model from a file.
        
        Args:
            model_path: Path to the model file
        """
        logger.log(Logger.SYSTEM, f"Loading scene model from {model_path}")
        
        try:
            # Use pyassimp to load the model
            processing_flags = (
                pyassimp.postprocess.aiProcess_Triangulate |
                pyassimp.postprocess.aiProcess_GenNormals
            )
            
            # Use with statement to properly handle the context manager
            with pyassimp.load(model_path, processing=processing_flags) as scene:
                if not scene or not scene.meshes:
                    logger.log(Logger.ERROR, f"No meshes found in {model_path}")
                    self._create_fallback_scene()
                    return
                
                # Create a copy of the scene data we need
                self.scene_model = scene
                logger.log(Logger.SYSTEM, f"Scene model loaded with {len(scene.meshes)} meshes")
        except Exception as e:
            logger.log(Logger.ERROR, f"Error loading scene model: {e}")
            self._create_fallback_scene()
    
    def _create_fallback_scene(self):
        """
        Create a fallback scene if the model loading fails.
        """
        logger.log(Logger.WARNING, "Creating fallback scene")
        # We'll create a simple grid as a fallback
        self.scene_model = None
    
    def setup_scene(self, marker_positions: Dict, camera_poses: Dict, models: Dict, scenes: Dict):
        """
        Set up the scene with marker positions, camera poses, and MR contents.
        
        Args:
            marker_positions: Dictionary of marker positions
            camera_poses: Dictionary of camera poses
            models: Dictionary of MR content models
            scenes: Dictionary of scene descriptions
        """
        self.marker_positions = marker_positions
        self.camera_poses = camera_poses
        self.models = models
        self.scenes = scenes
        
        logger.log(Logger.SYSTEM, "Scene setup complete")
    
    def start_render_loop(self):
        """
        Start the rendering loop.
        """
        logger.log(Logger.SYSTEM, "Starting render loop")
        glutMainLoop()
    
    def set_camera_view(self, timestamp):
        """
        Set the view to a specific camera pose.
        
        Args:
            timestamp: Timestamp of the camera pose to use
        """
        if timestamp in self.camera_poses:
            self.view_mode = "camera"
            self.current_camera_timestamp = timestamp
            glutPostRedisplay()
    
    def set_marker_view(self, marker_id):
        """
        Set the view to a specific marker.
        
        Args:
            marker_id: ID of the marker to view
        """
        if marker_id in self.marker_positions:
            self.view_mode = "marker"
            self.current_marker_id = marker_id
            glutPostRedisplay()
    
    def set_free_view(self):
        """
        Set the view to free navigation mode.
        """
        self.view_mode = "free"
        glutPostRedisplay()
    
    def _display_callback(self):
        """
        GLUT display callback function.
        """
        # Clear the screen
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.window_width / self.window_height, 0.1, 100.0)
        
        # Set up the modelview matrix based on view mode
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        if self.view_mode == "camera" and self.current_camera_timestamp in self.camera_poses:
            # Use the selected camera pose
            camera_pose = self.camera_poses[self.current_camera_timestamp]
            # Invert the camera pose to get the view matrix
            view_matrix = np.linalg.inv(camera_pose)
            glMultMatrixf(view_matrix.flatten('F'))
        elif self.view_mode == "marker" and self.current_marker_id in self.marker_positions:
            # Look at the selected marker
            marker_pos = self.marker_positions[self.current_marker_id]["pos"]
            marker_norm = self.marker_positions[self.current_marker_id]["norm"]
            # Position the camera along the marker normal
            eye = marker_pos + marker_norm * 0.5
            gluLookAt(eye[0], eye[1], eye[2], marker_pos[0], marker_pos[1], marker_pos[2], 0, 1, 0)
        else:
            # Free view mode
            # Apply zoom
            eye = self.camera_pos * self.zoom
            gluLookAt(eye[0], eye[1], eye[2], 
                      self.camera_target[0], self.camera_target[1], self.camera_target[2], 
                      self.camera_up[0], self.camera_up[1], self.camera_up[2])
            
            # Apply rotation
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 1, 0)
        
        # Draw the scene
        self._draw_scene()
        
        # Swap buffers
        glutSwapBuffers()
    
    def _draw_scene(self):
        """
        Draw the entire scene.
        """
        # Draw coordinate axes
        if self.show_axes:
            self._draw_axes()
        
        # Draw grid
        if self.show_grid:
            self._draw_grid()
        
        # Draw scene model
        if self.show_scene:
            self._draw_scene_model()
        
        # Draw markers
        if self.show_markers:
            self._draw_markers()
        
        # Draw camera positions
        if self.show_cameras:
            self._draw_cameras()
        
        # Draw MR contents
        if self.show_contents:
            self._draw_contents()
    
    def _draw_axes(self):
        """
        Draw coordinate axes.
        """
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glEnd()
        
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glEnd()
        
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def _draw_grid(self):
        """
        Draw a reference grid.
        """
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1.0)
        
        grid_size = 5
        grid_step = 0.2
        
        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            # X lines
            glVertex3f(i * grid_step, 0, -grid_size * grid_step)
            glVertex3f(i * grid_step, 0, grid_size * grid_step)
            
            # Z lines
            glVertex3f(-grid_size * grid_step, 0, i * grid_step)
            glVertex3f(grid_size * grid_step, 0, i * grid_step)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def _draw_scene_model(self):
        """
        Draw the scene model.
        """
        if self.scene_model is None:
            return
        
        glPushMatrix()
        
        # Set material properties for the scene model
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
        
        # Draw each mesh in the scene
        for mesh in self.scene_model.meshes:
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for vertex_idx in face:
                    # Set normal
                    if mesh.normals.size > 0:
                        normal = mesh.normals[vertex_idx]
                        glNormal3f(normal[0], normal[1], normal[2])
                    
                    # Set vertex
                    vertex = mesh.vertices[vertex_idx]
                    glVertex3f(vertex[0], vertex[1], vertex[2])
            glEnd()
        
        glPopMatrix()
    
    # def _draw_markers(self):
    #     """
    #     Draw the reference markers.
    #     """
    #     for marker_id, marker_data in self.marker_positions.items():
    #         pos = marker_data["pos"]
    #         norm = marker_data["norm"]
    #         tangent = marker_data["tangent"]
            
    #         glPushMatrix()
            
    #         # Translate to marker position
    #         glTranslatef(pos[0], pos[1], pos[2])
            
    #         # Draw marker as a small cube
    #         glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.0, 1.0])
    #         glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [1.0, 1.0, 0.0, 1.0])
    #         glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    #         glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0)
            
    #         # Draw a cube for the marker
    #         glutSolidCube(0.05)
            
    #         # Draw normal vector as an arrow
    #         glDisable(GL_LIGHTING)
    #         glLineWidth(2.0)
            
    #         # Normal vector (blue)
    #         glColor3f(0.0, 0.0, 1.0)
    #         glBegin(GL_LINES)
    #         glVertex3f(0.0, 0.0, 0.0)
    #         glVertex3f(norm[0] * 0.2, norm[1] * 0.2, norm[2] * 0.2)
    #         glEnd()
            
    #         # Tangent vector (red)
    #         glColor3f(1.0, 0.0, 0.0)
    #         glBegin(GL_LINES)
    #         glVertex3f(0.0, 0.0, 0.0)
    #         glVertex3f(tangent[0] * 0.2, tangent[1] * 0.2, tangent[2] * 0.2)
    #         glEnd()
            
    #         # Calculate bitangent (cross product of normal and tangent)
    #         bitangent = np.cross(norm, tangent)
            
    #         # Bitangent vector (green)
    #         glColor3f(0.0, 1.0, 0.0)
    #         glBegin(GL_LINES)
    #         glVertex3f(0.0, 0.0, 0.0)
    #         glVertex3f(bitangent[0] * 0.2, bitangent[1] * 0.2, bitangent[2] * 0.2)
    #         glEnd()
            
    #         glEnable(GL_LIGHTING)
            
    #         glPopMatrix()
    def _draw_markers(self):
        """
        Draw the reference markers.
        """
        for marker_id, marker_data in self.marker_positions.items():
            pos = marker_data["pos"]
            norm = marker_data["norm"]
            tangent = marker_data["tangent"]
            
            glPushMatrix()
            
            # Translate to marker position
            glTranslatef(pos[0], pos[1], pos[2])
            
            # Draw marker as a small cube
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.0, 1.0])
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [1.0, 1.0, 0.0, 1.0])
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0)
            
            # Draw a cube for the marker
            glutSolidCube(0.05)
            
            # Draw coordinate vectors
            glDisable(GL_LIGHTING)
            glLineWidth(2.0)
            
            # Disable depth testing to ensure all lines are visible
            glDisable(GL_DEPTH_TEST)
            
            # Calculate bitangent (cross product of normal and tangent)
            # Normalize to ensure consistent length with other vectors
            bitangent = np.cross(tangent, norm)
            bitangent_length = np.linalg.norm(bitangent)
            if bitangent_length > 0.001:  # Check for non-zero vector
                bitangent = bitangent / bitangent_length
            
            # Normal vector (blue) - draw last to avoid occlusion
            glColor3f(0.0, 0.0, 1.0)
            glBegin(GL_LINES)
            # Start from slightly offset position for better visibility
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(norm[0] * 0.2, norm[1] * 0.2, norm[2] * 0.2)
            glEnd()
            
            # Bitangent vector (green) - draw second
            glColor3f(0.0, 1.0, 0.0)
            glBegin(GL_LINES)
            # Start from slightly offset position for better visibility
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(bitangent[0] * 0.2, bitangent[1] * 0.2, bitangent[2] * 0.2)
            glEnd()
            
            # Tangent vector (red) - draw first
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(tangent[0] * 0.2, tangent[1] * 0.2, tangent[2] * 0.2)
            glEnd()
            
            x_axis = bitangent
            y_axis = tangent
            half_size = 0.086 / 2
            
            corners_3d = [
                (-half_size * x_axis - half_size * y_axis),     # Bottom-left
                (half_size * x_axis - half_size * y_axis),     # Bottom-right
                (half_size * x_axis + half_size * y_axis),     # Top-right
                (-half_size * x_axis + half_size * y_axis)    # Top-left
            ]
            # After calculating corners_3d

            # Define colors for each corner
            corner_colors = [
                [1.0, 0.0, 0.0, 1.0],  # Red - Top-left
                [0.0, 1.0, 0.0, 1.0],  # Green - Top-right
                [0.0, 0.0, 1.0, 1.0],  # Blue - Bottom-right
                [1.0, 1.0, 0.0, 1.0]   # Yellow - Bottom-left
            ]

            # Draw a small cube at each corner position
            cube_size = 0.01  # Size of the debug cubes

            # Enable lighting for the cubes
            glEnable(GL_LIGHTING)

            for i, corner in enumerate(corners_3d):
                glPushMatrix()
                
                # Move to the corner position (add marker position since corners are relative)
                corner_pos = corner
                glTranslatef(corner_pos[0], corner_pos[1], corner_pos[2])
                
                # Set material for this corner cube
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, corner_colors[i])
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0)
                
                # Draw a small cube
                glutSolidCube(cube_size)
                
                # Label the corner with its index (optional)
                glDisable(GL_LIGHTING)
                glColor3f(1.0, 1.0, 1.0)  # White text
                glRasterPos3f(0, cube_size, 0)
                
                # Draw the text using bitmap characters
                corner_text = str(i)
                for c in corner_text:
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))
                    
                glTranslatef(-corner_pos[0], -corner_pos[1], -corner_pos[2])
                
                glEnable(GL_LIGHTING)
                glPopMatrix()

            # Disable lighting again for the axes
            glDisable(GL_LIGHTING)
            
            
            # Re-enable depth testing for other rendering
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            
            glPopMatrix()
    
    # def _draw_cameras(self):
    #     """
    #     Draw the camera positions and view frustums.
    #     """
        
    #     for timestamp, pose in self.camera_poses.items():
    #         # Extract camera position from the pose matrix
    #         # The camera position is the negative of the translation vector
    #         # transformed by the rotation matrix
    #         rotation = pose[:3, :3]
    #         translation = pose[:3, 3]
    #         camera_pos = -np.dot(rotation.T, translation)
            
    #         glPushMatrix()
            
    #         # Translate to camera position
    #         glTranslatef(camera_pos[0], camera_pos[1], camera_pos[2])
            
    #         # Apply camera orientation
    #         # The rotation matrix in the pose is the inverse of the camera orientation
    #         camera_orientation = rotation.T
    #         orientation_matrix = np.eye(4)
    #         orientation_matrix[:3, :3] = camera_orientation
    #         glMultMatrixf(orientation_matrix.flatten('F'))
            
    #         # Draw camera as a small pyramid
    #         glDisable(GL_LIGHTING)
            
    #         # Camera body (cyan)
    #         glColor3f(0.0, 0.8, 0.8)
            
    #         # Draw camera frustum
    #         glBegin(GL_LINES)
    #         # Front face
    #         glVertex3f(0, 0, 0)
    #         glVertex3f(0.1, 0.1, -0.2)
            
    #         glVertex3f(0, 0, 0)
    #         glVertex3f(-0.1, 0.1, -0.2)
            
    #         glVertex3f(0, 0, 0)
    #         glVertex3f(-0.1, -0.1, -0.2)
            
    #         glVertex3f(0, 0, 0)
    #         glVertex3f(0.1, -0.1, -0.2)
            
    #         # Back face
    #         glVertex3f(0.1, 0.1, -0.2)
    #         glVertex3f(-0.1, 0.1, -0.2)
            
    #         glVertex3f(-0.1, 0.1, -0.2)
    #         glVertex3f(-0.1, -0.1, -0.2)
            
    #         glVertex3f(-0.1, -0.1, -0.2)
    #         glVertex3f(0.1, -0.1, -0.2)
            
    #         glVertex3f(0.1, -0.1, -0.2)
    #         glVertex3f(0.1, 0.1, -0.2)
    #         glEnd()
            
    #         # Draw viewing direction
    #         glColor3f(1.0, 1.0, 1.0)
    #         glBegin(GL_LINES)
    #         glVertex3f(0, 0, 0)
    #         glVertex3f(0, 0, -0.3)
    #         glEnd()
            
    #         glEnable(GL_LIGHTING)
            
    #         glPopMatrix()
    def _draw_cameras(self):
        """
        Draw the camera positions and view frustums with sequential numbering based on timestamps.
        """
        
        # Sort timestamps to ensure cameras are numbered in chronological order
        sorted_timestamps = sorted(self.camera_poses.keys())
        
        # Draw each camera in timestamp order
        for i, timestamp in enumerate(sorted_timestamps):
            pose = self.camera_poses[timestamp]

            # Extract camera position from the pose matrix
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            # camera_pos = -np.dot(rotation.T, translation)
            camera_pos = translation       
                 
            logger.log(Logger.DEBUG, f"pose{str(int(pd.Timestamp(timestamp).timestamp()))} in renderer :{translation}, {rotation}")
            
            
            glPushMatrix()      
            camera_orientation = rotation
            orientation_matrix = np.eye(4)
            orientation_matrix[:3, :3] = camera_orientation

            glMultMatrixf(orientation_matrix.flatten('F'))
            # glLoadIdentity()
            
            # Translate to camera position
            glTranslatef(camera_pos[0], camera_pos[1], camera_pos[2])

            
            # Apply camera orientation
            # camera_orientation = rotation.T
            
            m = glGetFloatv(GL_MODELVIEW_MATRIX)
            
            logger.log(Logger.DEBUG, np.array(m).reshape((4, 4)).T)


            # Draw camera as a small pyramid
            glDisable(GL_LIGHTING)
            
            # Camera body (cyan)
            glColor3f(0.0, 0.8, 0.8)
            
            # Draw camera frustum
            glBegin(GL_LINES)
            # Front face
            glVertex3f(0, 0, 0)
            glVertex3f(0.1, 0.1, -0.2)
            
            glVertex3f(0, 0, 0)
            glVertex3f(-0.1, 0.1, -0.2)
            
            glVertex3f(0, 0, 0)
            glVertex3f(-0.1, -0.1, -0.2)
            
            glVertex3f(0, 0, 0)
            glVertex3f(0.1, -0.1, -0.2)
            
            # Back face
            glVertex3f(0.1, 0.1, -0.2)
            glVertex3f(-0.1, 0.1, -0.2)
            
            glVertex3f(-0.1, 0.1, -0.2)
            glVertex3f(-0.1, -0.1, -0.2)
            
            glVertex3f(-0.1, -0.1, -0.2)
            glVertex3f(0.1, -0.1, -0.2)
            
            glVertex3f(0.1, -0.1, -0.2)
            glVertex3f(0.1, 0.1, -0.2)
            glEnd()
            
            # Draw viewing direction
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, -0.3)
            glEnd()
            
            # Draw camera number at frustum position
            glColor3f(1.0, 1.0, 0.0)  # Yellow text for visibility
            # Position the text at the center of the back face of the frustum
            glRasterPos3f(0, 0, -0.2)
            
            # Draw the camera index number (1-based for user readability)
            camera_number = str(pd.Timestamp(timestamp).timestamp())
            for c in camera_number:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))
            
            glEnable(GL_LIGHTING)
            
            glPopMatrix()
    
    
    def _draw_contents(self):
        """
        Draw the MR contents.
        """
        # Use the first scene if available
        if not self.scenes:
            return
        
        scene_name = next(iter(self.scenes))
        scene_data = self.scenes[scene_name]
        
        for obj_id, obj_data in scene_data.items():
            if obj_id in self.models:
                model_data = self.models[obj_id]
                
                if 'file_path' not in model_data:
                    continue
                
                model_path = model_data['file_path']
                
                # Check if model is already loaded
                if model_path not in self.model_cache:
                    self._load_content_model(model_path)
                
                # Skip if model loading failed
                if model_path not in self.model_cache:
                    continue
                
                glPushMatrix()
                
                # Apply object transform
                if 'position' in obj_data and 'rotation' in obj_data:
                    # Apply position
                    position = obj_data['position']
                    glTranslatef(position['x'], position['y'], position['z'])
                    
                    # Apply rotation
                    rotation = obj_data['rotation']
                    if 'w' in rotation:  # Quaternion
                        quat = pyrr.Quaternion([rotation['x'], rotation['y'], rotation['z'], rotation['w']])
                        rotation_matrix = quat.matrix33
                        glMultMatrixf(pyrr.matrix44.create_from_matrix33(rotation_matrix))
                    else:  # Euler angles
                        glRotatef(rotation.get('x', 0), 1, 0, 0)
                        glRotatef(rotation.get('y', 0), 0, 1, 0)
                        glRotatef(rotation.get('z', 0), 0, 0, 1)
                    
                    # Apply scale if provided
                    if 'scale' in obj_data:
                        scale = obj_data['scale']
                        glScalef(scale.get('x', 1.0), scale.get('y', 1.0), scale.get('z', 1.0))
                
                # Set material properties for the content model
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.0, 0.8, 0.0, 1.0])  # Green for MR contents
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0)
                
                # Draw the model
                model = self.model_cache[model_path]
                for mesh in model.meshes:
                    glBegin(GL_TRIANGLES)
                    for face in mesh.faces:
                        for vertex_idx in face:
                            # Set normal
                            if mesh.normals.size > 0:
                                normal = mesh.normals[vertex_idx]
                                glNormal3f(normal[0], normal[1], normal[2])
                            
                            # Set vertex
                            vertex = mesh.vertices[vertex_idx]
                            glVertex3f(vertex[0], vertex[1], vertex[2])
                    glEnd()
                
                glPopMatrix()
    
    def _load_content_model(self, model_path: str):
        """
        Load a content model from a file.
        
        Args:
            model_path: Path to the model file
        """
        logger.log(Logger.SYSTEM, f"Loading content model from {model_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                logger.log(Logger.WARNING, f"Model file does not exist: {model_path}")
                
                # Try different path variations
                possible_paths = [
                    # Try absolute path from project root
                    os.path.join(os.getcwd(), "LocalData", "Models", "Scene1", os.path.basename(model_path)),
                    # Try relative path from current directory
                    os.path.join("LocalData", "Models", "Scene1", os.path.basename(model_path)),
                    # Try just the filename in Scene1 directory
                    os.path.join("LocalData", "Models", "Scene1", os.path.basename(model_path).split('/')[-1]),
                    # Try with parent directory
                    os.path.join("..", "LocalData", "Models", "Scene1", os.path.basename(model_path))
                ]
                
                found = False
                for alt_path in possible_paths:
                    logger.log(Logger.DEBUG, f"Trying alternative path: {alt_path}")
                    if os.path.exists(alt_path):
                        logger.log(Logger.DEBUG, f"Found model at alternative path: {alt_path}")
                        model_path = alt_path
                        found = True
                        break
                
                if not found:
                    logger.log(Logger.ERROR, f"Could not find model file: {os.path.basename(model_path)}")
                    return
            
            # Use pyassimp to load the model
            processing_flags = (
                pyassimp.postprocess.aiProcess_Triangulate |
                pyassimp.postprocess.aiProcess_GenNormals
            )
            
            # Use with statement to properly handle the context manager
            with pyassimp.load(model_path, processing=processing_flags) as model:
                if not model or not model.meshes:
                    logger.log(Logger.ERROR, f"No meshes found in {model_path}")
                    return
                
                # Store the model for rendering
                self.model_cache[model_path] = model
                logger.log(Logger.SYSTEM, f"Content model loaded with {len(model.meshes)} meshes")
        except Exception as e:
            logger.log(Logger.ERROR, f"Error loading content model: {e}")
    
    def _reshape_callback(self, width, height):
        """
        GLUT reshape callback function.
        
        Args:
            width: New window width
            height: New window height
        """
        self.window_width = width
        self.window_height = height
        glViewport(0, 0, width, height)
        glutPostRedisplay()
    
    def _mouse_callback(self, button, state, x, y):
        """
        GLUT mouse callback function.
        
        Args:
            button: Mouse button (GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON)
            state: Button state (GLUT_UP, GLUT_DOWN)
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        self.mouse_x = x
        self.mouse_y = y
        self.mouse_button = button
        self.mouse_state = state
    
    def _motion_callback(self, x, y):
        """
        GLUT motion callback function.
        
        Args:
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        if self.mouse_state == GLUT_DOWN:
            dx = x - self.mouse_x
            dy = y - self.mouse_y
            
            if self.mouse_button == GLUT_LEFT_BUTTON:
                # Rotate the scene
                self.rotation_y += dx * 0.5
                self.rotation_x += dy * 0.5
            elif self.mouse_button == GLUT_RIGHT_BUTTON:
                # Zoom in/out
                self.zoom *= (1.0 + dy * 0.01)
                self.zoom = max(0.1, min(10.0, self.zoom))
        
        self.mouse_x = x
        self.mouse_y = y
        glutPostRedisplay()
    
    def _keyboard_callback(self, key, x, y):
        """
        GLUT keyboard callback function.
        
        Args:
            key: ASCII key
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        key = key.decode('utf-8')
        self.keys[key] = True
        
        # Toggle visualization options
        if key == 'm':
            self.show_markers = not self.show_markers
        elif key == 'c':
            self.show_cameras = not self.show_cameras
        elif key == 'o':
            self.show_contents = not self.show_contents
        elif key == 's':
            self.show_scene = not self.show_scene
        elif key == 'g':
            self.show_grid = not self.show_grid
        elif key == 'a':
            self.show_axes = not self.show_axes
        elif key == 'f':
            self.set_free_view()
        elif key == 'r':
            # Reset view
            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.zoom = 1.0
        elif key == 'q' or key == 27:  # ESC key
            # Exit the program
            glutLeaveMainLoop()
        
        glutPostRedisplay()
    
    def _special_key_callback(self, key, x, y):
        """
        GLUT special key callback function.
        
        Args:
            key: Special key code
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        # Handle arrow keys for navigation
        if key == GLUT_KEY_UP:
            self.rotation_x += 5.0
        elif key == GLUT_KEY_DOWN:
            self.rotation_x -= 5.0
        elif key == GLUT_KEY_LEFT:
            self.rotation_y -= 5.0
        elif key == GLUT_KEY_RIGHT:
            self.rotation_y += 5.0
        
        glutPostRedisplay()