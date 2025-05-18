import numpy as np
from typing import Dict, Any
import pyrr
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from Interfaces.Frame import Frame
from Logger import logger, Logger
from Systems.Renderer import Renderer, Model3D

class ContentsDepthCal:
    """
    ContentsDepthCal is responsible for calculating the depth map of MR contents.
    It gets the contents that are trying to render from the Renderer and
    generates a depth map from only the 3D objects that are trying to overlay in the camera scene.
    """
    
    def __init__(self, renderer: Renderer):
        """
        Initialize the ContentsDepthCal with a renderer.
        
        Args:
            renderer: The renderer instance used to render the MR contents
        """
        self.renderer = renderer
        self.initialize_opengl()
        
    def initialize_opengl(self):
        """
        Initialize OpenGL for depth rendering.
        """
        # Initialize GLUT if not already initialized
        try:
            glutInit()
        except:
            # GLUT might already be initialized by the renderer
            pass
            
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH)
        
        # Create a window for depth rendering (not visible)
        glutCreateWindow("Depth Renderer")
        
        # Set up OpenGL state for depth rendering
        glEnable(GL_DEPTH_TEST)
        
    def calculate_depth(self, frame: Frame, models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        """
        Calculate the depth map of MR contents.
        
        Args:
            frame: Frame object containing RGB and depth images
            models: Dictionary of 3D models with keys matching scene_data keys
                   Each entry should have a 'file_path' key with the path to the model file
            scene_data: Scene description dictionary with position and rotation for each model
            
        Returns:
            np.ndarray: Depth map of MR contents with the same dimensions as the frame
        """
        logger.log(Logger.DEBUG, f"Calculating depth map for MR contents")
        
        # Set up framebuffer for off-screen rendering
        width, height = frame.width, frame.height
        
        # Create a framebuffer object
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        # Create a texture to render to
        depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        # Attach texture to framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0)
        
        # We don't need color buffer for depth rendering
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        
        # Check framebuffer status
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logger.log(Logger.ERROR, "Framebuffer not complete")
            return np.zeros((height, width), dtype=np.float32)
        
        # Set up viewport
        glViewport(0, 0, width, height)
        
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height
        gluPerspective(60.0, aspect, 0.1, 100.0)
        
        # Set up modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
        
        # Clear depth buffer
        glClear(GL_DEPTH_BUFFER_BIT)
        
        # Render each object in the scene
        for obj_id, obj_data in scene_data.items():
            if obj_id in models:
                model_data = models[obj_id]
                
                if 'file_path' not in model_data:
                    logger.log(Logger.WARNING, f"No file_path specified for model {obj_id}")
                    continue
                    
                model_path = model_data['file_path']
                
                try:
                    # Load model
                    model = self.renderer.load_model(model_path)
                    
                    # Push matrix for this object
                    glPushMatrix()
                    
                    # Apply object transform
                    if 'position' in obj_data and 'rotation' in obj_data:
                        self._apply_transform(obj_data['position'], obj_data['rotation'])
                    else:
                        logger.log(Logger.WARNING, f"Missing position or rotation data for {obj_id}")
                    
                    # Render model
                    model.render()
                    
                    # Pop matrix
                    glPopMatrix()
                except Exception as e:
                    logger.log(Logger.ERROR, f"Error rendering model {obj_id} for depth calculation: {e}")
        
        # Read depth buffer
        depth_buffer = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth_array = np.frombuffer(depth_buffer, dtype=np.float32).reshape(height, width)
        
        # Flip image vertically (OpenGL has origin at bottom-left)
        depth_array = np.flipud(depth_array)
        
        # Convert normalized depth values [0,1] to actual depth values in meters
        # The depth values are normalized in the range [0,1] where 0 is near plane and 1 is far plane
        near, far = 0.1, 100.0
        depth_meters = far * near / (far - (far - near) * depth_array)
        
        # Clean up
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [depth_texture])
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        logger.log(Logger.DEBUG, f"Depth map calculation complete. Shape: {depth_meters.shape}")
        
        return depth_meters
    
    def _apply_transform(self, position: Dict[str, float], rotation: Dict[str, float]):
        """
        Apply a transform (position and rotation) to the current OpenGL matrix.
        
        Args:
            position: Dictionary with x, y, z position values
            rotation: Dictionary with x, y, z, w quaternion rotation values or x, y, z Euler angles
        """
        # Apply translation
        glTranslatef(position['x'], position['y'], position['z'])
        
        # Check if rotation is specified as quaternion or Euler angles
        if 'w' in rotation:
            # It's a quaternion
            quat = pyrr.Quaternion([rotation['x'], rotation['y'], rotation['z'], rotation['w']])
        else:
            # It's Euler angles (in degrees) - convert to quaternion
            import math
            
            # Extract Euler angles in degrees
            euler_x = math.radians(rotation.get('x', 0))
            euler_y = math.radians(rotation.get('y', 0))
            euler_z = math.radians(rotation.get('z', 0))
            
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
        
        # Apply rotation
        rotation_matrix = quat.matrix33
        glMultMatrixf(pyrr.matrix44.create_from_matrix33(rotation_matrix))