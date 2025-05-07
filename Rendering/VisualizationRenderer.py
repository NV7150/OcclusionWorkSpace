import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import pyrr
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pyassimp
import pyassimp.postprocess
from PIL import Image

from .ShaderManager import ShaderManager
from .BufferManager import BufferManager
from .TextureManager import TextureManager
from .Primitives import Primitives
from .Camera import Camera
from ..core.IRenderer import IRenderer
from ..DataLoaders.Frame import Frame
from ..Utils.TransformUtils import TransformUtils
from ..Logger.Logger import Logger


class VisualizationRenderer(IRenderer):
    """
    Renderer for visualizing MR scenes in 3D.
    
    This renderer is specifically designed for visualization purposes, rendering
    scene models, reference markers, camera positions, and MR contents.
    It implements the IRenderer interface and uses modern OpenGL with shaders.
    """
    
    # Shader sources
    PHONG_VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoord;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;
    out vec3 Normal;
    out vec2 TexCoord;
    
    void main()
    {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        TexCoord = aTexCoord;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
    """
    
    PHONG_FRAGMENT_SHADER = """
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    in vec2 TexCoord;
    
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;
    uniform vec4 objectColor;
    uniform float shininess;
    uniform bool useTexture;
    uniform sampler2D textureSampler;
    
    out vec4 FragColor;
    
    void main()
    {
        // Material properties
        vec4 baseColor = useTexture ? texture(textureSampler, TexCoord) : objectColor;
        
        // Ambient
        float ambientStrength = 0.2;
        vec3 ambient = ambientStrength * lightColor;
        
        // Diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = specularStrength * spec * lightColor;
        
        // Result
        vec3 result = (ambient + diffuse + specular) * baseColor.rgb;
        FragColor = vec4(result, baseColor.a);
    }
    """
    
    COLOR_VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    """
    
    COLOR_FRAGMENT_SHADER = """
    #version 330 core
    uniform vec4 color;
    
    out vec4 FragColor;
    
    void main()
    {
        FragColor = color;
    }
    """
    
    def __init__(self, 
                 shader_manager: ShaderManager,
                 buffer_manager: BufferManager,
                 texture_manager: TextureManager,
                 primitives: Primitives,
                 logger: Optional[Logger] = None):
        """
        Initialize the VisualizationRenderer.
        
        Args:
            shader_manager: ShaderManager instance for managing shaders
            buffer_manager: BufferManager instance for managing OpenGL buffers
            texture_manager: TextureManager instance for managing textures
            primitives: Primitives instance for rendering basic shapes
            logger: Optional logger instance
        """
        self._shader_manager = shader_manager
        self._buffer_manager = buffer_manager
        self._texture_manager = texture_manager
        self._primitives = primitives
        self._logger = logger
        
        # Scene data
        self._scene_model_path = None
        self._scene_model = None
        self._marker_positions = {}
        self._camera_poses = {}
        self._models = {}
        self._scenes = {}
        
        # Window properties
        self._window_width = 1024
        self._window_height = 768
        self._window_title = b"MR Scene Visualization"
        self._window_id = None
        
        # Camera properties
        self._camera = Camera()
        self._camera.position = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        self._camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._camera.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Mouse interaction
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_button = -1
        self._mouse_state = GLUT_UP
        
        # Rotation and zoom
        self._rotation_x = 0.0
        self._rotation_y = 0.0
        self._zoom = 1.0
        
        # Keyboard state
        self._keys = {}
        
        # Visualization options
        self._show_markers = True
        self._show_cameras = True
        self._show_contents = True
        self._show_scene = True
        self._show_grid = True
        self._show_axes = True
        
        # Current view mode
        self._view_mode = "free"  # "free", "camera", "marker"
        self._current_camera_timestamp = None
        self._current_marker_id = None
        
        # Light properties
        self._light_pos = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        self._light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        if self._logger:
            self._logger.log(Logger.RENDER, "VisualizationRenderer initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the renderer.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Initialize GLUT
            glutInit()
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
            glutInitWindowSize(self._window_width, self._window_height)
            self._window_id = glutCreateWindow(self._window_title)
            
            # Set up OpenGL state
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Initialize shaders
            self._init_shaders()
            
            # Set up callbacks
            glutDisplayFunc(self._display_callback)
            glutReshapeFunc(self._reshape_callback)
            glutMouseFunc(self._mouse_callback)
            glutMotionFunc(self._motion_callback)
            glutKeyboardFunc(self._keyboard_callback)
            glutSpecialFunc(self._special_key_callback)
            
            if self._logger:
                self._logger.log(Logger.RENDER, "Renderer initialized successfully")
            
            return True
        except Exception as e:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Failed to initialize renderer: {e}")
            return False
    
    def _init_shaders(self) -> None:
        """
        Initialize shader programs.
        """
        # Create shader programs
        self._shader_manager.create_program("phong", self.PHONG_VERTEX_SHADER, self.PHONG_FRAGMENT_SHADER)
        self._shader_manager.create_program("color", self.COLOR_VERTEX_SHADER, self.COLOR_FRAGMENT_SHADER)
        
        if self._logger:
            self._logger.log(Logger.RENDER, "Shader programs initialized")
    
    def shutdown(self) -> None:
        """
        Shutdown the renderer and release resources.
        """
        # Clean up resources
        self._buffer_manager.cleanup()
        self._texture_manager.cleanup()
        
        # Delete window
        if self._window_id is not None:
            glutDestroyWindow(self._window_id)
        
        if self._logger:
            self._logger.log(Logger.RENDER, "Renderer shut down")
    
    # IRenderer interface implementation - simplified for visualization renderer
    def render_frame(self, frame: Frame, occlusion_mask: np.ndarray, models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        return frame.rgb.copy()
    
    def render_depth_only(self, frame: Frame, models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        return frame.depth.copy()
    
    def save_rendered_image(self, image: np.ndarray, output_path: str) -> str:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(image).save(output_path)
            return output_path
        except Exception as e:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Failed to save rendered image: {e}")
            return ""
    
    def render_and_save_batch(self, frames: List[Frame], occlusion_masks: Dict[np.datetime64, np.ndarray],
                             models: Dict[str, Any], scene_data: Dict, output_dir: str, output_prefix: str = "frame") -> List[str]:
        return []
    
    def set_camera_parameters(self, fov: float, aspect_ratio: float, near_plane: float, far_plane: float) -> None:
        self._camera.fov = fov
        self._camera.aspect_ratio = aspect_ratio
        self._camera.near_plane = near_plane
        self._camera.far_plane = far_plane
    
    def set_lighting_parameters(self, ambient: List[float], diffuse: List[float], specular: List[float], light_position: List[float]) -> None:
        self._light_color = np.array(diffuse, dtype=np.float32)
        self._light_pos = np.array(light_position, dtype=np.float32)
    
    # Visualization-specific methods
    def setup_visualization_scene(self, scene_model_path: str, marker_positions: Dict,
                                camera_poses: Dict, models: Dict[str, Any], scenes: Dict[str, Dict]) -> None:
        """
        Set up the visualization scene.
        """
        self._scene_model_path = scene_model_path
        self._marker_positions = marker_positions
        self._camera_poses = camera_poses
        self._models = models
        self._scenes = scenes
        
        if self._logger:
            self._logger.log(Logger.RENDER, "Visualization scene set up")
    
    def start_visualization_loop(self) -> None:
        """
        Start the visualization rendering loop.
        """
        if self._logger:
            self._logger.log(Logger.RENDER, "Starting visualization loop")
        
        # Start the GLUT main loop
        glutMainLoop()
    
    def set_view_mode(self, mode: str, param: Any = None) -> None:
        """
        Set the view mode for the visualization.
        """
        self._view_mode = mode
        
        if mode == "camera" and param is not None:
            self._current_camera_timestamp = param
        elif mode == "marker" and param is not None:
            self._current_marker_id = param
        elif mode == "free":
            self._current_camera_timestamp = None
            self._current_marker_id = None
        
        # Trigger a redisplay
        glutPostRedisplay()
    
    # GLUT callbacks
    def _display_callback(self) -> None:
        """
        GLUT display callback function.
        """
        # Clear the screen
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up view matrix based on current view mode
        view_matrix = self._get_view_matrix()
        
        # Set up projection matrix
        projection_matrix = self._camera.get_projection_matrix()
        
        # Get eye position for lighting calculations
        eye_pos = self._camera.position
        
        # Draw the scene
        self._draw_scene(view_matrix, projection_matrix, eye_pos)
        
        # Swap buffers
        glutSwapBuffers()
    
    def _get_view_matrix(self) -> np.ndarray:
        """
        Get the view matrix based on the current view mode.
        """
        # Implementation would depend on view mode
        return self._camera.get_view_matrix()
    
    def _draw_scene(self, view_matrix: np.ndarray, projection_matrix: np.ndarray, eye_pos: np.ndarray) -> None:
        """
        Draw the entire scene.
        """
        # Draw axes if enabled
        if self._show_axes:
            self._primitives.render_axes("color", np.identity(4, dtype=np.float32))
        
        # Draw grid if enabled
        if self._show_grid:
            self._primitives.render_grid("color", np.identity(4, dtype=np.float32), [0.5, 0.5, 0.5, 1.0], 10, 10)
    
    def _reshape_callback(self, width: int, height: int) -> None:
        """
        GLUT reshape callback function.
        """
        self._window_width = width
        self._window_height = height
        glViewport(0, 0, width, height)
        self._camera.aspect_ratio = width / height
    
    def _mouse_callback(self, button: int, state: int, x: int, y: int) -> None:
        """
        GLUT mouse callback function.
        """
        self._mouse_button = button
        self._mouse_state = state
        self._mouse_x = x
        self._mouse_y = y
    
    def _motion_callback(self, x: int, y: int) -> None:
        """
        GLUT motion callback function.
        """
        if self._mouse_state == GLUT_DOWN:
            dx = x - self._mouse_x
            dy = y - self._mouse_y
            
            if self._mouse_button == GLUT_LEFT_BUTTON:
                # Rotate camera
                self._rotation_y += dx * 0.01
                self._rotation_x += dy * 0.01
                
                # Update camera position
                self._camera.position[0] = self._zoom * np.sin(self._rotation_y) * np.cos(self._rotation_x)
                self._camera.position[1] = self._zoom * np.sin(self._rotation_x)
                self._camera.position[2] = self._zoom * np.cos(self._rotation_y) * np.cos(self._rotation_x)
            
            elif self._mouse_button == GLUT_RIGHT_BUTTON:
                # Zoom camera
                self._zoom += dy * 0.01
                self._zoom = max(0.1, min(10.0, self._zoom))
                
                # Update camera position
                self._camera.position[0] = self._zoom * np.sin(self._rotation_y) * np.cos(self._rotation_x)
                self._camera.position[1] = self._zoom * np.sin(self._rotation_x)
                self._camera.position[2] = self._zoom * np.cos(self._rotation_y) * np.cos(self._rotation_x)
            
            self._mouse_x = x
            self._mouse_y = y
            
            # Trigger a redisplay
            glutPostRedisplay()
    
    def _keyboard_callback(self, key: bytes, x: int, y: int) -> None:
        """
        GLUT keyboard callback function.
        """
        key_char = key.decode('utf-8')
        
        if key_char == 'q' or key == b'\x1b':  # q or ESC
            # Exit the application
            glutLeaveMainLoop()
        
        elif key_char == 'm':
            # Toggle marker visibility
            self._show_markers = not self._show_markers
        
        elif key_char == 'c':
            # Toggle camera visibility
            self._show_cameras = not self._show_cameras
        
        elif key_char == 'o':
            # Toggle MR contents visibility
            self._show_contents = not self._show_contents
        
        elif key_char == 's':
            # Toggle scene model visibility
            self._show_scene = not self._show_scene
        
        elif key_char == 'g':
            # Toggle grid visibility
            self._show_grid = not self._show_grid
        
        elif key_char == 'a':
            # Toggle axes visibility
            self._show_axes = not self._show_axes
        
        elif key_char == 'f':
            # Reset to free view mode
            self.set_view_mode("free")
        
        elif key_char == 'r':
            # Reset view
            self._rotation_x = 0.0
            self._rotation_y = 0.0
            self._zoom = 1.0
            self._camera.position = np.array([0.0, 0.0, 3.0], dtype=np.float32)
            self._camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._camera.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Trigger a redisplay
        glutPostRedisplay()
    
    def _special_key_callback(self, key: int, x: int, y: int) -> None:
        """
        GLUT special key callback function.
        """
        # Handle arrow keys for camera rotation
        if key == GLUT_KEY_LEFT:
            self._rotation_y -= 0.1
        elif key == GLUT_KEY_RIGHT:
            self._rotation_y += 0.1
        elif key == GLUT_KEY_UP:
            self._rotation_x -= 0.1
        elif key == GLUT_KEY_DOWN:
            self._rotation_x += 0.1
        
        # Update camera position
        self._camera.position[0] = self._zoom * np.sin(self._rotation_y) * np.cos(self._rotation_x)
        self._camera.position[1] = self._zoom * np.sin(self._rotation_x)
        self._camera.position[2] = self._zoom * np.cos(self._rotation_y) * np.cos(self._rotation_x)
        
        # Trigger a redisplay
        glutPostRedisplay()