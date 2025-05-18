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

# Import core interfaces
from core.IRenderer import IRenderer
from core.IModel import IModel
from core.IScene import IScene

# Import utilities
from Utils.Logger import logger

# Default shader paths (assuming they are in a 'shaders' directory relative to this script)
DEFAULT_SHADER_DIR = os.path.join(os.path.dirname(__file__), "..", "Rendering", "shaders")
DEFAULT_VERTEX_SHADER_PATH = os.path.join(DEFAULT_SHADER_DIR, "default.vert")
DEFAULT_FRAGMENT_SHADER_PATH = os.path.join(DEFAULT_SHADER_DIR, "default.frag")
DEFAULT_SIMPLE_VERTEX_SHADER_PATH = os.path.join(DEFAULT_SHADER_DIR, "simple.vert")
DEFAULT_SIMPLE_FRAGMENT_SHADER_PATH = os.path.join(DEFAULT_SHADER_DIR, "simple.frag")


class Shader:
    """
    Shader class for managing GLSL shaders.
    """
    
    def __init__(self, vertex_source: str, fragment_source: str):
        """
        Initialize the shader program with vertex and fragment shader sources.
        
        Args:
            vertex_source: GLSL vertex shader source code
            fragment_source: GLSL fragment shader source code
        """
        self.program_id = None
        self._create_program(vertex_source, fragment_source)
    
    def _create_program(self, vertex_source: str, fragment_source: str):
        """
        Create a shader program from vertex and fragment shader sources.
        
        Args:
            vertex_source: GLSL vertex shader source code
            fragment_source: GLSL fragment shader source code
        """
        # Create shaders
        vertex_shader = self._compile_shader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = self._compile_shader(fragment_source, GL_FRAGMENT_SHADER)
        
        # Create program
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        
        # Check for linking errors
        if not glGetProgramiv(program, GL_LINK_STATUS):
            info_log = glGetProgramInfoLog(program)
            glDeleteProgram(program)
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)
            raise RuntimeError(f"Shader program linking failed: {info_log}")
        
        # Clean up
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        self.program_id = program
    
    def _compile_shader(self, source: str, shader_type: int) -> int:
        """
        Compile a shader from source.
        
        Args:
            source: GLSL shader source code
            shader_type: GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
            
        Returns:
            Shader ID
        """
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        # Check for compilation errors
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            shader_type_str = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
            raise RuntimeError(f"{shader_type_str} shader compilation failed: {info_log}")
        
        return shader
    
    def use(self):
        """
        Use this shader program.
        """
        glUseProgram(self.program_id)
    
    def set_uniform_matrix4fv(self, name: str, value: np.ndarray):
        """
        Set a uniform mat4 value.
        
        Args:
            name: Uniform name
            value: 4x4 matrix as numpy array
        """
        location = glGetUniformLocation(self.program_id, name)
        glUniformMatrix4fv(location, 1, GL_FALSE, value)

    def set_uniform_3fv(self, name: str, value: np.ndarray):
        """
        Set a uniform vec3 value.
        
        Args:
            name: Uniform name
            value: 3D vector as numpy array
        """
        location = glGetUniformLocation(self.program_id, name)
        glUniform3fv(location, 1, value)
    
    def set_uniform_1f(self, name: str, value: float):
        """
        Set a uniform float value.
        
        Args:
            name: Uniform name
            value: Float value
        """
        location = glGetUniformLocation(self.program_id, name)
        glUniform1f(location, value)
    
    def set_uniform_1i(self, name: str, value: int):
        """
        Set a uniform int value.
        
        Args:
            name: Uniform name
            value: Int value
        """
        location = glGetUniformLocation(self.program_id, name)
        glUniform1i(location, value)


class Mesh:
    """
    Mesh class for managing vertex data and rendering.
    """
    
    def __init__(self, vertices: np.ndarray, normals: np.ndarray, indices: np.ndarray):
        """
        Initialize the mesh with vertex data.
        
        Args:
            vertices: Vertex positions (Nx3 array)
            normals: Vertex normals (Nx3 array)
            indices: Triangle indices (Nx3 array)
        """
        self.vertices = vertices
        self.normals = normals
        self.indices = indices
        self.vao = None
        self.vbo_vertices = None
        self.vbo_normals = None
        self.ebo = None
        self._setup_mesh()
    
    def _setup_mesh(self):
        """
        Set up the mesh VAO and VBOs.
        """
        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create VBO for vertices
        self.vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Create VBO for normals
        self.vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        # Create EBO for indices
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def draw(self):
        """
        Draw the mesh.
        """
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices) * 3, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
    
    def delete(self):
        """
        Delete the mesh VAO and VBOs.
        """
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo_vertices])
        glDeleteBuffers(1, [self.vbo_normals])
        glDeleteBuffers(1, [self.ebo])


class Model:
    """
    Model class for managing a collection of meshes.
    """
    
    def __init__(self):
        """
        Initialize an empty model.
        """
        self.meshes = []
    
    def add_mesh(self, mesh: Mesh):
        """
        Add a mesh to the model.
        
        Args:
            mesh: Mesh to add
        """
        self.meshes.append(mesh)
    
    def draw(self):
        """
        Draw all meshes in the model.
        """
        for mesh in self.meshes:
            mesh.draw()
    
    def delete(self):
        """
        Delete all meshes in the model.
        """
        for mesh in self.meshes:
            mesh.delete()


class VisualizeModelRender(IRenderer):
    """
    VisualizeModelRender is responsible for rendering the MR scene visualization.
    It handles rendering the scene model, reference markers, camera positions, and MR contents.
    This implementation uses modern OpenGL with shaders and VBOs.
    
    Implements the IRenderer interface for consistent usage across the framework.
    """
    
    # Vertex shader source
    # REMOVED - Now loaded from file
    
    # Fragment shader source
    # REMOVED - Now loaded from file
    
    # Simple color shader for lines and basic shapes
    # REMOVED - Now loaded from file
    
    def __init__(self,
                 window_title: bytes = b"MR Scene Visualization",
                 window_width: int = 1024,
                 window_height: int = 768,
                 camera_pos: np.ndarray = np.array([0.0, 0.0, 3.0], dtype=np.float32),
                 camera_target: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 camera_up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32),
                 light_pos: np.ndarray = np.array([1.0, 1.0, 2.0], dtype=np.float32),
                 light_color: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32),
                 default_shader_paths: Tuple[str, str] = (DEFAULT_VERTEX_SHADER_PATH, DEFAULT_FRAGMENT_SHADER_PATH),
                 simple_shader_paths: Tuple[str, str] = (DEFAULT_SIMPLE_VERTEX_SHADER_PATH, DEFAULT_SIMPLE_FRAGMENT_SHADER_PATH),
                 fovy: float = 45.0,
                 near_clip: float = 0.1,
                 far_clip: float = 100.0,
                 # Lighting and Material
                 ambient_strength: float = 0.2,
                 specular_strength: float = 0.5,
                 default_shininess: float = 32.0,
                 # Colors
                 axes_colors: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
                     np.array([1.0, 0.0, 0.0], dtype=np.float32), # X-axis Red
                     np.array([0.0, 1.0, 0.0], dtype=np.float32), # Y-axis Green
                     np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z-axis Blue
                 ),
                 grid_color: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32),
                 marker_cube_color: np.ndarray = np.array([1.0, 1.0, 0.0], dtype=np.float32),
                 marker_normal_color: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32), # Blue
                 marker_tangent_color: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32), # Red
                 marker_bitangent_color: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32), # Green
                 camera_frustum_color: np.ndarray = np.array([0.0, 0.8, 0.8], dtype=np.float32), # Cyan
                 camera_text_color: Tuple[float, float, float] = (1.0, 1.0, 0.0), # Yellow
                 content_color: np.ndarray = np.array([0.0, 0.8, 0.0], dtype=np.float32), # Green
                 scene_model_color: np.ndarray = np.array([0.8, 0.8, 0.8], dtype=np.float32),
                 # Scales
                 marker_cube_scale: float = 0.05,
                 marker_arrow_scale: float = 0.2,
                 # Grid parameters
                 grid_render_size: int = 5, # Renamed from grid_size to avoid conflict if used elsewhere
                 grid_step_size: float = 0.2 # Renamed from grid_step
                ):
        """
        Initialize the VisualizeModelRender.
        """
        self.scene_model = None
        self.marker_positions = {}
        self.camera_poses = {}
        self.models = {}
        self.scenes = {}
        
        # Window properties
        self.window_width = window_width
        self.window_height = window_height
        self.window_title = window_title
        self.window_id = None
        
        # Camera properties
        self.camera_pos = camera_pos
        self.camera_target = camera_target
        self.camera_up = camera_up
        self.fovy = fovy
        self.near_clip = near_clip
        self.far_clip = far_clip
        
        # Lighting and Material
        self.ambient_strength = ambient_strength
        self.specular_strength = specular_strength
        self.default_shininess = default_shininess
        
        # Colors
        self.axes_colors = axes_colors
        self.grid_color = grid_color
        self.marker_cube_color = marker_cube_color
        self.marker_normal_color = marker_normal_color
        self.marker_tangent_color = marker_tangent_color
        self.marker_bitangent_color = marker_bitangent_color
        self.camera_frustum_color = camera_frustum_color
        self.camera_text_color = camera_text_color
        self.content_color = content_color
        self.scene_model_color = scene_model_color
        
        # Scales
        self.marker_cube_scale = marker_cube_scale
        self.marker_arrow_scale = marker_arrow_scale
        
        # Grid parameters
        self.grid_render_size = grid_render_size
        self.grid_step_size = grid_step_size
        
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
        
        # Shader programs
        self.shader = None
        self.simple_shader = None
        self.default_shader_paths = default_shader_paths
        self.simple_shader_paths = simple_shader_paths
        
        # Geometry for basic shapes - store VBO/EBO handles for cleanup
        self.cube_vao = None
        self.cube_vbo_vertices = None
        self.cube_vbo_normals = None
        self.cube_ebo = None
        
        self.grid_vao = None
        self.grid_vbo = None
        
        self.axes_vao = None
        self.axes_vbo = None
        
        self.arrow_vao = None
        self.arrow_vbo = None
        
        # Light properties
        self.light_pos = light_pos
        self.light_color = light_color
    
    def _load_shader_source(self, file_path: str) -> str:
        """Helper function to load shader source from a file."""
        if not os.path.exists(file_path):
            logger.log(logger.ERROR, f"Shader file not found: {file_path}")
            # Return a very basic fallback shader source to avoid crashing
            if file_path.endswith(".vert"):
                return """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); }
"""
            elif file_path.endswith(".frag"):
                return """
#version 330 core
out vec4 FragColor;
uniform vec3 objectColor;
void main() { FragColor = vec4(objectColor, 1.0); }
"""
            return "" # Should not happen
        with open(file_path, 'r') as f:
            return f.read()

    def initialize(self):
        """
        Initialize the renderer.
        """
        # Initialize GLUT
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.window_width, self.window_height)
        self.window_id = glutCreateWindow(self.window_title)
        
        # Set up OpenGL state
        glEnable(GL_DEPTH_TEST)
        
        # Initialize shaders
        vertex_source = self._load_shader_source(self.default_shader_paths[0])
        fragment_source = self._load_shader_source(self.default_shader_paths[1])
        self.shader = Shader(vertex_source, fragment_source)

        simple_vertex_source = self._load_shader_source(self.simple_shader_paths[0])
        simple_fragment_source = self._load_shader_source(self.simple_shader_paths[1])
        self.simple_shader = Shader(simple_vertex_source, simple_fragment_source)
        
        # Set up callbacks
        glutDisplayFunc(self._display_callback)
        glutReshapeFunc(self._reshape_callback)
        glutMouseFunc(self._mouse_callback)
        glutMotionFunc(self._motion_callback)
        glutKeyboardFunc(self._keyboard_callback)
        glutSpecialFunc(self._special_key_callback)
        
        # Create basic geometry
        self._create_basic_geometry()
        
        # Scene model is not loaded here by default, requires a specific call.
        # self._load_scene_model(scene_model_path) 
        
        logger.log(logger.SYSTEM, "Renderer initialized with modern OpenGL shaders")
    
    def _create_basic_geometry(self):
        """
        Create VAOs and VBOs for basic geometry (cube, grid, axes, arrow).
        """
        # Create cube
        self._create_cube()
        
        # Create grid
        self._create_grid()
        
        # Create axes
        self._create_axes()
        
        # Create arrow
        self._create_arrow()
    
    def _create_cube(self):
        """
        Create a cube VAO for markers and other simple objects.
        """
        # Cube vertices
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,  # 0
             0.5, -0.5,  0.5,  # 1
             0.5,  0.5,  0.5,  # 2
            -0.5,  0.5,  0.5,  # 3
            # Back face
            -0.5, -0.5, -0.5,  # 4
             0.5, -0.5, -0.5,  # 5
             0.5,  0.5, -0.5,  # 6
            -0.5,  0.5, -0.5,  # 7
        ], dtype=np.float32)
        
        # Cube normals
        normals = np.array([
            # Front face
             0.0,  0.0,  1.0,  # 0
             0.0,  0.0,  1.0,  # 1
             0.0,  0.0,  1.0,  # 2
             0.0,  0.0,  1.0,  # 3
            # Back face
             0.0,  0.0, -1.0,  # 4
             0.0,  0.0, -1.0,  # 5
             0.0,  0.0, -1.0,  # 6
             0.0,  0.0, -1.0,  # 7
        ], dtype=np.float32)
        
        # Cube indices
        indices = np.array([
            # Front face
            0, 1, 2, 2, 3, 0,
            # Right face
            1, 5, 6, 6, 2, 1,
            # Back face
            5, 4, 7, 7, 6, 5,
            # Left face
            4, 0, 3, 3, 7, 4,
            # Top face
            3, 2, 6, 6, 7, 3,
            # Bottom face
            4, 5, 1, 1, 0, 4
        ], dtype=np.uint32)
        
        # Create VAO
        self.cube_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cube_vao)
        
        # Create VBO for vertices
        self.cube_vbo_vertices = glGenBuffers(1) # Store handle
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Create VBO for normals
        self.cube_vbo_normals = glGenBuffers(1) # Store handle
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        # Create EBO for indices
        self.cube_ebo = glGenBuffers(1) # Store handle
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cube_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def _create_grid(self):
        """
        Create a grid VAO for the reference grid.
        """
        # Use configurable grid parameters
        grid_s = self.grid_render_size 
        grid_st = self.grid_step_size
        vertices = []
        
        # Create grid lines
        for i in range(-grid_s, grid_s + 1):
            # X lines
            vertices.extend([i * grid_st, 0, -grid_s * grid_st])
            vertices.extend([i * grid_st, 0, grid_s * grid_st])
            
            # Z lines
            vertices.extend([-grid_s * grid_st, 0, i * grid_st])
            vertices.extend([grid_s * grid_st, 0, i * grid_st])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Create VAO
        self.grid_vao = glGenVertexArrays(1)
        glBindVertexArray(self.grid_vao)
        
        # Create VBO for vertices
        self.grid_vbo = glGenBuffers(1) # Store handle
        glBindBuffer(GL_ARRAY_BUFFER, self.grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def _create_axes(self):
        """
        Create axes VAO for coordinate axes.
        """
        vertices = np.array([
            # X axis (red)
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            # Y axis (green)
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            # Z axis (blue)
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0
        ], dtype=np.float32)
        
        # Create VAO
        self.axes_vao = glGenVertexArrays(1)
        glBindVertexArray(self.axes_vao)
        
        # Create VBO for vertices
        self.axes_vbo = glGenBuffers(1) # Store handle
        glBindBuffer(GL_ARRAY_BUFFER, self.axes_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def _create_arrow(self):
        """
        Create arrow VAO for direction indicators.
        """
        # Simple arrow as a line with a small cone at the end
        vertices = np.array([
            # Line part
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.8,
            # Cone base (triangle fan)
            0.0, 0.0, 0.8,
            0.05, 0.0, 0.7,
            0.035, 0.035, 0.7,
            0.0, 0.05, 0.7,
            -0.035, 0.035, 0.7,
            -0.05, 0.0, 0.7,
            -0.035, -0.035, 0.7,
            0.0, -0.05, 0.7,
            0.035, -0.035, 0.7,
            0.05, 0.0, 0.7
        ], dtype=np.float32)
        
        # Create VAO
        self.arrow_vao = glGenVertexArrays(1)
        glBindVertexArray(self.arrow_vao)
        
        # Create VBO for vertices
        self.arrow_vbo = glGenBuffers(1) # Store handle
        glBindBuffer(GL_ARRAY_BUFFER, self.arrow_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def _load_scene_model(self, model_path: str):
        """
        Load the scene model from a file using modern OpenGL.
        
        Args:
            model_path: Path to the model file
        """
        logger.log(logger.SYSTEM, f"Loading scene model from {model_path}")
        
        try:
            # Use pyassimp to load the model
            processing_flags = (
                pyassimp.postprocess.aiProcess_Triangulate |
                pyassimp.postprocess.aiProcess_GenNormals
            )
            
            # Use with statement to properly handle the context manager
            with pyassimp.load(model_path, processing=processing_flags) as scene:
                if not scene or not scene.meshes:
                    logger.log(logger.ERROR, f"No meshes found in {model_path}")
                    self._create_fallback_scene()
                    return
                
                # Create a model from the scene
                model = Model()
                
                # Process each mesh
                for mesh in scene.meshes:
                    # Extract vertices, normals, and indices
                    vertices = mesh.vertices.astype(np.float32)
                    normals = mesh.normals.astype(np.float32)
                    
                    # Convert faces to indices
                    indices = np.array([idx for face in mesh.faces for idx in face], dtype=np.uint32)
                    
                    # Create mesh
                    mesh_obj = Mesh(vertices, normals, indices)
                    model.add_mesh(mesh_obj)
                
                self.scene_model = model
                logger.log(logger.SYSTEM, f"Scene model loaded with {len(scene.meshes)} meshes")
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading scene model: {e}")
            self._create_fallback_scene()
    
    def _create_fallback_scene(self):
        """
        Create a fallback scene if the model loading fails.
        """
        logger.log(logger.WARNING, "Creating fallback scene")
        # We'll create a simple grid as a fallback.
        # For now, setting to None. A proper fallback might involve creating a default Model object.
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
        
        logger.log(logger.SYSTEM, "Scene setup complete")
    
    def start_render_loop(self):
        """
        Start the rendering loop.
        """
        logger.log(logger.SYSTEM, "Starting render loop")
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
        aspect_ratio = self.window_width / self.window_height if self.window_height > 0 else 1.0
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=self.fovy,
            aspect=aspect_ratio,
            near=self.near_clip,
            far=self.far_clip
        )
        
        # Set up the view matrix based on view mode
        if self.view_mode == "camera" and self.current_camera_timestamp in self.camera_poses:
            # Use the selected camera pose
            camera_pose = self.camera_poses[self.current_camera_timestamp]
            # Invert the camera pose to get the view matrix
            view_matrix = np.linalg.inv(camera_pose)
            eye_pos = -np.dot(camera_pose[:3, :3].T, camera_pose[:3, 3])
        elif self.view_mode == "marker" and self.current_marker_id in self.marker_positions:
            # Look at the selected marker
            marker_pos = self.marker_positions[self.current_marker_id]["pos"]
            marker_norm = self.marker_positions[self.current_marker_id]["norm"]
            # Position the camera along the marker normal
            eye_pos = marker_pos + marker_norm * 0.5
            view_matrix = pyrr.matrix44.create_look_at(
                eye=eye_pos,
                target=marker_pos,
                up=[0, 1, 0]
            )
        else:
            # Free view mode
            # Apply zoom
            eye_pos = self.camera_pos * self.zoom
            view_matrix = pyrr.matrix44.create_look_at(
                eye=eye_pos,
                target=self.camera_target,
                up=self.camera_up
            )
            
            # Apply rotation
            rotation_x = pyrr.matrix44.create_from_x_rotation(np.radians(self.rotation_x))
            rotation_y = pyrr.matrix44.create_from_y_rotation(np.radians(self.rotation_y))
            view_matrix = pyrr.matrix44.multiply(rotation_x, view_matrix)
            view_matrix = pyrr.matrix44.multiply(rotation_y, view_matrix)
        
        # Draw the scene
        self._draw_scene(view_matrix, projection, eye_pos)
        
        # Swap buffers
        glutSwapBuffers()
    
    def _draw_scene(self, view_matrix, projection_matrix, eye_pos):
        """
        Draw the entire scene.
        
        Args:
            view_matrix: View matrix
            projection_matrix: Projection matrix
            eye_pos: Camera position
        """
        # Draw coordinate axes
        if self.show_axes:
            self._draw_axes(view_matrix, projection_matrix)
        
        # Draw grid
        if self.show_grid:
            self._draw_grid(view_matrix, projection_matrix)
        
        # Draw scene model
        if self.show_scene and self.scene_model:
            self._draw_scene_model(view_matrix, projection_matrix, eye_pos)
        
        # Draw markers
        if self.show_markers:
            self._draw_markers(view_matrix, projection_matrix, eye_pos)
        
        # Draw camera positions
        if self.show_cameras:
            self._draw_cameras(view_matrix, projection_matrix)
        
        # Draw MR contents
        if self.show_contents:
            self._draw_contents(view_matrix, projection_matrix, eye_pos)
    

    def _draw_axes(self, view_matrix, projection_matrix):
        """
        Draw coordinate axes using shaders.
        
        Args:
            view_matrix: View matrix
            projection_matrix: Projection matrix
        """
        # Use simple shader
        self.simple_shader.use()
        
        # Set matrices
        model_matrix = np.identity(4, dtype=np.float32)
        self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
        self.simple_shader.set_uniform_matrix4fv("view", view_matrix)
        self.simple_shader.set_uniform_matrix4fv("projection", projection_matrix)
        
        # Draw X axis (red)
        self.simple_shader.set_uniform_3fv("color", self.axes_colors[0])
        glBindVertexArray(self.axes_vao)
        glDrawArrays(GL_LINES, 0, 2)
        
        # Draw Y axis (green)
        self.simple_shader.set_uniform_3fv("color", self.axes_colors[1])
        glDrawArrays(GL_LINES, 2, 2)
        
        # Draw Z axis (blue)
        self.simple_shader.set_uniform_3fv("color", self.axes_colors[2])
        glDrawArrays(GL_LINES, 4, 2)
        
        glBindVertexArray(0)
    
    def _draw_grid(self, view_matrix, projection_matrix):
        """
        Draw reference grid using shaders.
        
        Args:
            view_matrix: View matrix
            projection_matrix: Projection matrix
        """
        # Use simple shader
        self.simple_shader.use()
        
        # Set matrices
        model_matrix = np.identity(4, dtype=np.float32)
        self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
        self.simple_shader.set_uniform_matrix4fv("view", view_matrix)
        self.simple_shader.set_uniform_matrix4fv("projection", projection_matrix)
        
        # Set color (gray)
        self.simple_shader.set_uniform_3fv("color", self.grid_color)
        
        # Draw grid
        glBindVertexArray(self.grid_vao)
        # Use configurable grid size for drawing count
        glDrawArrays(GL_LINES, 0, (self.grid_render_size * 2 + 1) * 4)
        glBindVertexArray(0)
    
    def _draw_scene_model(self, view_matrix, projection_matrix, eye_pos):
        """
        Draw the scene model using shaders.
        
        Args:
            view_matrix: View matrix
            projection_matrix: Projection matrix
            eye_pos: Camera position
        """
        if not self.scene_model:
            return
        
        # Use shader
        self.shader.use()
        
        # Set matrices
        model_matrix = np.identity(4, dtype=np.float32)
        self.shader.set_uniform_matrix4fv("model", model_matrix)
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)
        
        # Set lighting properties
        self.shader.set_uniform_3fv("lightPos", self.light_pos)
        self.shader.set_uniform_3fv("viewPos", eye_pos)
        self.shader.set_uniform_3fv("lightColor", self.light_color)
        self.shader.set_uniform_3fv("objectColor", self.scene_model_color)
        # Pass lighting parameters from config
        self.shader.set_uniform_1f("ambientStrength", self.ambient_strength)
        self.shader.set_uniform_1f("specularStrength", self.specular_strength)
        self.shader.set_uniform_1f("shininess", self.default_shininess)
        
        # Draw model
        self.scene_model.draw()
    
    def _draw_markers(self, view_matrix, projection_matrix, eye_pos):
        """
        Draw the reference markers using shaders.
        
        Args:
            view_matrix: View matrix
            projection_matrix: Projection matrix
            eye_pos: Camera position
        """
        for marker_id, marker_data in self.marker_positions.items():
            pos = marker_data["pos"]
            norm = marker_data["norm"]
            tangent = marker_data["tangent"]
            
            # Calculate bitangent (cross product of tangent and normal)
            bitangent = np.cross(tangent, norm)
            bitangent_length = np.linalg.norm(bitangent)
            if bitangent_length > 0.001:  # Check for non-zero vector
                bitangent = bitangent / bitangent_length
            
            # Draw marker cube
            self.shader.use()
            
            # Create model matrix for the marker
            model_matrix = np.identity(4, dtype=np.float32)
            # Apply translation
            model_matrix[0, 3] = pos[0]
            model_matrix[1, 3] = pos[1]
            model_matrix[2, 3] = pos[2]
            # Apply scale
            scale_val = self.marker_cube_scale
            scale_matrix = pyrr.matrix44.create_from_scale([scale_val, scale_val, scale_val])
            model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
            
            # Set matrices
            self.shader.set_uniform_matrix4fv("model", model_matrix)
            self.shader.set_uniform_matrix4fv("view", view_matrix)
            self.shader.set_uniform_matrix4fv("projection", projection_matrix)
            
            # Set lighting properties
            self.shader.set_uniform_3fv("lightPos", self.light_pos)
            self.shader.set_uniform_3fv("viewPos", eye_pos)
            self.shader.set_uniform_3fv("lightColor", self.light_color)
            self.shader.set_uniform_3fv("objectColor", self.marker_cube_color)
            self.shader.set_uniform_1f("shininess", self.default_shininess) # Use configured default
            # Pass lighting parameters from config
            self.shader.set_uniform_1f("ambientStrength", self.ambient_strength)
            self.shader.set_uniform_1f("specularStrength", self.specular_strength)
            
            # Draw cube
            glBindVertexArray(self.cube_vao)
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            
            # Draw coordinate vectors
            self.simple_shader.use()
            
            # Set matrices
            self.simple_shader.set_uniform_matrix4fv("view", view_matrix)
            self.simple_shader.set_uniform_matrix4fv("projection", projection_matrix)
            
            # Create model matrix for the normal vector
            model_matrix = np.identity(4, dtype=np.float32)
            # Apply translation
            model_matrix[0, 3] = pos[0]
            model_matrix[1, 3] = pos[1]
            model_matrix[2, 3] = pos[2]
            # Apply rotation to align with normal
            rotation_matrix = self._create_rotation_matrix_from_vectors(np.array([0, 0, 1]), norm)
            model_matrix = pyrr.matrix44.multiply(model_matrix, rotation_matrix)
            # Apply scale
            arrow_scale_val = self.marker_arrow_scale
            scale_matrix = pyrr.matrix44.create_from_scale([arrow_scale_val, arrow_scale_val, arrow_scale_val])
            model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
            
            # Set model matrix
            self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
            
            # Draw normal vector (blue)
            self.simple_shader.set_uniform_3fv("color", self.marker_normal_color)
            glBindVertexArray(self.arrow_vao)
            glDrawArrays(GL_LINES, 0, 2)  # Draw line part
            glDrawArrays(GL_TRIANGLE_FAN, 2, 10)  # Draw cone part
            glBindVertexArray(0)
            
            # Create model matrix for the tangent vector
            model_matrix = np.identity(4, dtype=np.float32)
            # Apply translation
            model_matrix[0, 3] = pos[0]
            model_matrix[1, 3] = pos[1]
            model_matrix[2, 3] = pos[2]
            # Apply rotation to align with tangent
            rotation_matrix = self._create_rotation_matrix_from_vectors(np.array([0, 0, 1]), tangent)
            model_matrix = pyrr.matrix44.multiply(model_matrix, rotation_matrix)
            # Apply scale
            scale_matrix = pyrr.matrix44.create_from_scale([arrow_scale_val, arrow_scale_val, arrow_scale_val])
            model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
            
            # Set model matrix
            self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
            
            # Draw tangent vector (red)
            self.simple_shader.set_uniform_3fv("color", self.marker_tangent_color)
            glBindVertexArray(self.arrow_vao)
            glDrawArrays(GL_LINES, 0, 2)  # Draw line part
            glDrawArrays(GL_TRIANGLE_FAN, 2, 10)  # Draw cone part
            glBindVertexArray(0)
            
            # Create model matrix for the bitangent vector
            model_matrix = np.identity(4, dtype=np.float32)
            # Apply translation
            model_matrix[0, 3] = pos[0]
            model_matrix[1, 3] = pos[1]
            model_matrix[2, 3] = pos[2]
            # Apply rotation to align with bitangent
            rotation_matrix = self._create_rotation_matrix_from_vectors(np.array([0, 0, 1]), bitangent)
            model_matrix = pyrr.matrix44.multiply(model_matrix, rotation_matrix)
            # Apply scale
            scale_matrix = pyrr.matrix44.create_from_scale([arrow_scale_val, arrow_scale_val, arrow_scale_val])
            model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
            
            # Set model matrix
            self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
            
            # Draw bitangent vector (green)
            self.simple_shader.set_uniform_3fv("color", self.marker_bitangent_color)
            glBindVertexArray(self.arrow_vao)
            glDrawArrays(GL_LINES, 0, 2)  # Draw line part
            glDrawArrays(GL_TRIANGLE_FAN, 2, 10)  # Draw cone part
            glBindVertexArray(0)
    
    def _create_rotation_matrix_from_vectors(self, source, target):
        """
        Create a rotation matrix that rotates from source vector to target vector.
        
        Args:
            source: Source vector
            target: Target vector
            
        Returns:
            4x4 rotation matrix
        """
        source = source / np.linalg.norm(source)
        target = target / np.linalg.norm(target)
        
        # Calculate the rotation axis
        axis = np.cross(source, target)
        axis_length = np.linalg.norm(axis)
        
        if axis_length < 1e-6:
            # Vectors are parallel
            if np.dot(source, target) > 0:
                # Same direction
                return np.identity(4, dtype=np.float32)
            else:
                # Opposite direction
                # Find a perpendicular vector to rotate around
                if abs(source[0]) < abs(source[1]):
                    if abs(source[0]) < abs(source[2]):
                        axis = np.array([1, 0, 0])
                    else:
                        axis = np.array([0, 0, 1])
                else:
                    if abs(source[1]) < abs(source[2]):
                        axis = np.array([0, 1, 0])
                    else:
                        axis = np.array([0, 0, 1])
                axis = np.cross(source, axis)
                axis = axis / np.linalg.norm(axis)
                angle = np.pi
        else:
            # Normalize the axis
            axis = axis / axis_length
            # Calculate the rotation angle
            angle = np.arccos(np.dot(source, target))
        
        # Create the rotation matrix
        rotation_matrix = pyrr.matrix44.create_from_axis_rotation(axis, angle)
        return rotation_matrix

    def _draw_cameras(self, view_matrix, projection_matrix):
        """
        Draw the camera positions and view frustums with sequential numbering.
        
        Args:
            view_matrix: View matrix
            projection_matrix: Projection matrix
        """
        # Sort timestamps to ensure cameras are numbered in chronological order
        sorted_timestamps = sorted(self.camera_poses.keys())
        
        # Draw each camera in timestamp order
        for i, timestamp in enumerate(sorted_timestamps):
            pose = self.camera_poses[timestamp]
            
            # Extract camera position from the pose matrix
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            camera_pos = translation
            logger.log(logger.DEBUG, f"Camera {i}: Position: {camera_pos}, Rotation: {rotation}")
            
            # Use simple shader for drawing the camera frustum
            self.simple_shader.use()
            
            # Create model matrix for the camera
            # In OpenGL, transformations are applied in reverse order
            # First create translation matrix
            # translation_matrix = pyrr.matrix44.create_from_translation(camera_pos)
            
            # # Create rotation matrix from the camera orientation
            # rotation_matrix = np.identity(4, dtype=np.float32)
            # rotation_matrix[:3, :3] = rotation
            
            # # Combine transformations: first rotate, then translate
            # model_matrix = pyrr.matrix44.multiply(translation_matrix, rotation_matrix)
            
            model_matrix = pose.T
            
            # Log the model matrix for debugging
            logger.log(logger.DEBUG, f"Camera {i} model matrix:\n{model_matrix}")
            
            # Set matrices
            self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
            self.simple_shader.set_uniform_matrix4fv("view", view_matrix)
            self.simple_shader.set_uniform_matrix4fv("projection", projection_matrix)
            
            # Draw camera frustum (cyan)
            self.simple_shader.set_uniform_3fv("color", self.camera_frustum_color)
            
            # Create and draw camera frustum lines
            # The camera's local coordinate system has:
            # - Z axis pointing forward (viewing direction)
            # - Y axis pointing up
            # - X axis pointing right
            vertices = np.array([
                # Front face (at camera position)
                0, 0, 0,
                0.1, 0.1, 0.2,  # Changed to positive Z for forward direction
                
                0, 0, 0,
                -0.1, 0.1, 0.2,
                
                0, 0, 0,
                -0.1, -0.1, 0.2,
                
                0, 0, 0,
                0.1, -0.1, 0.2,
                
                # Back face
                0.1, 0.1, 0.2,
                -0.1, 0.1, 0.2,
                
                -0.1, 0.1, 0.2,
                -0.1, -0.1, 0.2,
                
                -0.1, -0.1, 0.2,
                0.1, -0.1, 0.2,
                
                0.1, -0.1, 0.2,
                0.1, 0.1, 0.2,
                
                # Viewing direction
                0, 0, 0,
                0, 0, 0.3  # Changed to positive Z for forward direction
            ], dtype=np.float32)
            
            # Create temporary VAO for the frustum
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            
            # Create VBO for vertices
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)
            
            # Draw lines
            glDrawArrays(GL_LINES, 0, len(vertices) // 3)
            
            # Clean up
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(1, [vbo])
            
            # Draw camera number at frustum position
            # For simplicity, we'll use a timestamp-based identifier
            # In a real application, you might want to use a more user-friendly identifier
            camera_id = str(pd.Timestamp(timestamp).timestamp())
            
            # This part is tricky with modern OpenGL
            # For text rendering, we'll use the legacy OpenGL functions
            # In a real application, you might want to use a proper text rendering library
            glUseProgram(0)  # Disable shaders
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadMatrixf(projection_matrix.flatten('F'))
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            # Use the same model matrix as for the frustum
            combined_matrix = pyrr.matrix44.multiply(view_matrix, model_matrix)
            glLoadMatrixf(combined_matrix.flatten('F'))
            
            glColor3f(self.camera_text_color[0], self.camera_text_color[1], self.camera_text_color[2])  # Use configured Yellow text for visibility
            glRasterPos3f(0, 0, 0.2)  # Position the text at the center of the back face (positive Z)
            
            for c in camera_id:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            
            # Re-enable shaders
            self.simple_shader.use()
    
    def _draw_contents(self, view_matrix, projection_matrix, eye_pos):
        """
        Draw the MR contents using shaders.
        
        Args:
            view_matrix: View matrix
            projection_matrix: Projection matrix
            eye_pos: Camera position
        """
        # Use the first scene if available
        # self.scenes is expected to be the dictionary loaded from a single scene JSON file,
        # which should contain an "objects" key.
        if not self.scenes or not isinstance(self.scenes, dict):
            logger.log(logger.DEBUG, "_draw_contents: self.scenes is empty or not a dict.")
            return
        
        scene_objects = self.scenes.get("objects", {})
        if not isinstance(scene_objects, dict):
            logger.log(logger.WARNING, f"_draw_contents: 'objects' key in scene_data is not a dictionary or is missing.")
            return
        
        if not scene_objects:
            logger.log(logger.DEBUG, "_draw_contents: No objects found in scene_objects.")
            return

        # scene_name = next(iter(self.scenes)) # Old logic
        # scene_data = self.scenes[scene_name] # Old logic
        
        # for obj_id, obj_data in scene_data.items(): # Old logic, scene_data could be a string
        for obj_id, obj_data in scene_objects.items(): # New logic
            if not isinstance(obj_data, dict):
                logger.log(logger.WARNING, f"_draw_contents: Object data for '{obj_id}' is not a dictionary, skipping.")
                continue

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
                
                # Use shader
                self.shader.use()
                
                # Create model matrix for the object
                model_matrix = np.identity(4, dtype=np.float32)
                
                # Apply object transform
                if 'position' in obj_data and 'rotation' in obj_data:
                    # Apply position
                    position = obj_data['position']
                    translation_matrix = pyrr.matrix44.create_from_translation(
                        [position['x'], position['y'], position['z']]
                    )
                    model_matrix = pyrr.matrix44.multiply(model_matrix, translation_matrix)
                    
                    # Apply rotation
                    rotation = obj_data['rotation']
                    if 'w' in rotation:  # Quaternion
                        quat = pyrr.Quaternion([rotation['x'], rotation['y'], rotation['z'], rotation['w']])
                        rotation_matrix = pyrr.matrix44.create_from_quaternion(quat)
                        model_matrix = pyrr.matrix44.multiply(model_matrix, rotation_matrix)
                    else:  # Euler angles
                        rotation_x = pyrr.matrix44.create_from_x_rotation(np.radians(rotation.get('x', 0)))
                        rotation_y = pyrr.matrix44.create_from_y_rotation(np.radians(rotation.get('y', 0)))
                        rotation_z = pyrr.matrix44.create_from_z_rotation(np.radians(rotation.get('z', 0)))
                        rotation_matrix = pyrr.matrix44.multiply(rotation_x, rotation_y)
                        rotation_matrix = pyrr.matrix44.multiply(rotation_matrix, rotation_z)
                        model_matrix = pyrr.matrix44.multiply(model_matrix, rotation_matrix)
                    
                    # Apply scale if provided
                    if 'scale' in obj_data:
                        scale = obj_data['scale']
                        scale_matrix = pyrr.matrix44.create_from_scale(
                            [scale.get('x', 1.0), scale.get('y', 1.0), scale.get('z', 1.0)]
                        )
                        model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
                
                # Set matrices
                self.shader.set_uniform_matrix4fv("model", model_matrix)
                self.shader.set_uniform_matrix4fv("view", view_matrix)
                self.shader.set_uniform_matrix4fv("projection", projection_matrix)
                
                # Set lighting properties
                self.shader.set_uniform_3fv("lightPos", self.light_pos)
                self.shader.set_uniform_3fv("viewPos", eye_pos)
                self.shader.set_uniform_3fv("lightColor", self.light_color)
                self.shader.set_uniform_3fv("objectColor", self.content_color)  # Use configured Green for MR contents
                self.shader.set_uniform_1f("shininess", self.default_shininess) # Use configured default
                # Pass lighting parameters from config
                self.shader.set_uniform_1f("ambientStrength", self.ambient_strength)
                self.shader.set_uniform_1f("specularStrength", self.specular_strength)
                
                # Draw model
                model = self.model_cache[model_path]
                model.draw()
    
    def _load_content_model(self, model_path: str):
        """
        Load a content model from a file.
        
        Args:
            model_path: Path to the model file
        """
        logger.log(logger.SYSTEM, f"Loading content model from {model_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                logger.log(logger.WARNING, f"Model file does not exist: {model_path}")
                
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
                    logger.log(logger.DEBUG, f"Trying alternative path: {alt_path}")
                    if os.path.exists(alt_path):
                        logger.log(logger.DEBUG, f"Found model at alternative path: {alt_path}")
                        model_path = alt_path
                        found = True
                        break
                
                if not found:
                    logger.log(logger.ERROR, f"Could not find model file: {os.path.basename(model_path)}")
                    return
            
            # Use pyassimp to load the model
            processing_flags = (
                pyassimp.postprocess.aiProcess_Triangulate |
                pyassimp.postprocess.aiProcess_GenNormals
            )
            
            # Use with statement to properly handle the context manager
            with pyassimp.load(model_path, processing=processing_flags) as scene:
                if not scene or not scene.meshes:
                    logger.log(logger.ERROR, f"No meshes found in {model_path}")
                    return
                
                # Create a model from the scene
                model = Model()
                
                # Process each mesh
                for mesh in scene.meshes:
                    # Extract vertices, normals, and indices
                    vertices = mesh.vertices.astype(np.float32)
                    normals = mesh.normals.astype(np.float32)
                    
                    # Convert faces to indices
                    indices = np.array([idx for face in mesh.faces for idx in face], dtype=np.uint32)
                    
                    # Create mesh
                    mesh_obj = Mesh(vertices, normals, indices)
                    model.add_mesh(mesh_obj)
                
                # Store the model for rendering
                self.model_cache[model_path] = model
                logger.log(logger.SYSTEM, f"Content model loaded with {len(scene.meshes)} meshes")
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading content model: {e}")

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
        elif key == 'q' or key == chr(27):  # ESC key
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

    # --- IRenderer Interface Methods ---

    def render_frame(self, frame, models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        """
        Render a frame with the given models and scene data.
        This visualization renderer primarily uses its internal state set via
        setup_scene, set_camera_view etc. It will attempt to render
        based on that. The arguments might be used in a future refactor
        to more directly control what's rendered per call.

        Returns:
            Rendered image as a numpy array
        """
        logger.log(logger.SYSTEM, "render_frame called. For VisualizeModelRender, this triggers a redraw and returns the buffer.")
        
        # Ensure display callback updates the buffer
        self._display_callback() # This will draw to the back buffer

        # Read pixels from the front buffer (after swap in _display_callback)
        # or back buffer if called mid-drawing cycle.
        # For simplicity and consistency with typical GLUT loops, we'll assume _display_callback
        # has been called and buffers swapped.
        # Reading from GL_FRONT might be problematic if called out of sync.
        # A more robust way would be to render to an FBO.
        
        glReadBuffer(GL_BACK) # Or GL_FRONT, depending on when this is called relative to glutSwapBuffers
        image_buffer = glReadPixels(0, 0, self.window_width, self.window_height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(self.window_height, self.window_width, 3)
        
        # OpenGL reads pixels from bottom-left, so flip it vertically
        return np.flipud(image)

    def render_depth(self, models: Dict[str, Any], scene_data: Dict, camera_matrix: np.ndarray) -> np.ndarray:
        """
        Render a depth map for the given models and scene data.
        NOTE: This is a stub. A proper implementation would require a depth-specific
        shader and rendering pass, possibly to an FBO.
        
        Args:
            models: Dictionary mapping model IDs to model objects
            scene_data: Dictionary containing scene description
            camera_matrix: Camera matrix for the view (currently not used by this stub)
            
        Returns:
            Depth map as a numpy array (currently a placeholder)
        """
        logger.log(logger.WARNING, "render_depth is not fully implemented. Returning a placeholder.")
        # This would involve setting up the view from camera_matrix,
        # rendering geometry (possibly with a specific depth shader),
        # and then reading the depth buffer.
        
        # For now, let's trigger a normal display callback and try to read its depth.
        # This is NOT a correct way to get a depth map for arbitrary camera_matrix.
        self._display_callback()

        depth_buffer = glReadPixels(0, 0, self.window_width, self.window_height, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth_image = np.frombuffer(depth_buffer, dtype=np.float32).reshape(self.window_height, self.window_width)
        
        # OpenGL reads pixels from bottom-left, so flip it vertically
        return np.flipud(depth_image)

    def set_camera(self, position: Tuple[float, float, float], 
                  target: Tuple[float, float, float], 
                  up: Tuple[float, float, float]) -> None:
        """
        Set the camera position, target, and up vector for free view.
        
        Args:
            position: Camera position as (x, y, z)
            target: Camera target as (x, y, z)
            up: Camera up vector as (x, y, z)
        """
        self.camera_pos = np.array(position, dtype=np.float32)
        self.camera_target = np.array(target, dtype=np.float32)
        self.camera_up = np.array(up, dtype=np.float32)
        self.view_mode = "free" # Assume setting camera this way implies free view
        self.rotation_x = 0.0 # Reset rotation if camera is set directly
        self.rotation_y = 0.0
        self.zoom = 1.0
        logger.log(logger.SYSTEM, f"Camera set to: pos={position}, target={target}, up={up}")
        if self.window_id is not None: # Only if window exists
             glutPostRedisplay()

    def set_projection(self, fovy: float, aspect: float, near: float, far: float) -> None:
        """
        Set the projection parameters.
        
        Args:
            fovy: Field of view in degrees
            aspect: Aspect ratio (width / height) - Note: aspect is usually derived from window, but can be overridden.
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self.fovy = fovy
        # Aspect ratio will be recalculated based on window size in _display_callback,
        # but we can store a preferred aspect if needed, or use the one passed.
        # For now, fovy, near, far are stored. Aspect is dynamic.
        self.near_clip = near
        self.far_clip = far
        logger.log(logger.SYSTEM, f"Projection set to: fovy={fovy}, near={near}, far={far}")
        if self.window_id is not None: # Only if window exists
            glutPostRedisplay()

    def cleanup(self) -> None:
        """
        Clean up resources used by the renderer.
        """
        logger.log(logger.SYSTEM, "Cleaning up VisualizeModelRender resources.")
        
        # Delete scene model and cached models
        if self.scene_model:
            self.scene_model.delete()
            self.scene_model = None
        
        for model_path in list(self.model_cache.keys()): # list() to avoid modification during iteration
            model = self.model_cache.pop(model_path)
            if model:
                model.delete()
        self.model_cache.clear()

        # Delete basic geometry VAOs (VBOs/EBOs are typically associated with VAOs or deleted separately if not)
        # Assuming these are simple VAOs with their buffers.
        # A more robust cleanup would involve storing VBO/EBO handles and deleting them explicitly.
        if self.cube_vao:
            glDeleteVertexArrays(1, [self.cube_vao])
            self.cube_vao = None
        if self.cube_vbo_vertices:
            glDeleteBuffers(1, [self.cube_vbo_vertices])
            self.cube_vbo_vertices = None
        if self.cube_vbo_normals:
            glDeleteBuffers(1, [self.cube_vbo_normals])
            self.cube_vbo_normals = None
        if self.cube_ebo:
            glDeleteBuffers(1, [self.cube_ebo])
            self.cube_ebo = None
            
        if self.grid_vao:
            glDeleteVertexArrays(1, [self.grid_vao])
            self.grid_vao = None
        if self.grid_vbo:
            glDeleteBuffers(1, [self.grid_vbo])
            self.grid_vbo = None
            
        if self.axes_vao:
            glDeleteVertexArrays(1, [self.axes_vao])
            self.axes_vao = None
        if self.axes_vbo:
            glDeleteBuffers(1, [self.axes_vbo])
            self.axes_vbo = None
            
        if self.arrow_vao:
            glDeleteVertexArrays(1, [self.arrow_vao])
            self.arrow_vao = None
        if self.arrow_vbo:
            glDeleteBuffers(1, [self.arrow_vbo])
            self.arrow_vbo = None

        # Delete shaders
        if self.shader and self.shader.program_id is not None:
            glDeleteProgram(self.shader.program_id)
            self.shader = None
        if self.simple_shader and self.simple_shader.program_id is not None:
            glDeleteProgram(self.simple_shader.program_id)
            self.simple_shader = None
            
        # If GLUT window was created and we are responsible for it
        if self.window_id is not None:
            # This can sometimes cause issues if called from a thread not owning the context
            # or if glutMainLoop is still technically running.
            # glutDestroyWindow(self.window_id) # This might be too aggressive or cause errors.
            # glutLeaveMainLoop() might be called by user (e.g. on 'q' press)
            logger.log(logger.SYSTEM, "GLUT window cleanup would happen here if applicable (e.g., glutDestroyWindow).")
            self.window_id = None
        
        logger.log(logger.SYSTEM, "VisualizeModelRender cleanup finished.")

    def load_scene(self, scene_model_path: str):
        """
        Loads the main scene model.
        """
        self._load_scene_model(scene_model_path)
        if self.window_id is not None:
            glutPostRedisplay()
