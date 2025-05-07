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


class VisualizeModelRender:
    """
    VisualizeModelRender is responsible for rendering the MR scene visualization.
    It handles rendering the scene model, reference markers, camera positions, and MR contents.
    This implementation uses modern OpenGL with shaders and VBOs.
    """
    
    # Vertex shader source
    VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;
    out vec3 Normal;
    
    void main()
    {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
    """
    
    # Fragment shader source
    FRAGMENT_SHADER = """
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;
    uniform vec3 objectColor;
    uniform float shininess;
    
    out vec4 FragColor;
    
    void main()
    {
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
        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
    """
    
    # Simple color shader for lines and basic shapes
    SIMPLE_VERTEX_SHADER = """
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
    
    SIMPLE_FRAGMENT_SHADER = """
    #version 330 core
    uniform vec3 color;
    
    out vec4 FragColor;
    
    void main()
    {
        FragColor = vec4(color, 1.0);
    }
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
        
        # Shader programs
        self.shader = None
        self.simple_shader = None
        
        # Geometry for basic shapes
        self.cube_vao = None
        self.grid_vao = None
        self.axes_vao = None
        self.arrow_vao = None
        
        # Light properties
        self.light_pos = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        self.light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
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
        
        # Initialize shaders
        self.shader = Shader(self.VERTEX_SHADER, self.FRAGMENT_SHADER)
        self.simple_shader = Shader(self.SIMPLE_VERTEX_SHADER, self.SIMPLE_FRAGMENT_SHADER)
        
        # Set up callbacks
        glutDisplayFunc(self._display_callback)
        glutReshapeFunc(self._reshape_callback)
        glutMouseFunc(self._mouse_callback)
        glutMotionFunc(self._motion_callback)
        glutKeyboardFunc(self._keyboard_callback)
        glutSpecialFunc(self._special_key_callback)
        
        # Create basic geometry
        self._create_basic_geometry()
        
        # Load scene model
        self._load_scene_model(scene_model_path)
        
        logger.log(Logger.SYSTEM, "Renderer initialized with modern OpenGL shaders")
    
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
        vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Create VBO for normals
        vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        # Create EBO for indices
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def _create_grid(self):
        """
        Create a grid VAO for the reference grid.
        """
        grid_size = 5
        grid_step = 0.2
        vertices = []
        
        # Create grid lines
        for i in range(-grid_size, grid_size + 1):
            # X lines
            vertices.extend([i * grid_step, 0, -grid_size * grid_step])
            vertices.extend([i * grid_step, 0, grid_size * grid_step])
            
            # Z lines
            vertices.extend([-grid_size * grid_step, 0, i * grid_step])
            vertices.extend([grid_size * grid_step, 0, i * grid_step])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Create VAO
        self.grid_vao = glGenVertexArrays(1)
        glBindVertexArray(self.grid_vao)
        
        # Create VBO for vertices
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
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
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
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
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
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
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=45.0,
            aspect=self.window_width / self.window_height,
            near=0.1,
            far=100.0
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
        self.simple_shader.set_uniform_3fv("color", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        glBindVertexArray(self.axes_vao)
        glDrawArrays(GL_LINES, 0, 2)
        
        # Draw Y axis (green)
        self.simple_shader.set_uniform_3fv("color", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        glDrawArrays(GL_LINES, 2, 2)
        
        # Draw Z axis (blue)
        self.simple_shader.set_uniform_3fv("color", np.array([0.0, 0.0, 1.0], dtype=np.float32))
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
        self.simple_shader.set_uniform_3fv("color", np.array([0.5, 0.5, 0.5], dtype=np.float32))
        
        # Draw grid
        glBindVertexArray(self.grid_vao)
        grid_size = 5
        glDrawArrays(GL_LINES, 0, (grid_size * 2 + 1) * 4)
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
        self.shader.set_uniform_3fv("objectColor", np.array([0.8, 0.8, 0.8], dtype=np.float32))
        self.shader.set_uniform_1f("shininess", 32.0)
        
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
            scale_matrix = pyrr.matrix44.create_from_scale([0.05, 0.05, 0.05])
            model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
            
            # Set matrices
            self.shader.set_uniform_matrix4fv("model", model_matrix)
            self.shader.set_uniform_matrix4fv("view", view_matrix)
            self.shader.set_uniform_matrix4fv("projection", projection_matrix)
            
            # Set lighting properties
            self.shader.set_uniform_3fv("lightPos", self.light_pos)
            self.shader.set_uniform_3fv("viewPos", eye_pos)
            self.shader.set_uniform_3fv("lightColor", self.light_color)
            self.shader.set_uniform_3fv("objectColor", np.array([1.0, 1.0, 0.0], dtype=np.float32))
            self.shader.set_uniform_1f("shininess", 100.0)
            
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
            scale_matrix = pyrr.matrix44.create_from_scale([0.2, 0.2, 0.2])
            model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
            
            # Set model matrix
            self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
            
            # Draw normal vector (blue)
            self.simple_shader.set_uniform_3fv("color", np.array([0.0, 0.0, 1.0], dtype=np.float32))
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
            scale_matrix = pyrr.matrix44.create_from_scale([0.2, 0.2, 0.2])
            model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
            
            # Set model matrix
            self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
            
            # Draw tangent vector (red)
            self.simple_shader.set_uniform_3fv("color", np.array([1.0, 0.0, 0.0], dtype=np.float32))
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
            scale_matrix = pyrr.matrix44.create_from_scale([0.2, 0.2, 0.2])
            model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)
            
            # Set model matrix
            self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
            
            # Draw bitangent vector (green)
            self.simple_shader.set_uniform_3fv("color", np.array([0.0, 1.0, 0.0], dtype=np.float32))
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
            logger.log(Logger.DEBUG, f"Camera {i}: Position: {camera_pos}, Rotation: {rotation}")
            
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
            logger.log(Logger.DEBUG, f"Camera {i} model matrix:\n{model_matrix}")
            
            # Set matrices
            self.simple_shader.set_uniform_matrix4fv("model", model_matrix)
            self.simple_shader.set_uniform_matrix4fv("view", view_matrix)
            self.simple_shader.set_uniform_matrix4fv("projection", projection_matrix)
            
            # Draw camera frustum (cyan)
            self.simple_shader.set_uniform_3fv("color", np.array([0.0, 0.8, 0.8], dtype=np.float32))
            
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
            
            glColor3f(1.0, 1.0, 0.0)  # Yellow text for visibility
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
                self.shader.set_uniform_3fv("objectColor", np.array([0.0, 0.8, 0.0], dtype=np.float32))  # Green for MR contents
                self.shader.set_uniform_1f("shininess", 100.0)
                
                # Draw model
                model = self.model_cache[model_path]
                model.draw()
    
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
            with pyassimp.load(model_path, processing=processing_flags) as scene:
                if not scene or not scene.meshes:
                    logger.log(Logger.ERROR, f"No meshes found in {model_path}")
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
                logger.log(Logger.SYSTEM, f"Content model loaded with {len(scene.meshes)} meshes")
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
