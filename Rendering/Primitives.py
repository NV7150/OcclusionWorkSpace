import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from OpenGL.GL import *
import math
import ctypes

from .BufferManager import BufferManager
from .ShaderManager import ShaderManager
from ..Logger.Logger import Logger


class Primitives:
    """
    Utility class for rendering basic 3D primitives.
    
    This class provides methods for creating and rendering common geometric primitives
    such as cubes, spheres, cylinders, planes, and coordinate axes. These are useful
    for debugging, visualization, and creating simple scenes.
    """
    
    def __init__(self, buffer_manager: BufferManager, shader_manager: ShaderManager, 
                logger: Optional[Logger] = None):
        """
        Initialize the Primitives utility.
        
        Args:
            buffer_manager: BufferManager instance for managing OpenGL buffers
            shader_manager: ShaderManager instance for managing shaders
            logger: Optional logger instance
        """
        self._buffer_manager = buffer_manager
        self._shader_manager = shader_manager
        self._logger = logger
        
        # Track which primitives have been initialized
        self._initialized_primitives = set()
        
        if self._logger:
            self._logger.log(Logger.RENDER, "Primitives utility initialized")
    
    def init_cube(self) -> None:
        """
        Initialize a unit cube centered at the origin.
        """
        if "cube" in self._initialized_primitives:
            return
        
        # Vertices for a unit cube centered at the origin
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,  # Bottom-left
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,  # Bottom-right
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,  # Top-right
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,  # Top-left
            
            # Back face
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,  # Bottom-left
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,  # Bottom-right
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,  # Top-right
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,  # Top-left
            
            # Left face
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 0.0,  # Bottom-left
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,  # Bottom-right
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 1.0,  # Top-right
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,  # Top-left
            
            # Right face
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 0.0,  # Bottom-left
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 0.0,  # Bottom-right
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0,  # Top-right
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 1.0,  # Top-left
            
            # Bottom face
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 0.0,  # Bottom-left
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 0.0,  # Bottom-right
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 1.0,  # Top-right
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 1.0,  # Top-left
            
            # Top face
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 0.0,  # Bottom-left
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,  # Bottom-right
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 1.0,  # Top-right
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0   # Top-left
        ], dtype=np.float32)
        
        # Indices for drawing the cube with triangles
        indices = np.array([
            0,  1,  2,  2,  3,  0,   # Front face
            4,  5,  6,  6,  7,  4,   # Back face
            8,  9,  10, 10, 11, 8,   # Left face
            12, 13, 14, 14, 15, 12,  # Right face
            16, 17, 18, 18, 19, 16,  # Bottom face
            20, 21, 22, 22, 23, 20   # Top face
        ], dtype=np.uint32)
        
        # Create VAO for the cube
        vao = self._buffer_manager.create_vao("cube")
        glBindVertexArray(vao)
        
        # Create VBO for vertices
        self._buffer_manager.create_vbo("cube_vertices", vertices)
        
        # Create EBO for indices
        self._buffer_manager.create_ebo("cube_indices", indices)
        
        # Set up vertex attributes
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        # Texture coordinate attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        # Unbind VAO
        glBindVertexArray(0)
        
        self._initialized_primitives.add("cube")
        
        if self._logger:
            self._logger.log(Logger.RENDER, "Initialized cube primitive")
    
    def init_grid(self, size: int = 10, divisions: int = 10) -> None:
        """
        Initialize a grid on the XZ plane.
        
        Args:
            size: Size of the grid (default: 10)
            divisions: Number of divisions (default: 10)
        """
        if "grid" in self._initialized_primitives:
            return
        
        # Generate grid vertices
        vertices = []
        
        # Calculate step size
        step = size / divisions
        half_size = size / 2
        
        # Generate vertices for grid lines along X axis
        for i in range(divisions + 1):
            x = -half_size + i * step
            
            # Line along Z axis
            vertices.extend([x, 0.0, -half_size, 0.0, 1.0, 0.0, 0.0, 0.0])
            vertices.extend([x, 0.0, half_size, 0.0, 1.0, 0.0, 1.0, 0.0])
        
        # Generate vertices for grid lines along Z axis
        for i in range(divisions + 1):
            z = -half_size + i * step
            
            # Line along X axis
            vertices.extend([-half_size, 0.0, z, 0.0, 1.0, 0.0, 0.0, 0.0])
            vertices.extend([half_size, 0.0, z, 0.0, 1.0, 0.0, 1.0, 0.0])
        
        # Convert to numpy array
        vertices_array = np.array(vertices, dtype=np.float32)
        
        # Create VAO for the grid
        vao = self._buffer_manager.create_vao("grid")
        glBindVertexArray(vao)
        
        # Create VBO for vertices
        self._buffer_manager.create_vbo("grid_vertices", vertices_array)
        
        # Set up vertex attributes
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        # Texture coordinate attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        # Unbind VAO
        glBindVertexArray(0)
        
        self._initialized_primitives.add("grid")
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Initialized grid primitive with size {size} and {divisions} divisions")
    
    def init_axes(self, length: float = 1.0) -> None:
        """
        Initialize coordinate axes.
        
        Args:
            length: Length of each axis (default: 1.0)
        """
        if "axes" in self._initialized_primitives:
            return
        
        # Vertices for coordinate axes
        vertices = np.array([
            # X axis (red)
            0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0,  # Origin
            length, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0,  # X end
            
            # Y axis (green)
            0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0,  # Origin
            0.0, length, 0.0,  0.0, 1.0, 0.0,  1.0, 0.0,  # Y end
            
            # Z axis (blue)
            0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0,  # Origin
            0.0, 0.0, length,  0.0, 0.0, 1.0,  1.0, 0.0   # Z end
        ], dtype=np.float32)
        
        # Create VAO for the axes
        vao = self._buffer_manager.create_vao("axes")
        glBindVertexArray(vao)
        
        # Create VBO for vertices
        self._buffer_manager.create_vbo("axes_vertices", vertices)
        
        # Set up vertex attributes
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Color attribute (using normal attribute location)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        # Texture coordinate attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        # Unbind VAO
        glBindVertexArray(0)
        
        self._initialized_primitives.add("axes")
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Initialized axes primitive with length {length}")
    
    def render_cube(self, shader_name: str, model_matrix: np.ndarray, 
                   color: List[float] = [1.0, 1.0, 1.0, 1.0]) -> None:
        """
        Render a cube.
        
        Args:
            shader_name: Name of the shader program to use
            model_matrix: 4x4 model matrix
            color: RGBA color (default: white)
        """
        # Initialize cube if not already done
        self.init_cube()
        
        # Use shader program
        self._shader_manager.use(shader_name)
        
        # Set uniforms
        self._shader_manager.set_uniform_matrix4fv(shader_name, "model", model_matrix)
        self._shader_manager.set_uniform_4fv(shader_name, "color", color)
        
        # Bind VAO and draw
        self._buffer_manager.bind_vao("cube")
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def render_grid(self, shader_name: str, model_matrix: np.ndarray, 
                   color: List[float] = [0.5, 0.5, 0.5, 1.0], 
                   size: int = 10, divisions: int = 10) -> None:
        """
        Render a grid.
        
        Args:
            shader_name: Name of the shader program to use
            model_matrix: 4x4 model matrix
            color: RGBA color (default: gray)
            size: Size of the grid (default: 10)
            divisions: Number of divisions (default: 10)
        """
        # Initialize grid if not already done
        self.init_grid(size, divisions)
        
        # Use shader program
        self._shader_manager.use(shader_name)
        
        # Set uniforms
        self._shader_manager.set_uniform_matrix4fv(shader_name, "model", model_matrix)
        self._shader_manager.set_uniform_4fv(shader_name, "color", color)
        
        # Bind VAO and draw
        self._buffer_manager.bind_vao("grid")
        num_vertices = (divisions + 1) * 4  # 2 vertices per line, 2 * (divisions + 1) lines
        glDrawArrays(GL_LINES, 0, num_vertices)
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def render_axes(self, shader_name: str, model_matrix: np.ndarray, length: float = 1.0) -> None:
        """
        Render coordinate axes.
        
        Args:
            shader_name: Name of the shader program to use
            model_matrix: 4x4 model matrix
            length: Length of each axis (default: 1.0)
        """
        # Initialize axes if not already done
        self.init_axes(length)
        
        # Use shader program
        self._shader_manager.use(shader_name)
        
        # Set uniforms
        self._shader_manager.set_uniform_matrix4fv(shader_name, "model", model_matrix)
        
        # Bind VAO and draw
        self._buffer_manager.bind_vao("axes")
        glDrawArrays(GL_LINES, 0, 6)  # 3 axes, 2 vertices each
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        # Delete all VAOs, VBOs, and EBOs
        for primitive in self._initialized_primitives:
            self._buffer_manager.delete_vao(primitive)
            self._buffer_manager.delete_vbo(f"{primitive}_vertices")
            if primitive == "cube":
                self._buffer_manager.delete_ebo(f"{primitive}_indices")
        
        self._initialized_primitives.clear()
        
        if self._logger:
            self._logger.log(Logger.RENDER, "Primitives utility cleaned up")
