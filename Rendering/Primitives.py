import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from OpenGL.GL import *
from Utils.Logger import logger

class Primitives:
    """
    Class for creating and rendering basic geometric primitives.
    
    This class provides methods for creating and rendering basic shapes like
    cubes, spheres, cylinders, planes, and grids for debugging and visualization.
    """
    
    def __init__(self, buffer_manager=None, shader_manager=None):
        """
        Initialize the Primitives.
        
        Args:
            buffer_manager: BufferManager instance for creating and managing buffers
            shader_manager: ShaderManager instance for managing shaders
        """
        self.buffer_manager = buffer_manager
        self.shader_manager = shader_manager
        self.primitives = {}  # Dictionary mapping primitive names to primitive data
        
        # Initialize primitives if buffer manager is provided
        if buffer_manager is not None:
            self._initialize_primitives()
    
    def _initialize_primitives(self) -> None:
        """
        Initialize all primitive shapes.
        """
        self._create_cube()
        self._create_sphere()
        self._create_cylinder()
        self._create_plane()
        self._create_grid()
        self._create_axes()
    
    def _create_cube(self) -> None:
        """
        Create a cube primitive.
        """
        # Cube vertices (8 corners)
        vertices = np.array([
            # Front face
            [-0.5, -0.5,  0.5],  # Bottom-left
            [ 0.5, -0.5,  0.5],  # Bottom-right
            [ 0.5,  0.5,  0.5],  # Top-right
            [-0.5,  0.5,  0.5],  # Top-left
            
            # Back face
            [-0.5, -0.5, -0.5],  # Bottom-left
            [ 0.5, -0.5, -0.5],  # Bottom-right
            [ 0.5,  0.5, -0.5],  # Top-right
            [-0.5,  0.5, -0.5]   # Top-left
        ], dtype=np.float32)
        
        # Cube normals
        normals = np.array([
            # Front face
            [0.0, 0.0, 1.0],  # Forward
            [0.0, 0.0, 1.0],  # Forward
            [0.0, 0.0, 1.0],  # Forward
            [0.0, 0.0, 1.0],  # Forward
            
            # Back face
            [0.0, 0.0, -1.0],  # Backward
            [0.0, 0.0, -1.0],  # Backward
            [0.0, 0.0, -1.0],  # Backward
            [0.0, 0.0, -1.0]   # Backward
        ], dtype=np.float32)
        
        # Cube texture coordinates
        uvs = np.array([
            # Front face
            [0.0, 0.0],  # Bottom-left
            [1.0, 0.0],  # Bottom-right
            [1.0, 1.0],  # Top-right
            [0.0, 1.0],  # Top-left
            
            # Back face
            [1.0, 0.0],  # Bottom-left
            [0.0, 0.0],  # Bottom-right
            [0.0, 1.0],  # Top-right
            [1.0, 1.0]   # Top-left
        ], dtype=np.float32)
        
        # Cube indices (6 faces, 2 triangles per face, 3 indices per triangle)
        indices = np.array([
            # Front face
            0, 1, 2,
            0, 2, 3,
            
            # Right face
            1, 5, 6,
            1, 6, 2,
            
            # Back face
            5, 4, 7,
            5, 7, 6,
            
            # Left face
            4, 0, 3,
            4, 3, 7,
            
            # Top face
            3, 2, 6,
            3, 6, 7,
            
            # Bottom face
            4, 5, 1,
            4, 1, 0
        ], dtype=np.uint32)
        
        # Create buffers
        self.buffer_manager.create_mesh_buffers("cube", vertices, indices, normals, uvs)
        
        # Store primitive data
        self.primitives["cube"] = {
            "vertex_count": len(vertices),
            "index_count": len(indices),
            "vao_name": "cube"
        }
    
    def _create_sphere(self, radius: float = 1.0, segments: int = 32) -> None:
        """
        Create a sphere primitive.
        
        Args:
            radius: Radius of the sphere
            segments: Number of segments (resolution)
        """
        vertices = []
        normals = []
        uvs = []
        indices = []
        
        # Generate vertices, normals, and UVs
        for y in range(segments + 1):
            v = y / segments
            phi = v * math.pi
            
            for x in range(segments + 1):
                u = x / segments
                theta = u * 2 * math.pi
                
                # Vertex position
                px = radius * math.sin(phi) * math.cos(theta)
                py = radius * math.cos(phi)
                pz = radius * math.sin(phi) * math.sin(theta)
                vertices.extend([px, py, pz])
                
                # Normal (normalized vertex position)
                nx = math.sin(phi) * math.cos(theta)
                ny = math.cos(phi)
                nz = math.sin(phi) * math.sin(theta)
                normals.extend([nx, ny, nz])
                
                # Texture coordinates
                uvs.extend([u, v])
        
        # Generate indices
        for y in range(segments):
            for x in range(segments):
                # Get indices of quad corners
                i1 = y * (segments + 1) + x
                i2 = i1 + 1
                i3 = i1 + (segments + 1)
                i4 = i3 + 1
                
                # Add two triangles
                indices.extend([i1, i2, i3])
                indices.extend([i2, i4, i3])
        
        # Convert to numpy arrays
        vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        normals = np.array(normals, dtype=np.float32).reshape(-1, 3)
        uvs = np.array(uvs, dtype=np.float32).reshape(-1, 2)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create buffers
        self.buffer_manager.create_mesh_buffers("sphere", vertices, indices, normals, uvs)
        
        # Store primitive data
        self.primitives["sphere"] = {
            "vertex_count": len(vertices),
            "index_count": len(indices),
            "vao_name": "sphere"
        }
    
    def _create_cylinder(self, radius: float = 0.5, height: float = 1.0, segments: int = 32) -> None:
        """
        Create a cylinder primitive.
        
        Args:
            radius: Radius of the cylinder
            height: Height of the cylinder
            segments: Number of segments (resolution)
        """
        vertices = []
        normals = []
        uvs = []
        indices = []
        
        half_height = height / 2
        
        # Generate vertices, normals, and UVs for the sides
        for i in range(segments + 1):
            angle = i * 2 * math.pi / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            u = i / segments
            
            # Bottom vertex
            vertices.extend([x, -half_height, z])
            normals.extend([x / radius, 0, z / radius])
            uvs.extend([u, 0])
            
            # Top vertex
            vertices.extend([x, half_height, z])
            normals.extend([x / radius, 0, z / radius])
            uvs.extend([u, 1])
        
        # Generate indices for the sides
        for i in range(segments):
            i1 = i * 2
            i2 = i1 + 1
            i3 = (i + 1) * 2
            i4 = i3 + 1
            
            indices.extend([i1, i2, i3])
            indices.extend([i2, i4, i3])
        
        # Generate vertices, normals, and UVs for the top and bottom caps
        # Top cap (center vertex)
        top_center_index = len(vertices) // 3
        vertices.extend([0, half_height, 0])
        normals.extend([0, 1, 0])
        uvs.extend([0.5, 0.5])
        
        # Top cap (perimeter vertices)
        for i in range(segments):
            angle = i * 2 * math.pi / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            u = 0.5 + 0.5 * math.cos(angle)
            v = 0.5 + 0.5 * math.sin(angle)
            
            vertices.extend([x, half_height, z])
            normals.extend([0, 1, 0])
            uvs.extend([u, v])
        
        # Bottom cap (center vertex)
        bottom_center_index = len(vertices) // 3
        vertices.extend([0, -half_height, 0])
        normals.extend([0, -1, 0])
        uvs.extend([0.5, 0.5])
        
        # Bottom cap (perimeter vertices)
        for i in range(segments):
            angle = i * 2 * math.pi / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            u = 0.5 + 0.5 * math.cos(angle)
            v = 0.5 + 0.5 * math.sin(angle)
            
            vertices.extend([x, -half_height, z])
            normals.extend([0, -1, 0])
            uvs.extend([u, v])
        
        # Generate indices for the top cap
        for i in range(segments):
            i1 = top_center_index
            i2 = top_center_index + 1 + i
            i3 = top_center_index + 1 + ((i + 1) % segments)
            
            indices.extend([i1, i2, i3])
        
        # Generate indices for the bottom cap
        for i in range(segments):
            i1 = bottom_center_index
            i2 = bottom_center_index + 1 + ((i + 1) % segments)
            i3 = bottom_center_index + 1 + i
            
            indices.extend([i1, i2, i3])
        
        # Convert to numpy arrays
        vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        normals = np.array(normals, dtype=np.float32).reshape(-1, 3)
        uvs = np.array(uvs, dtype=np.float32).reshape(-1, 2)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create buffers
        self.buffer_manager.create_mesh_buffers("cylinder", vertices, indices, normals, uvs)
        
        # Store primitive data
        self.primitives["cylinder"] = {
            "vertex_count": len(vertices),
            "index_count": len(indices),
            "vao_name": "cylinder"
        }
    
    def _create_plane(self, size: float = 1.0) -> None:
        """
        Create a plane primitive.
        
        Args:
            size: Size of the plane (width and height)
        """
        half_size = size / 2
        
        # Plane vertices (4 corners)
        vertices = np.array([
            [-half_size, 0, -half_size],  # Bottom-left
            [ half_size, 0, -half_size],  # Bottom-right
            [ half_size, 0,  half_size],  # Top-right
            [-half_size, 0,  half_size]   # Top-left
        ], dtype=np.float32)
        
        # Plane normals (all pointing up)
        normals = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        # Plane texture coordinates
        uvs = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        
        # Plane indices (2 triangles)
        indices = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32)
        
        # Create buffers
        self.buffer_manager.create_mesh_buffers("plane", vertices, indices, normals, uvs)
        
        # Store primitive data
        self.primitives["plane"] = {
            "vertex_count": len(vertices),
            "index_count": len(indices),
            "vao_name": "plane"
        }
    
    def _create_grid(self, size: float = 10.0, divisions: int = 10) -> None:
        """
        Create a grid primitive.
        
        Args:
            size: Size of the grid
            divisions: Number of divisions
        """
        vertices = []
        indices = []
        
        # Generate grid lines
        half_size = size / 2
        step = size / divisions
        
        # Generate vertices and indices for the grid lines
        index = 0
        
        # X-axis lines
        for i in range(divisions + 1):
            x = -half_size + i * step
            
            # Line from (x, 0, -half_size) to (x, 0, half_size)
            vertices.extend([x, 0, -half_size])
            vertices.extend([x, 0, half_size])
            
            indices.extend([index, index + 1])
            index += 2
        
        # Z-axis lines
        for i in range(divisions + 1):
            z = -half_size + i * step
            
            # Line from (-half_size, 0, z) to (half_size, 0, z)
            vertices.extend([-half_size, 0, z])
            vertices.extend([half_size, 0, z])
            
            indices.extend([index, index + 1])
            index += 2
        
        # Convert to numpy arrays
        vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create buffers
        self.buffer_manager.create_buffer("grid_vertices", GL_ARRAY_BUFFER, vertices)
        self.buffer_manager.create_buffer("grid_indices", GL_ELEMENT_ARRAY_BUFFER, indices)
        
        # Create VAO
        vao_id = self.buffer_manager.create_vao("grid")
        self.buffer_manager.bind_vao("grid")
        
        # Bind vertex buffer
        self.buffer_manager.bind_buffer("grid_vertices")
        
        # Set vertex attribute pointer
        self.buffer_manager.set_vertex_attrib_pointer(0, 3, GL_FLOAT, False, 0, 0)
        
        # Bind index buffer
        self.buffer_manager.bind_buffer("grid_indices")
        
        # Unbind VAO
        self.buffer_manager.unbind_vao()
        
        # Store primitive data
        self.primitives["grid"] = {
            "vertex_count": len(vertices),
            "index_count": len(indices),
            "vao_name": "grid",
            "line_count": len(indices) // 2
        }
    
    def _create_axes(self, size: float = 1.0) -> None:
        """
        Create axes primitive.
        
        Args:
            size: Size of the axes
        """
        # Axes vertices (origin and endpoints)
        vertices = np.array([
            [0, 0, 0],  # Origin
            [size, 0, 0],  # X-axis endpoint
            [0, 0, 0],  # Origin
            [0, size, 0],  # Y-axis endpoint
            [0, 0, 0],  # Origin
            [0, 0, size]   # Z-axis endpoint
        ], dtype=np.float32)
        
        # Axes colors (red for X, green for Y, blue for Z)
        colors = np.array([
            [1, 0, 0, 1],  # Red (X-axis)
            [1, 0, 0, 1],  # Red (X-axis)
            [0, 1, 0, 1],  # Green (Y-axis)
            [0, 1, 0, 1],  # Green (Y-axis)
            [0, 0, 1, 1],  # Blue (Z-axis)
            [0, 0, 1, 1]   # Blue (Z-axis)
        ], dtype=np.float32)
        
        # Axes indices (3 lines)
        indices = np.array([
            0, 1,  # X-axis
            2, 3,  # Y-axis
            4, 5   # Z-axis
        ], dtype=np.uint32)
        
        # Create buffers
        self.buffer_manager.create_buffer("axes_vertices", GL_ARRAY_BUFFER, vertices)
        self.buffer_manager.create_buffer("axes_colors", GL_ARRAY_BUFFER, colors)
        self.buffer_manager.create_buffer("axes_indices", GL_ELEMENT_ARRAY_BUFFER, indices)
        
        # Create VAO
        vao_id = self.buffer_manager.create_vao("axes")
        self.buffer_manager.bind_vao("axes")
        
        # Bind vertex buffer
        self.buffer_manager.bind_buffer("axes_vertices")
        
        # Set vertex attribute pointer
        self.buffer_manager.set_vertex_attrib_pointer(0, 3, GL_FLOAT, False, 0, 0)
        
        # Bind color buffer
        self.buffer_manager.bind_buffer("axes_colors")
        
        # Set color attribute pointer
        self.buffer_manager.set_vertex_attrib_pointer(1, 4, GL_FLOAT, False, 0, 0)
        
        # Bind index buffer
        self.buffer_manager.bind_buffer("axes_indices")
        
        # Unbind VAO
        self.buffer_manager.unbind_vao()
        
        # Store primitive data
        self.primitives["axes"] = {
            "vertex_count": len(vertices),
            "index_count": len(indices),
            "vao_name": "axes",
            "line_count": len(indices) // 2
        }
    
    def render_cube(self, shader_name: str = "default") -> None:
        """
        Render a cube using the specified shader.
        
        Args:
            shader_name: Name of the shader to use
        """
        if "cube" not in self.primitives:
            logger.log(logger.ERROR, "Cube primitive not initialized")
            return
        
        if self.shader_manager is not None:
            self.shader_manager.use_program(shader_name)
        
        primitive = self.primitives["cube"]
        self.buffer_manager.bind_vao(primitive["vao_name"])
        glDrawElements(GL_TRIANGLES, primitive["index_count"], GL_UNSIGNED_INT, None)
        self.buffer_manager.unbind_vao()
    
    def render_sphere(self, shader_name: str = "default") -> None:
        """
        Render a sphere using the specified shader.
        
        Args:
            shader_name: Name of the shader to use
        """
        if "sphere" not in self.primitives:
            logger.log(logger.ERROR, "Sphere primitive not initialized")
            return
        
        if self.shader_manager is not None:
            self.shader_manager.use_program(shader_name)
        
        primitive = self.primitives["sphere"]
        self.buffer_manager.bind_vao(primitive["vao_name"])
        glDrawElements(GL_TRIANGLES, primitive["index_count"], GL_UNSIGNED_INT, None)
        self.buffer_manager.unbind_vao()
    
    def render_cylinder(self, shader_name: str = "default") -> None:
        """
        Render a cylinder using the specified shader.
        
        Args:
            shader_name: Name of the shader to use
        """
        if "cylinder" not in self.primitives:
            logger.log(logger.ERROR, "Cylinder primitive not initialized")
            return
        
        if self.shader_manager is not None:
            self.shader_manager.use_program(shader_name)
        
        primitive = self.primitives["cylinder"]
        self.buffer_manager.bind_vao(primitive["vao_name"])
        glDrawElements(GL_TRIANGLES, primitive["index_count"], GL_UNSIGNED_INT, None)
        self.buffer_manager.unbind_vao()
    
    def render_plane(self, shader_name: str = "default") -> None:
        """
        Render a plane using the specified shader.
        
        Args:
            shader_name: Name of the shader to use
        """
        if "plane" not in self.primitives:
            logger.log(logger.ERROR, "Plane primitive not initialized")
            return
        
        if self.shader_manager is not None:
            self.shader_manager.use_program(shader_name)
        
        primitive = self.primitives["plane"]
        self.buffer_manager.bind_vao(primitive["vao_name"])
        glDrawElements(GL_TRIANGLES, primitive["index_count"], GL_UNSIGNED_INT, None)
        self.buffer_manager.unbind_vao()
    
    def render_grid(self) -> None:
        """
        Render a grid.
        """
        if "grid" not in self.primitives:
            logger.log(logger.ERROR, "Grid primitive not initialized")
            return
        
        primitive = self.primitives["grid"]
        self.buffer_manager.bind_vao(primitive["vao_name"])
        
        # Set line properties
        glLineWidth(1.0)
        glColor4f(0.5, 0.5, 0.5, 1.0)
        
        # Draw grid lines
        glDrawElements(GL_LINES, primitive["index_count"], GL_UNSIGNED_INT, None)
        
        self.buffer_manager.unbind_vao()
    
    def render_axes(self) -> None:
        """
        Render coordinate axes.
        """
        if "axes" not in self.primitives:
            logger.log(logger.ERROR, "Axes primitive not initialized")
            return
        
        primitive = self.primitives["axes"]
        self.buffer_manager.bind_vao(primitive["vao_name"])
        
        # Set line properties
        glLineWidth(3.0)
        
        # Draw axes lines
        glDrawElements(GL_LINES, primitive["index_count"], GL_UNSIGNED_INT, None)
        
        self.buffer_manager.unbind_vao()