import numpy as np
from typing import Dict, List, Optional, Tuple

from ..core.IModel import IModel, IMesh, IMaterial
from .Mesh import Mesh
from .Material import Material


class Model(IModel):
    """
    Implementation of the IModel interface.
    
    Represents a complete 3D model consisting of one or more meshes,
    materials, and its transformation in space.
    """
    
    def __init__(self, name: str):
        """
        Initialize a Model with the given name.
        
        Args:
            name: Name/identifier of the model
        """
        self._name = name
        self._meshes: List[IMesh] = []
        self._materials: Dict[str, IMaterial] = {}
        self._transform = np.identity(4, dtype=np.float32)  # Identity matrix (no transformation)
    
    def get_name(self) -> str:
        """Returns the name or identifier of the model."""
        return self._name
    
    def get_meshes(self) -> List[IMesh]:
        """Returns a list of all meshes that make up this model."""
        return self._meshes
    
    def get_materials(self) -> Dict[str, IMaterial]:
        """Returns a dictionary of materials used by this model, keyed by material name."""
        return self._materials
    
    def get_transform(self) -> np.ndarray:
        """
        Get the model's transformation matrix.
        
        Returns:
            4x4 transformation matrix as numpy array
        """
        return self._transform
    
    def set_transform(self, transform: np.ndarray) -> None:
        """
        Set the model's transformation matrix.
        
        Args:
            transform: 4x4 transformation matrix as numpy array
        """
        self._transform = transform
    
    def get_bounding_box(self) -> tuple:
        """
        Returns the axis-aligned bounding box of the model as (min_point, max_point).
        
        This method computes the bounding box by considering all vertices of all meshes,
        transformed by the model's transformation matrix.
        
        Returns:
            Tuple of two numpy arrays representing the min and max points of the bounding box
        """
        if not self._meshes:
            # Return a default bounding box (origin point) if there are no meshes
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        
        # Initialize min and max points with the first vertex of the first mesh
        all_vertices = []
        for mesh in self._meshes:
            vertices = mesh.get_vertices()
            # Transform vertices by the model's transformation matrix
            homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
            transformed_vertices = np.dot(homogeneous_vertices, self._transform.T)[:, :3]
            all_vertices.append(transformed_vertices)
        
        all_vertices = np.vstack(all_vertices)
        min_point = np.min(all_vertices, axis=0)
        max_point = np.max(all_vertices, axis=0)
        
        return min_point, max_point
    
    def add_mesh(self, mesh: IMesh) -> None:
        """
        Add a mesh to the model.
        
        Args:
            mesh: The mesh to add
        """
        self._meshes.append(mesh)
    
    def add_material(self, material: IMaterial) -> None:
        """
        Add a material to the model.
        
        Args:
            material: The material to add
        """
        self._materials[material.get_name()] = material
    
    def set_position(self, position: np.ndarray) -> None:
        """
        Set the position of the model.
        
        Args:
            position: 3D position vector [x, y, z]
        """
        # Create a new transform matrix with the given position
        transform = np.identity(4, dtype=np.float32)
        transform[:3, 3] = position
        
        # Preserve rotation and scale from the current transform
        # by copying the 3x3 rotation/scale matrix
        transform[:3, :3] = self._transform[:3, :3]
        
        self._transform = transform
    
    def set_rotation(self, rotation: np.ndarray) -> None:
        """
        Set the rotation of the model using a rotation matrix.
        
        Args:
            rotation: 3x3 rotation matrix
        """
        # Create a new transform matrix with the current position
        transform = np.identity(4, dtype=np.float32)
        transform[:3, 3] = self._transform[:3, 3]
        
        # Set the rotation part
        transform[:3, :3] = rotation
        
        self._transform = transform
    
    def set_scale(self, scale: float) -> None:
        """
        Set the uniform scale of the model.
        
        Args:
            scale: Uniform scale factor
        """
        # Get the current rotation without scale by normalizing each column
        current_rotation = self._transform[:3, :3].copy()
        for i in range(3):
            col_length = np.linalg.norm(current_rotation[:, i])
            if col_length > 0:
                current_rotation[:, i] /= col_length
        
        # Apply the new scale
        scaled_rotation = current_rotation * scale
        
        # Create a new transform matrix with the current position and new scaled rotation
        transform = np.identity(4, dtype=np.float32)
        transform[:3, 3] = self._transform[:3, 3]
        transform[:3, :3] = scaled_rotation
        
        self._transform = transform
    
    def create_default_cube(size: float = 1.0) -> 'Model':
        """
        Create a default cube model.
        
        Args:
            size: Size of the cube (default: 1.0)
            
        Returns:
            A Model instance with a single cube mesh
        """
        model = Model("Cube")
        
        # Define cube vertices
        half_size = size / 2
        vertices = np.array([
            # Front face
            [-half_size, -half_size,  half_size],
            [ half_size, -half_size,  half_size],
            [ half_size,  half_size,  half_size],
            [-half_size,  half_size,  half_size],
            
            # Back face
            [-half_size, -half_size, -half_size],
            [ half_size, -half_size, -half_size],
            [ half_size,  half_size, -half_size],
            [-half_size,  half_size, -half_size]
        ], dtype=np.float32)
        
        # Define cube indices (triangles)
        indices = np.array([
            # Front face
            [0, 1, 2], [0, 2, 3],
            
            # Back face
            [4, 5, 6], [4, 6, 7],
            
            # Left face
            [0, 3, 7], [0, 7, 4],
            
            # Right face
            [1, 5, 6], [1, 6, 2],
            
            # Top face
            [3, 2, 6], [3, 6, 7],
            
            # Bottom face
            [0, 4, 5], [0, 5, 1]
        ], dtype=np.uint32)
        
        # Define normals
        normals = np.array([
            # Front face
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            
            # Back face
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]
        ], dtype=np.float32)
        
        # Create a simple material
        material = Material("DefaultMaterial")
        model.add_material(material)
        
        # Create the mesh
        mesh = Mesh(
            "CubeMesh",
            vertices=vertices,
            normals=normals,
            indices=indices,
            material_name="DefaultMaterial"
        )
        model.add_mesh(mesh)
        
        return model
    
    def __str__(self) -> str:
        """String representation of the model."""
        return (f"Model({self._name}, {len(self._meshes)} meshes, "
                f"{len(self._materials)} materials)")
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return (f"Model({self._name}, meshes: {[str(m) for m in self._meshes]}, "
                f"materials: {[str(m) for m in self._materials.values()]})")