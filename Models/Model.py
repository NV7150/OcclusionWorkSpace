from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from core.IModel import IModel

class Model(IModel):
    """
    Class representing a 3D model with geometry, materials, and textures.
    
    This class implements the IModel interface and provides methods for
    accessing model data such as vertices, normals, UVs, indices, materials,
    and textures.
    """
    
    def __init__(self, name: str):
        """
        Initialize a Model with a name.
        
        Args:
            name: Name of the model
        """
        self.name = name
        self.meshes = []
        self.materials = {}
        self.textures = {}
        self._bounding_box = None
    
    def add_mesh(self, mesh):
        """
        Add a mesh to the model.
        
        Args:
            mesh: Mesh object to add
        """
        self.meshes.append(mesh)
        self._bounding_box = None  # Reset bounding box cache
    
    def add_material(self, name: str, material):
        """
        Add a material to the model.
        
        Args:
            name: Name of the material
            material: Material object to add
        """
        self.materials[name] = material
    
    def add_texture(self, name: str, texture):
        """
        Add a texture to the model.
        
        Args:
            name: Name of the texture
            texture: Texture object to add
        """
        self.textures[name] = texture
    
    def get_vertices(self) -> List[float]:
        """
        Get the model's vertices.
        
        Returns:
            List of vertex coordinates (flattened)
        """
        vertices = []
        for mesh in self.meshes:
            vertices.extend(mesh.get_vertices())
        return vertices
    
    def get_normals(self) -> List[float]:
        """
        Get the model's normals.
        
        Returns:
            List of normal vectors (flattened)
        """
        normals = []
        for mesh in self.meshes:
            normals.extend(mesh.get_normals())
        return normals
    
    def get_uvs(self) -> List[float]:
        """
        Get the model's texture coordinates.
        
        Returns:
            List of texture coordinates (flattened)
        """
        uvs = []
        for mesh in self.meshes:
            uvs.extend(mesh.get_uvs())
        return uvs
    
    def get_indices(self) -> List[int]:
        """
        Get the model's face indices.
        
        Returns:
            List of vertex indices for faces
        """
        indices = []
        offset = 0
        for mesh in self.meshes:
            mesh_indices = mesh.get_indices()
            # Adjust indices for the combined vertex array
            adjusted_indices = [idx + offset for idx in mesh_indices]
            indices.extend(adjusted_indices)
            offset += len(mesh.get_vertices()) // 3  # Assuming 3 floats per vertex
        return indices
    
    def get_materials(self) -> Dict[str, Any]:
        """
        Get the model's materials.
        
        Returns:
            Dictionary mapping material names to material properties
        """
        return self.materials
    
    def get_textures(self) -> Dict[str, Any]:
        """
        Get the model's textures.
        
        Returns:
            Dictionary mapping texture names to texture data
        """
        return self.textures
    
    def get_meshes(self) -> List[Any]:
        """
        Get the model's meshes.
        
        Returns:
            List of mesh objects
        """
        return self.meshes
    
    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get the model's axis-aligned bounding box.
        
        Returns:
            Tuple of (min_point, max_point), where each point is a tuple of (x, y, z)
        """
        if self._bounding_box is None:
            # Calculate bounding box from all meshes
            if not self.meshes:
                return ((0, 0, 0), (0, 0, 0))
            
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            for mesh in self.meshes:
                mesh_min, mesh_max = mesh.get_bounding_box()
                min_x = min(min_x, mesh_min[0])
                min_y = min(min_y, mesh_min[1])
                min_z = min(min_z, mesh_min[2])
                max_x = max(max_x, mesh_max[0])
                max_y = max(max_y, mesh_max[1])
                max_z = max(max_z, mesh_max[2])
            
            self._bounding_box = ((min_x, min_y, min_z), (max_x, max_y, max_z))
        
        return self._bounding_box
    
    def get_center(self) -> Tuple[float, float, float]:
        """
        Get the center of the model's bounding box.
        
        Returns:
            Center point as (x, y, z)
        """
        min_point, max_point = self.get_bounding_box()
        return (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2
        )
    
    def get_size(self) -> Tuple[float, float, float]:
        """
        Get the size of the model's bounding box.
        
        Returns:
            Size as (width, height, depth)
        """
        min_point, max_point = self.get_bounding_box()
        return (
            max_point[0] - min_point[0],
            max_point[1] - min_point[1],
            max_point[2] - min_point[2]
        )