from typing import List, Tuple, Optional, Dict, Any
import numpy as np

class Mesh:
    """
    Class representing a mesh component of a 3D model.
    
    A mesh consists of vertices, normals, texture coordinates, and face indices.
    It may also have a material associated with it.
    """
    
    def __init__(self, name: str):
        """
        Initialize a Mesh with a name.
        
        Args:
            name: Name of the mesh
        """
        self.name = name
        self.vertices = []  # List of floats (x, y, z, x, y, z, ...)
        self.normals = []   # List of floats (nx, ny, nz, nx, ny, nz, ...)
        self.uvs = []       # List of floats (u, v, u, v, ...)
        self.indices = []   # List of ints (i1, i2, i3, i1, i2, i3, ...)
        self.material_name = None
        self._bounding_box = None
    
    def set_vertices(self, vertices: List[float]):
        """
        Set the mesh's vertices.
        
        Args:
            vertices: List of vertex coordinates (flattened)
        """
        self.vertices = vertices
        self._bounding_box = None  # Reset bounding box cache
    
    def set_normals(self, normals: List[float]):
        """
        Set the mesh's normals.
        
        Args:
            normals: List of normal vectors (flattened)
        """
        self.normals = normals
    
    def set_uvs(self, uvs: List[float]):
        """
        Set the mesh's texture coordinates.
        
        Args:
            uvs: List of texture coordinates (flattened)
        """
        self.uvs = uvs
    
    def set_indices(self, indices: List[int]):
        """
        Set the mesh's face indices.
        
        Args:
            indices: List of vertex indices for faces
        """
        self.indices = indices
    
    def set_material(self, material_name: str):
        """
        Set the mesh's material.
        
        Args:
            material_name: Name of the material to use
        """
        self.material_name = material_name
    
    def get_vertices(self) -> List[float]:
        """
        Get the mesh's vertices.
        
        Returns:
            List of vertex coordinates (flattened)
        """
        return self.vertices
    
    def get_normals(self) -> List[float]:
        """
        Get the mesh's normals.
        
        Returns:
            List of normal vectors (flattened)
        """
        return self.normals
    
    def get_uvs(self) -> List[float]:
        """
        Get the mesh's texture coordinates.
        
        Returns:
            List of texture coordinates (flattened)
        """
        return self.uvs
    
    def get_indices(self) -> List[int]:
        """
        Get the mesh's face indices.
        
        Returns:
            List of vertex indices for faces
        """
        return self.indices
    
    def get_material_name(self) -> Optional[str]:
        """
        Get the mesh's material name.
        
        Returns:
            Name of the material or None if no material is assigned
        """
        return self.material_name
    
    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get the mesh's axis-aligned bounding box.
        
        Returns:
            Tuple of (min_point, max_point), where each point is a tuple of (x, y, z)
        """
        if self._bounding_box is None:
            # Calculate bounding box from vertices
            if not self.vertices:
                return ((0, 0, 0), (0, 0, 0))
            
            # Reshape vertices into (n, 3) array
            vertices_array = np.array(self.vertices).reshape(-1, 3)
            
            # Get min and max coordinates
            min_point = tuple(vertices_array.min(axis=0))
            max_point = tuple(vertices_array.max(axis=0))
            
            self._bounding_box = (min_point, max_point)
        
        return self._bounding_box
    
    def calculate_normals(self):
        """
        Calculate normals for the mesh if they are not provided.
        
        This method calculates per-vertex normals based on the face normals.
        """
        if self.normals:
            return  # Normals already exist
        
        if not self.vertices or not self.indices:
            return  # No vertices or indices to calculate normals from
        
        # Reshape vertices into (n, 3) array
        vertices_array = np.array(self.vertices).reshape(-1, 3)
        
        # Initialize normals array
        normals_array = np.zeros_like(vertices_array)
        
        # Process each triangle
        for i in range(0, len(self.indices), 3):
            if i + 2 >= len(self.indices):
                break  # Skip incomplete triangles
            
            # Get vertex indices for this triangle
            i1, i2, i3 = self.indices[i], self.indices[i+1], self.indices[i+2]
            
            # Get vertices
            v1 = vertices_array[i1]
            v2 = vertices_array[i2]
            v3 = vertices_array[i3]
            
            # Calculate face normal using cross product
            edge1 = v2 - v1
            edge2 = v3 - v1
            face_normal = np.cross(edge1, edge2)
            
            # Normalize the face normal
            length = np.linalg.norm(face_normal)
            if length > 0:
                face_normal = face_normal / length
            
            # Add face normal to each vertex normal
            normals_array[i1] += face_normal
            normals_array[i2] += face_normal
            normals_array[i3] += face_normal
        
        # Normalize all vertex normals
        for i in range(len(normals_array)):
            length = np.linalg.norm(normals_array[i])
            if length > 0:
                normals_array[i] = normals_array[i] / length
        
        # Flatten the normals array
        self.normals = normals_array.flatten().tolist()