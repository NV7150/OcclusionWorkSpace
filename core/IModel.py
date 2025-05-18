from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

class IModel(ABC):
    """
    Interface for 3D model representation.
    
    This interface defines the contract for classes that represent 3D models
    with geometry, materials, and textures.
    """
    
    @abstractmethod
    def get_vertices(self) -> List[float]:
        """
        Get the model's vertices.
        
        Returns:
            List of vertex coordinates (flattened)
        """
        pass
    
    @abstractmethod
    def get_normals(self) -> List[float]:
        """
        Get the model's normals.
        
        Returns:
            List of normal vectors (flattened)
        """
        pass
    
    @abstractmethod
    def get_uvs(self) -> List[float]:
        """
        Get the model's texture coordinates.
        
        Returns:
            List of texture coordinates (flattened)
        """
        pass
    
    @abstractmethod
    def get_indices(self) -> List[int]:
        """
        Get the model's face indices.
        
        Returns:
            List of vertex indices for faces
        """
        pass
    
    @abstractmethod
    def get_materials(self) -> Dict[str, Any]:
        """
        Get the model's materials.
        
        Returns:
            Dictionary mapping material names to material properties
        """
        pass
    
    @abstractmethod
    def get_textures(self) -> Dict[str, Any]:
        """
        Get the model's textures.
        
        Returns:
            Dictionary mapping texture names to texture data
        """
        pass
    
    @abstractmethod
    def get_meshes(self) -> List[Any]:
        """
        Get the model's meshes.
        
        Returns:
            List of mesh objects
        """
        pass
    
    @abstractmethod
    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get the model's axis-aligned bounding box.
        
        Returns:
            Tuple of (min_point, max_point), where each point is a tuple of (x, y, z)
        """
        pass