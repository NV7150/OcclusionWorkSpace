from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple


class IModel(ABC):
    """
    Interface for 3D models.
    
    Defines methods that all model classes must implement to provide a consistent
    way to access and manipulate 3D model data across different implementations.
    """
    
    @abstractmethod
    def get_id(self) -> str:
        """
        Get the unique identifier for this model.
        
        Returns:
            Model ID string
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the display name of this model.
        
        Returns:
            Model name string
        """
        pass
    
    @abstractmethod
    def get_file_path(self) -> str:
        """
        Get the file path from which this model was loaded.
        
        Returns:
            File path string
        """
        pass
    
    @abstractmethod
    def get_meshes(self) -> List[Any]:
        """
        Get all meshes in this model.
        
        Returns:
            List of mesh objects
        """
        pass
    
    @abstractmethod
    def get_mesh_by_name(self, name: str) -> Optional[Any]:
        """
        Get a mesh by its name.
        
        Args:
            name: Name of the mesh to retrieve
            
        Returns:
            Mesh object or None if not found
        """
        pass
    
    @abstractmethod
    def get_mesh_by_index(self, index: int) -> Optional[Any]:
        """
        Get a mesh by its index.
        
        Args:
            index: Index of the mesh to retrieve
            
        Returns:
            Mesh object or None if index is out of range
        """
        pass
    
    @abstractmethod
    def get_materials(self) -> List[Any]:
        """
        Get all materials in this model.
        
        Returns:
            List of material objects
        """
        pass
    
    @abstractmethod
    def get_material_by_name(self, name: str) -> Optional[Any]:
        """
        Get a material by its name.
        
        Args:
            name: Name of the material to retrieve
            
        Returns:
            Material object or None if not found
        """
        pass
    
    @abstractmethod
    def get_material_by_index(self, index: int) -> Optional[Any]:
        """
        Get a material by its index.
        
        Args:
            index: Index of the material to retrieve
            
        Returns:
            Material object or None if index is out of range
        """
        pass
    
    @abstractmethod
    def get_textures(self) -> List[Any]:
        """
        Get all textures in this model.
        
        Returns:
            List of texture objects
        """
        pass
    
    @abstractmethod
    def get_texture_by_name(self, name: str) -> Optional[Any]:
        """
        Get a texture by its name.
        
        Args:
            name: Name of the texture to retrieve
            
        Returns:
            Texture object or None if not found
        """
        pass
    
    @abstractmethod
    def get_bounding_box(self) -> Tuple[List[float], List[float]]:
        """
        Get the axis-aligned bounding box of the model.
        
        Returns:
            Tuple of (min_point, max_point) where each point is [x, y, z]
        """
        pass
    
    @abstractmethod
    def get_center(self) -> List[float]:
        """
        Get the center point of the model.
        
        Returns:
            Center point as [x, y, z]
        """
        pass
    
    @abstractmethod
    def get_scale(self) -> List[float]:
        """
        Get the scale of the model.
        
        Returns:
            Scale as [x, y, z]
        """
        pass
    
    @abstractmethod
    def set_scale(self, scale: List[float]) -> None:
        """
        Set the scale of the model.
        
        Args:
            scale: Scale as [x, y, z]
        """
        pass
    
    @abstractmethod
    def prepare_for_rendering(self) -> bool:
        """
        Prepare the model for rendering (e.g., upload to GPU).
        
        Returns:
            True if preparation was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def release_resources(self) -> None:
        """
        Release any resources used by the model (e.g., GPU memory).
        """
        pass