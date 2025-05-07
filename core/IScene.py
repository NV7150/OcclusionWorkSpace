from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

from .IModel import IModel


class IScene(ABC):
    """
    Interface for scene management.
    
    Defines methods that all scene management classes must implement to provide
    a consistent way to access and manipulate scene data across different implementations.
    """
    
    @abstractmethod
    def load_models_and_scenes(self) -> Tuple[Dict[str, IModel], Dict[str, Dict]]:
        """
        Load all 3D models and scene descriptions from the specified directories.
        
        Returns:
            Tuple containing:
            - Dictionary mapping model IDs to model objects
            - Dictionary mapping scene names to scene descriptions
        """
        pass
    
    @abstractmethod
    def get_model_by_id(self, model_id: str) -> Optional[IModel]:
        """
        Get a model by its ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Model object or None if not found
        """
        pass
    
    @abstractmethod
    def get_all_models(self) -> Dict[str, IModel]:
        """
        Get all loaded models.
        
        Returns:
            Dictionary mapping model IDs to model objects
        """
        pass
    
    @abstractmethod
    def get_scene_by_name(self, scene_name: str) -> Optional[Dict]:
        """
        Get a scene description by its name.
        
        Args:
            scene_name: Name of the scene to retrieve
            
        Returns:
            Scene description dictionary or None if not found
        """
        pass
    
    @abstractmethod
    def get_all_scenes(self) -> Dict[str, Dict]:
        """
        Get all loaded scenes.
        
        Returns:
            Dictionary mapping scene names to scene descriptions
        """
        pass
    
    @abstractmethod
    def get_object_transform(self, scene_name: str, object_id: str) -> Optional[Dict]:
        """
        Get the transform (position, rotation, scale) of an object in a scene.
        
        Args:
            scene_name: Name of the scene
            object_id: ID of the object
            
        Returns:
            Dictionary containing position, rotation, and scale or None if not found
        """
        pass
    
    @abstractmethod
    def set_object_transform(self, scene_name: str, object_id: str, 
                            position: Optional[Dict[str, float]] = None,
                            rotation: Optional[Dict[str, float]] = None,
                            scale: Optional[Dict[str, float]] = None) -> bool:
        """
        Set the transform of an object in a scene.
        
        Args:
            scene_name: Name of the scene
            object_id: ID of the object
            position: Position dictionary with x, y, z keys (optional)
            rotation: Rotation dictionary with x, y, z keys (Euler angles) or x, y, z, w keys (quaternion) (optional)
            scale: Scale dictionary with x, y, z keys (optional)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def add_object_to_scene(self, scene_name: str, object_id: str, 
                           position: Dict[str, float],
                           rotation: Dict[str, float],
                           scale: Optional[Dict[str, float]] = None) -> bool:
        """
        Add an object to a scene.
        
        Args:
            scene_name: Name of the scene
            object_id: ID of the object
            position: Position dictionary with x, y, z keys
            rotation: Rotation dictionary with x, y, z keys (Euler angles) or x, y, z, w keys (quaternion)
            scale: Scale dictionary with x, y, z keys (optional)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def remove_object_from_scene(self, scene_name: str, object_id: str) -> bool:
        """
        Remove an object from a scene.
        
        Args:
            scene_name: Name of the scene
            object_id: ID of the object
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def create_scene(self, scene_name: str) -> bool:
        """
        Create a new scene.
        
        Args:
            scene_name: Name of the scene to create
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_scene(self, scene_name: str) -> bool:
        """
        Delete a scene.
        
        Args:
            scene_name: Name of the scene to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def save_scene(self, scene_name: str, file_path: str) -> bool:
        """
        Save a scene to a file.
        
        Args:
            scene_name: Name of the scene to save
            file_path: Path where the scene should be saved
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_scene(self, file_path: str) -> Optional[str]:
        """
        Load a scene from a file.
        
        Args:
            file_path: Path to the scene file
            
        Returns:
            Name of the loaded scene or None if loading failed
        """
        pass