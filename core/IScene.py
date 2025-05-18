from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

class IScene(ABC):
    """
    Interface for scene management.
    
    This interface defines the contract for classes that manage 3D scenes,
    including model instances, transformations, and scene graphs.
    """
    
    @abstractmethod
    def load_scene(self, scene_data: Dict) -> None:
        """
        Load a scene from scene data.
        
        Args:
            scene_data: Dictionary containing scene description
        """
        pass
    
    @abstractmethod
    def add_model_instance(self, model_id: str, instance_id: str, 
                          position: Tuple[float, float, float], 
                          rotation: Tuple[float, float, float, float], 
                          scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        """
        Add a model instance to the scene.
        
        Args:
            model_id: ID of the model
            instance_id: ID for this instance
            position: Position as (x, y, z)
            rotation: Rotation as quaternion (x, y, z, w)
            scale: Scale as (x, y, z), default is (1.0, 1.0, 1.0)
        """
        pass
    
    @abstractmethod
    def remove_model_instance(self, instance_id: str) -> None:
        """
        Remove a model instance from the scene.
        
        Args:
            instance_id: ID of the instance to remove
        """
        pass
    
    @abstractmethod
    def get_model_instances(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all model instances in the scene.
        
        Returns:
            Dictionary mapping instance IDs to instance data
        """
        pass
    
    @abstractmethod
    def get_instance_transform(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the transform of a model instance.
        
        Args:
            instance_id: ID of the instance
            
        Returns:
            Dictionary containing position, rotation, and scale, or None if not found
        """
        pass
    
    @abstractmethod
    def set_instance_transform(self, instance_id: str, 
                              position: Optional[Tuple[float, float, float]] = None, 
                              rotation: Optional[Tuple[float, float, float, float]] = None, 
                              scale: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Set the transform of a model instance.
        
        Args:
            instance_id: ID of the instance
            position: Position as (x, y, z), or None to keep current position
            rotation: Rotation as quaternion (x, y, z, w), or None to keep current rotation
            scale: Scale as (x, y, z), or None to keep current scale
        """
        pass
    
    @abstractmethod
    def get_scene_data(self) -> Dict:
        """
        Get the scene data.
        
        Returns:
            Dictionary containing scene description
        """
        pass