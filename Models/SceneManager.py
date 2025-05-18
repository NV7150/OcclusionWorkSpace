from typing import Dict, List, Optional, Any, Tuple
import os
import numpy as np
from core.IScene import IScene
from core.IModel import IModel
from .BaseSceneLoader import BaseSceneLoader
from Utils.TransformUtils import create_transformation_matrix
from Utils.Logger import logger

class SceneManager(IScene):
    """
    Class for managing 3D scenes.
    
    This class implements the IScene interface and provides methods for
    managing model instances, transformations, and scene graphs.
    """
    
    def __init__(self, models: Dict[str, IModel] = None):
        """
        Initialize a SceneManager with optional models.
        
        Args:
            models: Dictionary mapping model IDs to model objects
        """
        self.models = models or {}
        self.instances = {}  # Dictionary mapping instance IDs to instance data
        self.scene_data = {}  # Raw scene data
    
    def load_scene(self, scene_data: Dict) -> None:
        """
        Load a scene from scene data.
        
        Args:
            scene_data: Dictionary containing scene description
        """
        self.scene_data = scene_data
        self.instances = {}
        
        # Process objects in the scene
        for obj_id, obj_data in scene_data.get('objects', {}).items():
            # Add the object as an instance
            self.add_model_instance(
                obj_id,  # Use object ID as model ID
                obj_id,  # Use object ID as instance ID
                obj_data.get('position', (0.0, 0.0, 0.0)),
                obj_data.get('rotation', (0.0, 0.0, 0.0, 1.0)),
                obj_data.get('scale', (1.0, 1.0, 1.0))
            )
    
    def load_scene_from_file(self, file_path: str) -> bool:
        """
        Load a scene from a file.
        
        Args:
            file_path: Path to the scene file
            
        Returns:
            True if the scene was loaded successfully, False otherwise
        """
        # Get the appropriate loader for the file
        loader = BaseSceneLoader.create_loader_for_file(file_path)
        if not loader:
            logger.log(logger.ERROR, f"No suitable loader found for scene file: {file_path}")
            return False
        
        # Load the scene
        scene_id = os.path.splitext(os.path.basename(file_path))[0]
        scene_data = loader.load_scene(file_path, scene_id)
        if not scene_data:
            return False
        
        # Load the scene data
        self.load_scene(scene_data)
        return True
    
    def save_scene_to_file(self, file_path: str) -> bool:
        """
        Save the scene to a file.
        
        Args:
            file_path: Path to save the scene to
            
        Returns:
            True if the scene was saved successfully, False otherwise
        """
        # Get the appropriate loader for the file
        loader = BaseSceneLoader.create_loader_for_file(file_path)
        if not loader:
            logger.log(logger.ERROR, f"No suitable loader found for scene file: {file_path}")
            return False
        
        # Save the scene
        return loader.save_scene(self.get_scene_data(), file_path)
    
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
        # Create instance data
        instance_data = {
            'model_id': model_id,
            'position': position,
            'rotation': rotation,
            'scale': scale,
            'transform': self._calculate_transform(position, rotation, scale)
        }
        
        # Add the instance
        self.instances[instance_id] = instance_data
        
        # Update scene data
        if 'objects' not in self.scene_data:
            self.scene_data['objects'] = {}
        
        self.scene_data['objects'][instance_id] = {
            'position': position,
            'rotation': rotation,
            'scale': scale
        }
    
    def remove_model_instance(self, instance_id: str) -> None:
        """
        Remove a model instance from the scene.
        
        Args:
            instance_id: ID of the instance to remove
        """
        if instance_id in self.instances:
            del self.instances[instance_id]
        
        # Update scene data
        if 'objects' in self.scene_data and instance_id in self.scene_data['objects']:
            del self.scene_data['objects'][instance_id]
    
    def get_model_instances(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all model instances in the scene.
        
        Returns:
            Dictionary mapping instance IDs to instance data
        """
        return self.instances
    
    def get_instance_transform(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the transform of a model instance.
        
        Args:
            instance_id: ID of the instance
            
        Returns:
            Dictionary containing position, rotation, and scale, or None if not found
        """
        if instance_id not in self.instances:
            return None
        
        instance = self.instances[instance_id]
        return {
            'position': instance['position'],
            'rotation': instance['rotation'],
            'scale': instance['scale']
        }
    
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
        if instance_id not in self.instances:
            logger.log(logger.WARNING, f"Instance not found: {instance_id}")
            return
        
        instance = self.instances[instance_id]
        
        # Update position if provided
        if position is not None:
            instance['position'] = position
        
        # Update rotation if provided
        if rotation is not None:
            instance['rotation'] = rotation
        
        # Update scale if provided
        if scale is not None:
            instance['scale'] = scale
        
        # Recalculate transform matrix
        instance['transform'] = self._calculate_transform(
            instance['position'],
            instance['rotation'],
            instance['scale']
        )
        
        # Update scene data
        if 'objects' in self.scene_data and instance_id in self.scene_data['objects']:
            if position is not None:
                self.scene_data['objects'][instance_id]['position'] = position
            if rotation is not None:
                self.scene_data['objects'][instance_id]['rotation'] = rotation
            if scale is not None:
                self.scene_data['objects'][instance_id]['scale'] = scale
    
    def get_scene_data(self) -> Dict:
        """
        Get the scene data.
        
        Returns:
            Dictionary containing scene description
        """
        return self.scene_data
    
    def get_model_for_instance(self, instance_id: str) -> Optional[IModel]:
        """
        Get the model for a specific instance.
        
        Args:
            instance_id: ID of the instance
            
        Returns:
            Model object or None if not found
        """
        if instance_id not in self.instances:
            return None
        
        model_id = self.instances[instance_id]['model_id']
        return self.models.get(model_id)
    
    def get_instance_matrix(self, instance_id: str) -> Optional[np.ndarray]:
        """
        Get the transformation matrix for a specific instance.
        
        Args:
            instance_id: ID of the instance
            
        Returns:
            4x4 transformation matrix or None if not found
        """
        if instance_id not in self.instances:
            return None
        
        return self.instances[instance_id]['transform']
    
    def _calculate_transform(self, position: Tuple[float, float, float], 
                            rotation: Tuple[float, float, float, float], 
                            scale: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate the transformation matrix from position, rotation, and scale.
        
        Args:
            position: Position as (x, y, z)
            rotation: Rotation as quaternion (x, y, z, w)
            scale: Scale as (x, y, z)
            
        Returns:
            4x4 transformation matrix
        """
        # Create transformation matrix from position and rotation
        transform = create_transformation_matrix(position, rotation)
        
        # Apply scale
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale[0]
        scale_matrix[1, 1] = scale[1]
        scale_matrix[2, 2] = scale[2]
        
        return transform @ scale_matrix