import os
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from ..core.IModel import IModel
from ..core.IScene import IScene, ISceneNode


class SceneNode(ISceneNode):
    """
    Implementation of the ISceneNode interface.
    
    Represents a node in the scene graph hierarchy, with transformation
    and parent-child relationships.
    """
    
    def __init__(self, name: str, transform: Optional[np.ndarray] = None):
        """
        Initialize a SceneNode with the given name and optional transform.
        
        Args:
            name: Name of the node
            transform: 4x4 transformation matrix (default: identity)
        """
        self._name = name
        self._transform = transform if transform is not None else np.identity(4, dtype=np.float32)
        self._children: List[ISceneNode] = []
        self._parent: Optional[ISceneNode] = None
        
    @property
    def name(self) -> str:
        """Get the name of the node."""
        return self._name
    
    @property
    def parent(self) -> Optional[ISceneNode]:
        """Get the parent node."""
        return self._parent
    
    @parent.setter
    def parent(self, parent: Optional[ISceneNode]) -> None:
        """Set the parent node."""
        self._parent = parent
    
    def get_transform(self) -> np.ndarray:
        """Returns the transformation matrix for this node."""
        return self._transform
    
    def set_transform(self, transform: np.ndarray) -> None:
        """Set the transformation matrix for this node."""
        self._transform = transform
    
    def get_children(self) -> List[ISceneNode]:
        """Returns a list of child nodes."""
        return self._children
    
    def add_child(self, child: ISceneNode) -> None:
        """Adds a child node to this node."""
        self._children.append(child)
        # Set parent relationship if child is a SceneNode
        if isinstance(child, SceneNode):
            child.parent = self
    
    def remove_child(self, child: ISceneNode) -> bool:
        """
        Removes a child node from this node.
        
        Returns:
            True if the child was successfully removed, False otherwise
        """
        if child in self._children:
            self._children.remove(child)
            # Clear parent relationship if child is a SceneNode
            if isinstance(child, SceneNode):
                child.parent = None
            return True
        return False
    
    def get_world_transform(self) -> np.ndarray:
        """
        Calculate the world transformation matrix for this node.
        
        This combines this node's transform with all parent transforms
        to get the final world-space transformation.
        
        Returns:
            4x4 world transformation matrix
        """
        if self._parent is None:
            return self._transform
        
        # Recursively combine with parent transforms
        parent_transform = self._parent.get_world_transform()
        return np.dot(parent_transform, self._transform)


class ModelNode(SceneNode):
    """
    A specialized SceneNode that contains a model.
    """
    
    def __init__(self, name: str, model: IModel, transform: Optional[np.ndarray] = None):
        """
        Initialize a ModelNode with the given name, model, and optional transform.
        
        Args:
            name: Name of the node
            model: Model associated with this node
            transform: 4x4 transformation matrix (default: identity)
        """
        super().__init__(name, transform)
        self._model = model
        
    @property
    def model(self) -> IModel:
        """Get the model associated with this node."""
        return self._model
    
    def update_model_transform(self) -> None:
        """
        Update the model's transform to match the node's world transform.
        
        This ensures the model is rendered in world space correctly.
        """
        self._model.set_transform(self.get_world_transform())


class SceneManager(IScene):
    """
    Implementation of the IScene interface.
    
    Manages the scene graph, including loading scene descriptions,
    and managing model instances and their transformations.
    """
    
    def __init__(self):
        """Initialize the SceneManager."""
        self._root_node = SceneNode("Root")
        self._model_nodes: Dict[str, ModelNode] = {}  # Maps instance IDs to ModelNodes
        self._model_lookup: Dict[str, IModel] = {}    # Maps model IDs to Model objects
    
    def load_scene_description(self, scene_path: str) -> bool:
        """
        Load a scene description from a JSON file.
        
        Args:
            scene_path: Path to the scene description file
            
        Returns:
            True if loading was successful, False otherwise
        """
        if not os.path.exists(scene_path):
            print(f"Scene file not found: {scene_path}")
            return False
        
        try:
            with open(scene_path, 'r') as file:
                scene_data = json.load(file)
            
            # Process each model in the scene
            for model_id, model_info in scene_data.items():
                if model_id not in self._model_lookup:
                    print(f"Warning: Model '{model_id}' referenced in scene is not loaded")
                    continue
                
                model = self._model_lookup[model_id]
                
                # Extract position
                position = np.array([
                    model_info.get('position', {}).get('x', 0.0),
                    model_info.get('position', {}).get('y', 0.0),
                    model_info.get('position', {}).get('z', 0.0)
                ], dtype=np.float32)
                
                # Extract rotation (as quaternion [x,y,z,w])
                rotation_dict = model_info.get('rotation', {})
                rotation_quat = np.array([
                    rotation_dict.get('x', 0.0),
                    rotation_dict.get('y', 0.0),
                    rotation_dict.get('z', 0.0),
                    rotation_dict.get('w', 1.0)
                ], dtype=np.float32)
                
                # Convert quaternion to rotation matrix
                rotation_matrix = self._quaternion_to_rotation_matrix(rotation_quat)
                
                # Create transformation matrix
                transform = np.identity(4, dtype=np.float32)
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = position
                
                # Add model instance to scene
                instance_id = str(uuid.uuid4())
                model_node = ModelNode(f"Instance_{instance_id}", model, transform)
                self._root_node.add_child(model_node)
                self._model_nodes[instance_id] = model_node
                
                # Update model's transform
                model_node.update_model_transform()
            
            return True
        
        except Exception as e:
            print(f"Error loading scene description: {e}")
            return False
    
    def register_model(self, model_id: str, model: IModel) -> None:
        """
        Register a model with the scene manager.
        
        Args:
            model_id: ID to associate with the model
            model: Model object
        """
        self._model_lookup[model_id] = model
    
    def add_model_instance(self, model_id: str, model: IModel, 
                          position: np.ndarray, rotation: np.ndarray) -> str:
        """
        Add a model instance to the scene.
        
        Args:
            model_id: ID of the model
            model: Model object
            position: 3D position vector [x, y, z]
            rotation: Rotation as quaternion [x, y, z, w] or euler angles [x, y, z]
            
        Returns:
            Unique instance ID for the added model
        """
        # Check if rotation is quaternion (4 elements) or euler angles (3 elements)
        if rotation.size == 4:
            # Quaternion rotation
            rotation_matrix = self._quaternion_to_rotation_matrix(rotation)
        else:
            # Euler angles rotation (assuming XYZ order)
            rotation_matrix = self._euler_to_rotation_matrix(rotation)
        
        # Create transformation matrix
        transform = np.identity(4, dtype=np.float32)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = position
        
        # Register model if not already registered
        if model_id not in self._model_lookup:
            self._model_lookup[model_id] = model
        
        # Create instance
        instance_id = str(uuid.uuid4())
        model_node = ModelNode(f"Instance_{instance_id}", model, transform)
        self._root_node.add_child(model_node)
        self._model_nodes[instance_id] = model_node
        
        # Update model's transform
        model_node.update_model_transform()
        
        return instance_id
    
    def remove_model_instance(self, instance_id: str) -> bool:
        """
        Remove a model instance from the scene.
        
        Args:
            instance_id: ID of the instance to remove
            
        Returns:
            True if removal was successful, False otherwise
        """
        if instance_id not in self._model_nodes:
            print(f"Warning: Instance '{instance_id}' not found in scene")
            return False
        
        model_node = self._model_nodes[instance_id]
        parent = model_node.parent
        
        if parent is None:
            parent = self._root_node
        
        result = parent.remove_child(model_node)
        if result:
            del self._model_nodes[instance_id]
        
        return result
    
    def get_model_instances(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all model instances in the scene.
        
        Returns:
            Dictionary mapping instance IDs to dictionaries containing model ID, 
            model object, position, and rotation
        """
        result = {}
        for instance_id, model_node in self._model_nodes.items():
            transform = model_node.get_world_transform()
            position = transform[:3, 3]
            
            # Extract rotation as quaternion
            rotation_matrix = transform[:3, :3]
            quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
            
            # Find model ID
            model_id = "unknown"
            for mid, model in self._model_lookup.items():
                if model_node.model == model:
                    model_id = mid
                    break
            
            result[instance_id] = {
                'model_id': model_id,
                'model': model_node.model,
                'position': position,
                'rotation': quaternion
            }
        
        return result
    
    def get_model_instance_transform(self, instance_id: str) -> Optional[np.ndarray]:
        """
        Get the transformation matrix of a model instance.
        
        Args:
            instance_id: ID of the instance
            
        Returns:
            4x4 transformation matrix as numpy array, or None if instance not found
        """
        if instance_id not in self._model_nodes:
            return None
        
        return self._model_nodes[instance_id].get_world_transform()
    
    def set_model_instance_transform(self, instance_id: str, transform: np.ndarray) -> bool:
        """
        Set the transformation matrix of a model instance.
        
        Args:
            instance_id: ID of the instance
            transform: 4x4 transformation matrix as numpy array
            
        Returns:
            True if setting was successful, False otherwise
        """
        if instance_id not in self._model_nodes:
            return False
        
        model_node = self._model_nodes[instance_id]
        model_node.set_transform(transform)
        model_node.update_model_transform()
        
        return True
    
    def set_model_instance_position(self, instance_id: str, position: np.ndarray) -> bool:
        """
        Set the position of a model instance.
        
        Args:
            instance_id: ID of the instance
            position: 3D position vector
            
        Returns:
            True if setting was successful, False otherwise
        """
        if instance_id not in self._model_nodes:
            return False
        
        model_node = self._model_nodes[instance_id]
        transform = model_node.get_transform()
        transform[:3, 3] = position
        model_node.set_transform(transform)
        model_node.update_model_transform()
        
        return True
    
    def set_model_instance_rotation(self, instance_id: str, rotation: np.ndarray) -> bool:
        """
        Set the rotation of a model instance.
        
        Args:
            instance_id: ID of the instance
            rotation: Rotation as quaternion [x, y, z, w] or euler angles [x, y, z]
            
        Returns:
            True if setting was successful, False otherwise
        """
        if instance_id not in self._model_nodes:
            return False
        
        model_node = self._model_nodes[instance_id]
        transform = model_node.get_transform()
        
        # Check if rotation is quaternion (4 elements) or euler angles (3 elements)
        if rotation.size == 4:
            # Quaternion rotation
            rotation_matrix = self._quaternion_to_rotation_matrix(rotation)
        else:
            # Euler angles rotation (assuming XYZ order)
            rotation_matrix = self._euler_to_rotation_matrix(rotation)
        
        transform[:3, :3] = rotation_matrix
        model_node.set_transform(transform)
        model_node.update_model_transform()
        
        return True
    
    def _quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Convert a quaternion to a rotation matrix.
        
        Args:
            quaternion: Quaternion as [x, y, z, w]
            
        Returns:
            3x3 rotation matrix
        """
        x, y, z, w = quaternion
        
        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to rotation matrix
        xx, xy, xz = x*x, x*y, x*z
        yy, yz, zz = y*y, y*z, z*z
        wx, wy, wz = w*x, w*y, w*z
        
        rotation_matrix = np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        ], dtype=np.float32)
        
        return rotation_matrix
    
    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to a rotation matrix.
        
        Args:
            euler_angles: Euler angles as [x, y, z] in radians (XYZ order)
            
        Returns:
            3x3 rotation matrix
        """
        x, y, z = euler_angles
        
        # Rotation around X axis
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ], dtype=np.float32)
        
        # Rotation around Y axis
        ry = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ], dtype=np.float32)
        
        # Rotation around Z axis
        rz = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combine rotations (XYZ order)
        rotation_matrix = np.dot(rz, np.dot(ry, rx))
        
        return rotation_matrix
    
    def _rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert a rotation matrix to a quaternion.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Quaternion as [x, y, z, w]
        """
        m = rotation_matrix
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w], dtype=np.float32)