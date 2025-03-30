import os
import json
import glob
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class ModelLoader:
    """
    ModelLoader is responsible for loading 3D object files and scene descriptions
    from specified directories.
    """
    
    def __init__(self, model_dirs: List[str]):
        """
        Initialize the ModelLoader with a list of model directories.
        
        Args:
            model_dirs: List of directory paths containing the models and scene descriptions
        """
        self.model_dirs = model_dirs
        self.models = {}  # Dictionary to store models by ID
        self.scenes = {}  # Dictionary to store scene descriptions by directory
        
    def load_models_and_scenes(self) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
        """
        Load all 3D models and scene descriptions from the specified directories.
        
        Returns:
            Tuple containing:
            - Dictionary mapping model IDs to model objects
            - Dictionary mapping directory names to scene descriptions
        """
        for model_dir in self.model_dirs:
            # Load scene description JSON file
            json_files = glob.glob(os.path.join(model_dir, '*.json'))
            
            if not json_files:
                print(f"Warning: No JSON scene description found in {model_dir}")
                continue
                
            # Use the first JSON file found (there should be only one per directory)
            scene_file = json_files[0]
            dir_name = os.path.basename(model_dir)
            
            try:
                with open(scene_file, 'r') as f:
                    scene_data = json.load(f)
                    self.scenes[dir_name] = scene_data
            except Exception as e:
                print(f"Error loading scene description from {scene_file}: {e}")
                continue
                
            # Load 3D model files (both OBJ and FBX)
            obj_files = glob.glob(os.path.join(model_dir, '*.obj'))
            fbx_files = glob.glob(os.path.join(model_dir, '*.fbx'))
            model_files = obj_files + fbx_files
            
            for model_file in model_files:
                model_id = os.path.splitext(os.path.basename(model_file))[0]
                file_ext = os.path.splitext(model_file)[1].lower()
                
                # Check if this model ID is referenced in the scene
                if model_id in scene_data:
                    try:
                        # Store the file path and format
                        self.models[model_id] = {
                            'file_path': model_file,
                            'format': file_ext[1:],  # 'obj' or 'fbx' without the dot
                        }
                    except Exception as e:
                        print(f"Error loading model from {model_file}: {e}")
        
        return self.models, self.scenes
    
    def get_model_by_id(self, model_id: str) -> Optional[Any]:
        """
        Get a model by its ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Model object or None if not found
        """
        return self.models.get(model_id)
    
    def get_scene_by_directory(self, directory: str) -> Optional[Dict]:
        """
        Get a scene description by its directory name.
        
        Args:
            directory: Name of the directory containing the scene
            
        Returns:
            Scene description dictionary or None if not found
        """
        return self.scenes.get(directory)
    
    def get_object_transform(self, scene_name: str, object_id: str) -> Optional[Dict]:
        """
        Get the transform (position, rotation) of an object in a scene.
        
        Args:
            scene_name: Name of the scene
            object_id: ID of the object
            
        Returns:
            Dictionary containing position and rotation or None if not found
        """
        scene = self.scenes.get(scene_name)
        if not scene:
            return None
            
        object_data = scene.get(object_id)
        if not object_data:
            return None
        
        position = object_data.get('position', {'x': 0, 'y': 0, 'z': 0})
        rotation = object_data.get('rotation', {'x': 0, 'y': 0, 'z': 0, 'w': 1})
        
        # Pass the rotation data as is - Renderer will handle the conversion
        return {
            'position': position,
            'rotation': rotation
        }