from typing import Optional, Dict, List, Any, Tuple
import os
import json
from .BaseSceneLoader import BaseSceneLoader
from Utils.Logger import logger

class JsonSceneLoader(BaseSceneLoader):
    """
    Loader for JSON scene description files.
    
    This class implements the BaseSceneLoader interface for loading scene descriptions
    from JSON files. It handles parsing the JSON and converting it to the internal
    scene representation.
    """
    
    def supports_extension(self, extension: str) -> bool:
        """
        Check if this loader supports a specific file extension.
        
        Args:
            extension: File extension (e.g., 'json')
            
        Returns:
            True if the extension is supported, False otherwise
        """
        return extension.lower() == 'json'
    
    def load_scene(self, file_path: str, scene_id: str) -> Optional[Dict]:
        """
        Load a scene description from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            scene_id: ID to assign to the loaded scene
            
        Returns:
            Dictionary containing scene description or None if loading failed
        """
        if not os.path.exists(file_path):
            logger.log(logger.ERROR, f"Scene file not found: {file_path}")
            return None
        
        try:
            # Load the JSON file
            with open(file_path, 'r') as f:
                scene_data = json.load(f)
            
            # Validate the scene data
            if not isinstance(scene_data, dict):
                logger.log(logger.ERROR, f"Invalid scene format in {file_path}: Root must be an object")
                return None
            
            # Process the scene data
            processed_scene = self._process_scene_data(scene_data, scene_id)
            
            logger.log(logger.MODEL, f"Loaded scene: {file_path}")
            return processed_scene
            
        except json.JSONDecodeError as e:
            logger.log(logger.ERROR, f"Error parsing JSON in {file_path}: {str(e)}")
            return None
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading scene from {file_path}: {str(e)}")
            return None
    
    def save_scene(self, scene_data: Dict, file_path: str) -> bool:
        """
        Save a scene description to a JSON file.
        
        Args:
            scene_data: Dictionary containing scene description
            file_path: Path to save the scene description to
            
        Returns:
            True if the scene was saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert the scene data to JSON format
            json_data = self._convert_scene_to_json(scene_data)
            
            # Save the JSON file
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            logger.log(logger.MODEL, f"Saved scene to: {file_path}")
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error saving scene to {file_path}: {str(e)}")
            return False
    
    def _process_scene_data(self, scene_data: Dict, scene_id: str) -> Dict:
        """
        Process the raw scene data from JSON.
        
        Args:
            scene_data: Raw scene data from JSON
            scene_id: ID to assign to the scene
            
        Returns:
            Processed scene data
        """
        processed_scene = {
            'id': scene_id,
            'objects': {}
        }
        
        # Process each object in the scene
        for obj_id, obj_data in scene_data.items():
            # Skip non-object entries
            if not isinstance(obj_data, dict):
                continue
            
            # Process object data
            processed_obj = {}
            
            # Process position
            if 'position' in obj_data and isinstance(obj_data['position'], dict):
                pos = obj_data['position']
                processed_obj['position'] = (
                    float(pos.get('x', 0.0)),
                    float(pos.get('y', 0.0)),
                    float(pos.get('z', 0.0))
                )
            else:
                processed_obj['position'] = (0.0, 0.0, 0.0)
            
            # Process rotation
            if 'rotation' in obj_data and isinstance(obj_data['rotation'], dict):
                rot = obj_data['rotation']
                
                # Check if rotation is in Euler angles or quaternion
                if 'w' in rot:
                    # Quaternion
                    processed_obj['rotation'] = (
                        float(rot.get('x', 0.0)),
                        float(rot.get('y', 0.0)),
                        float(rot.get('z', 0.0)),
                        float(rot.get('w', 1.0))
                    )
                else:
                    # Euler angles (convert to quaternion)
                    from Utils.TransformUtils import euler_to_quaternion
                    euler = (
                        float(rot.get('x', 0.0)),
                        float(rot.get('y', 0.0)),
                        float(rot.get('z', 0.0))
                    )
                    processed_obj['rotation'] = euler_to_quaternion(euler)
            else:
                processed_obj['rotation'] = (0.0, 0.0, 0.0, 1.0)
            
            # Process scale
            if 'scale' in obj_data and isinstance(obj_data['scale'], dict):
                scale = obj_data['scale']
                processed_obj['scale'] = (
                    float(scale.get('x', 1.0)),
                    float(scale.get('y', 1.0)),
                    float(scale.get('z', 1.0))
                )
            else:
                processed_obj['scale'] = (1.0, 1.0, 1.0)
            
            # Add the processed object to the scene
            processed_scene['objects'][obj_id] = processed_obj
        
        return processed_scene
    
    def _convert_scene_to_json(self, scene_data: Dict) -> Dict:
        """
        Convert the internal scene representation to JSON format.
        
        Args:
            scene_data: Internal scene representation
            
        Returns:
            Scene data in JSON format
        """
        json_data = {}
        
        # Process each object in the scene
        for obj_id, obj_data in scene_data.get('objects', {}).items():
            json_obj = {}
            
            # Process position
            if 'position' in obj_data:
                pos = obj_data['position']
                json_obj['position'] = {
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2]
                }
            
            # Process rotation
            if 'rotation' in obj_data:
                rot = obj_data['rotation']
                if len(rot) == 4:
                    # Quaternion
                    json_obj['rotation'] = {
                        'x': rot[0],
                        'y': rot[1],
                        'z': rot[2],
                        'w': rot[3]
                    }
                else:
                    # Euler angles
                    json_obj['rotation'] = {
                        'x': rot[0],
                        'y': rot[1],
                        'z': rot[2]
                    }
            
            # Process scale
            if 'scale' in obj_data:
                scale = obj_data['scale']
                json_obj['scale'] = {
                    'x': scale[0],
                    'y': scale[1],
                    'z': scale[2]
                }
            
            # Add the JSON object to the scene
            json_data[obj_id] = json_obj
        
        return json_data