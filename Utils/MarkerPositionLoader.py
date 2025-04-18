import json
import os
import numpy as np
from typing import Dict, Any, Optional, Union
from Logger import logger, Logger

class MarkerPositionLoader:
    """
    Utility class for loading and parsing marker position data from JSON files.
    Specifically designed for tracking applications that require marker positions and normals.
    """
    
    @staticmethod
    def load_marker_positions(file_path: str) -> Dict[Union[str, int], Dict[str, np.ndarray]]:
        """
        Load marker positions from a JSON file with the format:
        {
            "{id}": {
                "pos": [x, y, z],
                "norm": [x, y, z]
            }
        }
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary mapping marker IDs to their positions and normals
            
        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If the file contains invalid JSON
            KeyError: If the JSON data is missing required keys
        """
        if not os.path.exists(file_path):
            logger.log(Logger.ERROR, f"Marker positions file not found: {file_path}")
            raise FileNotFoundError(f"Marker positions file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.log(Logger.ERROR, f"Invalid JSON format in {file_path}: {e}")
            raise
        
        # Convert the loaded data to the appropriate format
        marker_positions = {}
        for marker_id, marker_data in data.items():
            # Validate the marker data structure
            if "pos" not in marker_data or "norm" not in marker_data:
                logger.log(Logger.ERROR, f"Marker {marker_id} is missing 'pos' or 'norm' data")
                raise KeyError(f"Marker {marker_id} is missing 'pos' or 'norm' data")
            
            # Convert string IDs to integers if they are numeric
            try:
                id_key = int(marker_id)
            except ValueError:
                id_key = marker_id
            
            # Convert position and normal to numpy arrays
            try:
                pos = np.array(marker_data["pos"], dtype=np.float32)
                norm = np.array(marker_data["norm"], dtype=np.float32)
                
                if pos.shape != (3,) or norm.shape != (3,):
                    logger.log(Logger.ERROR, f"Marker {marker_id} has invalid position or normal dimensions")
                    raise ValueError(f"Marker {marker_id} has invalid position or normal dimensions")
                
                marker_positions[id_key] = {
                    "pos": pos,
                    "norm": norm
                }
            except (TypeError, ValueError) as e:
                logger.log(Logger.ERROR, f"Invalid position or normal data for marker {marker_id}: {e}")
                raise
        
        logger.log(Logger.DEBUG, f"Loaded {len(marker_positions)} marker positions from {file_path}")
        return marker_positions
    
    @staticmethod
    def save_marker_positions(marker_positions: Dict[Union[str, int], Dict[str, np.ndarray]], file_path: str) -> bool:
        """
        Save marker positions to a JSON file.
        
        Args:
            marker_positions: Dictionary mapping marker IDs to their positions and normals
            file_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            output_data = {}
            for marker_id, marker_data in marker_positions.items():
                output_data[str(marker_id)] = {
                    "pos": marker_data["pos"].tolist(),
                    "norm": marker_data["norm"].tolist()
                }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            
            logger.log(Logger.DEBUG, f"Saved marker positions to {file_path}")
            return True
        except Exception as e:
            logger.log(Logger.ERROR, f"Failed to save marker positions to {file_path}: {e}")
            return False