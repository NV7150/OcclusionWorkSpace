from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from core.IScene import IScene

class BaseSceneLoader(ABC):
    """
    Abstract base class for scene loaders.
    
    This class defines the interface for loading scene descriptions from various file formats.
    Concrete implementations should handle specific file formats like JSON, YAML, etc.
    """
    
    @abstractmethod
    def load_scene(self, file_path: str, scene_id: str) -> Optional[Dict]:
        """
        Load a scene description from a file.
        
        Args:
            file_path: Path to the scene description file
            scene_id: ID to assign to the loaded scene
            
        Returns:
            Dictionary containing scene description or None if loading failed
        """
        pass
    
    @abstractmethod
    def save_scene(self, scene_data: Dict, file_path: str) -> bool:
        """
        Save a scene description to a file.
        
        Args:
            scene_data: Dictionary containing scene description
            file_path: Path to save the scene description to
            
        Returns:
            True if the scene was saved successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def supports_extension(self, extension: str) -> bool:
        """
        Check if this loader supports a specific file extension.
        
        Args:
            extension: File extension (e.g., 'json', 'yaml')
            
        Returns:
            True if the extension is supported, False otherwise
        """
        pass
    
    @staticmethod
    def get_extension(file_path: str) -> str:
        """
        Get the extension of a file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File extension (lowercase, without the dot)
        """
        import os
        _, ext = os.path.splitext(file_path)
        return ext.lower()[1:] if ext else ""
    
    @staticmethod
    def create_loader_for_file(file_path: str) -> Optional['BaseSceneLoader']:
        """
        Create an appropriate loader for a given file based on its extension.
        
        Args:
            file_path: Path to the scene description file
            
        Returns:
            Appropriate loader instance or None if no suitable loader is found
        """
        from .JsonSceneLoader import JsonSceneLoader
        
        extension = BaseSceneLoader.get_extension(file_path)
        
        # Try each loader
        loaders = [JsonSceneLoader()]
        
        for loader in loaders:
            if loader.supports_extension(extension):
                return loader
        
        return None