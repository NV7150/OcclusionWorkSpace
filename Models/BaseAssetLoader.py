from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from .Model import Model

class BaseAssetLoader(ABC):
    """
    Abstract base class for asset loaders.
    
    This class defines the interface for loading 3D model assets from various file formats.
    Concrete implementations should handle specific file formats like OBJ, FBX, etc.
    """
    
    @abstractmethod
    def load_model(self, file_path: str, model_id: str) -> Optional[Model]:
        """
        Load a 3D model from a file.
        
        Args:
            file_path: Path to the model file
            model_id: ID to assign to the loaded model
            
        Returns:
            Loaded Model object or None if loading failed
        """
        pass
    
    @abstractmethod
    def supports_extension(self, extension: str) -> bool:
        """
        Check if this loader supports a specific file extension.
        
        Args:
            extension: File extension (e.g., 'obj', 'fbx')
            
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
    def create_loader_for_file(file_path: str) -> Optional['BaseAssetLoader']:
        """
        Create an appropriate loader for a given file based on its extension.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Appropriate loader instance or None if no suitable loader is found
        """
        from .FbxLoader import FbxLoader
        from .ObjLoader import ObjLoader
        
        extension = BaseAssetLoader.get_extension(file_path)
        
        # Try each loader
        loaders = [FbxLoader(), ObjLoader()]
        
        for loader in loaders:
            if loader.supports_extension(extension):
                return loader
        
        return None