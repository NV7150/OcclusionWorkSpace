import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Set

from ..core.IModel import IModel


class BaseAssetLoader(ABC):
    """
    Abstract base class for asset loaders.
    
    Provides the interface and common functionality for loading 3D model assets
    from various file formats (e.g. OBJ, FBX).
    """
    
    def __init__(self, supported_extensions: Set[str]):
        """
        Initialize the BaseAssetLoader with supported file extensions.
        
        Args:
            supported_extensions: Set of supported file extensions (e.g. {'.obj', '.fbx'})
        """
        self._supported_extensions = {ext.lower() for ext in supported_extensions}
    
    @abstractmethod
    def load_model(self, file_path: str) -> IModel:
        """
        Load a model from the specified file path.
        
        To be implemented by concrete subclasses for specific file formats.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Loaded model as an IModel instance
        
        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file extension is not supported
            Exception: If loading fails
        """
        pass
    
    def is_supported(self, file_path: str) -> bool:
        """
        Check if the given file is supported by this loader.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file extension is supported, False otherwise
        """
        _, extension = os.path.splitext(file_path)
        return extension.lower() in self._supported_extensions
    
    @staticmethod
    def get_model_id_from_path(file_path: str) -> str:
        """
        Extract a model ID from a file path.
        
        By default, uses the filename without extension as the model ID.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Model ID as a string
        """
        basename = os.path.basename(file_path)
        return os.path.splitext(basename)[0]
    
    @staticmethod
    def check_file_exists(file_path: str) -> None:
        """
        Check if a file exists, and raise an exception if it doesn't.
        
        Args:
            file_path: Path to the file to check
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    @abstractmethod
    def get_supported_extensions(self) -> Set[str]:
        """
        Get the set of file extensions supported by this loader.
        
        Returns:
            Set of supported file extensions (e.g. {'.obj', '.fbx'})
        """
        pass