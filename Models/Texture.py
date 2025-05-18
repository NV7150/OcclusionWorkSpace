from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
import os

class Texture:
    """
    Class representing a texture for 3D models.
    
    A texture is an image that can be applied to a 3D model's surface.
    """
    
    def __init__(self, name: str):
        """
        Initialize a Texture with a name.
        
        Args:
            name: Name of the texture
        """
        self.name = name
        self.path = None
        self.data = None
        self.width = 0
        self.height = 0
        self.channels = 0
        self.gl_texture_id = None  # OpenGL texture ID (set by renderer)
    
    def load_from_file(self, path: str) -> bool:
        """
        Load texture data from a file.
        
        Args:
            path: Path to the texture file
            
        Returns:
            True if the texture was loaded successfully, False otherwise
        """
        if not os.path.exists(path):
            print(f"Texture file not found: {path}")
            return False
        
        try:
            # Load image using PIL
            image = Image.open(path)
            
            # Convert to RGBA if needed
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Get image data
            self.data = np.array(image)
            self.width, self.height = image.size
            self.channels = 4  # RGBA
            self.path = path
            
            return True
        except Exception as e:
            print(f"Error loading texture from {path}: {e}")
            return False
    
    def load_from_data(self, data: np.ndarray) -> bool:
        """
        Load texture data from a numpy array.
        
        Args:
            data: Numpy array containing texture data (height, width, channels)
            
        Returns:
            True if the texture was loaded successfully, False otherwise
        """
        if data is None or data.size == 0:
            print("Invalid texture data")
            return False
        
        try:
            # Set texture data
            self.data = data
            self.height, self.width = data.shape[:2]
            self.channels = data.shape[2] if len(data.shape) > 2 else 1
            
            return True
        except Exception as e:
            print(f"Error loading texture from data: {e}")
            return False
    
    def get_data(self) -> Optional[np.ndarray]:
        """
        Get the texture data.
        
        Returns:
            Numpy array containing texture data or None if not loaded
        """
        return self.data
    
    def get_size(self) -> Tuple[int, int]:
        """
        Get the texture size.
        
        Returns:
            Tuple of (width, height)
        """
        return (self.width, self.height)
    
    def get_channels(self) -> int:
        """
        Get the number of channels in the texture.
        
        Returns:
            Number of channels (e.g., 3 for RGB, 4 for RGBA)
        """
        return self.channels
    
    def get_path(self) -> Optional[str]:
        """
        Get the path to the texture file.
        
        Returns:
            Path to the texture file or None if not loaded from a file
        """
        return self.path
    
    def set_gl_texture_id(self, texture_id: int):
        """
        Set the OpenGL texture ID.
        
        Args:
            texture_id: OpenGL texture ID
        """
        self.gl_texture_id = texture_id
    
    def get_gl_texture_id(self) -> Optional[int]:
        """
        Get the OpenGL texture ID.
        
        Returns:
            OpenGL texture ID or None if not set
        """
        return self.gl_texture_id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the texture to a dictionary.
        
        Returns:
            Dictionary representation of the texture
        """
        return {
            'name': self.name,
            'path': self.path,
            'width': self.width,
            'height': self.height,
            'channels': self.channels
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Texture':
        """
        Create a texture from a dictionary.
        
        Args:
            data: Dictionary containing texture properties
            
        Returns:
            Texture object
        """
        texture = cls(data['name'])
        
        if 'path' in data and data['path']:
            texture.load_from_file(data['path'])
        
        return texture