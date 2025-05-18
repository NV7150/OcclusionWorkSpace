import os
import numpy as np
from typing import Dict, Optional, Tuple
from PIL import Image
from OpenGL.GL import *
from Utils.Logger import logger

class TextureManager:
    """
    Class for managing OpenGL textures.
    
    This class handles loading, binding, and managing textures for rendering.
    """
    
    def __init__(self):
        """
        Initialize the TextureManager.
        """
        self.textures = {}  # Dictionary mapping texture names to texture IDs
        self.current_texture = None  # Currently active texture
    
    def load_texture(self, name: str, image_path: str) -> int:
        """
        Load a texture from an image file.
        
        Args:
            name: Name to assign to the texture
            image_path: Path to the image file
            
        Returns:
            OpenGL texture ID
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                logger.log(logger.ERROR, f"Texture file not found: {image_path}")
                return 0
            
            # Load image using PIL
            image = Image.open(image_path)
            
            # Convert to RGBA if needed
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Get image data
            img_data = np.array(image)
            width, height = image.size
            
            # Create texture
            texture_id = self.create_texture_from_data(name, img_data, width, height)
            
            logger.log(logger.RENDER, f"Loaded texture '{name}' from {image_path} with ID {texture_id}")
            return texture_id
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading texture '{name}' from {image_path}: {str(e)}")
            return 0
    
    def create_texture_from_data(self, name: str, data: np.ndarray, width: int, height: int) -> int:
        """
        Create a texture from image data.
        
        Args:
            name: Name to assign to the texture
            data: Image data as a numpy array
            width: Width of the image
            height: Height of the image
            
        Returns:
            OpenGL texture ID
        """
        try:
            # Generate texture ID
            texture_id = glGenTextures(1)
            
            # Bind texture
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
            
            # Generate mipmaps
            glGenerateMipmap(GL_TEXTURE_2D)
            
            # Store texture
            self.textures[name] = texture_id
            
            return texture_id
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating texture '{name}': {str(e)}")
            return 0
    
    def create_depth_texture(self, name: str, width: int, height: int) -> int:
        """
        Create a depth texture for shadow mapping or depth rendering.
        
        Args:
            name: Name to assign to the texture
            width: Width of the texture
            height: Height of the texture
            
        Returns:
            OpenGL texture ID
        """
        try:
            # Generate texture ID
            texture_id = glGenTextures(1)
            
            # Bind texture
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
            
            # Create empty texture
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
            
            # Store texture
            self.textures[name] = texture_id
            
            logger.log(logger.RENDER, f"Created depth texture '{name}' with ID {texture_id}")
            return texture_id
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating depth texture '{name}': {str(e)}")
            return 0
    
    def create_color_texture(self, name: str, width: int, height: int) -> int:
        """
        Create a color texture for framebuffer rendering.
        
        Args:
            name: Name to assign to the texture
            width: Width of the texture
            height: Height of the texture
            
        Returns:
            OpenGL texture ID
        """
        try:
            # Generate texture ID
            texture_id = glGenTextures(1)
            
            # Bind texture
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Create empty texture
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            
            # Store texture
            self.textures[name] = texture_id
            
            logger.log(logger.RENDER, f"Created color texture '{name}' with ID {texture_id}")
            return texture_id
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating color texture '{name}': {str(e)}")
            return 0
    
    def bind_texture(self, name: str, texture_unit: int = 0) -> bool:
        """
        Bind a texture to a texture unit.
        
        Args:
            name: Name of the texture to bind
            texture_unit: Texture unit to bind to (default: 0)
            
        Returns:
            True if the texture was successfully bound, False otherwise
        """
        if name not in self.textures:
            logger.log(logger.ERROR, f"Texture '{name}' not found")
            return False
        
        texture_id = self.textures[name]
        
        # Activate texture unit
        glActiveTexture(GL_TEXTURE0 + texture_unit)
        
        # Bind texture
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        self.current_texture = name
        
        return True
    
    def unbind_texture(self, texture_unit: int = 0) -> None:
        """
        Unbind the current texture from a texture unit.
        
        Args:
            texture_unit: Texture unit to unbind from (default: 0)
        """
        # Activate texture unit
        glActiveTexture(GL_TEXTURE0 + texture_unit)
        
        # Unbind texture
        glBindTexture(GL_TEXTURE_2D, 0)
        
        self.current_texture = None
    
    def get_texture(self, name: str) -> Optional[int]:
        """
        Get a texture by name.
        
        Args:
            name: Name of the texture
            
        Returns:
            OpenGL texture ID or None if not found
        """
        return self.textures.get(name)
    
    def delete_texture(self, name: str) -> bool:
        """
        Delete a texture.
        
        Args:
            name: Name of the texture to delete
            
        Returns:
            True if the texture was successfully deleted, False otherwise
        """
        if name not in self.textures:
            logger.log(logger.ERROR, f"Texture '{name}' not found")
            return False
        
        texture_id = self.textures[name]
        
        # Delete texture
        glDeleteTextures(1, [texture_id])
        
        # Remove from dictionary
        del self.textures[name]
        
        # Reset current texture if it was the deleted one
        if self.current_texture == name:
            self.current_texture = None
        
        logger.log(logger.RENDER, f"Deleted texture '{name}'")
        return True
    
    def cleanup(self) -> None:
        """
        Clean up all textures.
        """
        # Delete all textures
        for name, texture_id in self.textures.items():
            glDeleteTextures(1, [texture_id])
            
        self.textures = {}
        self.current_texture = None
        
        logger.log(logger.RENDER, "Cleaned up all textures")