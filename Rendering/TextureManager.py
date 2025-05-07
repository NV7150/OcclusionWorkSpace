import os
import numpy as np
from PIL import Image # Import Image directly
from OpenGL.GL import *
from typing import Dict, Optional

# Make sure we can import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Logger.Logger import Logger


class TextureManager:
    """
    Manages OpenGL textures, including loading, binding, and cleanup.
    Serves as a central repository for all textures used in the application.
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        if logger:
            self._logger = logger
        else:
            self._logger = Logger(log_level="INFO", log_file_name="texture_manager.log")
            self._logger.warning("TextureManager initialized with a default logger.")
        self.textures: Dict[str, int] = {}
        
    def load_texture(self, name: str, file_path: str, flip_vertical: bool = False) -> int:
        """
        Loads a texture from a file and stores it with the given name.
        
        Args:
            name (str): Name to reference this texture
            file_path (str): Path to the texture file
            flip_vertical (bool): Whether to flip the image vertically upon loading.
                                  OpenGL typically expects textures with origin at bottom-left.
            
        Returns:
            int: OpenGL texture ID if successful, 0 otherwise
        """
        if name in self.textures:
            self._logger.info(f"Texture '{name}' already loaded. Using existing texture ID: {self.textures[name]}.")
            return self.textures[name]
            
        if not os.path.exists(file_path):
            self._logger.error(f"Texture file '{file_path}' not found for texture '{name}'.")
            return 0
        
        texture_id = 0 # Initialize to ensure it's defined in case of early exit
        try:
            self._logger.debug(f"Loading texture '{name}' from '{file_path}'. Flip vertical: {flip_vertical}")
            image = Image.open(file_path)
            if flip_vertical:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Determine format based on image
            if image.mode == 'RGB':
                img_data = np.array(image, dtype=np.uint8)
                format = GL_RGB
                internal_format = GL_RGB8 # Specify sized internal format
            elif image.mode == 'RGBA':
                img_data = np.array(image, dtype=np.uint8)
                format = GL_RGBA
                internal_format = GL_RGBA8 # Specify sized internal format
            elif image.mode == 'L': # Grayscale
                img_data = np.array(image, dtype=np.uint8)
                # For single channel, use GL_RED. Shaders can swizzle: texture(sampler, uv).rrr for grayscale.
                format = GL_RED 
                internal_format = GL_R8 # Store as 8-bit red channel
            else:
                self._logger.warning(f"Texture '{name}' from '{file_path}' has mode '{image.mode}'. Converting to RGBA.")
                image = image.convert('RGBA')
                img_data = np.array(image, dtype=np.uint8)
                format = GL_RGBA
                internal_format = GL_RGBA8
                
            # Create OpenGL texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Crucial for non-standard image widths (not multiple of 4 bytes)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1) 
            
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, image.width, image.height, 
                         0, format, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            
            self.textures[name] = texture_id
            self._logger.info(f"Loaded texture '{name}' (ID: {texture_id}) from {file_path} ({image.width}x{image.height}, mode: {image.mode}).")
            
            image.close()
            glBindTexture(GL_TEXTURE_2D, 0) # Unbind texture after setup
            return texture_id
            
        except Exception as e:
            self._logger.error(f"Error loading texture '{name}' from '{file_path}': {str(e)}")
            if texture_id: # Check if texture_id was generated
                glDeleteTextures(1, [texture_id])
            return 0
    
    def create_texture_from_data(self, name: str, data: np.ndarray, width: int, height: int, 
                                 format=GL_RGB, internal_format=GL_RGB8, # Default to sized format
                                 data_type=GL_UNSIGNED_BYTE, 
                                 min_filter=GL_LINEAR, mag_filter=GL_LINEAR, 
                                 wrap_s=GL_CLAMP_TO_EDGE, wrap_t=GL_CLAMP_TO_EDGE,
                                 generate_mipmaps: bool = False) -> int:
        """
        Creates a texture from provided data.
        """
        if name in self.textures:
            self._logger.info(f"Texture '{name}' already exists. Deleting old texture (ID: {self.textures[name]}) before creating new one.")
            self.delete_texture(name) # Use the class method for proper cleanup
        
        texture_id = 0
        try:
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1) # Common for numpy arrays
            
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 
                         0, format, data_type, data)
            
            if generate_mipmaps or \
               min_filter in (GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST_MIPMAP_LINEAR, 
                              GL_LINEAR_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_NEAREST):
                self._logger.debug(f"Generating mipmaps for texture '{name}'.")
                glGenerateMipmap(GL_TEXTURE_2D)
            
            self.textures[name] = texture_id
            self._logger.info(f"Created texture '{name}' (ID: {texture_id}) from data ({width}x{height}, format: {format}, internal: {internal_format}).")
            
            glBindTexture(GL_TEXTURE_2D, 0) # Unbind texture
            return texture_id
            
        except Exception as e:
            self._logger.error(f"Error creating texture '{name}' from data: {str(e)}")
            if texture_id:
                glDeleteTextures(1, [texture_id])
            return 0
    
    def bind_texture(self, name: str, texture_unit: int = GL_TEXTURE0) -> bool:
        """
        Binds a texture to the specified texture unit.
        """
        if name not in self.textures:
            self._logger.error(f"Cannot bind texture '{name}': not found.")
            return False
            
        glActiveTexture(texture_unit)
        glBindTexture(GL_TEXTURE_2D, self.textures[name])
        # self._logger.debug(f"Bound texture '{name}' (ID: {self.textures[name]}) to unit {texture_unit - GL_TEXTURE0}") # Can be verbose
        return True
    
    def unbind_texture(self, texture_unit: int = GL_TEXTURE0) -> None:
        """Unbinds texture from the specified texture unit by binding texture 0."""
        glActiveTexture(texture_unit)
        glBindTexture(GL_TEXTURE_2D, 0)
        # self._logger.debug(f"Unbound texture from unit {texture_unit - GL_TEXTURE0}")

    def get_texture_id(self, name: str) -> Optional[int]:
        """
        Returns the OpenGL texture ID for the given name.
        """
        texture_id = self.textures.get(name)
        if texture_id is None:
            self._logger.debug(f"Texture ID for '{name}' not found in manager.")
        return texture_id
    
    def delete_texture(self, name: str) -> bool:
        """
        Deletes a texture from GPU memory and removes it from the manager.
        """
        if name not in self.textures:
            self._logger.warning(f"Cannot delete texture '{name}': not found in manager.")
            return False
            
        texture_id_to_delete = self.textures[name]
        glDeleteTextures(1, [texture_id_to_delete]) # Pass as a list or array
        del self.textures[name]
        self._logger.info(f"Deleted texture '{name}' (ID: {texture_id_to_delete}).")
        return True
    
    def cleanup(self):
        """
        Deletes all textures from GPU memory. Should be called during shutdown.
        """
        if not self.textures:
            self._logger.info("TextureManager: No textures to clean up.")
            return

        self._logger.info(f"Cleaning up TextureManager: Deleting {len(self.textures)} textures.")
        texture_ids = list(self.textures.values())
        if texture_ids: # Ensure list is not empty
            glDeleteTextures(len(texture_ids), texture_ids)
        
        self.textures.clear()
        self._logger.info("TextureManager cleanup complete.")