import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from OpenGL.GL import *
from ..Logger.Logger import Logger


class Framebuffer:
    """
    Manages OpenGL Framebuffer Objects (FBOs).
    
    This class provides a high-level interface for creating and managing framebuffers,
    which are used for off-screen rendering, post-processing effects, and depth map generation.
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the Framebuffer manager.
        
        Args:
            logger: Optional logger instance
        """
        self._logger = logger
        self._framebuffers: Dict[str, int] = {}  # FBO IDs by name
        self._color_textures: Dict[str, int] = {}  # Color attachment texture IDs by FBO name
        self._depth_textures: Dict[str, int] = {}  # Depth attachment texture IDs by FBO name
        self._renderbuffers: Dict[str, int] = {}  # Renderbuffer IDs by FBO name
        self._dimensions: Dict[str, Tuple[int, int]] = {}  # Width, height by FBO name
        
        if self._logger:
            self._logger.log(Logger.RENDER, "Framebuffer manager initialized")
    
    def create_framebuffer(self, name: str, width: int, height: int,
                          color_attachment: bool = True,
                          depth_attachment: bool = True,
                          depth_only: bool = False,
                          multisample: int = 0) -> int:
        """
        Create a framebuffer object with specified attachments.
        
        Args:
            name: Name to identify this framebuffer
            width: Width of the framebuffer in pixels
            height: Height of the framebuffer in pixels
            color_attachment: Whether to create a color attachment (default: True)
            depth_attachment: Whether to create a depth attachment (default: True)
            depth_only: If True, create a depth-only framebuffer (default: False)
            multisample: Number of multisamples for anti-aliasing (default: 0, no multisampling)
            
        Returns:
            OpenGL framebuffer ID
        """
        if name in self._framebuffers:
            if self._logger:
                self._logger.log(Logger.WARNING, f"Framebuffer '{name}' already exists, deleting and recreating")
            self.delete_framebuffer(name)
        
        # Create framebuffer
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        self._framebuffers[name] = fbo
        self._dimensions[name] = (width, height)
        
        # Create attachments
        if color_attachment and not depth_only:
            self._create_color_attachment(name, width, height, multisample)
        
        if depth_attachment:
            self._create_depth_attachment(name, width, height, multisample)
        
        # Set draw buffers
        if depth_only:
            glDrawBuffer(GL_NONE)
            glReadBuffer(GL_NONE)
        
        # Check framebuffer status
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Framebuffer '{name}' is not complete: {status}")
            self.delete_framebuffer(name)
            return 0
        
        # Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        if self._logger:
            attachments = []
            if color_attachment and not depth_only:
                attachments.append("color")
            if depth_attachment:
                attachments.append("depth")
            self._logger.log(Logger.RENDER, 
                           f"Created framebuffer '{name}' ({width}x{height}) with {', '.join(attachments)} attachments")
        
        return fbo
    
    def _create_color_attachment(self, name: str, width: int, height: int, multisample: int) -> None:
        """
        Create a color attachment for a framebuffer.
        
        Args:
            name: Name of the framebuffer
            width: Width in pixels
            height: Height in pixels
            multisample: Number of multisamples
        """
        # Create texture for color attachment
        if multisample > 0:
            # Create a multisampled texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture)
            glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, multisample, GL_RGBA, width, height, GL_TRUE)
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
            
            # Attach texture to framebuffer
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, texture, 0)
        else:
            # Create a regular texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glBindTexture(GL_TEXTURE_2D, 0)
            
            # Attach texture to framebuffer
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        
        self._color_textures[name] = texture
    
    def _create_depth_attachment(self, name: str, width: int, height: int, multisample: int) -> None:
        """
        Create a depth attachment for a framebuffer.
        
        Args:
            name: Name of the framebuffer
            width: Width in pixels
            height: Height in pixels
            multisample: Number of multisamples
        """
        if multisample > 0:
            # Create a multisampled renderbuffer for depth
            rbo = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, rbo)
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, multisample, GL_DEPTH_COMPONENT24, width, height)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)
            
            # Attach renderbuffer to framebuffer
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
            
            self._renderbuffers[name] = rbo
        else:
            # Create a depth texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)
            
            # Attach texture to framebuffer
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0)
            
            self._depth_textures[name] = texture
    
    def bind(self, name: str) -> bool:
        """
        Bind a framebuffer for rendering.
        
        Args:
            name: Name of the framebuffer to bind
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._framebuffers:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot bind framebuffer '{name}': not found")
            return False
        
        glBindFramebuffer(GL_FRAMEBUFFER, self._framebuffers[name])
        
        # Set viewport to match framebuffer dimensions
        width, height = self._dimensions[name]
        glViewport(0, 0, width, height)
        
        return True
    
    def unbind(self) -> None:
        """
        Unbind the currently bound framebuffer and return to default framebuffer.
        """
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def get_color_texture(self, name: str) -> Optional[int]:
        """
        Get the color attachment texture ID for a framebuffer.
        
        Args:
            name: Name of the framebuffer
            
        Returns:
            OpenGL texture ID or None if not found
        """
        return self._color_textures.get(name)
    
    def get_depth_texture(self, name: str) -> Optional[int]:
        """
        Get the depth attachment texture ID for a framebuffer.
        
        Args:
            name: Name of the framebuffer
            
        Returns:
            OpenGL texture ID or None if not found
        """
        return self._depth_textures.get(name)
    
    def get_dimensions(self, name: str) -> Optional[Tuple[int, int]]:
        """
        Get the dimensions of a framebuffer.
        
        Args:
            name: Name of the framebuffer
            
        Returns:
            Tuple of (width, height) or None if not found
        """
        return self._dimensions.get(name)
    
    def resize(self, name: str, width: int, height: int) -> bool:
        """
        Resize a framebuffer and its attachments.
        
        Args:
            name: Name of the framebuffer
            width: New width in pixels
            height: New height in pixels
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._framebuffers:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot resize framebuffer '{name}': not found")
            return False
        
        # Get current properties
        has_color = name in self._color_textures
        has_depth = name in self._depth_textures
        has_renderbuffer = name in self._renderbuffers
        depth_only = not has_color
        multisample = 0  # We would need to store this information
        
        # Delete and recreate
        self.delete_framebuffer(name)
        self.create_framebuffer(name, width, height, has_color, has_depth or has_renderbuffer, depth_only, multisample)
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Resized framebuffer '{name}' to {width}x{height}")
        
        return True
    
    def read_pixels(self, name: str, x: int = 0, y: int = 0, 
                   width: Optional[int] = None, height: Optional[int] = None,
                   format: int = GL_RGBA, type: int = GL_UNSIGNED_BYTE) -> np.ndarray:
        """
        Read pixels from a framebuffer.
        
        Args:
            name: Name of the framebuffer
            x: X offset in pixels (default: 0)
            y: Y offset in pixels (default: 0)
            width: Width to read in pixels (default: full width)
            height: Height to read in pixels (default: full height)
            format: Format of the pixel data (default: GL_RGBA)
            type: Data type of the pixel data (default: GL_UNSIGNED_BYTE)
            
        Returns:
            Numpy array containing the pixel data
        """
        if name not in self._framebuffers:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot read pixels from framebuffer '{name}': not found")
            return np.array([])
        
        # Get dimensions if not specified
        if width is None or height is None:
            fb_width, fb_height = self._dimensions[name]
            width = width or fb_width
            height = height or fb_height
        
        # Bind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self._framebuffers[name])
        
        # Determine the size of the output array
        if format == GL_RGBA:
            channels = 4
        elif format == GL_RGB:
            channels = 3
        elif format == GL_DEPTH_COMPONENT:
            channels = 1
        else:
            channels = 4  # Default
        
        # Determine the data type
        if type == GL_UNSIGNED_BYTE:
            dtype = np.uint8
        elif type == GL_FLOAT:
            dtype = np.float32
        else:
            dtype = np.uint8  # Default
        
        # Read pixels
        if channels == 1:
            data = glReadPixels(x, y, width, height, format, type)
            array = np.frombuffer(data, dtype=dtype).reshape(height, width)
        else:
            data = glReadPixels(x, y, width, height, format, type)
            array = np.frombuffer(data, dtype=dtype).reshape(height, width, channels)
        
        # Flip vertically (OpenGL has origin at bottom-left)
        array = np.flipud(array)
        
        # Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        return array
    
    def read_depth(self, name: str, x: int = 0, y: int = 0,
                  width: Optional[int] = None, height: Optional[int] = None,
                  near: float = 0.1, far: float = 100.0) -> np.ndarray:
        """
        Read depth values from a framebuffer and convert to linear depth.
        
        Args:
            name: Name of the framebuffer
            x: X offset in pixels (default: 0)
            y: Y offset in pixels (default: 0)
            width: Width to read in pixels (default: full width)
            height: Height to read in pixels (default: full height)
            near: Near clipping plane distance (default: 0.1)
            far: Far clipping plane distance (default: 100.0)
            
        Returns:
            Numpy array containing the depth values in meters
        """
        # Read raw depth values
        depth = self.read_pixels(name, x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
        
        if depth.size == 0:
            return depth
        
        # Convert normalized depth values [0,1] to linear depth values
        linear_depth = far * near / (far - (far - near) * depth)
        
        return linear_depth
    
    def blit_to_default(self, name: str, 
                       src_x0: int = 0, src_y0: int = 0, 
                       src_x1: Optional[int] = None, src_y1: Optional[int] = None,
                       dst_x0: int = 0, dst_y0: int = 0, 
                       dst_x1: Optional[int] = None, dst_y1: Optional[int] = None,
                       mask: int = GL_COLOR_BUFFER_BIT, 
                       filter: int = GL_LINEAR) -> bool:
        """
        Blit (copy) framebuffer contents to the default framebuffer.
        
        Args:
            name: Name of the source framebuffer
            src_x0, src_y0: Source region lower-left corner
            src_x1, src_y1: Source region upper-right corner (default: framebuffer dimensions)
            dst_x0, dst_y0: Destination region lower-left corner
            dst_x1, dst_y1: Destination region upper-right corner (default: same as source)
            mask: Which buffers to copy (default: GL_COLOR_BUFFER_BIT)
            filter: Interpolation method (default: GL_LINEAR)
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._framebuffers:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot blit framebuffer '{name}': not found")
            return False
        
        # Get dimensions if not specified
        src_width, src_height = self._dimensions[name]
        src_x1 = src_x1 or src_width
        src_y1 = src_y1 or src_height
        
        # Set default destination dimensions if not specified
        dst_x1 = dst_x1 or (dst_x0 + (src_x1 - src_x0))
        dst_y1 = dst_y1 or (dst_y0 + (src_y1 - src_y0))
        
        # Bind read and draw framebuffers
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._framebuffers[name])
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        
        # Blit framebuffer
        glBlitFramebuffer(
            src_x0, src_y0, src_x1, src_y1,
            dst_x0, dst_y0, dst_x1, dst_y1,
            mask, filter
        )
        
        # Unbind framebuffers
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        return True
    
    def delete_framebuffer(self, name: str) -> bool:
        """
        Delete a framebuffer and its attachments.
        
        Args:
            name: Name of the framebuffer to delete
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._framebuffers:
            if self._logger:
                self._logger.log(Logger.WARNING, f"Cannot delete framebuffer '{name}': not found")
            return False
        
        # Delete color texture if it exists
        if name in self._color_textures:
            glDeleteTextures(1, [self._color_textures[name]])
            del self._color_textures[name]
        
        # Delete depth texture if it exists
        if name in self._depth_textures:
            glDeleteTextures(1, [self._depth_textures[name]])
            del self._depth_textures[name]
        
        # Delete renderbuffer if it exists
        if name in self._renderbuffers:
            glDeleteRenderbuffers(1, [self._renderbuffers[name]])
            del self._renderbuffers[name]
        
        # Delete framebuffer
        glDeleteFramebuffers(1, [self._framebuffers[name]])
        del self._framebuffers[name]
        
        # Remove dimensions
        if name in self._dimensions:
            del self._dimensions[name]
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Deleted framebuffer '{name}'")
        
        return True
    
    def cleanup(self) -> None:
        """
        Delete all framebuffers and clean up resources.
        """
        # Get a list of all framebuffer names
        framebuffer_names = list(self._framebuffers.keys())
        
        # Delete each framebuffer
        for name in framebuffer_names:
            self.delete_framebuffer(name)
        
        if self._logger:
            self._logger.log(Logger.RENDER, "Framebuffer manager cleaned up")
