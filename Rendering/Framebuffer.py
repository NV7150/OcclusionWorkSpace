from typing import Dict, Optional, Tuple, List
from OpenGL.GL import *
from Utils.Logger import logger
import numpy as np

class Framebuffer:
    """
    Class for managing OpenGL framebuffer objects.
    
    This class handles creating, binding, and managing framebuffer objects
    for off-screen rendering, such as shadow maps and post-processing effects.
    """
    
    def __init__(self):
        """
        Initialize the Framebuffer.
        """
        self.framebuffers = {}  # Dictionary mapping FBO names to FBO IDs
        self.renderbuffers = {}  # Dictionary mapping RBO names to RBO IDs
        self.current_fbo = None  # Currently bound FBO
    
    def create_framebuffer(self, name: str) -> int:
        """
        Create a framebuffer object (FBO).
        
        Args:
            name: Name to assign to the FBO
            
        Returns:
            OpenGL FBO ID
        """
        try:
            # Generate FBO
            fbo_id = glGenFramebuffers(1)
            
            # Store FBO
            self.framebuffers[name] = fbo_id
            
            logger.log(logger.RENDER, f"Created FBO '{name}' with ID {fbo_id}")
            return fbo_id
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating FBO '{name}': {str(e)}")
            return 0
    
    def bind_framebuffer(self, name: str) -> bool:
        """
        Bind a framebuffer object (FBO).
        
        Args:
            name: Name of the FBO to bind
            
        Returns:
            True if the FBO was successfully bound, False otherwise
        """
        if name not in self.framebuffers:
            logger.log(logger.ERROR, f"FBO '{name}' not found")
            return False
        
        fbo_id = self.framebuffers[name]
        
        # Bind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
        
        self.current_fbo = name
        
        return True
    
    def unbind_framebuffer(self) -> None:
        """
        Unbind the current framebuffer object (FBO).
        """
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.current_fbo = None
    
    def create_renderbuffer(self, name: str, width: int, height: int, internal_format: int = GL_DEPTH_COMPONENT) -> int:
        """
        Create a renderbuffer object (RBO).
        
        Args:
            name: Name to assign to the RBO
            width: Width of the renderbuffer
            height: Height of the renderbuffer
            internal_format: Internal format of the renderbuffer (default: GL_DEPTH_COMPONENT)
            
        Returns:
            OpenGL RBO ID
        """
        try:
            # Generate RBO
            rbo_id = glGenRenderbuffers(1)
            
            # Bind RBO
            glBindRenderbuffer(GL_RENDERBUFFER, rbo_id)
            
            # Set RBO storage
            glRenderbufferStorage(GL_RENDERBUFFER, internal_format, width, height)
            
            # Unbind RBO
            glBindRenderbuffer(GL_RENDERBUFFER, 0)
            
            # Store RBO
            self.renderbuffers[name] = rbo_id
            
            logger.log(logger.RENDER, f"Created RBO '{name}' with ID {rbo_id}")
            return rbo_id
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating RBO '{name}': {str(e)}")
            return 0
    
    def attach_texture_to_framebuffer(self, fbo_name: str, texture_id: int, attachment: int = GL_COLOR_ATTACHMENT0) -> bool:
        """
        Attach a texture to a framebuffer object.
        
        Args:
            fbo_name: Name of the FBO
            texture_id: OpenGL texture ID to attach
            attachment: Attachment point (default: GL_COLOR_ATTACHMENT0)
            
        Returns:
            True if the texture was successfully attached, False otherwise
        """
        if fbo_name not in self.framebuffers:
            logger.log(logger.ERROR, f"FBO '{fbo_name}' not found")
            return False
        
        fbo_id = self.framebuffers[fbo_name]
        
        try:
            # Bind FBO
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
            
            # Attach texture
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture_id, 0)
            
            # Unbind FBO
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error attaching texture to FBO '{fbo_name}': {str(e)}")
            return False
    
    def attach_renderbuffer_to_framebuffer(self, fbo_name: str, rbo_name: str, attachment: int = GL_DEPTH_ATTACHMENT) -> bool:
        """
        Attach a renderbuffer to a framebuffer object.
        
        Args:
            fbo_name: Name of the FBO
            rbo_name: Name of the RBO to attach
            attachment: Attachment point (default: GL_DEPTH_ATTACHMENT)
            
        Returns:
            True if the renderbuffer was successfully attached, False otherwise
        """
        if fbo_name not in self.framebuffers:
            logger.log(logger.ERROR, f"FBO '{fbo_name}' not found")
            return False
        
        if rbo_name not in self.renderbuffers:
            logger.log(logger.ERROR, f"RBO '{rbo_name}' not found")
            return False
        
        fbo_id = self.framebuffers[fbo_name]
        rbo_id = self.renderbuffers[rbo_name]
        
        try:
            # Bind FBO
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
            
            # Attach renderbuffer
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, rbo_id)
            
            # Unbind FBO
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error attaching RBO '{rbo_name}' to FBO '{fbo_name}': {str(e)}")
            return False
    
    def check_framebuffer_status(self, fbo_name: str) -> bool:
        """
        Check if a framebuffer object is complete.
        
        Args:
            fbo_name: Name of the FBO to check
            
        Returns:
            True if the FBO is complete, False otherwise
        """
        if fbo_name not in self.framebuffers:
            logger.log(logger.ERROR, f"FBO '{fbo_name}' not found")
            return False
        
        fbo_id = self.framebuffers[fbo_name]
        
        # Bind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
        
        # Check status
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        
        # Unbind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        if status != GL_FRAMEBUFFER_COMPLETE:
            logger.log(logger.ERROR, f"Framebuffer '{fbo_name}' is not complete: {status}")
            return False
        
        return True
    
    def create_color_depth_framebuffer(self, name: str, width: int, height: int, 
                                      color_texture_id: int, depth_format: int = GL_DEPTH_COMPONENT) -> bool:
        """
        Create a framebuffer with a color texture attachment and a depth renderbuffer.
        
        Args:
            name: Name to assign to the FBO and RBO
            width: Width of the framebuffer
            height: Height of the framebuffer
            color_texture_id: OpenGL texture ID for color attachment
            depth_format: Format for the depth renderbuffer (default: GL_DEPTH_COMPONENT)
            
        Returns:
            True if the framebuffer was successfully created, False otherwise
        """
        try:
            # Create FBO
            fbo_id = self.create_framebuffer(name)
            if fbo_id == 0:
                return False
            
            # Create depth RBO
            rbo_id = self.create_renderbuffer(f"{name}_depth", width, height, depth_format)
            if rbo_id == 0:
                return False
            
            # Bind FBO
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
            
            # Attach color texture
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture_id, 0)
            
            # Attach depth renderbuffer
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_id)
            
            # Check framebuffer status
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                logger.log(logger.ERROR, f"Framebuffer '{name}' is not complete: {status}")
                
                # Clean up
                self.delete_framebuffer(name)
                self.delete_renderbuffer(f"{name}_depth")
                
                return False
            
            # Unbind FBO
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            
            logger.log(logger.RENDER, f"Created color-depth framebuffer '{name}'")
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating color-depth framebuffer '{name}': {str(e)}")
            return False
    
    def create_depth_framebuffer(self, name: str, width: int, height: int, depth_texture_id: int) -> bool:
        """
        Create a framebuffer with a depth texture attachment.
        
        Args:
            name: Name to assign to the FBO
            width: Width of the framebuffer
            height: Height of the framebuffer
            depth_texture_id: OpenGL texture ID for depth attachment
            
        Returns:
            True if the framebuffer was successfully created, False otherwise
        """
        try:
            # Create FBO
            fbo_id = self.create_framebuffer(name)
            if fbo_id == 0:
                return False
            
            # Bind FBO
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
            
            # Attach depth texture
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture_id, 0)
            
            # No color buffer is drawn to
            glDrawBuffer(GL_NONE)
            glReadBuffer(GL_NONE)
            
            # Check framebuffer status
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                logger.log(logger.ERROR, f"Framebuffer '{name}' is not complete: {status}")
                
                # Clean up
                self.delete_framebuffer(name)
                
                return False
            
            # Unbind FBO
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            
            logger.log(logger.RENDER, f"Created depth framebuffer '{name}'")
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating depth framebuffer '{name}': {str(e)}")
            return False
    
    def delete_framebuffer(self, name: str) -> bool:
        """
        Delete a framebuffer object (FBO).
        
        Args:
            name: Name of the FBO to delete
            
        Returns:
            True if the FBO was successfully deleted, False otherwise
        """
        if name not in self.framebuffers:
            logger.log(logger.ERROR, f"FBO '{name}' not found")
            return False
        
        fbo_id = self.framebuffers[name]
        
        # Delete FBO
        glDeleteFramebuffers(1, [fbo_id])
        
        # Remove from dictionary
        del self.framebuffers[name]
        
        # Reset current FBO if it was the deleted one
        if self.current_fbo == name:
            self.current_fbo = None
        
        logger.log(logger.RENDER, f"Deleted FBO '{name}'")
        return True
    
    def delete_renderbuffer(self, name: str) -> bool:
        """
        Delete a renderbuffer object (RBO).
        
        Args:
            name: Name of the RBO to delete
            
        Returns:
            True if the RBO was successfully deleted, False otherwise
        """
        if name not in self.renderbuffers:
            logger.log(logger.ERROR, f"RBO '{name}' not found")
            return False
        
        rbo_id = self.renderbuffers[name]
        
        # Delete RBO
        glDeleteRenderbuffers(1, [rbo_id])
        
        # Remove from dictionary
        del self.renderbuffers[name]
        
        logger.log(logger.RENDER, f"Deleted RBO '{name}'")
        return True
    
    def cleanup(self) -> None:
        """
        Clean up all framebuffers and renderbuffers.
        """
        # Delete all FBOs
        for name, fbo_id in self.framebuffers.items():
            glDeleteFramebuffers(1, [fbo_id])
        
        # Delete all RBOs
        for name, rbo_id in self.renderbuffers.items():
            glDeleteRenderbuffers(1, [rbo_id])
        
        self.framebuffers = {}
        self.renderbuffers = {}
        self.current_fbo = None
        
        logger.log(logger.RENDER, "Cleaned up all framebuffers and renderbuffers")
    
    def read_color_buffer(self) -> np.ndarray:
        """
        Read the color buffer from the currently bound framebuffer.
        
        Returns:
            Color buffer as a numpy array
        """
        if self.current_fbo is None:
            logger.log(logger.ERROR, "No framebuffer is currently bound")
            return np.zeros((100, 100, 4), dtype=np.uint8)
        
        try:
            # Get the viewport dimensions
            viewport = glGetIntegerv(GL_VIEWPORT)
            width, height = viewport[2], viewport[3]
            
            # Read pixels from framebuffer
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            rendered_data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
            rendered_image = np.frombuffer(rendered_data, dtype=np.uint8).reshape(height, width, 4)
            
            # Flip image vertically (OpenGL has origin at bottom-left)
            rendered_image = np.flipud(rendered_image)
            
            return rendered_image
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error reading color buffer: {str(e)}")
            return np.zeros((100, 100, 4), dtype=np.uint8)