import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from OpenGL.GL import *
from ..Logger.Logger import Logger


class BufferManager:
    """
    Manages OpenGL buffer objects (VBOs, EBOs, VAOs).
    
    This class provides a centralized way to create, manage, and use OpenGL buffer objects,
    reducing code duplication and ensuring proper resource management.
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the BufferManager.
        
        Args:
            logger: Optional logger instance
        """
        self._logger = logger
        self._vaos: Dict[str, int] = {}  # VAO IDs by name
        self._vbos: Dict[str, int] = {}  # VBO IDs by name
        self._ebos: Dict[str, int] = {}  # EBO IDs by name
        
        if self._logger:
            self._logger.log(Logger.RENDER, "BufferManager initialized")
    
    def create_vao(self, name: str) -> int:
        """
        Create a Vertex Array Object (VAO).
        
        Args:
            name: Name to identify this VAO
            
        Returns:
            OpenGL VAO ID
        """
        if name in self._vaos:
            if self._logger:
                self._logger.log(Logger.WARNING, f"VAO '{name}' already exists, returning existing ID")
            return self._vaos[name]
        
        vao = glGenVertexArrays(1)
        self._vaos[name] = vao
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Created VAO '{name}' with ID {vao}")
        
        return vao
    
    def create_vbo(self, name: str, data: np.ndarray, usage: int = GL_STATIC_DRAW) -> int:
        """
        Create a Vertex Buffer Object (VBO).
        
        Args:
            name: Name to identify this VBO
            data: Numpy array containing the buffer data
            usage: OpenGL buffer usage hint (default: GL_STATIC_DRAW)
            
        Returns:
            OpenGL VBO ID
        """
        if name in self._vbos:
            if self._logger:
                self._logger.log(Logger.WARNING, f"VBO '{name}' already exists, deleting and recreating")
            glDeleteBuffers(1, [self._vbos[name]])
        
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, usage)
        self._vbos[name] = vbo
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Created VBO '{name}' with ID {vbo}, size {data.nbytes} bytes")
        
        return vbo
    
    def create_ebo(self, name: str, data: np.ndarray, usage: int = GL_STATIC_DRAW) -> int:
        """
        Create an Element Buffer Object (EBO).
        
        Args:
            name: Name to identify this EBO
            data: Numpy array containing the index data
            usage: OpenGL buffer usage hint (default: GL_STATIC_DRAW)
            
        Returns:
            OpenGL EBO ID
        """
        if name in self._ebos:
            if self._logger:
                self._logger.log(Logger.WARNING, f"EBO '{name}' already exists, deleting and recreating")
            glDeleteBuffers(1, [self._ebos[name]])
        
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, usage)
        self._ebos[name] = ebo
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Created EBO '{name}' with ID {ebo}, size {data.nbytes} bytes")
        
        return ebo
    
    def update_vbo(self, name: str, data: np.ndarray, offset: int = 0) -> bool:
        """
        Update data in an existing VBO.
        
        Args:
            name: Name of the VBO to update
            data: New data to upload
            offset: Byte offset into the buffer (default: 0)
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._vbos:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot update VBO '{name}': not found")
            return False
        
        vbo = self._vbos[name]
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferSubData(GL_ARRAY_BUFFER, offset, data.nbytes, data)
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Updated VBO '{name}' with {data.nbytes} bytes at offset {offset}")
        
        return True
    
    def update_ebo(self, name: str, data: np.ndarray, offset: int = 0) -> bool:
        """
        Update data in an existing EBO.
        
        Args:
            name: Name of the EBO to update
            data: New data to upload
            offset: Byte offset into the buffer (default: 0)
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._ebos:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot update EBO '{name}': not found")
            return False
        
        ebo = self._ebos[name]
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, offset, data.nbytes, data)
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Updated EBO '{name}' with {data.nbytes} bytes at offset {offset}")
        
        return True
    
    def bind_vao(self, name: str) -> bool:
        """
        Bind a VAO for use.
        
        Args:
            name: Name of the VAO to bind
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._vaos:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot bind VAO '{name}': not found")
            return False
        
        glBindVertexArray(self._vaos[name])
        return True
    
    def bind_vbo(self, name: str) -> bool:
        """
        Bind a VBO for use.
        
        Args:
            name: Name of the VBO to bind
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._vbos:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot bind VBO '{name}': not found")
            return False
        
        glBindBuffer(GL_ARRAY_BUFFER, self._vbos[name])
        return True
    
    def bind_ebo(self, name: str) -> bool:
        """
        Bind an EBO for use.
        
        Args:
            name: Name of the EBO to bind
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._ebos:
            if self._logger:
                self._logger.log(Logger.ERROR, f"Cannot bind EBO '{name}': not found")
            return False
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebos[name])
        return True
    
    def configure_vertex_attribute(self, attribute_index: int, size: int, data_type: int,
                                  normalized: bool, stride: int, offset: int) -> None:
        """
        Configure a vertex attribute pointer.
        
        Args:
            attribute_index: Index of the vertex attribute
            size: Number of components per vertex attribute (1-4)
            data_type: Data type of each component (e.g., GL_FLOAT)
            normalized: Whether fixed-point data should be normalized
            stride: Byte offset between consecutive vertex attributes
            offset: Byte offset of the first component
        """
        glVertexAttribPointer(attribute_index, size, data_type, normalized, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(attribute_index)
    
    def delete_vao(self, name: str) -> bool:
        """
        Delete a VAO.
        
        Args:
            name: Name of the VAO to delete
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._vaos:
            if self._logger:
                self._logger.log(Logger.WARNING, f"Cannot delete VAO '{name}': not found")
            return False
        
        glDeleteVertexArrays(1, [self._vaos[name]])
        del self._vaos[name]
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Deleted VAO '{name}'")
        
        return True
    
    def delete_vbo(self, name: str) -> bool:
        """
        Delete a VBO.
        
        Args:
            name: Name of the VBO to delete
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._vbos:
            if self._logger:
                self._logger.log(Logger.WARNING, f"Cannot delete VBO '{name}': not found")
            return False
        
        glDeleteBuffers(1, [self._vbos[name]])
        del self._vbos[name]
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Deleted VBO '{name}'")
        
        return True
    
    def delete_ebo(self, name: str) -> bool:
        """
        Delete an EBO.
        
        Args:
            name: Name of the EBO to delete
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._ebos:
            if self._logger:
                self._logger.log(Logger.WARNING, f"Cannot delete EBO '{name}': not found")
            return False
        
        glDeleteBuffers(1, [self._ebos[name]])
        del self._ebos[name]
        
        if self._logger:
            self._logger.log(Logger.RENDER, f"Deleted EBO '{name}'")
        
        return True
    
    def cleanup(self) -> None:
        """
        Delete all buffers and clean up resources.
        """
        # Delete all VAOs
        for vao in self._vaos.values():
            glDeleteVertexArrays(1, [vao])
        
        # Delete all VBOs
        for vbo in self._vbos.values():
            glDeleteBuffers(1, [vbo])
        
        # Delete all EBOs
        for ebo in self._ebos.values():
            glDeleteBuffers(1, [ebo])
        
        # Clear dictionaries
        self._vaos.clear()
        self._vbos.clear()
        self._ebos.clear()
        
        if self._logger:
            self._logger.log(Logger.RENDER, "BufferManager cleaned up")
