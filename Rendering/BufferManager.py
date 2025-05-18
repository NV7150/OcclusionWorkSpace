import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from OpenGL.GL import *
from Utils.Logger import logger

class BufferObject:
    """
    Class representing an OpenGL buffer object.
    
    This class encapsulates the ID and type of an OpenGL buffer object.
    """
    
    def __init__(self, buffer_id: int, buffer_type: int):
        """
        Initialize a BufferObject.
        
        Args:
            buffer_id: OpenGL buffer ID
            buffer_type: OpenGL buffer type (e.g., GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER)
        """
        self.id = buffer_id
        self.type = buffer_type

class BufferManager:
    """
    Class for managing OpenGL buffer objects.
    
    This class handles creating, binding, and managing vertex buffer objects (VBOs),
    vertex array objects (VAOs), and element buffer objects (EBOs).
    """
    
    def __init__(self):
        """
        Initialize the BufferManager.
        """
        self.vaos = {}  # Dictionary mapping VAO names to VAO IDs
        self.buffers = {}  # Dictionary mapping buffer names to BufferObject instances
        self.current_vao = None  # Currently bound VAO
    
    def create_vao(self, name: str) -> int:
        """
        Create a vertex array object (VAO).
        
        Args:
            name: Name to assign to the VAO
            
        Returns:
            OpenGL VAO ID
        """
        try:
            # Generate VAO
            vao_id = glGenVertexArrays(1)
            
            # Store VAO
            self.vaos[name] = vao_id
            
            logger.log(logger.RENDER, f"Created VAO '{name}' with ID {vao_id}")
            return vao_id
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating VAO '{name}': {str(e)}")
            return 0
    
    def bind_vao(self, name: str) -> bool:
        """
        Bind a vertex array object (VAO).
        
        Args:
            name: Name of the VAO to bind
            
        Returns:
            True if the VAO was successfully bound, False otherwise
        """
        if name not in self.vaos:
            logger.log(logger.ERROR, f"VAO '{name}' not found")
            return False
        
        vao_id = self.vaos[name]
        
        # Bind VAO
        glBindVertexArray(vao_id)
        
        self.current_vao = name
        
        return True
    
    def unbind_vao(self) -> None:
        """
        Unbind the current vertex array object (VAO).
        """
        glBindVertexArray(0)
        self.current_vao = None
    
    def create_buffer(self, name: str, buffer_type: int, data: np.ndarray, usage: int = GL_STATIC_DRAW) -> int:
        """
        Create a buffer object (VBO or EBO).
        
        Args:
            name: Name to assign to the buffer
            buffer_type: OpenGL buffer type (e.g., GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER)
            data: Buffer data as a numpy array
            usage: OpenGL buffer usage hint (default: GL_STATIC_DRAW)
            
        Returns:
            OpenGL buffer ID
        """
        try:
            # Generate buffer
            buffer_id = glGenBuffers(1)
            
            # Bind buffer
            glBindBuffer(buffer_type, buffer_id)
            
            # Upload data
            glBufferData(buffer_type, data, usage)
            
            # Store buffer
            self.buffers[name] = BufferObject(buffer_id, buffer_type)
            
            logger.log(logger.RENDER, f"Created buffer '{name}' with ID {buffer_id}")
            return buffer_id
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating buffer '{name}': {str(e)}")
            return 0
    
    def update_buffer(self, name: str, data: np.ndarray, offset: int = 0) -> bool:
        """
        Update the data in a buffer object.
        
        Args:
            name: Name of the buffer to update
            data: New buffer data as a numpy array
            offset: Offset in bytes (default: 0)
            
        Returns:
            True if the buffer was successfully updated, False otherwise
        """
        if name not in self.buffers:
            logger.log(logger.ERROR, f"Buffer '{name}' not found")
            return False
        
        buffer = self.buffers[name]
        
        try:
            # Bind buffer
            glBindBuffer(buffer.type, buffer.id)
            
            # Update data
            glBufferSubData(buffer.type, offset, data)
            
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error updating buffer '{name}': {str(e)}")
            return False
    
    def bind_buffer(self, name: str) -> bool:
        """
        Bind a buffer object.
        
        Args:
            name: Name of the buffer to bind
            
        Returns:
            True if the buffer was successfully bound, False otherwise
        """
        if name not in self.buffers:
            logger.log(logger.ERROR, f"Buffer '{name}' not found")
            return False
        
        buffer = self.buffers[name]
        
        # Bind buffer
        glBindBuffer(buffer.type, buffer.id)
        
        return True
    
    def unbind_buffer(self, buffer_type: int) -> None:
        """
        Unbind a buffer of the specified type.
        
        Args:
            buffer_type: OpenGL buffer type (e.g., GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER)
        """
        glBindBuffer(buffer_type, 0)
    
    def set_vertex_attrib_pointer(self, attrib_index: int, size: int, attrib_type: int, 
                                 normalized: bool, stride: int, offset: int) -> None:
        """
        Set a vertex attribute pointer.
        
        Args:
            attrib_index: Index of the vertex attribute
            size: Number of components per vertex attribute (1, 2, 3, or 4)
            attrib_type: Data type of each component (e.g., GL_FLOAT)
            normalized: Whether fixed-point data values should be normalized
            stride: Byte offset between consecutive vertex attributes
            offset: Byte offset of the first component
        """
        glVertexAttribPointer(attrib_index, size, attrib_type, normalized, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(attrib_index)
    
    def create_mesh_buffers(self, name: str, vertices: np.ndarray, indices: np.ndarray, 
                           normals: Optional[np.ndarray] = None, 
                           uvs: Optional[np.ndarray] = None) -> bool:
        """
        Create buffers for a mesh with vertices, indices, and optional normals and UVs.
        
        Args:
            name: Base name for the buffers and VAO
            vertices: Vertex data as a numpy array (shape: Nx3)
            indices: Index data as a numpy array
            normals: Normal data as a numpy array (shape: Nx3) or None
            uvs: Texture coordinate data as a numpy array (shape: Nx2) or None
            
        Returns:
            True if the buffers were successfully created, False otherwise
        """
        try:
            # Create VAO
            vao_id = self.create_vao(name)
            if vao_id == 0:
                return False
            
            # Bind VAO
            self.bind_vao(name)
            
            # Create vertex buffer
            vertex_buffer_id = self.create_buffer(f"{name}_vertices", GL_ARRAY_BUFFER, vertices.astype(np.float32))
            if vertex_buffer_id == 0:
                return False
            
            # Set vertex attribute pointer
            self.set_vertex_attrib_pointer(0, 3, GL_FLOAT, False, 0, 0)
            
            # Create normal buffer if provided
            if normals is not None:
                normal_buffer_id = self.create_buffer(f"{name}_normals", GL_ARRAY_BUFFER, normals.astype(np.float32))
                if normal_buffer_id == 0:
                    return False
                
                # Set normal attribute pointer
                self.set_vertex_attrib_pointer(1, 3, GL_FLOAT, False, 0, 0)
            
            # Create UV buffer if provided
            if uvs is not None:
                uv_buffer_id = self.create_buffer(f"{name}_uvs", GL_ARRAY_BUFFER, uvs.astype(np.float32))
                if uv_buffer_id == 0:
                    return False
                
                # Set UV attribute pointer
                self.set_vertex_attrib_pointer(2, 2, GL_FLOAT, False, 0, 0)
            
            # Create index buffer
            index_buffer_id = self.create_buffer(f"{name}_indices", GL_ELEMENT_ARRAY_BUFFER, indices.astype(np.uint32))
            if index_buffer_id == 0:
                return False
            
            # Unbind VAO
            self.unbind_vao()
            
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating mesh buffers '{name}': {str(e)}")
            return False
    
    def create_interleaved_mesh_buffers(self, name: str, data: np.ndarray, indices: np.ndarray, 
                                       attrib_sizes: List[int], attrib_offsets: List[int], 
                                       stride: int) -> bool:
        """
        Create buffers for a mesh with interleaved vertex attributes.
        
        Args:
            name: Base name for the buffers and VAO
            data: Interleaved vertex data as a numpy array
            indices: Index data as a numpy array
            attrib_sizes: List of sizes for each vertex attribute
            attrib_offsets: List of offsets for each vertex attribute
            stride: Byte stride between consecutive vertices
            
        Returns:
            True if the buffers were successfully created, False otherwise
        """
        try:
            # Create VAO
            vao_id = self.create_vao(name)
            if vao_id == 0:
                return False
            
            # Bind VAO
            self.bind_vao(name)
            
            # Create vertex buffer
            vertex_buffer_id = self.create_buffer(f"{name}_vertices", GL_ARRAY_BUFFER, data.astype(np.float32))
            if vertex_buffer_id == 0:
                return False
            
            # Set vertex attribute pointers
            for i, (size, offset) in enumerate(zip(attrib_sizes, attrib_offsets)):
                self.set_vertex_attrib_pointer(i, size, GL_FLOAT, False, stride, offset)
            
            # Create index buffer
            index_buffer_id = self.create_buffer(f"{name}_indices", GL_ELEMENT_ARRAY_BUFFER, indices.astype(np.uint32))
            if index_buffer_id == 0:
                return False
            
            # Unbind VAO
            self.unbind_vao()
            
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating interleaved mesh buffers '{name}': {str(e)}")
            return False
    
    def delete_vao(self, name: str) -> bool:
        """
        Delete a vertex array object (VAO).
        
        Args:
            name: Name of the VAO to delete
            
        Returns:
            True if the VAO was successfully deleted, False otherwise
        """
        if name not in self.vaos:
            logger.log(logger.ERROR, f"VAO '{name}' not found")
            return False
        
        vao_id = self.vaos[name]
        
        # Delete VAO
        glDeleteVertexArrays(1, [vao_id])
        
        # Remove from dictionary
        del self.vaos[name]
        
        # Reset current VAO if it was the deleted one
        if self.current_vao == name:
            self.current_vao = None
        
        logger.log(logger.RENDER, f"Deleted VAO '{name}'")
        return True
    
    def delete_buffer(self, name: str) -> bool:
        """
        Delete a buffer object.
        
        Args:
            name: Name of the buffer to delete
            
        Returns:
            True if the buffer was successfully deleted, False otherwise
        """
        if name not in self.buffers:
            logger.log(logger.ERROR, f"Buffer '{name}' not found")
            return False
        
        buffer = self.buffers[name]
        
        # Delete buffer
        glDeleteBuffers(1, [buffer.id])
        
        # Remove from dictionary
        del self.buffers[name]
        
        logger.log(logger.RENDER, f"Deleted buffer '{name}'")
        return True
    
    def cleanup(self) -> None:
        """
        Clean up all buffers and VAOs.
        """
        # Delete all buffers
        for name, buffer in self.buffers.items():
            glDeleteBuffers(1, [buffer.id])
        
        # Delete all VAOs
        for name, vao_id in self.vaos.items():
            glDeleteVertexArrays(1, [vao_id])
        
        self.buffers = {}
        self.vaos = {}
        self.current_vao = None
        
        logger.log(logger.RENDER, "Cleaned up all buffers and VAOs")