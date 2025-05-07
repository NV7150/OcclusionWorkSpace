import os
import numpy as np
from typing import Dict, Optional, List, Tuple
import OpenGL.GL as gl

# Make sure we can import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Logger.Logger import Logger # Assuming Logger is accessible


class ShaderManager:
    """
    Manages the compilation, linking and use of GLSL shader programs.
    
    This class handles the creation and management of shader programs,
    including vertex and fragment shaders, and provides uniform setting utilities.
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """Initialize the shader manager."""
        if logger:
            self._logger = logger
        else:
            # Fallback to a default logger if none is provided
            self._logger = Logger(log_level="INFO", log_file="shader_manager.log") 
            self._logger.warning("ShaderManager initialized with a default logger.")

        # Dictionary to hold shader programs: program_name -> program_id
        self._shader_programs: Dict[str, int] = {}
        # Dictionary to cache uniform locations: (program_name, uniform_name) -> location
        self._uniform_locations: Dict[Tuple[str, str], int] = {}
        # Currently active shader program
        self._active_program: Optional[str] = None
    
    def create_program(self, name: str, vertex_source: str, fragment_source: str, 
                      geometry_source: Optional[str] = None) -> bool:
        """
        Create a shader program from source code.
        
        Args:
            name: Unique name for the shader program
            vertex_source: GLSL vertex shader source code
            fragment_source: GLSL fragment shader source code
            geometry_source: Optional GLSL geometry shader source code
            
        Returns:
            bool: True if program creation was successful, False otherwise
        """
        # Check if program with this name already exists
        if name in self._shader_programs:
            self._logger.warning(f"Shader program '{name}' already exists. Overwriting.")
            # Delete existing program
            self.delete_program(name)
        
        # Create shader objects
        vertex_shader = self._compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
        if not vertex_shader:
            return False
            
        fragment_shader = self._compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
        if not fragment_shader:
            gl.glDeleteShader(vertex_shader)
            return False
            
        geometry_shader = None
        if geometry_source:
            geometry_shader = self._compile_shader(geometry_source, gl.GL_GEOMETRY_SHADER)
            if not geometry_shader:
                gl.glDeleteShader(vertex_shader)
                gl.glDeleteShader(fragment_shader)
                return False
        
        # Create program object
        program_id = gl.glCreateProgram()
        
        # Attach shaders to program
        gl.glAttachShader(program_id, vertex_shader)
        gl.glAttachShader(program_id, fragment_shader)
        if geometry_shader:
            gl.glAttachShader(program_id, geometry_shader)
            
        # Link program
        gl.glLinkProgram(program_id)
        
        # Check linking status
        link_status = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
        if not link_status:
            info_log = gl.glGetProgramInfoLog(program_id)
            self._logger.error(f"Error linking shader program '{name}':\n{info_log.decode('utf-8')}")
            
            # Clean up
            gl.glDeleteProgram(program_id)
            gl.glDeleteShader(vertex_shader)
            gl.glDeleteShader(fragment_shader)
            if geometry_shader:
                gl.glDeleteShader(geometry_shader)
                
            return False
            
        # Store program ID
        self._shader_programs[name] = program_id
        self._logger.info(f"Shader program '{name}' created and linked successfully.")
        
        # Detach and delete shader objects (no longer needed after linking)
        gl.glDetachShader(program_id, vertex_shader)
        gl.glDetachShader(program_id, fragment_shader)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        
        if geometry_shader:
            gl.glDetachShader(program_id, geometry_shader)
            gl.glDeleteShader(geometry_shader)
            
        return True
    
    def create_program_from_files(self, name: str, vertex_path: str, fragment_path: str,
                                geometry_path: Optional[str] = None) -> bool:
        """
        Create a shader program from shader files.
        
        Args:
            name: Unique name for the shader program
            vertex_path: Path to the vertex shader file
            fragment_path: Path to the fragment shader file
            geometry_path: Optional path to the geometry shader file
            
        Returns:
            bool: True if program creation was successful, False otherwise
        """
        try:
            # Read shader source code from files
            with open(vertex_path, 'r') as file:
                vertex_source = file.read()
                
            with open(fragment_path, 'r') as file:
                fragment_source = file.read()
                
            geometry_source = None
            if geometry_path:
                with open(geometry_path, 'r') as file:
                    geometry_source = file.read()
                    
            # Create program using source code
            return self.create_program(name, vertex_source, fragment_source, geometry_source)
            
        except FileNotFoundError as e:
            self._logger.error(f"Error loading shader files for program '{name}': {e}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error loading shader files for program '{name}': {e}")
            return False
    
    def use_program(self, name: str) -> bool:
        """
        Activate a shader program for use.
        
        Args:
            name: Name of the shader program to use
            
        Returns:
            bool: True if program was activated, False if not found
        """
        if name in self._shader_programs:
            gl.glUseProgram(self._shader_programs[name])
            self._active_program = name
            return True
        else:
            self._logger.error(f"Cannot use shader program '{name}': not found.")
            return False
    
    def get_active_program_name(self) -> Optional[str]:
        """
        Get the name of the currently active shader program.
        
        Returns:
            Optional[str]: Name of active program, or None if none active
        """
        return self._active_program

    def get_active_program_id(self) -> Optional[int]:
        """
        Get the ID of the currently active shader program.
        Returns:
            Optional[int]: ID of active program, or None if none active or not found.
        """
        if self._active_program and self._active_program in self._shader_programs:
            return self._shader_programs[self._active_program]
        return None
    
    def delete_program(self, name: str) -> bool:
        """
        Delete a shader program.
        
        Args:
            name: Name of the shader program to delete
            
        Returns:
            bool: True if program was deleted, False if not found
        """
        if name in self._shader_programs:
            program_id = self._shader_programs[name]
            gl.glDeleteProgram(program_id)
            del self._shader_programs[name]
            
            # Remove uniform locations for this program
            keys_to_remove = [key for key in self._uniform_locations.keys() if key[0] == name]
            for key in keys_to_remove:
                del self._uniform_locations[key]
                
            # Reset active program if it was the deleted one
            if self._active_program == name:
                self._active_program = None
                gl.glUseProgram(0) # Bind default program
            
            self._logger.info(f"Shader program '{name}' deleted.")
            return True
        else:
            self._logger.warning(f"Cannot delete non-existent shader program '{name}'.")
            return False
    
    def has_program(self, name: str) -> bool:
        """
        Check if a shader program exists.
        
        Args:
            name: Name of the shader program to check
            
        Returns:
            bool: True if program exists, False otherwise
        """
        return name in self._shader_programs
    
    def get_program_id(self, name: str) -> Optional[int]:
        """
        Get the OpenGL ID of a shader program.
        
        Args:
            name: Name of the shader program
            
        Returns:
            Optional[int]: OpenGL program ID, or None if not found
        """
        program_id = self._shader_programs.get(name)
        if program_id is None:
            self._logger.debug(f"Program ID for '{name}' not found.")
        return program_id
    
    def get_uniform_location(self, uniform_name: str, program_name: Optional[str] = None) -> int:
        """
        Get the location of a uniform variable in a shader program.
        Uses active program if program_name is not specified.
        
        Args:
            uniform_name: Name of the uniform variable
            program_name: Optional name of the shader program. Uses active program if None.
            
        Returns:
            int: Location of the uniform, or -1 if not found
        """
        target_program_name = program_name if program_name else self._active_program
        
        if not target_program_name:
            self._logger.error("Cannot get uniform location: No active or specified shader program.")
            return -1

        # Check cache first
        key = (target_program_name, uniform_name)
        if key in self._uniform_locations:
            return self._uniform_locations[key]
            
        # Not in cache, look up location
        if target_program_name not in self._shader_programs:
            self._logger.error(f"Cannot get uniform location for non-existent program '{target_program_name}'.")
            return -1
            
        program_id = self._shader_programs[target_program_name]
        location = gl.glGetUniformLocation(program_id, uniform_name)
        
        if location == -1:
            self._logger.debug(f"Uniform '{uniform_name}' not found in program '{target_program_name}'.")
        
        # Cache the result
        self._uniform_locations[key] = location
        
        return location
    
    def _set_uniform_check(self, uniform_name: str) -> int:
        if not self._active_program:
            self._logger.error(f"No active shader program when setting uniform '{uniform_name}'.")
            return -1
        location = self.get_uniform_location(uniform_name)
        if location == -1:
            pass
        return location

    def set_uniform_1f(self, uniform_name: str, value: float) -> bool:
        location = self._set_uniform_check(uniform_name)
        if location == -1: return False
        gl.glUniform1f(location, value)
        return True
    
    def set_uniform_2f(self, uniform_name: str, x: float, y: float) -> bool:
        location = self._set_uniform_check(uniform_name)
        if location == -1: return False
        gl.glUniform2f(location, x, y)
        return True

    def set_uniform_3f(self, uniform_name: str, x: float, y: float, z: float) -> bool:
        location = self._set_uniform_check(uniform_name)
        if location == -1: return False
        gl.glUniform3f(location, x, y, z)
        return True

    def set_uniform_4f(self, uniform_name: str, x: float, y: float, z: float, w: float) -> bool:
        location = self._set_uniform_check(uniform_name)
        if location == -1: return False
        gl.glUniform4f(location, x, y, z, w)
        return True

    def set_uniform_1i(self, uniform_name: str, value: int) -> bool:
        location = self._set_uniform_check(uniform_name)
        if location == -1: return False
        gl.glUniform1i(location, value)
        return True
    
    def set_uniform_matrix_4fv(self, uniform_name: str, matrix: np.ndarray, 
                              transpose: bool = False) -> bool:
        location = self._set_uniform_check(uniform_name)
        if location == -1: return False
            
        if isinstance(matrix, np.ndarray) and matrix.shape == (4, 4):
            gl_transpose = gl.GL_TRUE if transpose else gl.GL_FALSE
            gl.glUniformMatrix4fv(location, 1, gl_transpose, matrix.astype(np.float32))
            return True
        else:
            self._logger.error(f"Invalid matrix format for uniform '{uniform_name}'. Expected 4x4 np.ndarray.")
            return False
    
    def set_uniform_vec3_array(self, uniform_name: str, values: np.ndarray) -> bool:
        location = self._set_uniform_check(uniform_name)
        if location == -1: return False
            
        if isinstance(values, np.ndarray) and len(values.shape) == 2 and values.shape[1] == 3:
            gl.glUniform3fv(location, values.shape[0], values.astype(np.float32).flatten())
            return True
        elif isinstance(values, np.ndarray) and len(values.shape) == 1 and values.shape[0] % 3 == 0:
             gl.glUniform3fv(location, values.shape[0] // 3, values.astype(np.float32))
             return True
        else:
            self._logger.error(f"Invalid vec3 array format for uniform '{uniform_name}'. Expected Nx3 or 1D (Nx3) np.ndarray.")
            return False
    
    def _compile_shader(self, source: str, shader_type: int) -> int:
        """
        Compile a shader from source code.
        
        Args:
            source: GLSL shader source code
            shader_type: GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, or GL_GEOMETRY_SHADER
            
        Returns:
            int: Shader object ID, or 0 on failure
        """
        shader_id = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader_id, source)
        gl.glCompileShader(shader_id)
        
        # Check compilation status
        compile_status = gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS)
        if not compile_status:
            info_log = gl.glGetShaderInfoLog(shader_id)
            
            shader_type_name = {
                gl.GL_VERTEX_SHADER: "vertex",
                gl.GL_FRAGMENT_SHADER: "fragment",
                gl.GL_GEOMETRY_SHADER: "geometry"
            }.get(shader_type, "unknown")
            
            log_message = f"Error compiling {shader_type_name} shader:\n{info_log.decode('utf-8')}\n"
            log_message += "Shader source:\n"
            for i, line in enumerate(source.splitlines()):
                log_message += f"{i+1:4d}: {line}\n"
            self._logger.error(log_message)
            
            gl.glDeleteShader(shader_id)
            return 0
            
        return shader_id

    def cleanup(self):
        """Deletes all shader programs managed by this manager."""
        self._logger.info("Cleaning up ShaderManager: Deleting all shader programs.")
        for name in list(self._shader_programs.keys()):
            self.delete_program(name)
        self._shader_programs.clear()
        self._uniform_locations.clear()
        if self._active_program is not None:
             gl.glUseProgram(0)
             self._active_program = None
        self._logger.info("ShaderManager cleanup complete.")