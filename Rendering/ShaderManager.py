import os
from typing import Dict, Optional, Tuple
from OpenGL.GL import *
from OpenGL.GL import shaders
from Utils.Logger import logger

class ShaderManager:
    """
    Class for managing OpenGL shaders.
    
    This class handles loading, compiling, and linking GLSL shaders,
    as well as managing shader programs and uniforms.
    """
    
    def __init__(self):
        """
        Initialize the ShaderManager.
        """
        self.shader_programs = {}  # Dictionary mapping program names to program IDs
        self.current_program = None  # Currently active shader program
        
        # Default shader sources
        self.default_vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 texCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 fragPosition;
        out vec3 fragNormal;
        out vec2 fragTexCoord;
        
        void main() {
            fragPosition = vec3(model * vec4(position, 1.0));
            fragNormal = mat3(transpose(inverse(model))) * normal;
            fragTexCoord = texCoord;
            
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """
        
        self.default_fragment_shader = """
        #version 330 core
        in vec3 fragPosition;
        in vec3 fragNormal;
        in vec2 fragTexCoord;
        
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec4 objectColor;
        uniform sampler2D textureSampler;
        uniform bool useTexture;
        
        out vec4 fragColor;
        
        void main() {
            // Ambient
            float ambientStrength = 0.2;
            vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
            
            // Diffuse
            vec3 norm = normalize(fragNormal);
            vec3 lightDir = normalize(lightPos - fragPosition);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
            
            // Specular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(viewPos - fragPosition);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
            
            // Combine
            vec4 baseColor;
            if (useTexture) {
                baseColor = texture(textureSampler, fragTexCoord);
            } else {
                baseColor = objectColor;
            }
            
            vec3 result = (ambient + diffuse + specular) * baseColor.rgb;
            fragColor = vec4(result, baseColor.a);
        }
        """
        
        self.depth_vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 position;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """
        
        self.depth_fragment_shader = """
        #version 330 core
        out float fragDepth;
        
        void main() {
            // Output depth value (normalized to [0,1])
            fragDepth = gl_FragCoord.z;
        }
        """
    
    def create_program_from_source(self, name: str, vertex_source: str, fragment_source: str) -> int:
        """
        Create a shader program from source code.
        
        Args:
            name: Name to assign to the shader program
            vertex_source: GLSL source code for the vertex shader
            fragment_source: GLSL source code for the fragment shader
            
        Returns:
            OpenGL program ID
        """
        try:
            # Compile shaders
            vertex_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
            
            # Create program
            program = shaders.compileProgram(vertex_shader, fragment_shader)
            
            # Store program
            self.shader_programs[name] = program
            
            logger.log(logger.RENDER, f"Created shader program '{name}' with ID {program}")
            return program
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error creating shader program '{name}': {str(e)}")
            return 0
    
    def create_program_from_files(self, name: str, vertex_file: str, fragment_file: str) -> int:
        """
        Create a shader program from files.
        
        Args:
            name: Name to assign to the shader program
            vertex_file: Path to the vertex shader file
            fragment_file: Path to the fragment shader file
            
        Returns:
            OpenGL program ID
        """
        try:
            # Read shader files
            with open(vertex_file, 'r') as f:
                vertex_source = f.read()
                
            with open(fragment_file, 'r') as f:
                fragment_source = f.read()
            
            # Create program from source
            return self.create_program_from_source(name, vertex_source, fragment_source)
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading shader files for '{name}': {str(e)}")
            return 0
    
    def create_default_programs(self) -> None:
        """
        Create default shader programs.
        """
        # Create default shader program
        self.create_program_from_source("default", self.default_vertex_shader, self.default_fragment_shader)
        
        # Create depth shader program
        self.create_program_from_source("depth", self.depth_vertex_shader, self.depth_fragment_shader)
    
    def use_program(self, name: str) -> bool:
        """
        Use a shader program.
        
        Args:
            name: Name of the shader program to use
            
        Returns:
            True if the program was successfully activated, False otherwise
        """
        if name not in self.shader_programs:
            logger.log(logger.ERROR, f"Shader program '{name}' not found")
            return False
        
        program = self.shader_programs[name]
        glUseProgram(program)
        self.current_program = name
        
        return True
    
    def get_current_program(self) -> Optional[int]:
        """
        Get the currently active shader program.
        
        Returns:
            OpenGL program ID of the currently active program or None if no program is active
        """
        if self.current_program is None:
            return None
        
        return self.shader_programs.get(self.current_program)
    
    def get_program(self, name: str) -> Optional[int]:
        """
        Get a shader program by name.
        
        Args:
            name: Name of the shader program
            
        Returns:
            OpenGL program ID or None if not found
        """
        return self.shader_programs.get(name)
    
    def set_uniform_1i(self, name: str, value: int) -> bool:
        """
        Set an integer uniform value.
        
        Args:
            name: Name of the uniform
            value: Integer value
            
        Returns:
            True if the uniform was set successfully, False otherwise
        """
        program = self.get_current_program()
        if program is None:
            logger.log(logger.ERROR, "No active shader program")
            return False
        
        try:
            location = glGetUniformLocation(program, name)
            if location == -1:
                logger.log(logger.WARNING, f"Uniform '{name}' not found in current shader program")
                return False
            
            glUniform1i(location, value)
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error setting uniform '{name}': {str(e)}")
            return False
    
    def set_uniform_1f(self, name: str, value: float) -> bool:
        """
        Set a float uniform value.
        
        Args:
            name: Name of the uniform
            value: Float value
            
        Returns:
            True if the uniform was set successfully, False otherwise
        """
        program = self.get_current_program()
        if program is None:
            logger.log(logger.ERROR, "No active shader program")
            return False
        
        try:
            location = glGetUniformLocation(program, name)
            if location == -1:
                logger.log(logger.WARNING, f"Uniform '{name}' not found in current shader program")
                return False
            
            glUniform1f(location, value)
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error setting uniform '{name}': {str(e)}")
            return False
    
    def set_uniform_3f(self, name: str, x: float, y: float, z: float) -> bool:
        """
        Set a vec3 uniform value.
        
        Args:
            name: Name of the uniform
            x: X component
            y: Y component
            z: Z component
            
        Returns:
            True if the uniform was set successfully, False otherwise
        """
        program = self.get_current_program()
        if program is None:
            logger.log(logger.ERROR, "No active shader program")
            return False
        
        try:
            location = glGetUniformLocation(program, name)
            if location == -1:
                logger.log(logger.WARNING, f"Uniform '{name}' not found in current shader program")
                return False
            
            glUniform3f(location, x, y, z)
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error setting uniform '{name}': {str(e)}")
            return False
    
    def set_uniform_4f(self, name: str, x: float, y: float, z: float, w: float) -> bool:
        """
        Set a vec4 uniform value.
        
        Args:
            name: Name of the uniform
            x: X component
            y: Y component
            z: Z component
            w: W component
            
        Returns:
            True if the uniform was set successfully, False otherwise
        """
        program = self.get_current_program()
        if program is None:
            logger.log(logger.ERROR, "No active shader program")
            return False
        
        try:
            location = glGetUniformLocation(program, name)
            if location == -1:
                logger.log(logger.WARNING, f"Uniform '{name}' not found in current shader program")
                return False
            
            glUniform4f(location, x, y, z, w)
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error setting uniform '{name}': {str(e)}")
            return False
    
    def set_uniform_matrix4fv(self, name: str, matrix) -> bool:
        """
        Set a mat4 uniform value.
        
        Args:
            name: Name of the uniform
            matrix: 4x4 matrix (numpy array or list of 16 floats)
            
        Returns:
            True if the uniform was set successfully, False otherwise
        """
        program = self.get_current_program()
        if program is None:
            logger.log(logger.ERROR, "No active shader program")
            return False
        
        try:
            location = glGetUniformLocation(program, name)
            if location == -1:
                logger.log(logger.WARNING, f"Uniform '{name}' not found in current shader program")
                return False
            
            # Convert to the format expected by OpenGL
            if hasattr(matrix, 'flatten'):
                # It's a numpy array
                matrix_data = matrix.flatten('F')  # Column-major order
            else:
                # Assume it's already a flat list
                matrix_data = matrix
            
            glUniformMatrix4fv(location, 1, GL_FALSE, matrix_data)
            return True
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error setting uniform '{name}': {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """
        Clean up shader programs.
        """
        for name, program in self.shader_programs.items():
            glDeleteProgram(program)
            
        self.shader_programs = {}
        self.current_program = None