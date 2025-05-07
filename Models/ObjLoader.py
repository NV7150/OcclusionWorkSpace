import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
import pywavefront

from .BaseAssetLoader import BaseAssetLoader
from .Model import Model
from .Mesh import Mesh
from .Material import Material
from ..core.IModel import IModel


class ObjLoader(BaseAssetLoader):
    """
    Implementation of BaseAssetLoader for loading OBJ format 3D models.
    
    Uses PyWavefront to load OBJ files and converts them into Model objects.
    """
    
    def __init__(self):
        """Initialize the ObjLoader with supported extensions."""
        super().__init__({'.obj'})
        
    def get_supported_extensions(self) -> Set[str]:
        """
        Get the set of file extensions supported by this loader.
        
        Returns:
            Set of supported file extensions
        """
        return self._supported_extensions
    
    def load_model(self, file_path: str) -> IModel:
        """
        Load a model from an OBJ file.
        
        Args:
            file_path: Path to the OBJ file
            
        Returns:
            Loaded model as a Model instance
            
        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file is not an OBJ file
            Exception: If loading fails
        """
        # Check if file exists
        self.check_file_exists(file_path)
        
        # Check if file is supported
        if not self.is_supported(file_path):
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Create a model with the ID from the file name
        model_id = self.get_model_id_from_path(file_path)
        model = Model(model_id)
        
        try:
            # Load the model using PyWavefront
            wavefront_obj = pywavefront.Wavefront(
                file_path, 
                create_materials=True,
                collect_faces=True
            )
            
            # Process each material and its associated mesh
            for material_name, material in wavefront_obj.materials.items():
                # Create material
                mat = self._convert_material(material, material_name, os.path.dirname(file_path))
                model.add_material(mat)
                
                # Create mesh for this material
                if len(material.vertices) > 0:
                    mesh = self._convert_mesh(material, material_name)
                    model.add_mesh(mesh)
            
            return model
            
        except Exception as e:
            raise Exception(f"Failed to load OBJ file {file_path}: {e}")
    
    def _convert_material(self, wavefront_material: Any, material_name: str, model_dir: str) -> Material:
        """
        Convert a PyWavefront material to a Material instance.
        
        Args:
            wavefront_material: PyWavefront material object
            material_name: Name of the material
            model_dir: Directory containing the model file (for textures)
            
        Returns:
            Material instance
        """
        # Default material properties
        diffuse_color = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
        specular_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        ambient_color = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
        shininess = 32.0
        diffuse_texture_path = None
        
        # Extract wavefront material properties
        if hasattr(wavefront_material, 'diffuse'):
            diffuse = wavefront_material.diffuse
            # Make sure we have all 4 components (RGBA)
            if len(diffuse) == 3:
                diffuse = (*diffuse, 1.0)
            diffuse_color = np.array(diffuse, dtype=np.float32)
            
        if hasattr(wavefront_material, 'specular'):
            specular = wavefront_material.specular
            # Make sure we have all 4 components (RGBA)
            if len(specular) == 3:
                specular = (*specular, 1.0)
            specular_color = np.array(specular, dtype=np.float32)
            
        if hasattr(wavefront_material, 'ambient'):
            ambient = wavefront_material.ambient
            # Make sure we have all 4 components (RGBA)
            if len(ambient) == 3:
                ambient = (*ambient, 1.0)
            ambient_color = np.array(ambient, dtype=np.float32)
            
        if hasattr(wavefront_material, 'shininess'):
            shininess = float(wavefront_material.shininess)
        
        # Check for texture
        if hasattr(wavefront_material, 'texture'):
            texture_file = wavefront_material.texture
            if texture_file:
                # Look for the texture in the model directory
                diffuse_texture_path = os.path.join(model_dir, os.path.basename(texture_file))
                if not os.path.exists(diffuse_texture_path):
                    # If not found, try the full path from the OBJ file
                    diffuse_texture_path = texture_file
                    if not os.path.exists(diffuse_texture_path):
                        print(f"Warning: Texture file not found: {texture_file}")
                        diffuse_texture_path = None
        
        return Material(
            name=material_name,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            ambient_color=ambient_color,
            shininess=shininess,
            diffuse_texture_path=diffuse_texture_path
        )
    
    def _convert_mesh(self, wavefront_material: Any, material_name: str) -> Mesh:
        """
        Convert PyWavefront material vertex data to a Mesh instance.
        
        In PyWavefront, each material has its own vertex data.
        
        Args:
            wavefront_material: PyWavefront material object with vertex data
            material_name: Name of the material this mesh uses
            
        Returns:
            Mesh instance
        """
        # PyWavefront stores vertices as a flat array [x,y,z,nx,ny,nz,u,v, ...]
        # We need to extract vertices, normals, and texture coordinates
        vertices = []
        normals = []
        tex_coords = []
        
        # Stride and offsets depend on the vertex format of the material
        # PyWavefront vertex format: V, VT, VN, VTN
        stride = wavefront_material.vertex_size
        
        # Check if we have normals and/or texture coordinates
        has_normals = 'n' in wavefront_material.vertex_format
        has_texcoords = 't' in wavefront_material.vertex_format
        
        # Extract the vertices from the flat array
        vertex_data = np.array(wavefront_material.vertices, dtype=np.float32)
        
        # Reshape into stride-sized chunks
        vertex_data = vertex_data.reshape(-1, stride)
        
        # Extract position data (always first 3 values)
        vertices = vertex_data[:, 0:3]
        
        # Extract normal data if available
        if has_normals:
            # In PyWavefront, normals follow positions (and texture coordinates if present)
            normal_offset = 3
            if has_texcoords:
                normal_offset = 5  # 3 for position + 2 for texcoords
            normals = vertex_data[:, normal_offset:normal_offset+3]
        else:
            # If no normals, provide dummy normals
            normals = np.zeros_like(vertices)
        
        # Extract texture coordinate data if available
        if has_texcoords:
            # In PyWavefront, texture coordinates follow positions
            tex_coords = vertex_data[:, 3:5]
        else:
            tex_coords = None
        
        # Create indices for rendering - we'll use simple triangulation
        # Each triangle is defined by 3 consecutive vertices
        indices = np.array(
            [[i, i+1, i+2] for i in range(0, len(vertices), 3)],
            dtype=np.uint32
        )
        
        # Generate a name for the mesh
        mesh_name = f"mesh_{material_name}"
        
        return Mesh(
            name=mesh_name,
            vertices=vertices,
            normals=normals,
            tex_coords=tex_coords,
            indices=indices,
            material_name=material_name
        )