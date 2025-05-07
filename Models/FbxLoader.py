import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
import pyassimp
from pyassimp.postprocess import aiProcess_Triangulate, aiProcess_FlipUVs, aiProcess_CalcTangentSpace

from .BaseAssetLoader import BaseAssetLoader
from .Model import Model
from .Mesh import Mesh
from .Material import Material
from ..core.IModel import IModel


class FbxLoader(BaseAssetLoader):
    """
    Implementation of BaseAssetLoader for loading FBX format 3D models.
    
    Uses PyAssimp to load FBX files and converts them into Model objects.
    """
    
    def __init__(self):
        """Initialize the FbxLoader with supported extensions."""
        super().__init__({'.fbx'})
        
    def get_supported_extensions(self) -> Set[str]:
        """
        Get the set of file extensions supported by this loader.
        
        Returns:
            Set of supported file extensions
        """
        return self._supported_extensions
    
    def load_model(self, file_path: str) -> IModel:
        """
        Load a model from an FBX file.
        
        Args:
            file_path: Path to the FBX file
            
        Returns:
            Loaded model as a Model instance
            
        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file is not an FBX file
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
        
        # Processing flags for PyAssimp
        processing_flags = (
            aiProcess_Triangulate |      # Triangulate polygons
            aiProcess_FlipUVs |          # Flip UV coordinates
            aiProcess_CalcTangentSpace   # Calculate tangent space
        )
        
        try:
            # Load the model using PyAssimp
            with pyassimp.load(file_path, processing=processing_flags) as scene:
                # First, create all materials
                for i, material in enumerate(scene.materials):
                    mat = self._convert_material(material, i, os.path.dirname(file_path))
                    model.add_material(mat)
                
                # Process each mesh in the scene
                for i, mesh in enumerate(scene.meshes):
                    mesh_obj = self._convert_mesh(mesh, i)
                    model.add_mesh(mesh_obj)
        
        except Exception as e:
            raise Exception(f"Failed to load FBX file {file_path}: {e}")
        
        return model
    
    def _convert_material(self, assimp_material: Any, index: int, model_dir: str) -> Material:
        """
        Convert an Assimp material to a Material instance.
        
        Args:
            assimp_material: PyAssimp material object
            index: Index of the material in the scene
            model_dir: Directory containing the model file (for textures)
            
        Returns:
            Material instance
        """
        # Generate a name for the material
        material_name = f"material_{index}"
        
        # Try to get material properties
        diffuse_color = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
        specular_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        ambient_color = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
        shininess = 32.0
        diffuse_texture_path = None
        
        # Extract material properties
        properties = dir(assimp_material)
        
        # Diffuse color
        if 'diffuse' in properties and assimp_material.diffuse is not None:
            diffuse = assimp_material.diffuse
            diffuse_color = np.array([diffuse[0], diffuse[1], diffuse[2], 1.0], dtype=np.float32)
        
        # Specular color
        if 'specular' in properties and assimp_material.specular is not None:
            specular = assimp_material.specular
            specular_color = np.array([specular[0], specular[1], specular[2], 1.0], dtype=np.float32)
        
        # Ambient color
        if 'ambient' in properties and assimp_material.ambient is not None:
            ambient = assimp_material.ambient
            ambient_color = np.array([ambient[0], ambient[1], ambient[2], 1.0], dtype=np.float32)
        
        # Shininess
        if 'shininess' in properties and assimp_material.shininess is not None:
            shininess = float(assimp_material.shininess)
        
        # Diffuse texture
        if 'diffuse_texture' in properties and assimp_material.diffuse_texture is not None:
            texture_file = assimp_material.diffuse_texture.split('*')[-1].strip()
            if texture_file:
                diffuse_texture_path = os.path.join(model_dir, texture_file)
                if not os.path.exists(diffuse_texture_path):
                    print(f"Warning: Texture file not found: {diffuse_texture_path}")
                    diffuse_texture_path = None
        
        return Material(
            name=material_name,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            ambient_color=ambient_color,
            shininess=shininess,
            diffuse_texture_path=diffuse_texture_path
        )
    
    def _convert_mesh(self, assimp_mesh: Any, index: int) -> Mesh:
        """
        Convert an Assimp mesh to a Mesh instance.
        
        Args:
            assimp_mesh: PyAssimp mesh object
            index: Index of the mesh in the scene
            
        Returns:
            Mesh instance
        """
        # Generate a name for the mesh
        mesh_name = f"mesh_{index}"
        
        # Extract vertex data
        vertices = np.array(assimp_mesh.vertices, dtype=np.float32)
        
        # Extract normal data
        normals = np.array(assimp_mesh.normals, dtype=np.float32) if hasattr(assimp_mesh, 'normals') and len(assimp_mesh.normals) > 0 else np.zeros_like(vertices)
        
        # Extract texture coordinates
        tex_coords = None
        if hasattr(assimp_mesh, 'texturecoords') and len(assimp_mesh.texturecoords) > 0 and assimp_mesh.texturecoords[0] is not None:
            # PyAssimp can return 3D texture coordinates, but we only need 2D (UV)
            tex_coords = np.array([uv[:2] for uv in assimp_mesh.texturecoords[0]], dtype=np.float32)
        
        # Extract face indices
        indices = np.array([face for face in assimp_mesh.faces], dtype=np.uint32)
        
        # Get material name
        material_name = f"material_{assimp_mesh.materialindex}" if hasattr(assimp_mesh, 'materialindex') else None
        
        return Mesh(
            name=mesh_name,
            vertices=vertices,
            normals=normals,
            tex_coords=tex_coords,
            indices=indices,
            material_name=material_name
        )