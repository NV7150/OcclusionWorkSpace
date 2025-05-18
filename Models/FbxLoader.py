from typing import Optional, Dict, List, Any, Tuple
import os
import numpy as np
from .BaseAssetLoader import BaseAssetLoader
from .Model import Model
from .Mesh import Mesh
from .Material import Material
from .Texture import Texture
from Utils.Logger import logger

class FbxLoader(BaseAssetLoader):
    """
    Loader for FBX model files.
    
    This class implements the BaseAssetLoader interface for loading FBX format 3D models.
    It uses PyAssimp to load the model data.
    """
    
    def __init__(self):
        """
        Initialize the FBX loader.
        """
        # Import PyAssimp here to avoid dependency issues if not needed
        try:
            import pyassimp
            self.pyassimp = pyassimp
            self.available = True
        except ImportError:
            logger.log(logger.WARNING, "PyAssimp not available. FBX loading will be disabled.")
            self.available = False
    
    def supports_extension(self, extension: str) -> bool:
        """
        Check if this loader supports a specific file extension.
        
        Args:
            extension: File extension (e.g., 'fbx')
            
        Returns:
            True if the extension is supported, False otherwise
        """
        return extension.lower() == 'fbx' and self.available
    
    def load_model(self, file_path: str, model_id: str) -> Optional[Model]:
        """
        Load a 3D model from an FBX file.
        
        Args:
            file_path: Path to the FBX file
            model_id: ID to assign to the loaded model
            
        Returns:
            Loaded Model object or None if loading failed
        """
        if not self.available:
            logger.log(logger.ERROR, "PyAssimp not available. Cannot load FBX file.")
            return None
        
        if not os.path.exists(file_path):
            logger.log(logger.ERROR, f"FBX file not found: {file_path}")
            return None
        
        try:
            # Define processing flags for PyAssimp
            processing_flags = (
                self.pyassimp.postprocess.aiProcess_Triangulate |
                self.pyassimp.postprocess.aiProcess_GenSmoothNormals |
                self.pyassimp.postprocess.aiProcess_FlipUVs |
                self.pyassimp.postprocess.aiProcess_CalcTangentSpace |
                self.pyassimp.postprocess.aiProcess_JoinIdenticalVertices
            )
            
            # Load the model using PyAssimp
            scene = self.pyassimp.load(file_path, processing=processing_flags)
            
            if not scene:
                logger.log(logger.ERROR, f"Failed to load FBX file: {file_path}")
                return None
            
            # Create a new model
            model = Model(model_id)
            
            # Process materials
            self._process_materials(scene, model, os.path.dirname(file_path))
            
            # Process meshes
            self._process_meshes(scene, model)
            
            # Release the PyAssimp scene
            self.pyassimp.release(scene)
            
            logger.log(logger.MODEL, f"Loaded FBX model: {file_path}")
            return model
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading FBX file {file_path}: {str(e)}")
            return None
    
    def _process_materials(self, scene, model: Model, base_dir: str):
        """
        Process materials from the PyAssimp scene.
        
        Args:
            scene: PyAssimp scene
            model: Model to add materials to
            base_dir: Base directory for texture paths
        """
        for i, assimp_material in enumerate(scene.materials):
            material_name = f"material_{i}"
            
            # Try to get material name from properties
            if hasattr(assimp_material, 'properties'):
                for key, value in assimp_material.properties.items():
                    if key == 'name':
                        material_name = value
                        break
            
            # Create a new material
            material = Material(material_name)
            
            # Set material properties
            if hasattr(assimp_material, 'properties'):
                # Ambient color
                if 'ambient' in assimp_material.properties:
                    ambient = assimp_material.properties['ambient']
                    if len(ambient) == 3:
                        ambient = (ambient[0], ambient[1], ambient[2], 1.0)
                    material.set_ambient(ambient)
                
                # Diffuse color
                if 'diffuse' in assimp_material.properties:
                    diffuse = assimp_material.properties['diffuse']
                    if len(diffuse) == 3:
                        diffuse = (diffuse[0], diffuse[1], diffuse[2], 1.0)
                    material.set_diffuse(diffuse)
                
                # Specular color
                if 'specular' in assimp_material.properties:
                    specular = assimp_material.properties['specular']
                    if len(specular) == 3:
                        specular = (specular[0], specular[1], specular[2], 1.0)
                    material.set_specular(specular)
                
                # Emission color
                if 'emissive' in assimp_material.properties:
                    emission = assimp_material.properties['emissive']
                    if len(emission) == 3:
                        emission = (emission[0], emission[1], emission[2], 1.0)
                    material.set_emission(emission)
                
                # Shininess
                if 'shininess' in assimp_material.properties:
                    material.set_shininess(assimp_material.properties['shininess'])
                
                # Opacity
                if 'opacity' in assimp_material.properties:
                    material.set_opacity(assimp_material.properties['opacity'])
                
                # Textures
                self._process_textures(assimp_material, material, model, base_dir)
            
            # Add the material to the model
            model.add_material(material_name, material)
    
    def _process_textures(self, assimp_material, material: Material, model: Model, base_dir: str):
        """
        Process textures from the PyAssimp material.
        
        Args:
            assimp_material: PyAssimp material
            material: Material to add textures to
            model: Model to add textures to
            base_dir: Base directory for texture paths
        """
        # Check for texture properties
        if not hasattr(assimp_material, 'properties'):
            return
        
        # Diffuse texture
        if 'file' in assimp_material.properties:
            texture_path = assimp_material.properties['file']
            if texture_path:
                # Make the path absolute if it's relative
                if not os.path.isabs(texture_path):
                    texture_path = os.path.join(base_dir, texture_path)
                
                # Create a new texture
                texture_name = os.path.basename(texture_path)
                texture = Texture(texture_name)
                
                # Load the texture
                if texture.load_from_file(texture_path):
                    # Add the texture to the model
                    model.add_texture(texture_name, texture)
                    
                    # Set the texture in the material
                    material.set_diffuse_texture(texture_name)
        
        # Other texture types (normal, specular, etc.) would be processed similarly
        # but PyAssimp's handling of these can vary, so we'd need to adapt based on
        # the specific FBX files being used
    
    def _process_meshes(self, scene, model: Model):
        """
        Process meshes from the PyAssimp scene.
        
        Args:
            scene: PyAssimp scene
            model: Model to add meshes to
        """
        for i, assimp_mesh in enumerate(scene.meshes):
            mesh_name = assimp_mesh.name or f"mesh_{i}"
            
            # Create a new mesh
            mesh = Mesh(mesh_name)
            
            # Process vertices
            vertices = []
            for vertex in assimp_mesh.vertices:
                vertices.extend([vertex[0], vertex[1], vertex[2]])
            mesh.set_vertices(vertices)
            
            # Process normals if available
            if hasattr(assimp_mesh, 'normals') and len(assimp_mesh.normals) > 0:
                normals = []
                for normal in assimp_mesh.normals:
                    normals.extend([normal[0], normal[1], normal[2]])
                mesh.set_normals(normals)
            
            # Process texture coordinates if available
            if hasattr(assimp_mesh, 'texturecoords') and len(assimp_mesh.texturecoords) > 0:
                uvs = []
                for uv in assimp_mesh.texturecoords[0]:
                    uvs.extend([uv[0], uv[1]])
                mesh.set_uvs(uvs)
            
            # Process face indices
            indices = []
            for face in assimp_mesh.faces:
                indices.extend(face)
            mesh.set_indices(indices)
            
            # Set material if available
            if hasattr(assimp_mesh, 'materialindex'):
                material_name = f"material_{assimp_mesh.materialindex}"
                mesh.set_material(material_name)
            
            # Add the mesh to the model
            model.add_mesh(mesh)