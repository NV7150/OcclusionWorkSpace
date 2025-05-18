from typing import Optional, Dict, List, Any, Tuple
import os
import numpy as np
from .BaseAssetLoader import BaseAssetLoader
from .Model import Model
from .Mesh import Mesh
from .Material import Material
from .Texture import Texture
from Utils.Logger import logger

class ObjLoader(BaseAssetLoader):
    """
    Loader for OBJ model files.
    
    This class implements the BaseAssetLoader interface for loading OBJ format 3D models.
    It uses PyWavefront to load the model data.
    """
    
    def __init__(self):
        """
        Initialize the OBJ loader.
        """
        # Import PyWavefront here to avoid dependency issues if not needed
        try:
            import pywavefront
            self.pywavefront = pywavefront
            self.available = True
        except ImportError:
            logger.log(logger.WARNING, "PyWavefront not available. OBJ loading will be disabled.")
            self.available = False
    
    def supports_extension(self, extension: str) -> bool:
        """
        Check if this loader supports a specific file extension.
        
        Args:
            extension: File extension (e.g., 'obj')
            
        Returns:
            True if the extension is supported, False otherwise
        """
        return extension.lower() == 'obj' and self.available
    
    def load_model(self, file_path: str, model_id: str) -> Optional[Model]:
        """
        Load a 3D model from an OBJ file.
        
        Args:
            file_path: Path to the OBJ file
            model_id: ID to assign to the loaded model
            
        Returns:
            Loaded Model object or None if loading failed
        """
        if not self.available:
            logger.log(logger.ERROR, "PyWavefront not available. Cannot load OBJ file.")
            return None
        
        if not os.path.exists(file_path):
            logger.log(logger.ERROR, f"OBJ file not found: {file_path}")
            return None
        
        try:
            # Load the model using PyWavefront
            obj = self.pywavefront.Wavefront(
                file_path,
                create_materials=True,
                collect_faces=True,
                parse=True
            )
            
            if not obj:
                logger.log(logger.ERROR, f"Failed to load OBJ file: {file_path}")
                return None
            
            # Create a new model
            model = Model(model_id)
            
            # Process materials and textures
            self._process_materials(obj, model, os.path.dirname(file_path))
            
            # Process meshes
            self._process_meshes(obj, model)
            
            logger.log(logger.MODEL, f"Loaded OBJ model: {file_path}")
            return model
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading OBJ file {file_path}: {str(e)}")
            return None
    
    def _process_materials(self, obj, model: Model, base_dir: str):
        """
        Process materials from the PyWavefront object.
        
        Args:
            obj: PyWavefront object
            model: Model to add materials to
            base_dir: Base directory for texture paths
        """
        for material_name, pywf_material in obj.materials.items():
            # Create a new material
            material = Material(material_name)
            
            # Set material properties
            # Ambient color
            if hasattr(pywf_material, 'ambient') and pywf_material.ambient:
                ambient = pywf_material.ambient
                if len(ambient) == 3:
                    ambient = (ambient[0], ambient[1], ambient[2], 1.0)
                material.set_ambient(ambient)
            
            # Diffuse color
            if hasattr(pywf_material, 'diffuse') and pywf_material.diffuse:
                diffuse = pywf_material.diffuse
                if len(diffuse) == 3:
                    diffuse = (diffuse[0], diffuse[1], diffuse[2], 1.0)
                material.set_diffuse(diffuse)
            
            # Specular color
            if hasattr(pywf_material, 'specular') and pywf_material.specular:
                specular = pywf_material.specular
                if len(specular) == 3:
                    specular = (specular[0], specular[1], specular[2], 1.0)
                material.set_specular(specular)
            
            # Emission color
            if hasattr(pywf_material, 'emissive') and pywf_material.emissive:
                emission = pywf_material.emissive
                if len(emission) == 3:
                    emission = (emission[0], emission[1], emission[2], 1.0)
                material.set_emission(emission)
            
            # Shininess
            if hasattr(pywf_material, 'shininess') and pywf_material.shininess is not None:
                material.set_shininess(pywf_material.shininess)
            
            # Opacity
            if hasattr(pywf_material, 'transparency') and pywf_material.transparency is not None:
                material.set_opacity(1.0 - pywf_material.transparency)
            
            # Process textures
            self._process_textures(pywf_material, material, model, base_dir)
            
            # Add the material to the model
            model.add_material(material_name, material)
    
    def _process_textures(self, pywf_material, material: Material, model: Model, base_dir: str):
        """
        Process textures from the PyWavefront material.
        
        Args:
            pywf_material: PyWavefront material
            material: Material to add textures to
            model: Model to add textures to
            base_dir: Base directory for texture paths
        """
        # Diffuse texture
        if hasattr(pywf_material, 'texture') and pywf_material.texture:
            texture_path = pywf_material.texture.path
            
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
    
    def _process_meshes(self, obj, model: Model):
        """
        Process meshes from the PyWavefront object.
        
        Args:
            obj: PyWavefront object
            model: Model to add meshes to
        """
        for material_name, pywf_mesh in obj.meshes.items():
            # Create a new mesh
            mesh = Mesh(material_name)
            
            # Process vertices, normals, and UVs
            vertices = []
            normals = []
            uvs = []
            indices = []
            
            # PyWavefront stores vertices in a flattened format
            # We need to extract them and reorganize
            vertex_index = 0
            
            # Create a mapping from PyWavefront vertex to our vertex index
            vertex_map = {}
            
            for face in pywf_mesh.faces:
                face_indices = []
                
                for i in range(0, len(face), 3):  # Assuming 3 components per vertex (position, normal, uv)
                    # Get the vertex components
                    v_idx = face[i]
                    n_idx = face[i+1] if i+1 < len(face) else None
                    t_idx = face[i+2] if i+2 < len(face) else None
                    
                    # Create a unique key for this vertex combination
                    vertex_key = (v_idx, n_idx, t_idx)
                    
                    # Check if we've seen this vertex before
                    if vertex_key in vertex_map:
                        # Reuse the existing vertex
                        face_indices.append(vertex_map[vertex_key])
                    else:
                        # Add a new vertex
                        if v_idx is not None and v_idx < len(obj.vertices):
                            vertices.extend(obj.vertices[v_idx])
                        
                        # Add normal if available
                        if n_idx is not None and hasattr(obj, 'normals') and n_idx < len(obj.normals):
                            normals.extend(obj.normals[n_idx])
                        
                        # Add texture coordinate if available
                        if t_idx is not None and hasattr(obj, 'texcoords') and t_idx < len(obj.texcoords):
                            uvs.extend(obj.texcoords[t_idx])
                        
                        # Store the mapping
                        vertex_map[vertex_key] = vertex_index
                        face_indices.append(vertex_index)
                        vertex_index += 1
                
                # Add the face indices
                # Convert to triangles if necessary (assuming convex polygons)
                if len(face_indices) >= 3:
                    for i in range(1, len(face_indices) - 1):
                        indices.extend([face_indices[0], face_indices[i], face_indices[i+1]])
            
            # Set the mesh data
            mesh.set_vertices(vertices)
            
            if normals:
                mesh.set_normals(normals)
            else:
                # Calculate normals if not provided
                mesh.calculate_normals()
            
            if uvs:
                mesh.set_uvs(uvs)
            
            mesh.set_indices(indices)
            
            # Set material
            mesh.set_material(material_name)
            
            # Add the mesh to the model
            model.add_mesh(mesh)