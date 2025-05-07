import numpy as np
from typing import Optional

from ..core.IModel import IMesh


class Mesh(IMesh):
    """
    Implementation of the IMesh interface.
    
    Represents a single mesh component of a 3D model with vertices, normals, 
    texture coordinates, and faces.
    """
    
    def __init__(
        self,
        name: str,
        vertices: np.ndarray,
        normals: np.ndarray,
        tex_coords: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        material_name: Optional[str] = None
    ):
        """
        Initialize a Mesh with the given geometry data.
        
        Args:
            name: Name of the mesh
            vertices: Vertex data as numpy array (Nx3 float array)
            normals: Normal data as numpy array (Nx3 float array)
            tex_coords: Texture coordinates as numpy array (Nx2 float array) or None
            indices: Face indices as numpy array (Nx3 int array) or None
            material_name: Name of the material associated with this mesh or None
        """
        self._name = name
        self._vertices = vertices
        self._normals = normals
        self._tex_coords = tex_coords
        
        # If indices are not provided, create simple triangulation (assumes vertices are ordered)
        if indices is None and len(vertices) > 2:
            self._indices = np.array(
                [[i, i+1, i+2] for i in range(0, len(vertices) - 2, 3)], 
                dtype=np.uint32
            )
        else:
            self._indices = indices if indices is not None else np.array([], dtype=np.uint32)
            
        self._material_name = material_name
    
    @property
    def name(self) -> str:
        """Get the name of the mesh."""
        return self._name
    
    def get_vertices(self) -> np.ndarray:
        """Returns the vertex data of the mesh."""
        return self._vertices
    
    def get_normals(self) -> np.ndarray:
        """Returns the normal data of the mesh."""
        return self._normals
    
    def get_tex_coords(self) -> Optional[np.ndarray]:
        """Returns the texture coordinate data of the mesh."""
        return self._tex_coords
    
    def get_indices(self) -> np.ndarray:
        """Returns the index data (faces) of the mesh."""
        return self._indices
    
    def get_material_name(self) -> Optional[str]:
        """Returns the name of the material associated with this mesh, if any."""
        return self._material_name
    
    def __str__(self) -> str:
        """String representation of the mesh."""
        return (f"Mesh({self._name}, {len(self._vertices)} vertices, "
                f"{len(self._indices)} faces, material: {self._material_name or 'None'})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the mesh."""
        return self.__str__()