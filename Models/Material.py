import numpy as np
from typing import Optional

from ..core.IModel import IMaterial


class Material(IMaterial):
    """
    Implementation of the IMaterial interface.
    
    Represents material properties for a 3D model including colors, 
    shininess, and texture maps.
    """
    
    def __init__(
        self,
        name: str,
        diffuse_color: np.ndarray = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32),
        specular_color: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        ambient_color: np.ndarray = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32),
        shininess: float = 32.0,
        diffuse_texture_path: Optional[str] = None
    ):
        """
        Initialize a Material with the given properties.
        
        Args:
            name: Material name
            diffuse_color: RGBA diffuse color (default: light gray)
            specular_color: RGBA specular color (default: white)
            ambient_color: RGBA ambient color (default: dark gray)
            shininess: Shininess factor (default: 32.0)
            diffuse_texture_path: Path to the diffuse texture file (default: None)
        """
        self._name = name
        self._diffuse_color = diffuse_color
        self._specular_color = specular_color
        self._ambient_color = ambient_color
        self._shininess = shininess
        self._diffuse_texture_path = diffuse_texture_path
    
    def get_name(self) -> str:
        """Returns the name of the material."""
        return self._name
    
    def get_diffuse_color(self) -> np.ndarray:
        """Returns the diffuse color of the material (r,g,b,a)."""
        return self._diffuse_color
    
    def get_specular_color(self) -> np.ndarray:
        """Returns the specular color of the material (r,g,b,a)."""
        return self._specular_color
    
    def get_ambient_color(self) -> np.ndarray:
        """Returns the ambient color of the material (r,g,b,a)."""
        return self._ambient_color
    
    def get_shininess(self) -> float:
        """Returns the shininess factor of the material."""
        return self._shininess
    
    def get_diffuse_texture_path(self) -> Optional[str]:
        """Returns the file path of the diffuse texture map, if any."""
        return self._diffuse_texture_path
    
    def __str__(self) -> str:
        """String representation of the material."""
        texture_info = f", texture: {self._diffuse_texture_path}" if self._diffuse_texture_path else ""
        return (f"Material({self._name}, diffuse: {self._diffuse_color}"
                f"{texture_info})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the material."""
        return (f"Material({self._name}, "
                f"diffuse: {self._diffuse_color}, "
                f"specular: {self._specular_color}, "
                f"ambient: {self._ambient_color}, "
                f"shininess: {self._shininess}, "
                f"texture: {self._diffuse_texture_path or 'None'})")