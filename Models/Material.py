from typing import Dict, List, Optional, Tuple, Any

class Material:
    """
    Class representing a material for 3D models.
    
    A material defines the appearance of a surface, including properties like
    color, shininess, and textures.
    """
    
    def __init__(self, name: str):
        """
        Initialize a Material with a name.
        
        Args:
            name: Name of the material
        """
        self.name = name
        
        # Default material properties
        self.ambient = (0.2, 0.2, 0.2, 1.0)    # Ambient color (RGBA)
        self.diffuse = (0.8, 0.8, 0.8, 1.0)    # Diffuse color (RGBA)
        self.specular = (1.0, 1.0, 1.0, 1.0)   # Specular color (RGBA)
        self.emission = (0.0, 0.0, 0.0, 1.0)   # Emission color (RGBA)
        self.shininess = 32.0                  # Shininess factor
        self.opacity = 1.0                     # Opacity (1.0 = fully opaque)
        
        # Texture maps
        self.diffuse_texture = None            # Diffuse texture name
        self.specular_texture = None           # Specular texture name
        self.normal_texture = None             # Normal map texture name
        self.bump_texture = None               # Bump map texture name
        self.opacity_texture = None            # Opacity texture name
    
    def set_ambient(self, color: Tuple[float, float, float, float]):
        """
        Set the ambient color.
        
        Args:
            color: Ambient color as (R, G, B, A)
        """
        self.ambient = color
    
    def set_diffuse(self, color: Tuple[float, float, float, float]):
        """
        Set the diffuse color.
        
        Args:
            color: Diffuse color as (R, G, B, A)
        """
        self.diffuse = color
    
    def set_specular(self, color: Tuple[float, float, float, float]):
        """
        Set the specular color.
        
        Args:
            color: Specular color as (R, G, B, A)
        """
        self.specular = color
    
    def set_emission(self, color: Tuple[float, float, float, float]):
        """
        Set the emission color.
        
        Args:
            color: Emission color as (R, G, B, A)
        """
        self.emission = color
    
    def set_shininess(self, shininess: float):
        """
        Set the shininess factor.
        
        Args:
            shininess: Shininess factor
        """
        self.shininess = shininess
    
    def set_opacity(self, opacity: float):
        """
        Set the opacity.
        
        Args:
            opacity: Opacity value (0.0 = fully transparent, 1.0 = fully opaque)
        """
        self.opacity = opacity
    
    def set_diffuse_texture(self, texture_name: str):
        """
        Set the diffuse texture.
        
        Args:
            texture_name: Name of the diffuse texture
        """
        self.diffuse_texture = texture_name
    
    def set_specular_texture(self, texture_name: str):
        """
        Set the specular texture.
        
        Args:
            texture_name: Name of the specular texture
        """
        self.specular_texture = texture_name
    
    def set_normal_texture(self, texture_name: str):
        """
        Set the normal map texture.
        
        Args:
            texture_name: Name of the normal map texture
        """
        self.normal_texture = texture_name
    
    def set_bump_texture(self, texture_name: str):
        """
        Set the bump map texture.
        
        Args:
            texture_name: Name of the bump map texture
        """
        self.bump_texture = texture_name
    
    def set_opacity_texture(self, texture_name: str):
        """
        Set the opacity texture.
        
        Args:
            texture_name: Name of the opacity texture
        """
        self.opacity_texture = texture_name
    
    def get_ambient(self) -> Tuple[float, float, float, float]:
        """
        Get the ambient color.
        
        Returns:
            Ambient color as (R, G, B, A)
        """
        return self.ambient
    
    def get_diffuse(self) -> Tuple[float, float, float, float]:
        """
        Get the diffuse color.
        
        Returns:
            Diffuse color as (R, G, B, A)
        """
        return self.diffuse
    
    def get_specular(self) -> Tuple[float, float, float, float]:
        """
        Get the specular color.
        
        Returns:
            Specular color as (R, G, B, A)
        """
        return self.specular
    
    def get_emission(self) -> Tuple[float, float, float, float]:
        """
        Get the emission color.
        
        Returns:
            Emission color as (R, G, B, A)
        """
        return self.emission
    
    def get_shininess(self) -> float:
        """
        Get the shininess factor.
        
        Returns:
            Shininess factor
        """
        return self.shininess
    
    def get_opacity(self) -> float:
        """
        Get the opacity.
        
        Returns:
            Opacity value
        """
        return self.opacity
    
    def get_diffuse_texture(self) -> Optional[str]:
        """
        Get the diffuse texture.
        
        Returns:
            Name of the diffuse texture or None if not set
        """
        return self.diffuse_texture
    
    def get_specular_texture(self) -> Optional[str]:
        """
        Get the specular texture.
        
        Returns:
            Name of the specular texture or None if not set
        """
        return self.specular_texture
    
    def get_normal_texture(self) -> Optional[str]:
        """
        Get the normal map texture.
        
        Returns:
            Name of the normal map texture or None if not set
        """
        return self.normal_texture
    
    def get_bump_texture(self) -> Optional[str]:
        """
        Get the bump map texture.
        
        Returns:
            Name of the bump map texture or None if not set
        """
        return self.bump_texture
    
    def get_opacity_texture(self) -> Optional[str]:
        """
        Get the opacity texture.
        
        Returns:
            Name of the opacity texture or None if not set
        """
        return self.opacity_texture
    
    def has_textures(self) -> bool:
        """
        Check if the material has any textures.
        
        Returns:
            True if the material has at least one texture, False otherwise
        """
        return any([
            self.diffuse_texture,
            self.specular_texture,
            self.normal_texture,
            self.bump_texture,
            self.opacity_texture
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the material to a dictionary.
        
        Returns:
            Dictionary representation of the material
        """
        return {
            'name': self.name,
            'ambient': self.ambient,
            'diffuse': self.diffuse,
            'specular': self.specular,
            'emission': self.emission,
            'shininess': self.shininess,
            'opacity': self.opacity,
            'diffuse_texture': self.diffuse_texture,
            'specular_texture': self.specular_texture,
            'normal_texture': self.normal_texture,
            'bump_texture': self.bump_texture,
            'opacity_texture': self.opacity_texture
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Material':
        """
        Create a material from a dictionary.
        
        Args:
            data: Dictionary containing material properties
            
        Returns:
            Material object
        """
        material = cls(data['name'])
        
        if 'ambient' in data:
            material.set_ambient(data['ambient'])
        if 'diffuse' in data:
            material.set_diffuse(data['diffuse'])
        if 'specular' in data:
            material.set_specular(data['specular'])
        if 'emission' in data:
            material.set_emission(data['emission'])
        if 'shininess' in data:
            material.set_shininess(data['shininess'])
        if 'opacity' in data:
            material.set_opacity(data['opacity'])
        if 'diffuse_texture' in data and data['diffuse_texture']:
            material.set_diffuse_texture(data['diffuse_texture'])
        if 'specular_texture' in data and data['specular_texture']:
            material.set_specular_texture(data['specular_texture'])
        if 'normal_texture' in data and data['normal_texture']:
            material.set_normal_texture(data['normal_texture'])
        if 'bump_texture' in data and data['bump_texture']:
            material.set_bump_texture(data['bump_texture'])
        if 'opacity_texture' in data and data['opacity_texture']:
            material.set_opacity_texture(data['opacity_texture'])
        
        return material