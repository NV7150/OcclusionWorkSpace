from abc import ABCMeta, abstractmethod

class ITextureManager(metaclass=ABCMeta):
    """
    Interface for managing textures in the rendering system.
    This includes loading, binding, and managing texture resources.
    """

    @abstractmethod
    def load_texture(self, texture_path: str) -> int:
        """
        Load a texture from a file and return its OpenGL texture ID.

        Args:
            texture_path: Path to the texture image file.

        Returns:
            The OpenGL texture ID.
        """
        pass

    @abstractmethod
    def bind_texture(self, texture_id: int):
        """
        Bind the specified texture for rendering.

        Args:
            texture_id: The ID of the texture to bind.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up texture resources.
        """
        pass