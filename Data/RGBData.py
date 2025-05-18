from dataclasses import dataclass
import numpy as np

@dataclass
class RGBData:
    """
    RGB image data from a camera.
    """
    timestamp: np.datetime64
    image: np.ndarray
    
    @property
    def width(self) -> int:
        """Get the width of the RGB image."""
        return self.image.shape[1]
    
    @property
    def height(self) -> int:
        """Get the height of the RGB image."""
        return self.image.shape[0]