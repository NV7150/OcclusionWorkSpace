from dataclasses import dataclass
import numpy as np

@dataclass
class DepthData:
    """
    Depth image data from a depth sensor.
    """
    timestamp: np.datetime64
    depth: np.ndarray
    
    @property
    def width(self) -> int:
        """Get the width of the depth image."""
        return self.depth.shape[1]
    
    @property
    def height(self) -> int:
        """Get the height of the depth image."""
        return self.depth.shape[0]