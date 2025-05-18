from dataclasses import dataclass
import numpy as np

@dataclass
class OcclusionData:
    """
    Occlusion mask data.
    """
    timestamp: np.datetime64
    mask: np.ndarray  # Binary mask (1 = occluded, 0 = not occluded)