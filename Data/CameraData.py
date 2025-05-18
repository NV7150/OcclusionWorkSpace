from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class CameraData:
    """
    Camera intrinsic and extrinsic parameters.
    """
    timestamp: np.datetime64
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    camera_pose: Optional[np.ndarray] = None  # 4x4 transformation matrix (extrinsic)