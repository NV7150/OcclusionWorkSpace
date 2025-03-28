import numpy as np


## rgb: ndarray
class Frame:
    def __init__(self, rgb: np.ndarray, depth: np.ndarray, imu: np.ndarray):
        self.rgb = rgb
        self.width = rgb.size[0]
        self.height = rgb.size[1]
        self.depth = depth
        self.depth_width = rgb.size[0]
        self.depth_height = depth.size[1]
        self.imu  = imu
