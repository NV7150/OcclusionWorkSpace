import numpy as np


class Frame:
    def __init__(self, timestamp: np.datetime64, rgb: np.ndarray, depth: np.ndarray, acc: np.ndarray, gyro: np.ndarray, timestamp_depth: np.datetime64 = None, timestamp_imu: np.datetime64 = None):
        self.timestamp = timestamp
        self.rgb = rgb
        self.width = rgb.size[0]
        self.height = rgb.size[1]
        self.depth = depth
        self.depth_width = rgb.size[0]
        self.depth_height = depth.size[1]
        self.acc = acc
        self.gyro = gyro
