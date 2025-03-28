import numpy as np


class Frame:
    def __init__(self, timestamp: np.datetime64, rgb: np.ndarray, depth: np.ndarray, acc: np.ndarray, gyro: np.ndarray, timestamp_depth: np.datetime64 = None, timestamp_imu: np.datetime64 = None):
        self.timestamp = timestamp
        self.rgb = rgb
        self.width = rgb.shape[1]  # Width is the second dimension in numpy arrays (height, width, channels)
        self.height = rgb.shape[0]  # Height is the first dimension
        self.depth = depth
        self.depth_width = rgb.shape[1]
        self.depth_height = depth.shape[0]
        self.acc = acc
        self.gyro = gyro
