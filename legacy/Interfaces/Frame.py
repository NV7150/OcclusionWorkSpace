import numpy as np


class Frame:
    def __init__(self, timestamp: np.datetime64, rgb: np.ndarray, depth: np.ndarray, acc: np.ndarray, gyro: np.ndarray, timestamp_depth: np.datetime64 = None, timestamp_imu: np.datetime64 = None):
        """
        Initializes a Frame object with the given parameters.
        Args:
            timestamp (np.datetime64): Timestamp of the frame.
            rgb (np.ndarray): RGB image data as a numpy array.
            depth (np.ndarray): Depth image data as a numpy array (in meter).
            acc (np.ndarray): Accelerometer data as a numpy array (m/s^2).
            gyro (np.ndarray): Gyroscope data as a numpy array (rad/s).
            timestamp_depth (np.datetime64, optional): Timestamp for depth data. Defaults to None.
            timestamp_imu (np.datetime64, optional): Timestamp for IMU data. Defaults to None.
        """ 
        self.timestamp = timestamp
        self.rgb = rgb
        self.width = rgb.shape[1]  # Width is the second dimension in numpy arrays (height, width, channels)
        self.height = rgb.shape[0]  # Height is the first dimension
        self.depth = depth
        self.depth_width = rgb.shape[1]
        self.depth_height = depth.shape[0]
        self.acc = acc
        self.gyro = gyro
