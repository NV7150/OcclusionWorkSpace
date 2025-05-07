import numpy as np
from typing import Optional


class Frame:
    """
    A data structure representing a single frame of sensor data in MR applications.
    
    Holds RGB image, depth image, accelerometer and gyroscope data with their timestamps.
    """
    
    def __init__(
        self, 
        timestamp: np.datetime64, 
        rgb: np.ndarray, 
        depth: np.ndarray, 
        acc: np.ndarray, 
        gyro: np.ndarray, 
        timestamp_depth: Optional[np.datetime64] = None, 
        timestamp_imu: Optional[np.datetime64] = None
    ):
        """
        Initialize a Frame object with the given parameters.
        
        Args:
            timestamp: Timestamp of the frame
            rgb: RGB image data as a numpy array (height x width x channels)
            depth: Depth image data as a numpy array (in meters, height x width)
            acc: Accelerometer data as a numpy array (m/s^2) [x, y, z]
            gyro: Gyroscope data as a numpy array (rad/s) [x, y, z]
            timestamp_depth: Timestamp for depth data. Defaults to frame timestamp if None.
            timestamp_imu: Timestamp for IMU data. Defaults to frame timestamp if None.
        """ 
        self._timestamp = timestamp
        self._rgb = rgb
        self._depth = depth
        self._acc = acc
        self._gyro = gyro
        self._timestamp_depth = timestamp_depth if timestamp_depth is not None else timestamp
        self._timestamp_imu = timestamp_imu if timestamp_imu is not None else timestamp
    
    @property
    def timestamp(self) -> np.datetime64:
        """Main frame timestamp."""
        return self._timestamp
    
    @property
    def timestamp_depth(self) -> np.datetime64:
        """Timestamp for the depth data."""
        return self._timestamp_depth
    
    @property
    def timestamp_imu(self) -> np.datetime64:
        """Timestamp for the IMU data."""
        return self._timestamp_imu
    
    @property
    def rgb(self) -> np.ndarray:
        """RGB image data as a numpy array."""
        return self._rgb
    
    @property
    def depth(self) -> np.ndarray:
        """Depth image data as a numpy array (in meters)."""
        return self._depth
    
    @property
    def acc(self) -> np.ndarray:
        """Accelerometer data as a numpy array (m/s^2)."""
        return self._acc
    
    @property
    def gyro(self) -> np.ndarray:
        """Gyroscope data as a numpy array (rad/s)."""
        return self._gyro
    
    @property
    def width(self) -> int:
        """Width of the RGB image."""
        return self._rgb.shape[1]
    
    @property
    def height(self) -> int:
        """Height of the RGB image."""
        return self._rgb.shape[0]
    
    @property
    def depth_width(self) -> int:
        """Width of the depth image."""
        return self._depth.shape[1]
    
    @property
    def depth_height(self) -> int:
        """Height of the depth image."""
        return self._depth.shape[0]
    
    def __str__(self) -> str:
        """Return string representation of the frame."""
        return (f"Frame(timestamp={self._timestamp}, "
                f"rgb_shape={self._rgb.shape}, "
                f"depth_shape={self._depth.shape})")
    
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return self.__str__()