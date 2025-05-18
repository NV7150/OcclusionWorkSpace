from dataclasses import dataclass
import numpy as np

@dataclass
class IMUData:
    """
    Inertial Measurement Unit (IMU) data.
    """
    timestamp: np.datetime64
    acc: np.ndarray  # Acceleration data [x, y, z] in m/s^2
    gyro: np.ndarray  # Gyroscope data [x, y, z] in rad/s