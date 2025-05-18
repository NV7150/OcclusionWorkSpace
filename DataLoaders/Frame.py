import numpy as np
from typing import Type, TypeVar, Dict, Optional, Any
from core.IFrame import IFrame, SensorData
from Data import RGBData, DepthData, IMUData, CameraData, OcclusionData

# Type variable for generic sensor data types
T = TypeVar('T', bound=SensorData)

class Frame(IFrame):
    """
    Class representing a frame of sensor data.
    
    A frame encapsulates all sensor data (RGB, depth, IMU) for a specific timestamp.
    It uses a component/plugin approach where different types of sensor data can be
    added to the frame.
    """
    
    def __init__(self, timestamp: np.datetime64, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Frame with a timestamp.
        
        Args:
            timestamp: Timestamp of the frame
            metadata: Additional metadata for the frame
        """
        self._timestamp = timestamp
        self._metadata = metadata or {}
        self._sensors = {}
    
    @property
    def timestamp(self) -> np.datetime64:
        """
        Get the frame timestamp.
        
        Returns:
            Timestamp of the frame
        """
        return self._timestamp
    
    def add_sensor(self, data: SensorData) -> None:
        """
        Register a sensor's data.
        
        Args:
            data: Sensor data to register
        """
        self._sensors[type(data)] = data
    
    def get_sensor(self, sensor_type: Type[T]) -> Optional[T]:
        """
        Retrieve data by its class.
        
        Args:
            sensor_type: Type of sensor data to retrieve
            
        Returns:
            Sensor data of the specified type, or None if not found
        """
        return self._sensors.get(sensor_type)
    
    def has_sensor(self, sensor_type: Type[SensorData]) -> bool:
        """
        Check if the frame has data for a specific sensor type.
        
        Args:
            sensor_type: Type of sensor data to check
            
        Returns:
            True if the frame has data for the specified sensor type, False otherwise
        """
        return sensor_type in self._sensors
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value.
        
        Args:
            key: Metadata key
            default: Default value to return if key is not found
            
        Returns:
            Metadata value or default if not found
        """
        return self._metadata.get(key, default)
    
    # Convenience methods for common sensor data
    
    def get_rgb(self) -> Optional[np.ndarray]:
        """
        Get the RGB image.
        
        Returns:
            RGB image as a numpy array, or None if not available
        """
        rgb_data = self.get_sensor(RGBData)
        return rgb_data.image if rgb_data else None
    
    def get_depth(self) -> Optional[np.ndarray]:
        """
        Get the depth image.
        
        Returns:
            Depth image as a numpy array, or None if not available
        """
        depth_data = self.get_sensor(DepthData)
        return depth_data.depth if depth_data else None
    
    def get_acc(self) -> Optional[np.ndarray]:
        """
        Get the acceleration data.
        
        Returns:
            Acceleration data as a numpy array, or None if not available
        """
        imu_data = self.get_sensor(IMUData)
        return imu_data.acc if imu_data else None
    
    def get_gyro(self) -> Optional[np.ndarray]:
        """
        Get the gyroscope data.
        
        Returns:
            Gyroscope data as a numpy array, or None if not available
        """
        imu_data = self.get_sensor(IMUData)
        return imu_data.gyro if imu_data else None
    
    def get_camera_matrix(self) -> Optional[np.ndarray]:
        """
        Get the camera intrinsic matrix.
        
        Returns:
            Camera intrinsic matrix, or None if not available
        """
        camera_data = self.get_sensor(CameraData)
        return camera_data.camera_matrix if camera_data else None
    
    def set_camera_matrix(self, camera_matrix: np.ndarray) -> None:
        """
        Set the camera intrinsic matrix.
        
        Args:
            camera_matrix: Camera intrinsic matrix
        """
        camera_data = self.get_sensor(CameraData)
        if camera_data:
            camera_data.camera_matrix = camera_matrix
        else:
            self.add_sensor(CameraData(self.timestamp, camera_matrix))
    
    def get_camera_pose(self) -> Optional[np.ndarray]:
        """
        Get the camera pose.
        
        Returns:
            Camera pose as a 4x4 transformation matrix, or None if not available
        """
        camera_data = self.get_sensor(CameraData)
        return camera_data.camera_pose if camera_data else None
    
    def set_camera_pose(self, pose: np.ndarray) -> None:
        """
        Set the camera pose.
        
        Args:
            pose: Camera pose as a 4x4 transformation matrix
        """
        camera_data = self.get_sensor(CameraData)
        if camera_data:
            camera_data.camera_pose = pose
        else:
            # Create a default camera matrix if none exists
            camera_matrix = np.eye(3)
            self.add_sensor(CameraData(self.timestamp, camera_matrix, pose))
    
    def get_occlusion_mask(self) -> Optional[np.ndarray]:
        """
        Get the occlusion mask.
        
        Returns:
            Occlusion mask as a numpy array, or None if not available
        """
        occlusion_data = self.get_sensor(OcclusionData)
        return occlusion_data.mask if occlusion_data else None
    
    def set_occlusion_mask(self, mask: np.ndarray) -> None:
        """
        Set the occlusion mask.
        
        Args:
            mask: Occlusion mask as a numpy array
        """
        occlusion_data = self.get_sensor(OcclusionData)
        if occlusion_data:
            occlusion_data.mask = mask
        else:
            self.add_sensor(OcclusionData(self.timestamp, mask))
    
    @property
    def width(self) -> Optional[int]:
        """
        Get the width of the RGB image.
        
        Returns:
            Width in pixels, or None if no RGB image is available
        """
        rgb_data = self.get_sensor(RGBData)
        return rgb_data.width if rgb_data else None
    
    @property
    def height(self) -> Optional[int]:
        """
        Get the height of the RGB image.
        
        Returns:
            Height in pixels, or None if no RGB image is available
        """
        rgb_data = self.get_sensor(RGBData)
        return rgb_data.height if rgb_data else None
    
    @property
    def depth_width(self) -> Optional[int]:
        """
        Get the width of the depth image.
        
        Returns:
            Width in pixels, or None if no depth image is available
        """
        depth_data = self.get_sensor(DepthData)
        return depth_data.width if depth_data else None
    
    @property
    def depth_height(self) -> Optional[int]:
        """
        Get the height of the depth image.
        
        Returns:
            Height in pixels, or None if no depth image is available
        """
        depth_data = self.get_sensor(DepthData)
        return depth_data.height if depth_data else None
    
    @classmethod
    def create_from_data(cls, timestamp: np.datetime64, rgb: np.ndarray, depth: np.ndarray, 
                        acc: np.ndarray, gyro: np.ndarray, 
                        timestamp_depth: Optional[np.datetime64] = None, 
                        timestamp_imu: Optional[np.datetime64] = None,
                        camera_matrix: Optional[np.ndarray] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> 'Frame':
        """
        Create a Frame from sensor data.
        
        This is a convenience method for creating a Frame with all the common sensor data.
        
        Args:
            timestamp: Timestamp of the frame
            rgb: RGB image as a numpy array
            depth: Depth image as a numpy array
            acc: Acceleration data as a numpy array
            gyro: Gyroscope data as a numpy array
            timestamp_depth: Timestamp of the depth image (if different from frame timestamp)
            timestamp_imu: Timestamp of the IMU data (if different from frame timestamp)
            camera_matrix: Camera intrinsic matrix
            metadata: Additional metadata for the frame
            
        Returns:
            Frame object with all the sensor data
        """
        frame = cls(timestamp, metadata)
        
        # Add RGB data
        frame.add_sensor(RGBData(timestamp, rgb))
        
        # Add depth data
        depth_ts = timestamp_depth or timestamp
        frame.add_sensor(DepthData(depth_ts, depth))
        
        # Add IMU data
        imu_ts = timestamp_imu or timestamp
        frame.add_sensor(IMUData(imu_ts, acc, gyro))
        
        # Add camera data if available
        if camera_matrix is not None:
            frame.add_sensor(CameraData(timestamp, camera_matrix))
        
        return frame