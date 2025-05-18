from abc import ABC, abstractmethod
from typing import Type, TypeVar, Dict, Generic, Optional, Protocol, Any
import numpy as np

class SensorData(Protocol):
    """
    Protocol for any sensor data.
    
    All sensor data types must have a timestamp.
    """
    timestamp: np.datetime64

# Type variable for generic sensor data types
T = TypeVar('T', bound=SensorData)

class IFrame(ABC):
    """
    Interface for frame data representation.
    
    This interface defines the contract for classes that represent a frame of sensor data
    for a specific timestamp. It uses a component/plugin approach where different types
    of sensor data can be added to the frame.
    """
    
    @property
    @abstractmethod
    def timestamp(self) -> np.datetime64:
        """
        Get the frame timestamp.
        
        Returns:
            Timestamp of the frame
        """
        pass
    
    @abstractmethod
    def add_sensor(self, data: SensorData) -> None:
        """
        Register a sensor's data.
        
        Args:
            data: Sensor data to register
        """
        pass
    
    @abstractmethod
    def get_sensor(self, sensor_type: Type[T]) -> Optional[T]:
        """
        Retrieve data by its class.
        
        Args:
            sensor_type: Type of sensor data to retrieve
            
        Returns:
            Sensor data of the specified type, or None if not found
        """
        pass
    
    @abstractmethod
    def has_sensor(self, sensor_type: Type[SensorData]) -> bool:
        """
        Check if the frame has data for a specific sensor type.
        
        Args:
            sensor_type: Type of sensor data to check
            
        Returns:
            True if the frame has data for the specified sensor type, False otherwise
        """
        pass
    
    @abstractmethod
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        pass
    
    @abstractmethod
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value.
        
        Args:
            key: Metadata key
            default: Default value to return if key is not found
            
        Returns:
            Metadata value or default if not found
        """
        pass