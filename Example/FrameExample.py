import os
import sys
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataLoaders.Frame import Frame
from Data import RGBData, DepthData, IMUData, CameraData, OcclusionData

def main():
    """
    Example demonstrating the use of the new Frame structure.
    """
    # Create a timestamp
    timestamp = np.datetime64('2025-05-08T12:00:00')
    
    # Create a frame
    frame = Frame(timestamp)
    
    # Create sensor data
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
    rgb_image[100:200, 200:300, 0] = 255  # Red rectangle
    rgb_data = RGBData(timestamp, rgb_image)
    
    depth_image = np.ones((480, 640), dtype=np.float32) * 5.0  # 5 meters depth
    depth_image[100:200, 200:300] = 1.0  # Object at 1 meter
    depth_data = DepthData(timestamp, depth_image)
    
    acc_data = np.array([0.1, 0.2, 9.8], dtype=np.float32)  # Acceleration (m/s^2)
    gyro_data = np.array([0.01, 0.02, 0.03], dtype=np.float32)  # Angular velocity (rad/s)
    imu_data = IMUData(timestamp, acc_data, gyro_data)
    
    camera_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    camera_data = CameraData(timestamp, camera_matrix)
    
    # Add sensor data to the frame
    frame.add_sensor(rgb_data)
    frame.add_sensor(depth_data)
    frame.add_sensor(imu_data)
    frame.add_sensor(camera_data)
    
    # Add metadata
    frame.set_metadata("source", "example")
    frame.set_metadata("description", "Example frame")
    
    # Access sensor data
    print(f"Frame timestamp: {frame.timestamp}")
    print(f"RGB image shape: {frame.get_rgb().shape}")
    print(f"Depth image shape: {frame.get_depth().shape}")
    print(f"Acceleration data: {frame.get_acc()}")
    print(f"Gyroscope data: {frame.get_gyro()}")
    print(f"Camera matrix:\n{frame.get_camera_matrix()}")
    print(f"Metadata: {frame.get_metadata('source')}, {frame.get_metadata('description')}")
    
    # Check if the frame has specific sensor data
    print(f"Has RGB data: {frame.has_sensor(RGBData)}")
    print(f"Has IMU data: {frame.has_sensor(IMUData)}")
    print(f"Has occlusion data: {frame.has_sensor(OcclusionData)}")
    
    # Add occlusion data
    occlusion_mask = np.zeros((480, 640), dtype=np.uint8)
    occlusion_mask[100:200, 200:300] = 1  # Object is occluded
    frame.set_occlusion_mask(occlusion_mask)
    
    # Check again
    print(f"Has occlusion data: {frame.has_sensor(OcclusionData)}")
    print(f"Occlusion mask shape: {frame.get_occlusion_mask().shape}")
    
    # Alternative way to create a frame with all sensor data at once
    frame2 = Frame.create_from_data(
        timestamp=timestamp,
        rgb=rgb_image,
        depth=depth_image,
        acc=acc_data,
        gyro=gyro_data,
        camera_matrix=camera_matrix,
        metadata={"source": "example2"}
    )
    
    print("\nFrame 2:")
    print(f"Frame timestamp: {frame2.timestamp}")
    print(f"RGB image shape: {frame2.get_rgb().shape}")
    print(f"Depth image shape: {frame2.get_depth().shape}")
    print(f"Metadata: {frame2.get_metadata('source')}")

if __name__ == "__main__":
    main()