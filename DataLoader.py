import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional
from Interfaces.Frame import Frame


class DataLoader:
    """
    DataLoader is responsible for loading RGB images, depth images, and IMU data
    from specified directories.
    """
    
    def __init__(self, data_dirs: List[str]):
        """
        Initialize the DataLoader with a list of data directories.
        
        Args:
            data_dirs: List of directory paths containing the data
        """
        self.data_dirs = data_dirs
        self.frames = {}  # Dictionary to store frames by timestamp
        
    def load_data(self) -> Dict[np.datetime64, Frame]:
        """
        Load all RGB images, depth images, and IMU data from the specified directories.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        for data_dir in self.data_dirs:
            # Load IMU data
            imu_file = os.path.join(data_dir, 'imu.csv')
            if not os.path.exists(imu_file):
                print(f"Warning: IMU file not found in {data_dir}")
                continue
                
            imu_data = pd.read_csv(imu_file)
            
            # Get RGB and depth image files
            rgb_files = glob.glob(os.path.join(data_dir, 'rgb_*.jpg')) + glob.glob(os.path.join(data_dir, 'rgb_*.png'))
            depth_files = glob.glob(os.path.join(data_dir, 'depth_*.png'))
            
            # Sort files by timestamp
            rgb_files.sort()
            depth_files.sort()
            
            # Process each RGB image
            for rgb_file in rgb_files:
                # Extract timestamp from filename
                timestamp_str = os.path.basename(rgb_file).split('_')[1].split('.')[0]
                timestamp = np.datetime64(float(timestamp_str), 's')
                
                # Find corresponding depth image
                depth_file = None
                for df in depth_files:
                    if timestamp_str in df:
                        depth_file = df
                        break
                
                if depth_file is None:
                    print(f"Warning: No depth image found for timestamp {timestamp_str}")
                    continue
                
                # Load RGB and depth images
                rgb_image = np.array(Image.open(rgb_file))
                depth_image = np.array(Image.open(depth_file))
                
                # Find closest IMU data
                closest_imu_idx = self._find_closest_timestamp(float(timestamp_str), imu_data['timestamp'].values)
                if closest_imu_idx is None:
                    print(f"Warning: No IMU data found for timestamp {timestamp_str}")
                    continue
                    
                # Get acceleration and gyroscope data
                acc = np.array([
                    imu_data.iloc[closest_imu_idx]['accel_x'],
                    imu_data.iloc[closest_imu_idx]['accel_y'],
                    imu_data.iloc[closest_imu_idx]['accel_z']
                ])
                
                gyro = np.array([
                    imu_data.iloc[closest_imu_idx]['gyro_x'],
                    imu_data.iloc[closest_imu_idx]['gyro_y'],
                    imu_data.iloc[closest_imu_idx]['gyro_z']
                ])
                
                # Create Frame object
                timestamp_depth = np.datetime64(float(timestamp_str), 's')
                timestamp_imu = np.datetime64(imu_data.iloc[closest_imu_idx]['timestamp'], 's')
                
                frame = Frame(
                    timestamp=timestamp,
                    rgb=rgb_image,
                    depth=depth_image,
                    acc=acc,
                    gyro=gyro,
                    timestamp_depth=timestamp_depth,
                    timestamp_imu=timestamp_imu
                )
                
                # Store frame
                self.frames[timestamp] = frame
        
        return self.frames
    
    def _find_closest_timestamp(self, target_timestamp: float, timestamps: np.ndarray) -> Optional[int]:
        """
        Find the index of the closest timestamp in the array.
        
        Args:
            target_timestamp: Target timestamp to find
            timestamps: Array of timestamps to search in
            
        Returns:
            Index of the closest timestamp or None if the array is empty
        """
        if len(timestamps) == 0:
            return None
            
        idx = np.abs(timestamps - target_timestamp).argmin()
        return idx
    
    def get_frame_by_timestamp(self, timestamp: np.datetime64) -> Optional[Frame]:
        """
        Get a frame by its timestamp.
        
        Args:
            timestamp: Timestamp of the frame to retrieve
            
        Returns:
            Frame object or None if not found
        """
        return self.frames.get(timestamp)
    
    def get_frames_sorted(self) -> List[Frame]:
        """
        Get all frames sorted by timestamp.
        
        Returns:
            List of Frame objects sorted by timestamp
        """
        sorted_timestamps = sorted(self.frames.keys())
        return [self.frames[ts] for ts in sorted_timestamps]