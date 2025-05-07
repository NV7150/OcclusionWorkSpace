import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional
import csv

from .BaseFrameLoader import BaseFrameLoader
from .Frame import Frame


class UniformedFrameLoader(BaseFrameLoader):
    """
    A frame loader that loads data from directories with a uniform structure.
    
    This loader expects:
    - RGB images named as rgb_<timestamp>.(jpg|png)
    - Depth images as CSV files named depth_<timestamp>.csv
    - IMU data in a single CSV file called imu.csv
    """
    
    def __init__(self, data_dirs: List[str]):
        """
        Initialize the UniformedFrameLoader with a list of data directories.
        
        Args:
            data_dirs: List of directories containing the frame data
        """
        super().__init__()
        self._data_dirs = data_dirs
    
    def load_data(self) -> Dict[np.datetime64, Frame]:
        """
        Load all RGB images, depth images, and IMU data from the specified directories.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        # Load RGB, depth and IMU data
        rgb_images = self._load_rgb_images()
        depth_images = self._load_depth_images()
        imu_data = self._load_imu_data()
        
        # Create frames for each timestamp where we have RGB data
        for timestamp, rgb_image in rgb_images.items():
            timestamp_str = pd.Timestamp(timestamp).timestamp()
            
            # Find corresponding depth image
            depth_image = depth_images.get(timestamp)
            if depth_image is None:
                print(f"Warning: No depth image found for timestamp {timestamp}")
                continue
            
            # Find closest IMU data
            closest_imu_timestamp = self._find_closest_timestamp(timestamp, list(imu_data.keys()))
            if closest_imu_timestamp is None:
                print(f"Warning: No IMU data found for timestamp {timestamp}")
                continue
            
            # Get acceleration and gyroscope data from closest IMU timestamp
            acc = imu_data[closest_imu_timestamp]['acc']
            gyro = imu_data[closest_imu_timestamp]['gyro']
            
            # Create Frame object
            frame = Frame(
                timestamp=timestamp,
                rgb=rgb_image,
                depth=depth_image,
                acc=acc,
                gyro=gyro,
                timestamp_depth=timestamp,
                timestamp_imu=closest_imu_timestamp
            )
            
            # Store frame
            self._frames[timestamp] = frame
        
        return self._frames
    
    def _load_rgb_images(self) -> Dict[np.datetime64, np.ndarray]:
        """
        Load RGB images from all data directories.
        
        Returns:
            Dictionary mapping timestamps to RGB images
        """
        rgb_images = {}
        
        for data_dir in self._data_dirs:
            # Get RGB image files
            rgb_files = glob.glob(os.path.join(data_dir, 'rgb_*.jpg')) + glob.glob(os.path.join(data_dir, 'rgb_*.png'))
            
            # Process each RGB image file
            for rgb_file in rgb_files:
                try:
                    # Extract timestamp from filename
                    timestamp_str = os.path.basename(rgb_file).split('_')[1].split('.')[0]
                    # Convert Unix timestamp to datetime64
                    timestamp = np.datetime64(pd.Timestamp.fromtimestamp(float(timestamp_str), tz='UTC'))
                    
                    # Load RGB image
                    rgb_image = np.array(Image.open(rgb_file))
                    
                    # Store RGB image
                    rgb_images[timestamp] = rgb_image
                except Exception as e:
                    print(f"Error loading RGB image {rgb_file}: {e}")
        
        return rgb_images
    
    def _load_depth_images(self) -> Dict[np.datetime64, np.ndarray]:
        """
        Load depth images from all data directories.
        
        Returns:
            Dictionary mapping timestamps to depth images
        """
        depth_images = {}
        
        for data_dir in self._data_dirs:
            # Get depth image files
            depth_files = glob.glob(os.path.join(data_dir, 'depth_*.csv'))
            
            # Process each depth image file
            for depth_file in depth_files:
                try:
                    # Extract timestamp from filename
                    timestamp_str = os.path.basename(depth_file).split('_')[1].split('.')[0]
                    # Convert Unix timestamp to datetime64
                    timestamp = np.datetime64(pd.Timestamp.fromtimestamp(float(timestamp_str), tz='UTC'))
                    
                    # Load depth data
                    depth_image = self._load_depth_from_csv(depth_file)
                    
                    # Store depth image
                    depth_images[timestamp] = depth_image
                except Exception as e:
                    print(f"Error loading depth image {depth_file}: {e}")
        
        return depth_images
    
    def _load_imu_data(self) -> Dict[np.datetime64, Dict[str, np.ndarray]]:
        """
        Load IMU data from all data directories.
        
        Returns:
            Dictionary mapping timestamps to dictionaries with 'acc' and 'gyro' keys
        """
        imu_data = {}
        
        for data_dir in self._data_dirs:
            # Load IMU data from CSV file
            imu_file = os.path.join(data_dir, 'imu.csv')
            if not os.path.exists(imu_file):
                print(f"Warning: IMU file not found in {data_dir}")
                continue
            
            try:
                imu_df = pd.read_csv(imu_file)
                
                # Process each row in the IMU data
                for _, row in imu_df.iterrows():
                    timestamp_sec = row['timestamp']
                    timestamp = np.datetime64(pd.Timestamp.fromtimestamp(timestamp_sec, tz='UTC'))
                    
                    acc = np.array([row['accel_x'], row['accel_y'], row['accel_z']])
                    gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
                    
                    imu_data[timestamp] = {
                        'acc': acc,
                        'gyro': gyro
                    }
            except Exception as e:
                print(f"Error loading IMU data from {imu_file}: {e}")
        
        return imu_data
    
    def _load_depth_from_csv(self, csv_file: str) -> np.ndarray:
        """
        Load depth data from a CSV file.
        
        Args:
            csv_file: Path to the CSV file containing depth data
            
        Returns:
            Depth data as a numpy array
        """
        try:
            depth_data = []
            with open(csv_file, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    depth_data.append([float(x) for x in row])
            
            depth_array = np.array(depth_data, dtype=np.float32)
            return depth_array
        except Exception as e:
            print(f"Error loading depth from CSV {csv_file}: {e}")
            # Return an empty array in case of error
            return np.zeros((480, 640), dtype=np.float32)  # Default size
    
    def _find_closest_timestamp(self, target: np.datetime64, timestamps: List[np.datetime64]) -> Optional[np.datetime64]:
        """
        Find the closest timestamp to the target timestamp.
        
        Args:
            target: Target timestamp
            timestamps: List of timestamps to search in
            
        Returns:
            Closest timestamp or None if the list is empty
        """
        if not timestamps:
            return None
        
        # Convert timestamps to nanoseconds for numerical comparison
        target_ns = pd.Timestamp(target).value
        timestamps_ns = [pd.Timestamp(ts).value for ts in timestamps]
        
        # Find the closest timestamp
        closest_idx = min(range(len(timestamps_ns)), key=lambda i: abs(timestamps_ns[i] - target_ns))
        
        return timestamps[closest_idx]