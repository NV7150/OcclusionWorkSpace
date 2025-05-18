import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import csv
from typing import Dict, List, Tuple, Optional, Any
from .BaseFrameLoader import BaseFrameLoader
from .Frame import Frame
from Utils.Logger import logger
# Import sensor data types
from Data import RGBData, DepthData, IMUData, CameraData

class UniformedFrameLoader(BaseFrameLoader):
    """
    Loader for uniformed frame data format.
    
    This class implements the BaseFrameLoader interface for loading frame data
    from a directory structure where RGB images, depth images, and IMU data
    are stored in a uniform format.
    
    Expected directory structure:
    - data_dir/
      - rgb_*.jpg or rgb_*.png (RGB images with timestamp in filename)
      - depth_*.csv (Depth data in CSV format with timestamp in filename)
      - imu.csv (IMU data with timestamp column)
    """
    
    def __init__(self, data_dirs: List[str], camera_matrix_file: Optional[str] = None):
        """
        Initialize the UniformedFrameLoader with data directories.
        
        Args:
            data_dirs: List of directory paths containing the data
            camera_matrix_file: Path to the camera matrix file (optional)
        """
        super().__init__()
        self.data_dirs = data_dirs
        self.camera_matrix_file = camera_matrix_file
        
        # Load camera matrix if provided
        if camera_matrix_file and os.path.exists(camera_matrix_file):
            self._load_camera_matrix()
    
    def load_data(self) -> Dict[np.datetime64, Frame]:
        """
        Load all RGB images, depth images, and IMU data from the specified directories.
        
        Returns:
            Dictionary mapping timestamps to Frame objects
        """
        # Clear existing frames
        self.clear_frames()
        
        for data_dir in self.data_dirs:
            logger.log(logger.SYSTEM, f"Loading data from {data_dir}")
            
            # Load IMU data
            imu_file = os.path.join(data_dir, 'imu.csv')
            if not os.path.exists(imu_file):
                logger.log(logger.WARNING, f"IMU file not found in {data_dir}")
                continue
                
            try:
                imu_data = pd.read_csv(imu_file)
            except Exception as e:
                logger.log(logger.ERROR, f"Error loading IMU data from {imu_file}: {str(e)}")
                continue
            
            # Get RGB and depth image files
            rgb_files = glob.glob(os.path.join(data_dir, 'rgb_*.jpg')) + glob.glob(os.path.join(data_dir, 'rgb_*.png'))
            depth_files = glob.glob(os.path.join(data_dir, 'depth_*.csv')) + glob.glob(os.path.join(data_dir, 'depth_*.png'))
            
            if not rgb_files:
                logger.log(logger.WARNING, f"No RGB images found in {data_dir}")
                continue
                
            # Sort files by timestamp
            rgb_files.sort()
            depth_files.sort()
            
            logger.log(logger.SYSTEM, f"Found {len(rgb_files)} RGB images and {len(depth_files)} depth images")
            
            # Process each RGB image
            for rgb_file in rgb_files:
                # Extract timestamp from filename
                try:
                    timestamp_str = os.path.basename(rgb_file).split('_')[1].split('.')[0]
                    # Convert Unix timestamp to datetime64
                    timestamp = np.datetime64(pd.Timestamp.fromtimestamp(float(timestamp_str), tz='UTC'))
                except Exception as e:
                    logger.log(logger.WARNING, f"Error parsing timestamp from {rgb_file}: {str(e)}")
                    continue
                
                # Find corresponding depth image
                depth_file = self._find_matching_depth_file(depth_files, timestamp_str)
                
                if depth_file is None:
                    logger.log(logger.WARNING, f"No depth image found for timestamp {timestamp_str}")
                    continue
                
                # Wrap the entire frame processing in a general try-except
                try: 
                    # --- Section 1: Load images ---
                    rgb_image = None
                    depth_image = None
                    try:
                        rgb_image = np.array(Image.open(rgb_file))
                        if depth_file.endswith('.csv'):
                            depth_image = self._load_depth_from_csv(depth_file)
                        else:
                            depth_image = np.array(Image.open(depth_file))
                    except Exception as img_err:
                        logger.log(logger.ERROR, f"Error loading images for frame {timestamp_str}: {img_err}")
                        continue # Skip this frame if images can't be loaded

                    # --- Section 2: Process IMU ---
                    acc, gyro, timestamp_imu = None, None, None # Initialize
                    try:
                        closest_imu_idx = self._find_closest_timestamp(float(timestamp_str), imu_data['timestamp'].values)
                        if closest_imu_idx is None:
                            logger.log(logger.WARNING, f"No IMU data found for timestamp {timestamp_str}")
                            # Keep acc, gyro, timestamp_imu as None
                        else:
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
                            # Reinstate timestamp_imu definition:
                            timestamp_imu = np.datetime64(pd.Timestamp.fromtimestamp(imu_data.iloc[closest_imu_idx]['timestamp']))
                    except Exception as imu_err:
                         logger.log(logger.ERROR, f"Error processing IMU for frame {timestamp_str}: {imu_err}")
                         # Continue without IMU data, acc/gyro/timestamp_imu remain None

                    # --- Section 3: Create Frame and Add Sensors ---
                    frame = None # Initialize frame to None
                    try:
                        metadata = {
                            'rgb_file': rgb_file,
                            'depth_file': depth_file,
                            'data_dir': data_dir
                        }
                        frame = Frame(timestamp=timestamp, metadata=metadata)
                        if rgb_image is not None: frame.add_sensor(RGBData(timestamp, rgb_image))
                        if depth_image is not None: frame.add_sensor(DepthData(timestamp, depth_image))
                        if acc is not None and gyro is not None and timestamp_imu is not None: 
                            frame.add_sensor(IMUData(timestamp_imu, acc, gyro))
                        if self.camera_matrix is not None: frame.add_sensor(CameraData(timestamp, self.camera_matrix))
                    except Exception as frame_create_err:
                        logger.log(logger.ERROR, f"Error creating Frame or adding sensors for {timestamp_str}: {frame_create_err}")
                        continue # Skip this frame if it can't be created/populated

                    # --- Section 4: Add frame to collection ---
                    try:
                        if frame is not None: # Ensure frame was created successfully
                            self.add_frame(frame)
                        else:
                            # This case should ideally not be reached if previous continue statements work
                            logger.log(logger.WARNING, f"Frame object was None for timestamp {timestamp_str}, cannot add.")
                    except Exception as add_frame_err:
                        logger.log(logger.ERROR, f"Error adding frame {timestamp_str} to collection: {add_frame_err}")
                        # Decide if you want to continue or stop, continuing for now

                except Exception as e: # Catch-all for unexpected errors in the frame processing block
                    logger.log(logger.ERROR, f"Generic error processing frame {timestamp_str}: {str(e)}")
                    # --- DEBUG PRINT START ---
                    if 'frame' in locals() and frame is not None:
                        # Ensure logger.DEBUG is a valid key for your logger setup
                        logger.log(logger.DEBUG, f"DEBUG (Outer Catch): Offending frame object type: {type(frame)}")
                        logger.log(logger.DEBUG, f"DEBUG (Outer Catch): Offending frame module: {frame.__class__.__module__}")
                        logger.log(logger.DEBUG, f"DEBUG (Outer Catch): hasattr(frame, 'timestamp'): {hasattr(frame, 'timestamp')}")
                        logger.log(logger.DEBUG, f"DEBUG (Outer Catch): hasattr(frame, 'get_timestamp'): {hasattr(frame, 'get_timestamp')}")
                    else:
                        logger.log(logger.DEBUG, f"DEBUG (Outer Catch): 'frame' variable not available or None when error occurred.")
                    # --- DEBUG PRINT END ---
            
        logger.log(logger.SYSTEM, f"Loaded {len(self.frames)} frames")
        return self.frames
    
    def _find_matching_depth_file(self, depth_files: List[str], timestamp_str: str) -> Optional[str]:
        """
        Find the depth file matching the given timestamp.
        
        Args:
            depth_files: List of depth file paths
            timestamp_str: Timestamp string to match
            
        Returns:
            Path to the matching depth file or None if not found
        """
        for df in depth_files:
            if timestamp_str in df:
                return df
        return None
    
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
            
            # Keep the raw depth values in meters
            return depth_array
            
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading depth from CSV {csv_file}: {str(e)}")
            # Return an empty array in case of error
            return np.zeros((480, 640), dtype=np.float32)  # Default size
    
    def _load_camera_matrix(self) -> None:
        """
        Load camera matrix from file.
        """
        try:
            if self.camera_matrix_file.endswith('.csv'):
                # Load from CSV
                camera_matrix = np.loadtxt(self.camera_matrix_file, delimiter=',')
            elif self.camera_matrix_file.endswith('.npy'):
                # Load from NumPy file
                camera_matrix = np.load(self.camera_matrix_file)
            else:
                logger.log(logger.ERROR, f"Unsupported camera matrix file format: {self.camera_matrix_file}")
                return
            
            # Ensure it's a 3x3 matrix
            if camera_matrix.shape == (3, 3):
                self.set_camera_matrix(camera_matrix)
                logger.log(logger.SYSTEM, f"Loaded camera matrix from {self.camera_matrix_file}")
            else:
                logger.log(logger.ERROR, f"Invalid camera matrix shape: {camera_matrix.shape}, expected (3, 3)")
                
        except Exception as e:
            logger.log(logger.ERROR, f"Error loading camera matrix from {self.camera_matrix_file}: {str(e)}")