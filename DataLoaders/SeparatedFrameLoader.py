import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import csv
import json
import traceback # Added for detailed error logging
from typing import Dict, List, Tuple, Optional, Any
from .BaseFrameLoader import BaseFrameLoader
from .Frame import Frame
from Utils.Logger import logger
from Data import RGBData, DepthData, IMUData, CameraData

class SeparatedFrameLoader(BaseFrameLoader):
    """
    Loader for frame data stored in a specific separated directory structure.

    This loader expects the data to be organized as follows:
    - ${dataset_path}/
      - depth/
        - depth_{timestamp}.csv  (or nearest match within threshold)
      - imu/
        - *.csv (expects exactly one CSV file)
      - rgb/
        - rgb_{timestamp}.png
      - intrinsics/
        - *.json (expects exactly one JSON file)
    """
    
    def __init__(self, dataset_path: str, timestamp_threshold_ms: int = 500):
        """
        Initialize the SeparatedFrameLoader with the dataset path.

        Args:
            dataset_path: Path to the root directory of the dataset.
            timestamp_threshold_ms: Maximum allowed time difference in milliseconds
                                    between RGB and Depth frame timestamps for a valid match.
        """
        super().__init__()
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset path not found or is not a directory: {dataset_path}")
        self.dataset_path = dataset_path
        self.rgb_dir = os.path.join(dataset_path, 'rgb')
        self.depth_dir = os.path.join(dataset_path, 'depth')
        self.imu_dir = os.path.join(dataset_path, 'imu')
        self.intrinsics_dir = os.path.join(dataset_path, 'intrinsics')
        self.timestamp_threshold_s = timestamp_threshold_ms / 1000.0

        # --- Find IMU and Intrinsics files dynamically ---
        self.imu_file = self._find_single_file(self.imu_dir, '*.csv')
        self.intrinsics_file = self._find_single_file(self.intrinsics_dir, '*.json')
        # -------------------------------------------------

        # Pre-load intrinsics and IMU data - Raise error if critical load fails
        try:
            self.intrinsics_data = self._load_intrinsics()
            self.imu_data = self._load_imu_data()
        except (FileNotFoundError, json.JSONDecodeError, pd.errors.ParserError, ValueError) as e:
            logger.log(logger.ERROR, f"Failed to load critical data: {e}")
            raise RuntimeError(f"Failed to initialize SeparatedFrameLoader due to data loading error: {e}") from e

        # Set camera matrix from intrinsics if available
        if self.intrinsics_data and 'rgb_intrinsics' in self.intrinsics_data:
            intr = self.intrinsics_data['rgb_intrinsics']
            # Validate intrinsics structure
            if not all(k in intr for k in ('fx', 'fy', 'cx', 'cy')):
                raise ValueError("Invalid intrinsics JSON structure: missing fx, fy, cx, or cy")
            matrix = np.array([
                [intr['fx'], 0, intr['cx']],
                [0, intr['fy'], intr['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            self.set_camera_matrix(matrix)
            self.depth_intrinsics = self.intrinsics_data.get('depth_intrinsics')
        else:
            logger.log(logger.WARNING, "RGB Intrinsics not found in JSON file. Camera matrix not set.")
            self.depth_intrinsics = None

    def _find_single_file(self, directory: str, pattern: str) -> Optional[str]:
        """Finds a single file matching the pattern in a directory."""
        if not os.path.isdir(directory):
            logger.log(logger.ERROR, f"Directory not found: {directory}")
            return None
        files = glob.glob(os.path.join(directory, pattern))
        if len(files) == 0:
            logger.log(logger.ERROR, f"No file matching '{pattern}' found in {directory}")
            return None
        if len(files) > 1:
            logger.log(logger.WARNING, f"Multiple files matching '{pattern}' found in {directory}. Using first one: {files[0]}")
        return files[0]

    def _load_intrinsics(self) -> Dict:
        """
        Load camera intrinsics from the JSON file.
        Raises FileNotFoundError or json.JSONDecodeError on failure.
        """
        if not self.intrinsics_file:
            raise FileNotFoundError("Intrinsics file path was not determined.")
        # Removed Try/Except - Let exceptions propagate
        with open(self.intrinsics_file, 'r') as f:
            data = json.load(f) # Can raise json.JSONDecodeError
        logger.log(logger.SYSTEM, f"Loaded intrinsics from {self.intrinsics_file}")
        return data

    def _load_imu_data(self) -> pd.DataFrame:
        """
        Load IMU data from the CSV file.
        Raises FileNotFoundError or pd.errors.ParserError/ValueError on failure.
        """
        if not self.imu_file:
            raise FileNotFoundError("IMU file path was not determined.")
        # Removed Try/Except - Let exceptions propagate
        imu_df = pd.read_csv(self.imu_file) # Can raise ParserError, FileNotFoundError
        # Validate required columns exist
        required_cols = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        if not all(col in imu_df.columns for col in required_cols):
            raise ValueError(f"IMU CSV missing required columns in {self.imu_file}")

        # Convert Unix timestamp column to datetime64 immediately for easier comparison
        imu_df['timestamp_dt'] = pd.to_datetime(imu_df['timestamp'], unit='s', origin='unix') # Can raise ValueError
        imu_df.set_index('timestamp_dt', inplace=True)
        imu_df.sort_index(inplace=True)
        logger.log(logger.SYSTEM, f"Loaded and preprocessed IMU data from {self.imu_file}")
        return imu_df

    def _get_available_timestamps_for_sensor(self, data_dir: str, prefix: str, suffix: str) -> List[float]:
        """Scan a directory to find available frame timestamps (as floats) based on filename pattern."""
        timestamps = []
        if not os.path.isdir(data_dir):
            logger.log(logger.ERROR, f"Data directory not found: {data_dir}")
            return timestamps

        for filename in os.listdir(data_dir):
            if filename.startswith(prefix) and filename.endswith(suffix):
                try:
                    ts_str = filename.replace(prefix, '').replace(suffix, '')
                    timestamps.append(float(ts_str))
                except ValueError:
                    logger.log(logger.WARNING, f"Could not parse timestamp from filename: {filename} in directory {data_dir}")
        timestamps.sort()
        return timestamps

    def get_available_timestamps(self) -> List[float]:
        """Scan the rgb directory to find available frame timestamps (as floats)."""
        return self._get_available_timestamps_for_sensor(self.rgb_dir, 'rgb_', '.png')

    def _get_available_depth_timestamps(self) -> List[float]:
        """Scan the depth directory to find available depth timestamps (as floats)."""
        return self._get_available_timestamps_for_sensor(self.depth_dir, 'depth_', '.csv')

    @staticmethod
    def _find_nearest_timestamp(target_ts: float, available_timestamps: List[float]) -> Optional[Tuple[float, float]]:
        """
        Finds the nearest timestamp in a sorted list of available timestamps.

        Args:
            target_ts: The timestamp to find the nearest match for.
            available_timestamps: A sorted list of available timestamps.

        Returns:
            A tuple (nearest_timestamp, difference_in_seconds) or None if available_timestamps is empty.
        """
        if not available_timestamps:
            return None

        available_timestamps_np = np.array(available_timestamps)
        idx = np.searchsorted(available_timestamps_np, target_ts)

        if idx == 0:
            nearest_ts = available_timestamps_np[0]
        elif idx == len(available_timestamps_np):
            nearest_ts = available_timestamps_np[-1]
        else:
            prev_ts = available_timestamps_np[idx - 1]
            next_ts = available_timestamps_np[idx]
            if (target_ts - prev_ts) < (next_ts - target_ts):
                nearest_ts = prev_ts
            else:
                nearest_ts = next_ts
        
        difference = abs(target_ts - nearest_ts)
        return nearest_ts, difference

    @staticmethod
    def _convert_filename_ts_to_datetime64(ts_float: float) -> np.datetime64:
        """
        Convert filename float timestamp to numpy.datetime64.
        NOTE: Assumes the float timestamp represents seconds since the Unix epoch.
        This assumption might be incorrect and may need adjustment.
        """
        return np.datetime64(int(ts_float * 1e9), 'ns')

    def load_data(self) -> Dict[np.datetime64, Frame]:
        """
        Load all RGB, depth, IMU, and intrinsic data from the dataset path.
        RGB frames are primary. For each RGB frame, the nearest depth frame within
        `timestamp_threshold_s` is used.

        Returns:
            Dictionary mapping timestamps (np.datetime64 based on RGB) to Frame objects.
        """
        self.clear_frames()

        if self.imu_data is None or self.intrinsics_data is None:
            logger.log(logger.CRITICAL, "IMU data or Intrinsics data is None after initialization. This should not happen.")
            return self.frames

        available_rgb_ts_floats = self.get_available_timestamps()
        if not available_rgb_ts_floats:
            logger.log(logger.WARNING, f"No RGB frame timestamps found in {self.rgb_dir}")
            return self.frames

        available_depth_ts_floats = self._get_available_depth_timestamps()
        if not available_depth_ts_floats:
            logger.log(logger.WARNING, f"No depth frame timestamps found in {self.depth_dir}. Cannot match any frames.")
            return self.frames

        logger.log(logger.SYSTEM, f"Found {len(available_rgb_ts_floats)} potential RGB frames.")
        logger.log(logger.SYSTEM, f"Found {len(available_depth_ts_floats)} potential Depth frames. Matching with threshold: {self.timestamp_threshold_s * 1000:.0f} ms.")

        processed_count = 0
        for rgb_ts_float in available_rgb_ts_floats:
            rgb_timestamp_str = str(rgb_ts_float)
            # --- Frame Specific Processing Block --- 
            try:
                frame_timestamp_dt = self._convert_filename_ts_to_datetime64(rgb_ts_float)
                rgb_file = os.path.join(self.rgb_dir, f"rgb_{rgb_timestamp_str}.png")

                if not os.path.exists(rgb_file):
                    logger.log(logger.WARNING, f"RGB file missing for ts {rgb_timestamp_str}, skipping frame: {rgb_file}")
                    continue

                # Find nearest depth frame
                match_result = self._find_nearest_timestamp(rgb_ts_float, available_depth_ts_floats)
                
                if match_result is None: # Should not happen if available_depth_ts_floats is not empty
                    logger.log(logger.WARNING, f"Internal error: No depth timestamp could be found for RGB ts {rgb_ts_float} despite available depth files. Skipping frame.")
                    continue

                closest_depth_ts_float, diff_s = match_result
                
                if diff_s > self.timestamp_threshold_s:
                    logger.log(logger.WARNING, f"No suitable depth frame for RGB ts {rgb_ts_float}. "
                                               f"Nearest depth ts {closest_depth_ts_float} (diff: {diff_s*1000:.2f}ms) "
                                               f"exceeds threshold ({self.timestamp_threshold_s*1000:.0f}ms). Skipping frame.")
                    continue

                depth_timestamp_str = str(closest_depth_ts_float)
                depth_file = os.path.join(self.depth_dir, f"depth_{depth_timestamp_str}.csv")

                if not os.path.exists(depth_file):
                    logger.log(logger.WARNING, f"Matched Depth file missing for ts {depth_timestamp_str} (paired with RGB ts {rgb_timestamp_str}), skipping frame: {depth_file}")
                    continue
                
                logger.log(logger.DEBUG, f"Matched RGB ts {rgb_ts_float} with Depth ts {closest_depth_ts_float} (diff: {diff_s*1000:.2f}ms). Using depth file: {depth_file}")

                # Load RGB (can raise PIL errors)
                rgb_image = np.array(Image.open(rgb_file))

                # Load Depth (can raise ValueError, etc.)
                depth_image = self._load_depth_from_csv(depth_file)
                depth_timestamp_dt = self._convert_filename_ts_to_datetime64(closest_depth_ts_float)

                # Find corresponding IMU data
                closest_imu_index = self.imu_data.index.asof(frame_timestamp_dt)
                

                logger.log(logger.DEBUG, f"First IMU data timestamp: {self.imu_data.index[0]}, Current frame_timestamp_dt: {frame_timestamp_dt}")

                if pd.isna(closest_imu_index):
                    logger.log(logger.WARNING, f"No suitable IMU data found for timestamp {frame_timestamp_dt}, skipping frame.")
                    continue # Skip frame if no IMU data
                else:
                    imu_row = self.imu_data.loc[closest_imu_index]
                    acc = np.array([imu_row['accel_x'], imu_row['accel_y'], imu_row['accel_z']], dtype=np.float32)
                    gyro = np.array([imu_row['gyro_x'], imu_row['gyro_y'], imu_row['gyro_z']], dtype=np.float32)
                    imu_timestamp_dt = np.datetime64(closest_imu_index)

                # Create Frame and Add Sensors
                metadata = {
                    'rgb_file': rgb_file,
                    'depth_file': depth_file,
                    'dataset_path': self.dataset_path
                }
                frame = Frame(timestamp=frame_timestamp_dt, metadata=metadata)
                frame.add_sensor(RGBData(timestamp=frame_timestamp_dt, image=rgb_image))
                frame.add_sensor(DepthData(timestamp=depth_timestamp_dt, depth=depth_image))

                if self.intrinsics_data and 'rgb_intrinsics' in self.intrinsics_data:
                    intr = self.intrinsics_data['rgb_intrinsics']
                    cam_matrix = np.array([
                        [intr['fx'], 0, intr['cx']],
                        [0, intr['fy'], intr['cy']],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    frame.add_sensor(CameraData(timestamp=frame_timestamp_dt, camera_matrix=cam_matrix))
                    if self.depth_intrinsics:
                        frame.set_metadata('depth_intrinsics', self.depth_intrinsics)

                # IMU data already validated to exist before this point
                frame.add_sensor(IMUData(timestamp=imu_timestamp_dt, acc=acc, gyro=gyro))

                self.add_frame(frame)
                processed_count += 1

            # Catch errors specific to this frame's processing
            except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as frame_error:
                logger.log(logger.ERROR, f"Error processing frame for timestamp {rgb_timestamp_str}: {frame_error}")
                logger.log(logger.ERROR, traceback.format_exc())
                continue # Skip to the next frame
            except Exception as unexpected_error:
                # Catch any other unexpected errors for this frame
                logger.log(logger.CRITICAL, f"Unexpected error processing frame {rgb_timestamp_str}: {unexpected_error}")
                logger.log(logger.CRITICAL, traceback.format_exc())
                continue # Skip to the next frame
        logger.log(logger.SYSTEM, f"Successfully processed and loaded {processed_count} out of {len(available_rgb_ts_floats)} potential frames from {self.dataset_path}")
        return self.frames

    def _load_depth_from_csv(self, csv_file: str) -> np.ndarray:
        """
        Load depth data from a CSV file. Assumes comma-separated float values.
        Raises FileNotFoundError, ValueError, or other exceptions on failure.

        Args:
            csv_file: Path to the CSV file containing depth data

        Returns:
            Depth data as a numpy array.
        """
        # Removed Try/Except - Let exceptions like FileNotFoundError, ValueError propagate
        depth_data = []
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            for i, row in enumerate(csv_reader):
                try:
                    # Convert row to floats - raises ValueError if conversion fails
                    depth_data.append([float(val) for val in row])
                except ValueError as ve:
                    # Reraise with more context
                    raise ValueError(f"Error converting value in {os.path.basename(csv_file)}, row {i+1}: {ve}") from ve
        depth_array = np.array(depth_data, dtype=np.float32)
        return depth_array