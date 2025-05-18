#!/usr/bin/env python3
"""
Dataloader Validation Script

This script loads data using SeparatedFrameLoader and performs checks on the loaded frames:
- Availability of RGB, Depth, and IMU sensor modules.
- Matching of RGB and Depth image resolutions.
- Timestamp alignment between RGB, Depth, and IMU data.
- Visualizes a sample of RGB and Depth frame pairs using OpenCV.
"""

import os
import sys
import argparse
import numpy as np
import cv2 # Import OpenCV

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataLoaders.SeparatedFrameLoader import SeparatedFrameLoader
from Data import RGBData, DepthData, IMUData #, CameraData (Import if CameraData check is needed)
from Utils.Logger import logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate SeparatedFrameLoader output.')
    parser.add_argument('--dataset-path', required=True, type=str,
                        help='Path to the root directory of the dataset for SeparatedFrameLoader.')
    parser.add_argument('--threshold-ms', type=int, default=500,
                        help='Timestamp threshold in milliseconds for matching RGB and Depth frames (default: 500).')
    parser.add_argument('--num-frames-to-visualize', type=int, default=5,
                        help='Number of frame pairs to visualize (default: 5). Set to 0 to disable visualization.')
    parser.add_argument('--log-keys', nargs='+', default=['error', 'system', 'warning'],
                        help='Logging keys to enable (e.g., system, debug, error, warning).')
    parser.add_argument('--log-to-file', action='store_true', default=False,
                        help='Enable logging to file.')
    parser.add_argument('--log-file', default='dataloader_validation.log',
                        help='Path to log file (default: dataloader_validation.log).')
    return parser.parse_args()

def main():
    """Main function to run the validation."""
    args = parse_args()

    # Configure logger
    logger.configure(
        enabled_log_keys=args.log_keys,
        log_to_file=args.log_to_file,
        log_file_path=args.log_file
    )

    logger.log(logger.SYSTEM, "Starting Dataloader Validation Script (using OpenCV for visualization).")
    logger.log(logger.SYSTEM, f"Dataset path: {args.dataset_path}")
    logger.log(logger.SYSTEM, f"Timestamp threshold: {args.threshold_ms} ms")
    logger.log(logger.SYSTEM, f"Frames to visualize: {args.num_frames_to_visualize}")

    try:
        loader = SeparatedFrameLoader(
            dataset_path=args.dataset_path,
            timestamp_threshold_ms=args.threshold_ms
        )
        logger.log(logger.SYSTEM, "SeparatedFrameLoader initialized.")
    except Exception as e:
        logger.log(logger.ERROR, f"Failed to initialize SeparatedFrameLoader: {e}")
        return

    logger.log(logger.SYSTEM, "Loading data...")
    try:
        frames_dict = loader.load_data()
    except Exception as e:
        logger.log(logger.ERROR, f"Failed to load data using SeparatedFrameLoader: {e}")
        return
        
    if not frames_dict:
        logger.log(logger.WARNING, "No frames were loaded. Exiting.")
        return

    logger.log(logger.SYSTEM, f"Successfully loaded {len(frames_dict)} frames.")

    # --- Frame Iteration and Checks ---
    frame_timestamps_sorted = sorted(frames_dict.keys())
    
    for i, frame_ts_dt64 in enumerate(frame_timestamps_sorted):
        frame = frames_dict[frame_ts_dt64]
        frame_id_str = f"Frame {i+1} (RGB ts: {frame_ts_dt64})"
        logger.log(logger.DEBUG, f"--- Checking {frame_id_str} ---")

        # 1. Sensor Availability
        rgb_sensor = frame.get_sensor(RGBData)
        depth_sensor = frame.get_sensor(DepthData)
        imu_sensor = frame.get_sensor(IMUData)

        if rgb_sensor:
            logger.log(logger.DEBUG, f"  [Sensor Check] RGBData: Present (ts: {rgb_sensor.timestamp})")
        else:
            logger.log(logger.ERROR, f"  [Sensor Check] RGBData: MISSING for {frame_id_str}")
        
        if depth_sensor:
            logger.log(logger.DEBUG, f"  [Sensor Check] DepthData: Present (ts: {depth_sensor.timestamp})")
        else:
            logger.log(logger.ERROR, f"  [Sensor Check] DepthData: MISSING for {frame_id_str}")

        if imu_sensor:
            logger.log(logger.DEBUG, f"  [Sensor Check] IMUData: Present (ts: {imu_sensor.timestamp})")
        else:
            logger.log(logger.WARNING, f"  [Sensor Check] IMUData: MISSING for {frame_id_str}") # IMU might be optional for some use cases

        # 2. Resolution Match (if both RGB and Depth are present)
        if rgb_sensor and depth_sensor:
            rgb_shape = rgb_sensor.image.shape
            depth_shape = depth_sensor.depth.shape
            if rgb_shape[:2] == depth_shape[:2]: # Compare height and width
                logger.log(logger.DEBUG, f"  [Resolution Check] RGB {rgb_shape[:2]} matches Depth {depth_shape[:2]}.")
            else:
                logger.log(logger.WARNING, f"  [Resolution Check] MISMATCH: RGB {rgb_shape[:2]} vs Depth {depth_shape[:2]} for {frame_id_str}.")
        
        # 3. Timestamp Match Verification (if all relevant sensors are present)
        if rgb_sensor and depth_sensor:
            rgb_ts_ns = rgb_sensor.timestamp.astype(np.int64)
            depth_ts_ns = depth_sensor.timestamp.astype(np.int64)
            diff_rgb_depth_ms = (depth_ts_ns - rgb_ts_ns) / 1e6 # nanoseconds to milliseconds
            logger.log(logger.DEBUG, f"  [Timestamp Check] RGB ts: {rgb_sensor.timestamp}, Depth ts: {depth_sensor.timestamp}")
            logger.log(logger.DEBUG, f"    Depth - RGB timestamp diff: {diff_rgb_depth_ms:.3f} ms")

        if rgb_sensor and imu_sensor:
            rgb_ts_ns = rgb_sensor.timestamp.astype(np.int64)
            imu_ts_ns = imu_sensor.timestamp.astype(np.int64)
            diff_rgb_imu_ms = (imu_ts_ns - rgb_ts_ns) / 1e6 # nanoseconds to milliseconds
            logger.log(logger.DEBUG, f"  [Timestamp Check] RGB ts: {rgb_sensor.timestamp}, IMU ts: {imu_sensor.timestamp}")
            logger.log(logger.DEBUG, f"    IMU - RGB timestamp diff: {diff_rgb_imu_ms:.3f} ms")
        elif rgb_sensor and not imu_sensor:
             logger.log(logger.DEBUG, f"  [Timestamp Check] IMU data not present for {frame_id_str}, skipping IMU timestamp comparison.")


    # --- Visualization with OpenCV ---
    if args.num_frames_to_visualize > 0 and len(frames_dict) > 0:
        logger.log(logger.SYSTEM, f"Visualizing up to {args.num_frames_to_visualize} frames using OpenCV...")
        
        frames_to_show_timestamps = frame_timestamps_sorted[:min(args.num_frames_to_visualize, len(frame_timestamps_sorted))]
        
        if not frames_to_show_timestamps:
            logger.log(logger.WARNING, "No frames available to visualize.")
        else:
            for idx, frame_ts_dt64 in enumerate(frames_to_show_timestamps):
                frame = frames_dict[frame_ts_dt64]
                rgb_s = frame.get_sensor(RGBData)
                depth_s = frame.get_sensor(DepthData)

                display_images = []

                if rgb_s is not None and rgb_s.image is not None:
                    rgb_display = cv2.cvtColor(rgb_s.image, cv2.COLOR_RGB2BGR)
                    display_images.append(rgb_display)
                else:
                    logger.log(logger.DEBUG, f"RGB data not available for visualization for frame {idx+1}.")
                    # Create a placeholder if RGB is missing to maintain structure if depth is present
                    if depth_s is not None and depth_s.depth is not None:
                         display_images.append(np.zeros((depth_s.depth.shape[0], depth_s.depth.shape[1], 3), dtype=np.uint8))

                if depth_s is not None and depth_s.depth is not None:
                    # Normalize depth for display: Convert to 0-255 uint8
                    depth_normalized = cv2.normalize(depth_s.depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_display = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR) # Convert to BGR to stack with RGB
                    display_images.append(depth_display)
                else:
                    logger.log(logger.DEBUG, f"Depth data not available for visualization for frame {idx+1}.")
                    # Create a placeholder if depth is missing and RGB is present
                    if rgb_s is not None and rgb_s.image is not None and not display_images:
                        display_images.append(np.zeros((rgb_s.image.shape[0], rgb_s.image.shape[1], 3), dtype=np.uint8))
                
                if display_images:
                    # Ensure consistent height for stacking if one image is missing and resolutions differ slightly
                    if len(display_images) == 2:
                        h1, w1, _ = display_images[0].shape
                        h2, w2, _ = display_images[1].shape
                        if h1 != h2:
                            # Resize the smaller one to match the height of the larger one, keeping aspect ratio for width
                            if h1 < h2:
                                new_w1 = int(w1 * (h2 / h1))
                                display_images[0] = cv2.resize(display_images[0], (new_w1, h2))
                            else:
                                new_w2 = int(w2 * (h1 / h2))
                                display_images[1] = cv2.resize(display_images[1], (new_w2, h1))
                        
                        combined_display = np.hstack(display_images)
                    elif len(display_images) == 1:
                        combined_display = display_images[0]
                    else: # No images to display
                        continue

                    window_title = f"Frame {idx+1} | RGB ts: {rgb_s.timestamp if rgb_s else 'N/A'} | Depth ts: {depth_s.timestamp if depth_s else 'N/A'}"
                    cv2.imshow(window_title, combined_display)
                    logger.log(logger.DEBUG, f"Showing {window_title}. Press any key to continue...")
                    if cv2.waitKey(0) == ord('q'): # Exit on 'q'
                        logger.log(logger.SYSTEM, "'q' pressed, stopping visualization.")
                        break 
            cv2.destroyAllWindows()
            logger.log(logger.SYSTEM, "OpenCV visualization finished.")

    logger.log(logger.SYSTEM, "Dataloader Validation Script finished.")

if __name__ == '__main__':
    main() 