# New Data Style Documentation

This document describes the file structure and formats for the new dataset style, exemplified by `LocalData/DepthIMUData3/still1`.

## Directory Structure

The dataset resides within a root directory (`${dataset_path}`) and contains the following subdirectories:

```
${dataset_path}/
  ├── depth/
  ├── imu/
  ├── rgb/
  └── intrinsics/
```

## File Formats

### 1. Depth Data

-   **Location**: `depth/`
-   **Filename Pattern**: `depth_{timestamp}.csv`
    -   `{timestamp}`: A floating-point number representing the frame's timestamp (e.g., `22826.846759083`). The exact epoch/origin of this timestamp is unclear but is consistent with RGB filenames.
-   **Format**: CSV (Comma Separated Values)
    -   Each row corresponds to a row of pixels in the depth image.
    -   Each value in a row represents the depth (in meters) for a specific pixel.
    -   No header row.
-   **Example Filename**: `depth_22826.846759083.csv`
-   **Example Content Snippet** (First row): `1.7832031,1.7841797,1.7695312,...`

### 2. RGB Data

-   **Location**: `rgb/`
-   **Filename Pattern**: `rgb_{timestamp}.png`
    -   `{timestamp}`: A floating-point number matching the corresponding depth file's timestamp (e.g., `22826.846759083`).
-   **Format**: Standard PNG image file.
-   **Example Filename**: `rgb_22826.846759083.png`

### 3. IMU Data

-   **Location**: `imu/`
-   **Filename**: `imu_data.csv` (Single file for the entire dataset)
-   **Format**: CSV with a header row.
    -   **Header**: `timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z`
    -   `timestamp`: Unix timestamp (seconds since epoch) with fractional seconds (e.g., `1746179287.738052`). Note this differs from the depth/RGB filename timestamps.
    -   Subsequent columns contain float values for accelerometer, gyroscope, and magnetometer readings.
-   **Example Content Snippet** (Second row): `1746179287.738052,-0.0069953...,0.02486...,0.01629...,...`

### 4. Camera Intrinsics

-   **Location**: `intrinsics/`
-   **Filename**: `camera_intrinsics.json` (Single file for the dataset)
-   **Format**: JSON object.
    -   Contains `rgb_intrinsics` and `depth_intrinsics` objects.
    -   Each intrinsics object contains `fx`, `fy`, `cx`, `cy` (focal lengths and principal point coordinates).
    -   Also includes a root-level `timestamp` field, likely a Unix timestamp (e.g., `1746180885.750066`), potentially indicating when the intrinsics were calibrated or recorded.
-   **Example Content Snippet**:
    ```json
    {
      "timestamp" : 1746180885.750066,
      "rgb_intrinsics" : {
        "fy" : 1592.4652099609375,
        "fx" : 1592.4652099609375,
        "cy" : 717.0284423828125,
        "cx" : 954.6080322265625
      },
      "depth_intrinsics" : {
        "cy" : 717.0284423828125,
        "fy" : 1592.4652099609375,
        "fx" : 1592.4652099609375,
        "cx" : 954.6080322265625
      }
    }
    ```

## Notes

-   The timestamps used in the `depth` and `rgb` filenames appear different from the Unix timestamps used within `imu_data.csv` and `camera_intrinsics.json`. Mapping between these might be necessary depending on the use case.
-   The `imu_data.csv` contains readings potentially spanning the time range of all RGB/depth frames, requiring filtering or interpolation to align with specific frame timestamps. 