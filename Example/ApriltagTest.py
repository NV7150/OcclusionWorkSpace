import os
import sys
import cv2
import numpy as np
import argparse
import pupil_apriltags as apriltags

# Add parent directory to path so we can import from other modules
sys.path.append('..')
from Logger import Logger, logger


def load_camera_matrix(file_path):
    """
    Load camera matrix from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the camera matrix
        
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix
        
    Raises:
        FileNotFoundError: If the camera matrix file does not exist
        ValueError: If the camera matrix has an invalid format or shape
    """
    if not os.path.exists(file_path):
        print(f"ERROR: Camera matrix file not found: {file_path}")
        raise FileNotFoundError(f"Camera matrix file not found: {file_path}")
    
    try:
        # Load the camera matrix from the CSV file
        matrix = np.loadtxt(file_path, delimiter=',')
        
        # Check if the loaded matrix has the correct shape
        if matrix.shape != (3, 3):
            print(f"ERROR: Invalid camera matrix shape in {file_path}. Expected (3, 3), got {matrix.shape}")
            raise ValueError(f"Invalid camera matrix shape in {file_path}. Expected (3, 3), got {matrix.shape}")
        
        print(f"Loaded camera matrix from {file_path}:")
        print(matrix)
        return matrix.astype(np.float32)
    except Exception as e:
        print(f"ERROR: Failed to load camera matrix from {file_path}: {e}")
        raise

def detect_apriltags(image_path, camera_matrix, tag_family, tag_size=0.05, debug_level=0):
    """
    Detect AprilTags in an image.
    
    Args:
        image_path (str): Path to the image file
        camera_matrix (np.ndarray): 3x3 camera intrinsic matrix
        tag_family (str): AprilTag family to detect
        tag_size (float): Size of the AprilTag in meters
        debug_level (int): Debug level for the detector (0-3)
        
    Returns:
        list: List of detected AprilTags
    """
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Print image information
    print(f"Loaded image from {image_path}")
    print(f"Image shape: {image.shape}")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Initialize the AprilTag detector
    detector = apriltags.Detector(
        families=tag_family,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=debug_level
    )
    
    # Detect AprilTags
    print(f"Detecting AprilTags with family: {tag_family}")
    detections = detector.detect(gray)
    
    # Print detection results
    print(f"Detected {len(detections)} AprilTags")
    for i, detection in enumerate(detections):
        print(f"Detection {i+1}:")
        print(f"  Tag ID: {detection.tag_id}")
        print(f"  Tag Family: {detection.tag_family.decode('utf-8')}")
        print(f"  Center: {detection.center}")
        print(f"  Corners: {detection.corners}")
        print(f"  Decision Margin: {detection.decision_margin}")
        print(f"  Hamming: {detection.hamming}")
        print(f"  Homography: {detection.homography}")
        
        # Draw detection on the image
        cv2.polylines(image, [np.int32(detection.corners)], True, (0, 255, 0), 2)
        cv2.putText(image, str(detection.tag_id), 
                    (int(detection.center[0]), int(detection.center[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the annotated image
    output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to {output_path}")
    
    return detections

def main():
    """
    Main function to test AprilTag detection.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test AprilTag detection')
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--camera-matrix', required=True, help='Path to the camera matrix CSV file')
    parser.add_argument('--tag-family', required=True, help='AprilTag family to detect (e.g., tag36h11, tag25h9, tag16h5)')
    parser.add_argument('--tag-size', type=float, default=0.05, help='Size of the AprilTag in meters')
    parser.add_argument('--debug-level', type=int, default=0, help='Debug level for the detector (0-3)')
    
    args = parser.parse_args()
    
    try:
        # Configure logger
        logger.configure(enabled_log_keys=[Logger.SYSTEM, Logger.DEBUG, Logger.ERROR])
        
        # Load camera matrix
        camera_matrix = load_camera_matrix(args.camera_matrix)
        
        # Detect AprilTags
        detections = detect_apriltags(
            args.image,
            camera_matrix,
            args.tag_family,
            args.tag_size,
            args.debug_level
        )
        
        # Print summary
        if detections:
            print(f"\nSUCCESS: Detected {len(detections)} AprilTags")
            print(f"Tag IDs: {[d.tag_id for d in detections]}")
        else:
            print("\nWARNING: No AprilTags detected")
            print("Try adjusting the parameters or using a different image")
            print("Available tag families: tag36h11, tag25h9, tag16h5, tagCircle21h7, tagCircle49h12, tagStandard41h12, tagStandard52h13, tagCustom48h12")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())