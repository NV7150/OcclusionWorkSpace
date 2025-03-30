import os
import sys
# Add parent directory to path so we can import from Systems and Occlusions
sys.path.append('..')
from Systems.BaseSystem import BaseSystem
from Occlusions.DepthThresholdOcclusion import DepthThresholdOcclusion, DepthGradientOcclusion


def main():
    """
    Example usage of the Occlusion Framework.
    """
    
    dir_id = input("enter the output dir >")
    file_id = input("enter the file id >")
    
    # Define directories
    data_dirs = [
        os.path.join('..', 'LocalData', 'DepthIMUData1', 'Fast2Slow'),
        os.path.join('..', 'LocalData', 'DepthIMUData1', 'Slow'),
    ]
    
    model_dirs = [
        os.path.join('..', 'LocalData', 'Models', 'Scene1'),
    ]
    
    output_dir = os.path.join('..', 'Output', dir_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create occlusion provider
    # You can choose between DepthThresholdOcclusion and DepthGradientOcclusion
    # or implement your own
    occlusion_provider = DepthThresholdOcclusion(threshold=0.3)
    # occlusion_provider = DepthGradientOcclusion(gradient_threshold=0.05)
    
    # Create and run the system
    system = BaseSystem(
        data_dirs=data_dirs,
        model_dirs=model_dirs,
        output_dir=output_dir,
        output_prefix=f"{file_id}",
        occlusion_provider=occlusion_provider
    )
    
    # Process all data
    system.process()
    
    print("Processing complete. Results saved to:", output_dir)


if __name__ == '__main__':
    main()