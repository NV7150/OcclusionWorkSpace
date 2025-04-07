import os
import sys
# Add parent directory to path so we can import from Systems and Occlusions
sys.path.append('..')
from Systems.BaseSystem import BaseSystem
from Logger import Logger, logger
from Occlusions.DepthThresholdOcclusion import DepthThresholdOcclusion, DepthGradientOcclusion
from Occlusions.SimpleOcclusion import SimpleOcclusion


def main():
    """
    Example usage of the Occlusion Framework.
    """
    
    dir_id = input("enter the output dir >")
    file_id = input("enter the file id >")
    
    # Configure logging
    log_options = input("Enter log options (comma-separated, leave empty for default) >").strip()
    if log_options:
        # Parse log options
        log_keys = [option.strip() for option in log_options.split(',')]
        logger.configure(enabled_log_keys=log_keys)
        logger.log(Logger.SYSTEM, f"Enabled log keys: {log_keys}")
    else:
        # Default to system logs and errors only
        logger.configure(enabled_log_keys=[Logger.SYSTEM, Logger.ERROR])
        logger.log(Logger.SYSTEM, "Using default log keys: system, error")
    
    # Define directories
    data_dirs = [
        os.path.join('..', 'LocalData', 'TestDepth'),
        # Uncomment below if you want to use the old data directories
        # os.path.join('..', 'LocalData', 'DepthIMUData1', 'Fast2Slow'),
        # os.path.join('..', 'LocalData', 'DepthIMUData1', 'Slow'),
    ]
    
    model_dirs = [
        os.path.join('..', 'LocalData', 'Models', 'Scene1'),
    ]
    
    output_dir = os.path.join('..', 'Output', dir_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create occlusion provider
    # You can choose between different occlusion providers:
    
    # 1. SimpleOcclusion - New provider that compares real camera depth with MR content depth
    occlusion_provider = SimpleOcclusion(max_depth=5.0)
    
    # 2. DepthThresholdOcclusion - Original provider that uses a threshold on depth
    # occlusion_provider = DepthThresholdOcclusion(threshold=0.3)
    
    # 3. DepthGradientOcclusion - Original provider that uses depth gradients
    # occlusion_provider = DepthGradientOcclusion(gradient_threshold=0.05)
    
    # Create and run the system
    system = BaseSystem(
        data_dirs=data_dirs,
        model_dirs=model_dirs,
        output_dir=output_dir,
        output_prefix=f"{file_id}",
        occlusion_provider=occlusion_provider,
        log_keys=logger.enabled_log_keys,
        log_to_file=False
    )
    
    # Process all data
    system.process()
    
    logger.log(Logger.SYSTEM, "Processing complete. Results saved to: " + output_dir)
    logger.log(Logger.SYSTEM, "Note: This example now uses the new SimpleOcclusion provider that compares")
    logger.log(Logger.SYSTEM, "      real camera depth with MR content depth from ContentsDepthCal.")


if __name__ == '__main__':
    main()