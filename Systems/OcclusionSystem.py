import os
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Union, Type
import importlib
import glob

# Core interfaces
from core.IFrameLoader import IFrameLoader
from core.IOcclusionProvider import IOcclusionProvider
from core.IRenderer import IRenderer
from core.IModel import IModel
from core.IScene import IScene
from core.ITracker import ITracker
from core.IFrame import IFrame

# Import system components
from Systems.OcclusionProcessor import OcclusionProcessor
from Utils.Logger import logger

class OcclusionSystem:
    """
    OcclusionSystem is the main class for the occlusion framework.
    
    It coordinates data loading, occlusion mask generation, and rendering
    using the refactored modular architecture.
    """
    
    def __init__(self,
                 frame_loader: IFrameLoader,
                 scene_manager: IScene,
                 renderer: IRenderer,
                 occlusion_provider: IOcclusionProvider,
                 output_dir: str = "Output",
                 output_prefix: str = "frame"):
        """
        Initialize the OcclusionSystem with components.
        
        Args:
            frame_loader: Frame loader for loading sensor data
            scene_manager: Scene manager for managing 3D models and scenes
            renderer: Renderer for rendering mixed reality scenes
            occlusion_provider: Occlusion provider for generating occlusion masks
            output_dir: Directory where rendered images will be saved
            output_prefix: Prefix for output filenames
        """
        self.frame_loader = frame_loader
        self.scene_manager = scene_manager
        self.renderer = renderer
        self.occlusion_provider = occlusion_provider
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        
        # Create occlusion processor
        self.occlusion_processor = OcclusionProcessor(renderer, occlusion_provider)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.log(logger.SYSTEM, f"Output directory: {output_dir}")
        
        # Data storage
        self.frames = {}
        self.occlusion_masks = {}
        
        logger.log(logger.SYSTEM, f"Initialized OcclusionSystem with {type(frame_loader).__name__}, "
                  f"{type(scene_manager).__name__}, {type(renderer).__name__}, "
                  f"and {type(occlusion_provider).__name__}")
    
    def load_data(self) -> None:
        """
        Load all data (frames, models, scenes).
        """
        logger.log(logger.SYSTEM, "Loading frames...")
        self.frames = self.frame_loader.load_data()
        logger.log(logger.SYSTEM, f"Loaded {len(self.frames)} frames")
    
    def generate_occlusion_masks(self, scene_name: Optional[str] = None) -> None:
        """
        Generate occlusion masks for all frames.
        
        Args:
            scene_name: Name of the scene to use for occlusion mask generation
        """
        logger.log(logger.SYSTEM, "Generating occlusion masks...")
        
        # Get scene data
        scene_data = self.scene_manager.get_scene_data()
        if not scene_data or 'objects' not in scene_data:
            logger.log(logger.ERROR, "No valid scene data loaded")
            return
        
        # Get models
        models = self.scene_manager.models
        if not models:
            logger.log(logger.ERROR, "No models loaded")
            return
        
        # Process frames to generate occlusion masks
        self.occlusion_masks = self.occlusion_processor.process_batch(
            self.frames, models, scene_data
        )
    
    def render_frames(self, scene_name: Optional[str] = None) -> None:
        """
        Render all frames with occlusion masks and save the results.
        
        Args:
            scene_name: Name of the scene to render
        """
        logger.log(logger.SYSTEM, f"Rendering frames...")
        
        # Get scene data
        scene_data = self.scene_manager.get_scene_data()
        if not scene_data or 'objects' not in scene_data:
            logger.log(logger.ERROR, "No valid scene data loaded")
            return
        
        # Get models
        models = self.scene_manager.models
        if not models:
            logger.log(logger.ERROR, "No models loaded")
            return
        
        # Get sorted frames
        frames_list = self.frame_loader.get_frames_sorted()
        logger.log(logger.DEBUG, f"Rendering {len(frames_list)} frames")
        
        # Render each frame
        output_paths = []
        for frame in frames_list:
            # Ensure frame has an occlusion mask
            if frame.get_metadata('occlusion_mask') is None:
                # Generate occlusion mask if not already generated
                occlusion_mask = self.occlusion_processor.process_frame(frame, models, scene_data)
                frame.set_metadata('occlusion_mask', occlusion_mask)
            
            # Render frame
            rendered_image = self.renderer.render_frame(frame, models, scene_data)
            
            # Save rendered image
            timestamp_str = str(frame.timestamp).replace(':', '-').replace(' ', '_')
            filename = f"{self.output_prefix}_{timestamp_str}.png"
            output_path = self.renderer.save_image(rendered_image, filename)
            output_paths.append(output_path)
        
        logger.log(logger.SYSTEM, f"Rendered {len(output_paths)} frames")
        logger.log(logger.SYSTEM, f"Output saved to {self.output_dir}")
    
    def process(self, scene_name: Optional[str] = None) -> None:
        """
        Process all data: load, generate occlusion masks, and render.
        
        Args:
            scene_name: Name of the scene to render
        """
        self.load_data()
        self.generate_occlusion_masks(scene_name)
        self.render_frames(scene_name)
        logger.log(logger.SYSTEM, "Processing complete")

    @classmethod
    def create(cls, 
              data_dirs: List[str],
              model_dirs: List[str],
              output_dir: str,
              output_prefix: str,
              occlusion_provider: Union[str, IOcclusionProvider],
              camera_matrix_file: Optional[str] = None) -> 'OcclusionSystem':
        """
        Create an OcclusionSystem with the specified parameters.
        
        Args:
            data_dirs: List of directories containing RGB, depth, and IMU data
            model_dirs: List of directories containing 3D models and scene descriptions
            output_dir: Directory where rendered images will be saved
            output_prefix: Prefix for output filenames
            occlusion_provider: OcclusionProvider instance or string specifying the provider class
            camera_matrix_file: Path to the camera matrix file (optional)
            
        Returns:
            OcclusionSystem instance
        """
        # Import concrete implementations only at creation time
        from DataLoaders.UniformedFrameLoader import UniformedFrameLoader
        from Models.SceneManager import SceneManager
        from Rendering.Renderer import Renderer
        
        # Create frame loader
        frame_loader = UniformedFrameLoader(data_dirs, camera_matrix_file)
        
        # Create renderer
        renderer = Renderer(output_dir)
        renderer.initialize()
        
        # Create scene manager
        scene_manager = SceneManager()
        
        # Load models
        models = {}
        for model_dir in model_dirs:
            # Find model files
            model_files = []
            for ext in ['obj', 'fbx']:
                model_files.extend(glob.glob(os.path.join(model_dir, f"*.{ext}")))
            
            # Load each model
            for model_file in model_files:
                model_id = os.path.splitext(os.path.basename(model_file))[0]
                from Models.BaseAssetLoader import BaseAssetLoader
                loader = BaseAssetLoader.create_loader_for_file(model_file)
                if loader:
                    model = loader.load_model(model_file, model_id)
                    if model:
                        models[model_id] = model
            
            # Find scene files
            scene_files = glob.glob(os.path.join(model_dir, "*.json"))
            
            # Load each scene
            for scene_file in scene_files:
                scene_id = os.path.splitext(os.path.basename(scene_file))[0]
                from Models.BaseSceneLoader import BaseSceneLoader
                loader = BaseSceneLoader.create_loader_for_file(scene_file)
                if loader:
                    scene_data = loader.load_scene(scene_file, scene_id)
                    if scene_data:
                        scene_manager.load_scene(scene_data)
        
        # Set models in scene manager
        scene_manager.models = models
        
        # Create occlusion provider
        if isinstance(occlusion_provider, str):
            # Import and instantiate the occlusion provider
            module_name, class_name = occlusion_provider.rsplit('.', 1)
            module = importlib.import_module(module_name)
            occlusion_provider_class = getattr(module, class_name)
            occlusion_provider = occlusion_provider_class()
        
        # Create OcclusionSystem
        return cls(
            frame_loader=frame_loader,
            scene_manager=scene_manager,
            renderer=renderer,
            occlusion_provider=occlusion_provider,
            output_dir=output_dir,
            output_prefix=output_prefix
        )