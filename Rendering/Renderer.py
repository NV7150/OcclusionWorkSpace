import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from OpenGL.GL import *
from OpenGL.GLUT import *
import pyrr
from PIL import Image

from core.IRenderer import IRenderer
from core.IModel import IModel
from core.IFrame import IFrame
from Data.RGBData import RGBData
from Data.OcclusionData import OcclusionData

from Models.Model import Model
from Models.SceneManager import SceneManager
from .ShaderManager import ShaderManager
from .TextureManager import TextureManager
from .BufferManager import BufferManager
from .Framebuffer import Framebuffer
from .Camera import Camera
from .Primitives import Primitives
from Utils.Logger import logger

class Renderer(IRenderer):
    """
    Main renderer class for 3D scenes using modern OpenGL.
    
    This class implements the IRenderer interface and provides methods for
    rendering 3D scenes with proper lighting, texturing, and occlusion.
    """
    
    def __init__(self, output_dir: str = "Output"):
        """
        Initialize the Renderer.
        
        Args:
            output_dir: Directory where rendered images will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # OpenGL context
        self.window_width = 800
        self.window_height = 600
        self.window_initialized = False
        
        # Managers
        self.shader_manager = None
        self.texture_manager = None
        self.buffer_manager = None
        self.framebuffer_manager = None
        
        # Camera
        self.camera = None
        
        # Primitives
        self.primitives = None
        
        # Scene
        self.scene_manager = None
        
        # Rendering state
        self.wireframe_mode = False
        self.show_grid = True
        self.show_axes = True
        
        # Lighting
        self.light_position = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        self.light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.ambient_strength = 0.2
        self.specular_strength = 0.5
        
        # Model cache
        self.model_buffers = {}  # Dictionary mapping model IDs to buffer names
    
    def initialize(self) -> None:
        """
        Initialize the renderer.
        
        This method sets up OpenGL, creates managers, and initializes resources.
        """
        # Initialize GLUT if not already initialized
        if not self.window_initialized:
            self._initialize_glut()
        
        # Create managers
        self.shader_manager = ShaderManager()
        self.texture_manager = TextureManager()
        self.buffer_manager = BufferManager()
        self.framebuffer_manager = Framebuffer()
        
        # Create default shaders
        self.shader_manager.create_default_programs()
        
        # Create camera
        self.camera = Camera(
            position=(0, 0, 5),
            target=(0, 0, 0),
            up=(0, 1, 0),
            fov=60.0,
            aspect=self.window_width / self.window_height,
            near=0.1,
            far=100.0
        )
        
        # Create primitives
        self.primitives = Primitives(self.buffer_manager, self.shader_manager)
        
        # Set up OpenGL state
        self._setup_opengl_state()
        
        logger.log(logger.RENDER, "Renderer initialized")
    
    def _initialize_glut(self) -> None:
        """
        Initialize GLUT and create a window.
        """
        # Initialize GLUT
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        
        # Create a window
        glutInitWindowSize(self.window_width, self.window_height)
        glutCreateWindow("MR Occlusion Renderer")
        
        # Set the window as initialized
        self.window_initialized = True
        
        logger.log(logger.RENDER, "GLUT initialized")
    
    def _setup_opengl_state(self) -> None:
        """
        Set up OpenGL state.
        """
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Enable backface culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set clear color
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        logger.log(logger.RENDER, "OpenGL state set up")
    
    def set_viewport(self, width: int, height: int) -> None:
        """
        Set the viewport size.
        
        Args:
            width: Viewport width
            height: Viewport height
        """
        self.window_width = width
        self.window_height = height
        
        # Update viewport
        glViewport(0, 0, width, height)
        
        # Update camera aspect ratio
        if self.camera is not None:
            self.camera.set_projection(
                self.camera.fov,
                width / height,
                self.camera.near,
                self.camera.far
            )
        
        logger.log(logger.RENDER, f"Viewport set to {width}x{height}")
    
    def render_frame(self, frame: IFrame, models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        """
        Render a frame with the given models and scene data.
        
        Args:
            frame: Frame object containing RGB and depth images
            models: Dictionary mapping model IDs to model objects
            scene_data: Dictionary containing scene description
            
        Returns:
            Rendered image as a numpy array
        """
        # Get RGB data from frame
        rgb_data = frame.get_sensor(RGBData)
        if rgb_data is None:
            logger.log(logger.ERROR, "No RGB data in frame")
            return np.zeros((100, 100, 4), dtype=np.uint8)
            
        # Get frame dimensions from RGB data
        width, height = rgb_data.width, rgb_data.height
        
        # Set viewport to match frame dimensions
        self.set_viewport(width, height)
        
        # Create color texture for framebuffer
        color_texture_id = self.texture_manager.create_color_texture("render_target", width, height)
        
        # Create framebuffer
        self.framebuffer_manager.create_color_depth_framebuffer("render_fbo", width, height, color_texture_id)
        
        # Bind framebuffer
        self.framebuffer_manager.bind_framebuffer("render_fbo")
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Get camera matrix from frame or create a default one
        camera_matrix = frame.get_metadata('camera_pose')
        if camera_matrix is None:
            logger.log(logger.WARNING, "No camera pose in frame, using default camera")
            # Create a default camera
            self.camera.set_default()
        else:
            # Set camera from matrix
            self.camera.set_from_matrix(camera_matrix)
        
        # Create scene manager if not already created
        if self.scene_manager is None:
            self.scene_manager = SceneManager(models)
        else:
            self.scene_manager.models = models
        
        # Load scene data
        self.scene_manager.load_scene(scene_data)
        
        # Render scene
        self._render_scene(self.scene_manager)
        
        # Read framebuffer
        rendered_image = self.framebuffer_manager.read_color_buffer()
        
        # Unbind framebuffer
        self.framebuffer_manager.unbind_framebuffer()
        
        # Apply occlusion mask if available
        occlusion_mask = frame.get_metadata('occlusion_mask')
        if occlusion_mask is not None:
            # Check dimensions and resize if necessary
            if occlusion_mask.shape[:2] != (height, width):
                # Import here to avoid circular imports
                from PIL import Image
                
                # Resize occlusion mask
                if occlusion_mask.ndim == 2:
                    resized_mask = np.array(Image.fromarray(occlusion_mask).resize((width, height), Image.NEAREST))
                else:
                    resized_mask = np.array(Image.fromarray(occlusion_mask[:,:,0] if occlusion_mask.shape[2] >= 1 else occlusion_mask).resize((width, height), Image.NEAREST))
                    if occlusion_mask.ndim == 3 and occlusion_mask.shape[2] > 1:
                        # Convert to 2D if it was 3D
                        resized_mask = resized_mask[:,:,0] if resized_mask.ndim == 3 else resized_mask
            else:
                resized_mask = occlusion_mask
                if resized_mask.ndim == 3:
                    # Convert to 2D if it's 3D
                    resized_mask = resized_mask[:,:,0] if resized_mask.shape[2] >= 1 else np.mean(resized_mask, axis=2)
            
            # Create alpha mask
            alpha_mask = np.zeros((height, width), dtype=np.uint8)
            alpha_mask[resized_mask == 0] = 255  # Areas not occluded
            
            # Apply alpha mask to rendered image
            rendered_image = rendered_image.copy()
            rendered_image[:, :, 3] = np.minimum(rendered_image[:, :, 3], alpha_mask)
        
        # Composite rendered image over original RGB image
        rgb_image = rgb_data.image.copy()
        
        # Convert RGB to RGBA
        if rgb_image.shape[2] == 3:
            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
            rgba_image[:, :, :3] = rgb_image
            rgba_image[:, :, 3] = 255
        else:
            rgba_image = rgb_image
        
        # Alpha blend rendered image over RGB image
        alpha = rendered_image[:, :, 3].astype(float) / 255.0
        alpha = alpha[:, :, np.newaxis]
        
        blended_image = (1.0 - alpha) * rgba_image + alpha * rendered_image
        blended_image = blended_image.astype(np.uint8)
        
        return blended_image
    
    def render_depth(self, models: Dict[str, Any], scene_data: Dict, camera_matrix: np.ndarray) -> np.ndarray:
        """
        Render a depth map for the given models and scene data.
        
        Args:
            models: Dictionary mapping model IDs to model objects
            scene_data: Dictionary containing scene description
            camera_matrix: Camera matrix for the view
            
        Returns:
            Depth map as a numpy array
        """
        # Create a framebuffer for depth rendering
        width, height = self.window_width, self.window_height
        
        # Create depth texture
        depth_texture_id = self.texture_manager.create_depth_texture("depth_target", width, height)
        
        # Create framebuffer
        self.framebuffer_manager.create_depth_framebuffer("depth_fbo", width, height, depth_texture_id)
        
        # Bind framebuffer
        self.framebuffer_manager.bind_framebuffer("depth_fbo")
        
        # Clear depth buffer
        glClear(GL_DEPTH_BUFFER_BIT)
        
        # Set up camera from matrix
        self.camera.set_from_matrix(camera_matrix)
        
        # Create scene manager if not already created
        if self.scene_manager is None:
            self.scene_manager = SceneManager(models)
        else:
            self.scene_manager.models = models
        
        # Load scene data
        self.scene_manager.load_scene(scene_data)
        
        # Use depth shader
        self.shader_manager.use_program("depth")
        
        # Set projection and view matrices
        self.shader_manager.set_uniform_matrix4fv("projection", self.camera.get_projection_matrix())
        self.shader_manager.set_uniform_matrix4fv("view", self.camera.get_view_matrix())
        
        # Render scene objects
        for instance_id, instance_data in self.scene_manager.get_model_instances().items():
            model_id = instance_data["model_id"]
            model = self.scene_manager.models.get(model_id)
            
            if model is None:
                logger.log(logger.WARNING, f"Model '{model_id}' not found")
                continue
            
            # Set model matrix
            self.shader_manager.set_uniform_matrix4fv("model", instance_data["transform"])
            
            # Render model
            self._render_model(model)
        
        # Read depth buffer
        depth_data = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth_map = np.frombuffer(depth_data, dtype=np.float32).reshape(height, width)
        
        # Flip depth map vertically (OpenGL has origin at bottom-left)
        depth_map = np.flipud(depth_map)
        
        # Unbind framebuffer
        self.framebuffer_manager.unbind_framebuffer()
        
        # Clean up
        self.texture_manager.delete_texture("depth_target")
        self.framebuffer_manager.delete_framebuffer("depth_fbo")
        
        return depth_map
    
    def _render_scene(self, scene_manager: SceneManager) -> None:
        """
        Render a scene.
        
        Args:
            scene_manager: SceneManager containing the scene to render
        """
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set wireframe mode if enabled
        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # Render grid if enabled
        if self.show_grid:
            # Use default shader
            self.shader_manager.use_program("default")
            
            # Set projection and view matrices
            self.shader_manager.set_uniform_matrix4fv("projection", self.camera.get_projection_matrix())
            self.shader_manager.set_uniform_matrix4fv("view", self.camera.get_view_matrix())
            
            # Set model matrix (identity)
            model_matrix = np.eye(4, dtype=np.float32)
            self.shader_manager.set_uniform_matrix4fv("model", model_matrix)
            
            # Set object color
            self.shader_manager.set_uniform_4f("objectColor", 0.5, 0.5, 0.5, 1.0)
            
            # Disable texturing
            self.shader_manager.set_uniform_1i("useTexture", 0)
            
            # Render grid
            self.primitives.render_grid()
        
        # Render axes if enabled
        if self.show_axes:
            # Use default shader
            self.shader_manager.use_program("default")
            
            # Set projection and view matrices
            self.shader_manager.set_uniform_matrix4fv("projection", self.camera.get_projection_matrix())
            self.shader_manager.set_uniform_matrix4fv("view", self.camera.get_view_matrix())
            
            # Set model matrix (identity)
            model_matrix = np.eye(4, dtype=np.float32)
            self.shader_manager.set_uniform_matrix4fv("model", model_matrix)
            
            # Render axes
            self.primitives.render_axes()
        
        # Use default shader for scene objects
        self.shader_manager.use_program("default")
        
        # Set projection and view matrices
        self.shader_manager.set_uniform_matrix4fv("projection", self.camera.get_projection_matrix())
        self.shader_manager.set_uniform_matrix4fv("view", self.camera.get_view_matrix())
        
        # Set lighting parameters
        self.shader_manager.set_uniform_3f("lightPos", self.light_position[0], self.light_position[1], self.light_position[2])
        self.shader_manager.set_uniform_3f("viewPos", self.camera.position[0], self.camera.position[1], self.camera.position[2])
        
        # Render scene objects
        for instance_id, instance_data in scene_manager.get_model_instances().items():
            model_id = instance_data["model_id"]
            model = scene_manager.models.get(model_id)
            
            if model is None:
                logger.log(logger.WARNING, f"Model '{model_id}' not found")
                continue
            
            # Set model matrix
            self.shader_manager.set_uniform_matrix4fv("model", instance_data["transform"])
            
            # Render model
            self._render_model(model)
    
    def _render_model(self, model: Union[Model, IModel]) -> None:
        """
        Render a model.
        
        Args:
            model: Model to render
        """
        # Check if model is an IModel instance
        if not isinstance(model, IModel):
            logger.log(logger.WARNING, f"Model is not an IModel instance: {type(model)}")
            return
        
        # Get model ID
        model_id = model.name if hasattr(model, 'name') else id(model)
        
        # Check if model buffers are already created
        if model_id not in self.model_buffers:
            # Create buffers for the model
            self._create_model_buffers(model, model_id)
        
        # Get model buffer names
        buffer_names = self.model_buffers.get(model_id)
        if buffer_names is None:
            logger.log(logger.WARNING, f"No buffers found for model '{model_id}'")
            return
        
        # Render each mesh
        for i, mesh_name in enumerate(buffer_names):
            # Bind VAO
            self.buffer_manager.bind_vao(mesh_name)
            
            # Get mesh
            mesh = model.get_meshes()[i]
            
            # Set material properties if available
            material_name = mesh.get_material_name()
            if material_name is not None:
                material = model.get_materials().get(material_name)
                if material is not None:
                    # Set material properties
                    diffuse = material.get_diffuse()
                    self.shader_manager.set_uniform_4f("objectColor", diffuse[0], diffuse[1], diffuse[2], diffuse[3])
                    
                    # Check if material has a diffuse texture
                    diffuse_texture = material.get_diffuse_texture()
                    if diffuse_texture is not None:
                        texture = model.get_textures().get(diffuse_texture)
                        if texture is not None:
                            # Bind texture
                            texture_id = texture.get_gl_texture_id()
                            if texture_id is not None:
                                self.texture_manager.bind_texture(diffuse_texture, 0)
                                self.shader_manager.set_uniform_1i("textureSampler", 0)
                                self.shader_manager.set_uniform_1i("useTexture", 1)
                            else:
                                self.shader_manager.set_uniform_1i("useTexture", 0)
                        else:
                            self.shader_manager.set_uniform_1i("useTexture", 0)
                    else:
                        self.shader_manager.set_uniform_1i("useTexture", 0)
                else:
                    # Use default material properties
                    self.shader_manager.set_uniform_4f("objectColor", 0.8, 0.8, 0.8, 1.0)
                    self.shader_manager.set_uniform_1i("useTexture", 0)
            else:
                # Use default material properties
                self.shader_manager.set_uniform_4f("objectColor", 0.8, 0.8, 0.8, 1.0)
                self.shader_manager.set_uniform_1i("useTexture", 0)
            
            # Draw mesh
            glDrawElements(GL_TRIANGLES, len(mesh.get_indices()), GL_UNSIGNED_INT, None)
            
            # Unbind VAO
            self.buffer_manager.unbind_vao()
    
    def _create_model_buffers(self, model: Union[Model, IModel], model_id: str) -> None:
        """
        Create buffers for a model.
        
        Args:
            model: Model to create buffers for
            model_id: ID of the model
        """
        # Get meshes
        meshes = model.get_meshes()
        
        # Create buffer names for each mesh
        buffer_names = []
        
        # Create buffers for each mesh
        for i, mesh in enumerate(meshes):
            # Create buffer name
            buffer_name = f"{model_id}_mesh_{i}"
            buffer_names.append(buffer_name)
            
            # Get mesh data
            vertices = np.array(mesh.get_vertices(), dtype=np.float32).reshape(-1, 3)
            indices = np.array(mesh.get_indices(), dtype=np.uint32)
            
            # Get normals if available
            normals = None
            if mesh.get_normals():
                normals = np.array(mesh.get_normals(), dtype=np.float32).reshape(-1, 3)
            
            # Get UVs if available
            uvs = None
            if mesh.get_uvs():
                uvs = np.array(mesh.get_uvs(), dtype=np.float32).reshape(-1, 2)
            
            # Create buffers
            self.buffer_manager.create_mesh_buffers(buffer_name, vertices, indices, normals, uvs)
        
        # Store buffer names
        self.model_buffers[model_id] = buffer_names
    
    def set_camera(self, position: Tuple[float, float, float], 
                  target: Tuple[float, float, float], 
                  up: Tuple[float, float, float]) -> None:
        """
        Set the camera position, target, and up vector.
        
        Args:
            position: Camera position as (x, y, z)
            target: Camera target as (x, y, z)
            up: Camera up vector as (x, y, z)
        """
        if self.camera is not None:
            self.camera.set_position(position)
            self.camera.set_target(target)
            self.camera.up = np.array(up, dtype=np.float32)
            self.camera._update_view_matrix()
    
    def set_projection(self, fov: float, aspect: float, near: float, far: float) -> None:
        """
        Set the projection parameters.
        
        Args:
            fov: Field of view in degrees
            aspect: Aspect ratio (width / height)
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        if self.camera is not None:
            self.camera.set_projection(fov, aspect, near, far)
    
    def toggle_wireframe(self) -> None:
        """
        Toggle wireframe rendering mode.
        """
        self.wireframe_mode = not self.wireframe_mode
        logger.log(logger.RENDER, f"Wireframe mode: {self.wireframe_mode}")
    
    def toggle_grid(self) -> None:
        """
        Toggle grid visibility.
        """
        self.show_grid = not self.show_grid
        logger.log(logger.RENDER, f"Grid visibility: {self.show_grid}")
    
    def toggle_axes(self) -> None:
        """
        Toggle axes visibility.
        """
        self.show_axes = not self.show_axes
        logger.log(logger.RENDER, f"Axes visibility: {self.show_axes}")
    
    def save_image(self, image: np.ndarray, filename: str) -> str:
        """
        Save an image to a file.
        
        Args:
            image: Image as a numpy array
            filename: Filename to save the image as
            
        Returns:
            Path to the saved image file
        """
        from PIL import Image
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create output path
        output_path = os.path.join(self.output_dir, filename)
        
        # Save image
        Image.fromarray(image).save(output_path)
        
        logger.log(logger.RENDER, f"Saved image to {output_path}")
        
        return output_path
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the renderer.
        """
        # Clean up managers
        if self.shader_manager is not None:
            self.shader_manager.cleanup()
        
        if self.texture_manager is not None:
            self.texture_manager.cleanup()
        
        if self.buffer_manager is not None:
            self.buffer_manager.cleanup()
        
        if self.framebuffer_manager is not None:
            self.framebuffer_manager.cleanup()
        
        # Reset state
        self.model_buffers = {}
        
        logger.log(logger.RENDER, "Renderer cleaned up")