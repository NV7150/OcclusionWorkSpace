import os
import numpy as np
import cv2 # For saving images, consider if this is the best place or if it should be a utility
from typing import List, Dict, Any, Optional, Union, Tuple

# Make sure we can import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.IRenderer import IRenderer
from core.IModel import IModel
# from Models.Model import Model # Specific model implementation, IRenderer should use IModel

from Logger.Logger import Logger
from Rendering.ShaderManager import ShaderManager
from Rendering.TextureManager import TextureManager
from Rendering.BufferManager import BufferManager
from Rendering.Camera import Camera
from Rendering.Framebuffer import Framebuffer
from Rendering.Primitives import Primitives

# Conditional import for OpenGL, helps in environments without it for basic logic
try:
    import OpenGL.GL as gl
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    # Mock GL functions if needed for type hinting or basic flow
    class gl:
        @staticmethod
        def glClearColor(*args): pass
        @staticmethod
        def glClear(*args): pass
        @staticmethod
        def glEnable(*args): pass
        @staticmethod
        def glViewport(*args): pass
        GL_COLOR_BUFFER_BIT = 0
        GL_DEPTH_BUFFER_BIT = 0
        GL_DEPTH_TEST = 0


class Renderer(IRenderer):
    """
    Main rendering class using modern OpenGL.
    Orchestrates ShaderManager, TextureManager, BufferManager, Camera, Framebuffer, and Primitives
    to render scenes.
    """

    def __init__(self, width: int = 640, height: int = 480, output_dir: str = "Output/"):
        self._logger = Logger()
        self._logger.info("Initializing Renderer...")

        if not OPENGL_AVAILABLE:
            self._logger.warning("OpenGL not found. Renderer will operate in a limited mode.")

        self._width = width
        self._height = height
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

        self._initialized = False

        # Rendering components
        self._shader_manager = ShaderManager(self._logger) if OPENGL_AVAILABLE else None
        self._texture_manager = TextureManager(self._logger) if OPENGL_AVAILABLE else None
        self._buffer_manager = BufferManager(self._logger) if OPENGL_AVAILABLE else None # Assuming BufferManager also takes logger
        self._camera = Camera() # Camera typically doesn't need GL context for matrix math
        self._primitives = Primitives(self._shader_manager, self._buffer_manager, self._logger) if OPENGL_AVAILABLE else None # Assuming Primitives takes logger

        # Framebuffers (e.g., for offscreen rendering)
        self._main_fbo: Optional[Framebuffer] = None
        self._depth_fbo: Optional[Framebuffer] = None

        # Default clear color
        self._clear_color = (0.1, 0.1, 0.1, 1.0)

    def initialize(self) -> bool:
        """
        Initialize the OpenGL context and resources.
        """
        if not OPENGL_AVAILABLE:
            self._logger.error("Cannot initialize Renderer: OpenGL is not available.")
            return False

        if self._initialized:
            self._logger.info("Renderer already initialized.")
            return True

        self._logger.info(f"Initializing Renderer with viewport {self._width}x{self._height}")
        try:
            # Basic OpenGL setup
            gl.glViewport(0, 0, self._width, self._height)
            gl.glClearColor(self._clear_color[0], self._clear_color[1], self._clear_color[2], self._clear_color[3])
            gl.glEnable(gl.GL_DEPTH_TEST)
            # TODO: Add other GL setup like blending, culling, etc.

            # Initialize managers (e.g., load default shaders)
            # self._shader_manager.load_default_shaders() # Example

            # Setup main FBO for rendering if needed, or render to default window buffer
            # self._main_fbo = Framebuffer(self._width, self._height)
            # self._main_fbo.create()

            # Setup FBO for depth-only rendering
            # self._depth_fbo = Framebuffer(self._width, self._height, depth_only=True)
            # self._depth_fbo.create()


            self._initialized = True
            self._logger.info("Renderer initialized successfully.")
            return True
        except Exception as e:
            self._logger.error(f"Error initializing Renderer: {e}")
            self._initialized = False
            return False

    def set_camera(self, camera: Camera) -> None:
        """
        Set the active camera for rendering.
        Args:
            camera: The Camera object to use.
        """
        if not isinstance(camera, Camera):
            self._logger.error("Invalid camera object provided to set_camera.")
            return
        self._camera = camera
        # The renderer will use self._camera.get_view_matrix() and self._camera.get_projection_matrix()

    def set_camera_matrices(self, view_matrix: np.ndarray, projection_matrix: np.ndarray) -> None:
        """
        Set camera view and projection matrices directly.
        This is an alternative to providing a full Camera object.
        """
        self._camera.set_view_matrix_directly(view_matrix) # Assuming Camera has such a method
        self._camera.set_projection_matrix_directly(projection_matrix) # Assuming Camera has such a method


    def render_frame(self,
                     models: List[IModel],
                     rgb_image: Optional[np.ndarray] = None,
                     depth_image: Optional[np.ndarray] = None,
                     occlusion_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Render the models, optionally compositing with an RGB image and applying an occlusion mask.
        """
        if not self._initialized or not OPENGL_AVAILABLE:
            self._logger.error("Renderer not initialized or OpenGL not available. Cannot render frame.")
            return rgb_image # Return original image if provided, else None

        try:
            # 1. Bind appropriate FBO (e.g., self._main_fbo or default window buffer)
            # self._main_fbo.bind() if self._main_fbo else gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # 2. Handle background (e.g., draw rgb_image as a textured quad)
            if rgb_image is not None:
                # TODO: Implement drawing of rgb_image as background
                # This might involve a specific shader and a full-screen quad
                pass

            # 3. Get camera matrices
            view_matrix = self._camera.get_view_matrix()
            projection_matrix = self._camera.get_projection_matrix()

            # 4. Render models
            for model_instance in models:
                if not isinstance(model_instance, IModel):
                    self._logger.warning("Skipping non-IModel object in render list.")
                    continue

                # TODO: High-level steps for rendering a model:
                # model_transform = model_instance.get_transform()
                # meshes = model_instance.get_meshes() # Assuming IModel has get_meshes()
                # for mesh in meshes:
                #     material = mesh.get_material() # Assuming Mesh has get_material()
                #     shader_name = material.get_shader_name() # Or determine shader based on material properties

                #     self._shader_manager.use_program(shader_name)
                #     active_shader_id = self._shader_manager.get_active_program_id()

                #     # Set camera uniforms
                #     self._shader_manager.set_uniform_matrix_4fv("viewMatrix", view_matrix)
                #     self._shader_manager.set_uniform_matrix_4fv("projectionMatrix", projection_matrix)

                #     # Set model transform uniform
                #     self._shader_manager.set_uniform_matrix_4fv("modelMatrix", model_transform)

                #     # Set material uniforms (colors, textures)
                #     # if material.has_diffuse_texture():
                #     #    texture_id = self._texture_manager.get_texture_id(material.get_diffuse_texture_name())
                #     #    gl.glActiveTexture(gl.GL_TEXTURE0)
                #     #    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
                #     #    self._shader_manager.set_uniform_1i("diffuseTexture", 0)

                #     # Bind VAO for the mesh (managed by BufferManager)
                #     # vao_id = self._buffer_manager.get_vao_for_mesh(mesh.get_id())
                #     # gl.glBindVertexArray(vao_id)

                #     # Draw call
                #     # gl.glDrawElements(...) or gl.glDrawArrays(...)

                #     # Unbind VAO
                #     # gl.glBindVertexArray(0)
                pass # Placeholder for model rendering loop

            # 5. Apply occlusion mask (if provided and applicable, might involve post-processing or stencil buffer)
            if occlusion_mask is not None:
                # TODO: Implement occlusion mask application
                pass

            # 6. Read back the rendered frame if rendering to an FBO
            rendered_image = None
            # if self._main_fbo:
            #     rendered_image = self._main_fbo.read_pixels() # Assuming FBO has read_pixels
            # else: # Reading from default window buffer (more complex)
            #     data = gl.glReadPixels(0, 0, self._width, self._height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            #     rendered_image = np.frombuffer(data, dtype=np.uint8).reshape(self._height, self._width, 4)
            #     rendered_image = cv2.flip(rendered_image, 0) # OpenGL origin is bottom-left

            # For now, if rgb_image was given, return a modified version or itself
            # This part needs to be refined based on actual rendering output
            if rendered_image is not None:
                return rendered_image
            elif rgb_image is not None:
                # Placeholder: if we didn't actually render to rgb_image, just return it
                # In a real scenario, models would be rendered onto/into the rgb_image
                return rgb_image
            else:
                # If no rgb_image and no FBO readback, create a dummy image
                return np.zeros((self._height, self._width, 3), dtype=np.uint8)


        except Exception as e:
            self._logger.error(f"Error rendering frame: {e}")
            # Fallback: return original rgb_image if available
            return rgb_image if rgb_image is not None else None


    def render_depth_only(self, models: List[IModel]) -> Optional[np.ndarray]:
        """
        Render only the depth of the models.
        """
        if not self._initialized or not OPENGL_AVAILABLE:
            self._logger.error("Renderer not initialized or OpenGL not available. Cannot render depth.")
            return None

        try:
            # 1. Bind depth FBO
            # if not self._depth_fbo:
            #     self._logger.error("Depth FBO not available for depth-only rendering.")
            #     return None
            # self._depth_fbo.bind()
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT) # Only clear depth

            # 2. Get camera matrices
            view_matrix = self._camera.get_view_matrix()
            projection_matrix = self._camera.get_projection_matrix()

            # 3. Render models with a depth-specific shader
            # depth_shader_name = "depth_shader" # Assume such a shader exists
            # self._shader_manager.use_program(depth_shader_name)

            for model_instance in models:
                if not isinstance(model_instance, IModel):
                    continue
                # TODO: Similar to render_frame but using depth shader
                # model_transform = model_instance.get_transform()
                # self._shader_manager.set_uniform_matrix_4fv("viewMatrix", view_matrix)
                # self._shader_manager.set_uniform_matrix_4fv("projectionMatrix", projection_matrix)
                # self._shader_manager.set_uniform_matrix_4fv("modelMatrix", model_transform)
                # ... bind VAO and draw ...
                pass

            # 4. Read back depth buffer from FBO
            # depth_buffer_data = self._depth_fbo.read_depth_data() # Assuming FBO can return depth
            # return depth_buffer_data
            return np.ones((self._height, self._width), dtype=np.float32) # Placeholder

        except Exception as e:
            self._logger.error(f"Error in render_depth_only: {e}")
            return None

    def save_frame_to_image(self, frame: np.ndarray, filename: str) -> bool:
        """
        Save a rendered frame (numpy array) to an image file.
        """
        if frame is None:
            self._logger.error("Cannot save frame: frame data is None.")
            return False
        
        output_path = os.path.join(self._output_dir, filename)
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # OpenCV expects BGR by default for imwrite, ensure frame is in correct format
            # If frame is RGBA, may need to convert to BGR or save as PNG to keep alpha
            if frame.shape[2] == 4: # RGBA
                img_to_save = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            elif frame.shape[2] == 3: # RGB
                img_to_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                img_to_save = frame

            cv2.imwrite(output_path, img_to_save)
            self._logger.debug(f"Frame saved to {output_path}")
            return True
        except Exception as e:
            self._logger.error(f"Error saving frame to {output_path}: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up OpenGL resources."""
        self._logger.info("Cleaning up Renderer...")
        if not OPENGL_AVAILABLE:
            self._logger.info("No OpenGL resources to clean up (OpenGL not available).")
            return

        if self._shader_manager:
            self._shader_manager.cleanup()
        if self._texture_manager:
            self._texture_manager.cleanup()
        if self._buffer_manager:
            self._buffer_manager.cleanup()
        if self._primitives:
            self._primitives.cleanup()

        if self._main_fbo:
            self._main_fbo.delete()
        if self._depth_fbo:
            self._depth_fbo.delete()

        # TODO: Any other OpenGL specific cleanup
        self._initialized = False
        self._logger.info("Renderer cleaned up.")

    # --- Helper methods ---
    def set_clear_color(self, r: float, g: float, b: float, a: float = 1.0):
        self._clear_color = (r, g, b, a)
        if self._initialized and OPENGL_AVAILABLE:
            gl.glClearColor(r, g, b, a)

    def get_dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    def resize_viewport(self, width: int, height: int):
        self._width = width
        self._height = height
        if self._initialized and OPENGL_AVAILABLE:
            gl.glViewport(0, 0, self._width, self._height)
            # May need to resize FBOs as well
            if self._main_fbo:
                self._main_fbo.resize(width, height)
            if self._depth_fbo:
                self._depth_fbo.resize(width, height)
        # Camera aspect ratio might also need update
        self._camera.set_aspect_ratio(width / height) # Assuming Camera has set_aspect_ratio
        self._logger.info(f"Renderer viewport resized to {width}x{height}")

# Example usage (conceptual)
if __name__ == '__main__':
    if not OPENGL_AVAILABLE:
        print("OpenGL not available, cannot run Renderer example.")
    else:
        # This example would require a valid OpenGL context (e.g., from GLFW, Pyglet)
        # For simplicity, we're just showing class instantiation and method calls.
        print("Conceptual Renderer Usage (requires OpenGL context):")
        
        # Initialize a windowing library and create an OpenGL context first.
        # For example, using Pyglet:
        # import pyglet
        # window = pyglet.window.Window(width=800, height=600, caption="Renderer Test")
        
        renderer = Renderer(width=800, height=600)
        
        # @window.event
        # def on_draw():
        #     if renderer._initialized:
        #         # renderer.render_frame(models=[...])
        #         pass # Actual rendering call
        
        if renderer.initialize():
            print("Renderer initialized.")
            
            # Setup camera
            cam = Camera()
            cam.set_position(np.array([0, 0, 5]))
            cam.look_at(np.array([0,0,0])) # Assuming Camera has look_at
            renderer.set_camera(cam)
            
            # Create some dummy models (IModel interface)
            class DummyModel(IModel):
                def get_transform(self) -> np.ndarray: return np.eye(4)
                def get_meshes(self) -> List[Any]: return [] # Replace Any with IMesh if defined
                def get_name(self) -> str: return "Dummy"

            models_to_render = [DummyModel()]
            
            # Simulate a render loop iteration
            # In a real app, this would be inside the window's draw event
            # output_frame = renderer.render_frame(models=models_to_render)
            # if output_frame is not None:
            #     renderer.save_frame_to_image(output_frame, "test_render.png")

            renderer.cleanup()
            print("Renderer cleaned up.")
        else:
            print("Renderer failed to initialize.")
        
        # if 'window' in locals():
        #    pyglet.app.run()

