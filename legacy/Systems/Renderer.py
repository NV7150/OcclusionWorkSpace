import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
import pyrr
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pywavefront
from pywavefront import visualization
import pyassimp
import pyassimp.postprocess
from Interfaces.Frame import Frame
from Logger import logger, Logger


class Model3D:
    """
    Base class for 3D models.
    """
    def render(self):
        """
        Render the model using OpenGL.
        """
        pass


class ObjModel(Model3D):
    """
    Class for loading and rendering 3D OBJ models using PyWavefront.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the OBJ model from a file.
        
        Args:
            file_path: Path to the OBJ file
        """
        self.file_path = file_path
        self.model = pywavefront.Wavefront(
            file_path,
            create_materials=True,
            collect_faces=True
        )
        
    def render(self):
        """
        Render the OBJ model using PyWavefront's visualization module.
        """
        visualization.draw(self.model)


class FbxModel(Model3D):
    """
    Class for loading and rendering 3D FBX models using PyAssimp.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the FBX model from a file.
        
        Args:
            file_path: Path to the FBX file
        """
        self.file_path = file_path
        logger.log(Logger.RENDER, f"Loading FBX model from {file_path}")
        
        # Store vertices, faces, and materials for rendering
        self.vertices = []
        self.faces = []
        self.materials = []
        
        try:
            # Use pyassimp to load the model
            with pyassimp.load(file_path, processing=pyassimp.postprocess.aiProcess_Triangulate) as scene:
                if not scene or not scene.meshes:
                    raise ValueError(f"No meshes found in {file_path}")
                
                # Extract mesh data for rendering
                for mesh in scene.meshes:
                    # Store vertices
                    vertices = []
                    for v in mesh.vertices:
                        vertices.append([v[0], v[1], v[2]])
                    
                    # Store faces
                    faces = []
                    for face in mesh.faces:
                        if len(face) == 3:  # Only use triangular faces
                            faces.append([face[0], face[1], face[2]])
                    
                    # Store material index
                    material_index = mesh.materialindex if hasattr(mesh, 'materialindex') else 0
                    
                    self.vertices.append(vertices)
                    self.faces.append(faces)
                    self.materials.append(material_index)
                
                logger.log(Logger.RENDER, f"Loaded model with {len(self.vertices)} meshes")
        except Exception as e:
            logger.log(Logger.ERROR, f"Error loading FBX model: {e}")
            # Create a simple cube as fallback
            self._create_fallback_cube()
    
    def _create_fallback_cube(self):
        """Create a simple cube as fallback if model loading fails"""
        logger.log(Logger.WARNING, "Creating fallback cube model")
        # Define the vertices of a cube
        vertices = [
            # Front face
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            # Back face
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]
        ]
        
        # Define the faces of the cube as triangles
        faces = [
            # Front face
            [0, 1, 2], [0, 2, 3],
            # Right face
            [1, 5, 6], [1, 6, 2],
            # Back face
            [5, 4, 7], [5, 7, 6],
            # Left face
            [4, 0, 3], [4, 3, 7],
            # Top face
            [3, 2, 6], [3, 6, 7],
            # Bottom face
            [4, 5, 1], [4, 1, 0]
        ]
        
        self.vertices = [vertices]
        self.faces = [faces]
        self.materials = [0]
        
    def render(self):
        """
        Render the FBX model using OpenGL.
        """
        # Set default material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)
        
        # Render each mesh
        for mesh_idx in range(len(self.vertices)):
            vertices = self.vertices[mesh_idx]
            faces = self.faces[mesh_idx]
            
            # Render mesh triangles
            glBegin(GL_TRIANGLES)
            for face in faces:
                # Calculate face normal for lighting
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                
                # Calculate normal vector using cross product
                ux = v1[0] - v0[0]
                uy = v1[1] - v0[1]
                uz = v1[2] - v0[2]
                
                vx = v2[0] - v0[0]
                vy = v2[1] - v0[1]
                vz = v2[2] - v0[2]
                
                nx = uy * vz - uz * vy
                ny = uz * vx - ux * vz
                nz = ux * vy - uy * vx
                
                # Normalize the normal vector
                length = (nx * nx + ny * ny + nz * nz) ** 0.5
                if length > 0:
                    nx /= length
                    ny /= length
                    nz /= length
                
                # Set normal for all vertices of this face
                glNormal3f(nx, ny, nz)
                
                # Render vertices
                for vertex_idx in face:
                    vertex = vertices[vertex_idx]
                    glVertex3f(vertex[0], vertex[1], vertex[2])
            glEnd()


class Renderer:
    """
    Renderer is responsible for rendering mixed reality scenes based on
    occlusion masks and scene descriptions.
    """
    
    def __init__(self, output_dir: str, tracker=None):
        """
        Initialize the Renderer with an output directory and optional tracker.
        
        Args:
            output_dir: Directory path where rendered images will be saved
            tracker: Optional Tracker object for camera pose estimation
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models_cache = {}  # Cache for loaded models
        self.tracker = tracker  # Store the tracker
        self.initialize_opengl()
        
    def initialize_opengl(self):
        """
        Initialize OpenGL for rendering.
        """
        # Initialize GLUT
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        
        # Create a window
        glutCreateWindow("MR Renderer")
        
        # Set up OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Set up light
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
    def load_model(self, model_path: str) -> Model3D:
        """
        Load a 3D model from a file, using cache if available.
        Automatically detects the file type from the file extension.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
        """
        if model_path in self.models_cache:
            return self.models_cache[model_path]
        
        # Determine file type from extension
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext == '.obj':
            # Load OBJ file using PyWavefront
            model = ObjModel(model_path)
        elif file_ext == '.fbx':
            # Load FBX file using PyAssimp
            model = FbxModel(model_path)
        else:
            raise ValueError(f"Unsupported model file format: {file_ext}")
            
        self.models_cache[model_path] = model
        return model
        
    def setup_camera(self, width: int, height: int, fov: float = 60.0):
        """
        Set up the camera for rendering.
        
        Args:
            width: Width of the viewport
            height: Height of the viewport
            fov: Field of view in degrees
        """
        # Set up viewport
        glViewport(0, 0, width, height)
        
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height
        gluPerspective(fov, aspect, 0.1, 100.0)
        
        # Set up modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
    def apply_transform(self, position: Dict[str, float], rotation: Dict[str, float], scale: Optional[Dict[str, float]] = None):
        """
        Apply a transform (position, rotation, and optional scale) to the current OpenGL matrix.
        
        Args:
            position: Dictionary with x, y, z position values
            rotation: Dictionary with x, y, z, w quaternion rotation values or x, y, z Euler angles
            scale: Optional dictionary with x, y, z scale values
        """
        # Apply scale first if provided
        if scale:
            glScalef(scale.get('x', 1.0), scale.get('y', 1.0), scale.get('z', 1.0))
            logger.log(Logger.DEBUG, f"Applied scale: ({scale.get('x', 1.0)}, {scale.get('y', 1.0)}, {scale.get('z', 1.0)})")
        
        # Check if rotation is specified as quaternion or Euler angles
        if 'w' in rotation:
            # It's a quaternion
            quat = pyrr.Quaternion([rotation['x'], rotation['y'], rotation['z'], rotation['w']])
        else:
            # It's Euler angles (in degrees) - convert to quaternion
            import math
            
            # Extract Euler angles in degrees
            euler_x = math.radians(rotation.get('x', 0))
            euler_y = math.radians(rotation.get('y', 0))
            euler_z = math.radians(rotation.get('z', 0))
            
            # Calculate quaternion components
            # This uses the ZYX rotation order (roll, pitch, yaw)
            cy = math.cos(euler_z * 0.5)
            sy = math.sin(euler_z * 0.5)
            cp = math.cos(euler_y * 0.5)
            sp = math.sin(euler_y * 0.5)
            cr = math.cos(euler_x * 0.5)
            sr = math.sin(euler_x * 0.5)
            
            # Create quaternion
            quat = pyrr.Quaternion([
                sr * cp * cy - cr * sp * sy,  # x
                cr * sp * cy + sr * cp * sy,  # y
                cr * cp * sy - sr * sp * cy,  # z
                cr * cp * cy + sr * sp * sy   # w
            ])
            
            logger.log(Logger.DEBUG, f"Converted Euler angles ({rotation.get('x', 0)}, {rotation.get('y', 0)}, {rotation.get('z', 0)}) to quaternion: {quat}")
        
        # Apply rotation
        rotation_matrix = quat.matrix33
        glMultMatrixf(pyrr.matrix44.create_from_matrix33(rotation_matrix))
        
        # Apply translation last
        glTranslatef(position['x'], position['y'], position['z'])
        
    # def apply_transform(self, position: Dict[str, float], rotation: Dict[str, float], scale: Optional[Dict[str, float]] = None):
    #     """
    #     Apply a transform (position, rotation, and optional scale) to the current OpenGL matrix.
        
    #     Args:
    #         position: Dictionary with x, y, z position values
    #         rotation: Dictionary with x, y, z, w quaternion rotation values or x, y, z Euler angles
    #         scale: Optional dictionary with x, y, z scale values
    #     """
    #     # Apply translation
    #     glTranslatef(position['x'], position['y'], position['z'])
        
    #     # Check if rotation is specified as quaternion or Euler angles
    #     if 'w' in rotation:
    #         # It's a quaternion
    #         quat = pyrr.Quaternion([rotation['x'], rotation['y'], rotation['z'], rotation['w']])
    #     else:
    #         # It's Euler angles (in degrees) - convert to quaternion
    #         import math
            
    #         # Extract Euler angles in degrees
    #         euler_x = math.radians(rotation.get('x', 0))
    #         euler_y = math.radians(rotation.get('y', 0))
    #         euler_z = math.radians(rotation.get('z', 0))
            
    #         # Calculate quaternion components
    #         # This uses the ZYX rotation order (roll, pitch, yaw)
    #         cy = math.cos(euler_z * 0.5)
    #         sy = math.sin(euler_z * 0.5)
    #         cp = math.cos(euler_y * 0.5)
    #         sp = math.sin(euler_y * 0.5)
    #         cr = math.cos(euler_x * 0.5)
    #         sr = math.sin(euler_x * 0.5)
            
    #         # Create quaternion
    #         quat = pyrr.Quaternion([
    #             sr * cp * cy - cr * sp * sy,  # x
    #             cr * sp * cy + sr * cp * sy,  # y
    #             cr * cp * sy - sr * sp * cy,  # z
    #             cr * cp * cy + sr * sp * sy   # w
    #         ])
            
    #         logger.log(Logger.DEBUG, f"Converted Euler angles ({rotation.get('x', 0)}, {rotation.get('y', 0)}, {rotation.get('z', 0)}) to quaternion: {quat}")
        
    #     # Apply rotation
    #     rotation_matrix = quat.matrix33
    #     glMultMatrixf(pyrr.matrix44.create_from_matrix33(rotation_matrix))
        
    #     # Apply scale if provided
    #     if scale:
    #         glScalef(scale.get('x', 1.0), scale.get('y', 1.0), scale.get('z', 1.0))
    #         logger.log(Logger.DEBUG, f"Applied scale: ({scale.get('x', 1.0)}, {scale.get('y', 1.0)}, {scale.get('z', 1.0)})")
        
    def render_frame(self, frame: Frame, occlusion_mask: np.ndarray,
                     models: Dict[str, Any], scene_data: Dict) -> np.ndarray:
        """
        Render a mixed reality frame based on the occlusion mask and scene description.
        
        Args:
            frame: Frame object containing RGB and depth images
            occlusion_mask: Binary mask indicating occluded areas
            models: Dictionary of 3D models with keys matching scene_data keys
                    Each entry should have a 'file_path' key with the path to the model file
            scene_data: Scene description dictionary with position and rotation for each model
            
        Returns:
            Rendered image as a numpy array
        """
        # Set up framebuffer for off-screen rendering
        width, height = frame.width, frame.height
        
        # Create a framebuffer object
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        # Create a texture to render to
        render_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, render_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Attach texture to framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_texture, 0)
        
        # Create a renderbuffer for depth
        depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb)
        
        # Check framebuffer status
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer not complete")
        
        # Set up camera
        self.setup_camera(width, height)
        
        # Clear buffers
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up camera position using tracker if available
        glLoadIdentity()
        
        if self.tracker:
            # Get camera pose from tracker
            camera_pose = self.tracker.track(frame)
            
            # Apply camera pose matrix directly
            # OpenGL uses column-major order, so we need to transpose the matrix
            camera_pose_gl = camera_pose.T.flatten()
            glLoadMatrixf(camera_pose_gl)
        else:
            # Fallback to default camera position if no tracker is available
            gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
        
        # Print debug info about models and scene data
        logger.log(Logger.DEBUG, f"Scene data contains {len(scene_data)} objects")
        logger.log(Logger.DEBUG, f"Models dictionary contains {len(models)} entries")
        
        # Render each object in the scene
        for obj_id, obj_data in scene_data.items():
            logger.log(Logger.DEBUG, f"Processing object: {obj_id}")
            
            if obj_id in models:
                model_data = models[obj_id]
                
                if 'file_path' not in model_data:
                    logger.log(Logger.WARNING, f"No file_path specified for model {obj_id}")
                    continue
                    
                model_path = model_data['file_path']
                logger.log(Logger.DEBUG, f"Loading model from: {model_path}")
                
                # Check if file exists
                if not os.path.exists(model_path):
                    logger.log(Logger.WARNING, f"Model file does not exist: {model_path}")
                    
                    # Try different path variations
                    possible_paths = [
                        # Try absolute path from project root
                        os.path.join(os.getcwd(), "LocalData", "Models", "Scene1", os.path.basename(model_path)),
                        # Try relative path from current directory
                        os.path.join("LocalData", "Models", "Scene1", os.path.basename(model_path)),
                        # Try just the filename in Scene1 directory
                        os.path.join("LocalData", "Models", "Scene1", os.path.basename(model_path).split('/')[-1]),
                        # Try with parent directory
                        os.path.join("..", "LocalData", "Models", "Scene1", os.path.basename(model_path))
                    ]
                    
                    found = False
                    for alt_path in possible_paths:
                        logger.log(Logger.DEBUG, f"Trying alternative path: {alt_path}")
                        if os.path.exists(alt_path):
                            logger.log(Logger.DEBUG, f"Found model at alternative path: {alt_path}")
                            model_path = alt_path
                            found = True
                            break
                    
                    if not found:
                        logger.log(Logger.ERROR, f"Could not find model file: {os.path.basename(model_path)}")
                        continue
                
                try:
                    # Load model
                    model = self.load_model(model_path)
                    
                    # Push matrix for this object
                    glPushMatrix()
                    
                    # Apply object transform
                    if 'position' in obj_data and 'rotation' in obj_data:
                        # Check if scale is provided
                        scale = obj_data.get('scale')
                        self.apply_transform(obj_data['position'], obj_data['rotation'], scale)
                    else:
                        logger.log(Logger.WARNING, f"Missing position or rotation data for {obj_id}")
                    
                    # Render model
                    model.render()
                    
                    # Pop matrix
                    glPopMatrix()
                except Exception as e:
                    logger.log(Logger.ERROR, f"Error rendering model {obj_id}: {e}")
            else:
                logger.log(Logger.WARNING, f"Object {obj_id} specified in scene data but not found in models dictionary")
        
        # Read pixels from framebuffer
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        rendered_data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        rendered_image = np.frombuffer(rendered_data, dtype=np.uint8).reshape(height, width, 4)
        
        # Flip image vertically (OpenGL has origin at bottom-left)
        rendered_image = np.flipud(rendered_image)
        
        # Clean up
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [render_texture])
        glDeleteRenderbuffers(1, [depth_rb])
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Apply occlusion mask to the rendered image
        # Convert occlusion mask to alpha channel (0 = fully occluded, 255 = not occluded)
        alpha_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Log dimensions for debugging
        logger.log(Logger.DEBUG, f"Frame dimensions: {frame.rgb.shape}")
        logger.log(Logger.DEBUG, f"Occlusion mask dimensions: {occlusion_mask.shape}")
        
        # Resize occlusion mask to match frame dimensions if needed
        if occlusion_mask.shape[:2] != (height, width):
            logger.log(Logger.DEBUG, f"Resizing occlusion mask from {occlusion_mask.shape[:2]} to {(height, width)}")
            # Use a simple resize approach for testing
            from PIL import Image
            if occlusion_mask.ndim == 2:
                # 2D mask
                resized_mask = np.array(Image.fromarray(occlusion_mask).resize((width, height), Image.NEAREST))
            else:
                # 3D mask
                resized_mask = np.array(Image.fromarray(occlusion_mask[:,:,0] if occlusion_mask.shape[2] >= 1 else occlusion_mask).resize((width, height), Image.NEAREST))
                if occlusion_mask.ndim == 3 and occlusion_mask.shape[2] > 1:
                    # Convert to 2D if it was 3D
                    resized_mask = resized_mask[:,:,0] if resized_mask.ndim == 3 else resized_mask
        else:
            resized_mask = occlusion_mask
            if resized_mask.ndim == 3:
                # Convert to 2D if it's 3D
                resized_mask = resized_mask[:,:,0] if resized_mask.shape[2] >= 1 else np.mean(resized_mask, axis=2)
        
        # Now use the resized mask
        alpha_mask[resized_mask == 0] = 255  # Areas not occluded
        
        # Apply alpha mask to rendered image
        # Create a copy of the rendered image to make it writable
        rendered_image = rendered_image.copy()
        rendered_image[:, :, 3] = np.minimum(rendered_image[:, :, 3], alpha_mask)
        
        # Composite rendered image over original RGB image
        rgb_image = frame.rgb.copy()
        
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
    
    def save_rendered_image(self, image: np.ndarray, timestamp: np.datetime64, file_prefix: str) -> str:
        """
        Save a rendered image to the output directory.
        
        Args:
            image: Rendered image as a numpy array
            timestamp: Timestamp of the frame
            
        Returns:
            Path to the saved image file
        """
        # Convert timestamp to string for filename
        timestamp_str = str(timestamp).replace(':', '-').replace(' ', '_')
        
        # Create output filename
        output_filename = f"{file_prefix}_{timestamp_str}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Save image
        Image.fromarray(image).save(output_path)
        logger.log(Logger.RENDER, f"Saved rendered image to {output_path}")
        
        return output_path
    
    def render_and_save_batch(self, frames: List[Frame], 
                             occlusion_masks: Dict[np.datetime64, np.ndarray],
                             models: Dict[str, Any], 
                             scene_data: Dict,
                             output_prefix: str
                             ) -> List[str]:
        """
        Render and save a batch of frames.
        
        Args:
            frames: List of Frame objects
            occlusion_masks: Dictionary mapping timestamps to occlusion masks
            models: Dictionary of 3D models
            scene_data: Scene description dictionary
            
        Returns:
            List of paths to the saved image files
        """
        output_paths = []
        
        for frame in frames:
            # Get occlusion mask for this frame
            occlusion_mask = occlusion_masks.get(frame.timestamp)
            if occlusion_mask is None:
                logger.log(Logger.WARNING, f"No occlusion mask found for timestamp {frame.timestamp}")
                continue
                
            # Render frame
            rendered_image = self.render_frame(frame, occlusion_mask, models, scene_data)
            
            # Save rendered image
            output_path = self.save_rendered_image(rendered_image, frame.timestamp, output_prefix)
            output_paths.append(output_path)
            
        return output_paths