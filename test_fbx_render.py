#!/usr/bin/env python3
"""
Simple test script to load and render an FBX file.
This isolates the FBX loading and rendering functionality from the rest of the system.
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Try to import PIL for image creation
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Cannot create images.")

# Try to import pyassimp for model loading
try:
    import pyassimp
    import pyassimp.postprocess
    PYASSIMP_AVAILABLE = True
except ImportError:
    PYASSIMP_AVAILABLE = False
    print("pyassimp not available. Cannot load 3D models.")


class Vector3:
    """Simple 3D vector class"""
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return Vector3(self.x / length, self.y / length, self.z / length)
        return Vector3()
    
    def to_tuple(self):
        return (self.x, self.y, self.z)


class Matrix4x4:
    """Simple 4x4 matrix class for transformations"""
    def __init__(self):
        self.data = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
    
    @staticmethod
    def identity():
        return Matrix4x4()
    
    @staticmethod
    def translation(x, y, z):
        m = Matrix4x4()
        m.data[3] = x
        m.data[7] = y
        m.data[11] = z
        return m
    
    @staticmethod
    def rotation_y(angle_degrees):
        angle_radians = angle_degrees * np.pi / 180.0
        cos_val = np.cos(angle_radians)
        sin_val = np.sin(angle_radians)
        
        m = Matrix4x4()
        m.data[0] = cos_val
        m.data[2] = sin_val
        m.data[8] = -sin_val
        m.data[10] = cos_val
        return m
    
    @staticmethod
    def rotation_x(angle_degrees):
        angle_radians = angle_degrees * np.pi / 180.0
        cos_val = np.cos(angle_radians)
        sin_val = np.sin(angle_radians)
        
        m = Matrix4x4()
        m.data[5] = cos_val
        m.data[6] = -sin_val
        m.data[9] = sin_val
        m.data[10] = cos_val
        return m
    
    @staticmethod
    def perspective(fov_degrees, aspect, near, far):
        fov_radians = fov_degrees * np.pi / 180.0
        f = 1.0 / np.tan(fov_radians / 2.0)
        
        m = Matrix4x4()
        m.data[0] = f / aspect
        m.data[5] = f
        m.data[10] = (far + near) / (near - far)
        m.data[11] = (2 * far * near) / (near - far)
        m.data[14] = -1
        m.data[15] = 0
        return m
    
    def multiply(self, other):
        result = Matrix4x4()
        for row in range(4):
            for col in range(4):
                result.data[row * 4 + col] = 0
                for i in range(4):
                    result.data[row * 4 + col] += self.data[row * 4 + i] * other.data[i * 4 + col]
        return result
    
    def transform_vector(self, v):
        x = v.x * self.data[0] + v.y * self.data[1] + v.z * self.data[2] + self.data[3]
        y = v.x * self.data[4] + v.y * self.data[5] + v.z * self.data[6] + self.data[7]
        z = v.x * self.data[8] + v.y * self.data[9] + v.z * self.data[10] + self.data[11]
        w = v.x * self.data[12] + v.y * self.data[13] + v.z * self.data[14] + self.data[15]
        
        if w != 0:
            return Vector3(x / w, y / w, z / w)
        return Vector3(x, y, z)


class Triangle:
    """Triangle class for rendering"""
    def __init__(self, v1, v2, v3, color=(200, 200, 200)):
        self.vertices = [v1, v2, v3]
        self.color = color
        
        # Calculate normal
        edge1 = v2 - v1
        edge2 = v3 - v1
        self.normal = edge1.cross(edge2).normalize()
    
    def transform(self, matrix):
        return Triangle(
            matrix.transform_vector(self.vertices[0]),
            matrix.transform_vector(self.vertices[1]),
            matrix.transform_vector(self.vertices[2]),
            self.color
        )
    
    def calculate_lighting(self, light_dir):
        # Simple diffuse lighting
        light_dir = light_dir.normalize()
        intensity = max(0.1, self.normal.dot(light_dir))  # Ambient + diffuse
        
        r = int(self.color[0] * intensity)
        g = int(self.color[1] * intensity)
        b = int(self.color[2] * intensity)
        
        return (min(255, r), min(255, g), min(255, b))


class Mesh:
    """Mesh class for storing model data"""
    def __init__(self):
        self.triangles = []
    
    def add_triangle(self, v1, v2, v3, color=(200, 200, 200)):
        self.triangles.append(Triangle(v1, v2, v3, color))
    
    def transform(self, matrix):
        result = Mesh()
        for triangle in self.triangles:
            result.triangles.append(triangle.transform(matrix))
        return result


class SoftwareRenderer:
    """Software renderer for 3D models"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.z_buffer = np.full((height, width), float('inf'))
        
        if PIL_AVAILABLE:
            self.image = Image.new("RGB", (width, height), (50, 50, 50))
            self.draw = ImageDraw.Draw(self.image)
        else:
            self.image = None
            self.draw = None
    
    def clear(self):
        """Clear the image and z-buffer"""
        if self.draw:
            self.draw.rectangle([(0, 0), (self.width, self.height)], fill=(50, 50, 50))
        self.z_buffer.fill(float('inf'))
    
    def draw_triangle(self, triangle, light_dir=Vector3(0, 0, -1)):
        """Draw a triangle using the painter's algorithm with z-buffer"""
        if not self.draw:
            return
        
        # Calculate lighting
        color = triangle.calculate_lighting(light_dir)
        
        # Convert 3D coordinates to screen coordinates
        screen_points = []
        for vertex in triangle.vertices:
            # Map from 3D space to screen space
            screen_x = int((vertex.x + 1) * self.width / 2)
            screen_y = int((1 - vertex.y) * self.height / 2)  # Y is flipped in screen space
            screen_points.append((screen_x, screen_y))
            
        # Draw the triangle
        self.draw.polygon(screen_points, fill=color, outline=None)
    
    def render_mesh(self, mesh, camera_pos, light_dir=Vector3(0, 0, -1)):
        """Render a mesh with depth sorting"""
        if not self.draw:
            return
        
        # Sort triangles by average z-coordinate (painter's algorithm)
        sorted_triangles = sorted(
            mesh.triangles,
            key=lambda t: -(t.vertices[0].z + t.vertices[1].z + t.vertices[2].z) / 3
        )
        
        # Draw triangles from back to front
        for triangle in sorted_triangles:
            # Simple backface culling
            view_dir = (camera_pos - triangle.vertices[0]).normalize()
            if triangle.normal.dot(view_dir) > 0:  # Triangle is facing the camera
                self.draw_triangle(triangle, light_dir)
    
    def save_image(self, filename):
        """Save the rendered image to a file"""
        if self.image:
            self.image.save(filename)
            print(f"Rendered image saved to: {filename}")
            return True
        return False


class FbxModel:
    """Class to load and store FBX model data"""
    
    def __init__(self, fbx_path):
        """
        Initialize the FBX model
        
        Args:
            fbx_path (str): Path to the FBX file
        """
        self.fbx_path = fbx_path
        self.vertices = []
        self.faces = []
        self.materials = []
        self.mesh = Mesh()
        self.load_model()
    
    def load_model(self):
        """Load the FBX model"""
        print(f"Loading FBX model from: {self.fbx_path}")
        
        if not PYASSIMP_AVAILABLE:
            print("pyassimp not available. Cannot load model.")
            self._create_fallback_cube()
            return
        
        try:
            # Try to load the model with pyassimp
            print("Using pyassimp to load the model...")
            self._load_with_pyassimp()
            
            if not self.vertices:
                # If pyassimp failed, create a fallback cube
                print("Failed to load model with pyassimp, creating fallback cube")
                self._create_fallback_cube()
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a fallback cube
            self._create_fallback_cube()
    
    def _load_with_pyassimp(self):
        """Load the model using pyassimp"""
        try:
            # Use pyassimp to load the model
            processing_flags = (
                pyassimp.postprocess.aiProcess_Triangulate | 
                pyassimp.postprocess.aiProcess_GenNormals
            )
            
            with pyassimp.load(self.fbx_path, processing=processing_flags) as scene:
                if not scene or not scene.meshes:
                    print("No meshes found in the model")
                    return
                
                print(f"Model loaded successfully with {len(scene.meshes)} meshes")
                
                # Extract mesh data for rendering
                for mesh_idx, mesh in enumerate(scene.meshes):
                    print(f"Processing mesh {mesh_idx} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
                    
                    # Store vertices
                    vertices = []
                    for v in mesh.vertices:
                        vertices.append(Vector3(v[0], v[1], v[2]))
                    
                    # Store faces and create triangles
                    faces = []
                    for face in mesh.faces:
                        if len(face) == 3:  # Only use triangular faces
                            faces.append([face[0], face[1], face[2]])
                            
                            # Add triangle to mesh
                            self.mesh.add_triangle(
                                vertices[face[0]],
                                vertices[face[1]],
                                vertices[face[2]]
                            )
                    
                    # Store material index
                    material_index = mesh.materialindex if hasattr(mesh, 'materialindex') else 0
                    
                    self.vertices.append(vertices)
                    self.faces.append(faces)
                    self.materials.append(material_index)
                
                print(f"Extracted {len(self.vertices)} meshes for rendering")
                
        except Exception as e:
            print(f"Error in pyassimp loading: {e}")
            # Let the caller handle the fallback
            raise
    
    def _create_fallback_cube(self):
        """Create a simple cube as fallback if model loading fails"""
        print("Creating fallback cube model")
        
        # Define the vertices of a cube
        vertices = [
            Vector3(-0.5, -0.5, 0.5),   # 0: front bottom left
            Vector3(0.5, -0.5, 0.5),    # 1: front bottom right
            Vector3(0.5, 0.5, 0.5),     # 2: front top right
            Vector3(-0.5, 0.5, 0.5),    # 3: front top left
            Vector3(-0.5, -0.5, -0.5),  # 4: back bottom left
            Vector3(0.5, -0.5, -0.5),   # 5: back bottom right
            Vector3(0.5, 0.5, -0.5),    # 6: back top right
            Vector3(-0.5, 0.5, -0.5)    # 7: back top left
        ]
        
        # Define the faces of the cube as triangles
        # Front face (red)
        self.mesh.add_triangle(vertices[0], vertices[1], vertices[2], (200, 100, 100))
        self.mesh.add_triangle(vertices[0], vertices[2], vertices[3], (200, 100, 100))
        
        # Right face (green)
        self.mesh.add_triangle(vertices[1], vertices[5], vertices[6], (100, 200, 100))
        self.mesh.add_triangle(vertices[1], vertices[6], vertices[2], (100, 200, 100))
        
        # Back face (blue)
        self.mesh.add_triangle(vertices[5], vertices[4], vertices[7], (100, 100, 200))
        self.mesh.add_triangle(vertices[5], vertices[7], vertices[6], (100, 100, 200))
        
        # Left face (yellow)
        self.mesh.add_triangle(vertices[4], vertices[0], vertices[3], (200, 200, 100))
        self.mesh.add_triangle(vertices[4], vertices[3], vertices[7], (200, 200, 100))
        
        # Top face (cyan)
        self.mesh.add_triangle(vertices[3], vertices[2], vertices[6], (100, 200, 200))
        self.mesh.add_triangle(vertices[3], vertices[6], vertices[7], (100, 200, 200))
        
        # Bottom face (magenta)
        self.mesh.add_triangle(vertices[4], vertices[5], vertices[1], (200, 100, 200))
        self.mesh.add_triangle(vertices[4], vertices[1], vertices[0], (200, 100, 200))
        
        # Store vertices and faces for info display
        self.vertices = [vertices]
        self.faces = [[(0, 1, 2), (0, 2, 3), (1, 5, 6), (1, 6, 2), 
                       (5, 4, 7), (5, 7, 6), (4, 0, 3), (4, 3, 7),
                       (3, 2, 6), (3, 6, 7), (4, 5, 1), (4, 1, 0)]]
        self.materials = [0]
    
    def print_info(self):
        """Print information about the model"""
        print("\n=== Model Information ===")
        print(f"File: {self.fbx_path}")
        print(f"Number of meshes: {len(self.vertices)}")
        
        total_vertices = sum(len(mesh) for mesh in self.vertices)
        total_faces = sum(len(mesh) for mesh in self.faces)
        total_triangles = len(self.mesh.triangles)
        
        print(f"Total vertices: {total_vertices}")
        print(f"Total faces: {total_faces}")
        print(f"Total triangles: {total_triangles}")
        
        for i, (vertices, faces) in enumerate(zip(self.vertices, self.faces)):
            print(f"\nMesh {i+1}:")
            print(f"  Vertices: {len(vertices)}")
            print(f"  Faces: {len(faces)}")
            
            # Calculate bounding box
            if vertices:
                min_x = min(v.x for v in vertices)
                max_x = max(v.x for v in vertices)
                min_y = min(v.y for v in vertices)
                max_y = max(v.y for v in vertices)
                min_z = min(v.z for v in vertices)
                max_z = max(v.z for v in vertices)
                
                print(f"  Bounding box:")
                print(f"    X: {min_x:.2f} to {max_x:.2f}")
                print(f"    Y: {min_y:.2f} to {max_y:.2f}")
                print(f"    Z: {min_z:.2f} to {max_z:.2f}")
        
        print("\n=========================")


def find_fbx_file(fbx_path):
    """
    Find the FBX file by checking various possible paths
    
    Args:
        fbx_path (str): Initial path to check
        
    Returns:
        str: Valid path to the FBX file or None if not found
    """
    # Check if file exists at the given path
    if os.path.exists(fbx_path):
        return fbx_path
    
    print(f"Error: File not found: {fbx_path}")
    print(f"Checking for alternative paths...")
    
    # Try different path variations
    base_name = os.path.basename(fbx_path)
    possible_paths = [
        os.path.join("LocalData", "Models", "Scene1", base_name),
        os.path.join("..", "LocalData", "Models", "Scene1", base_name),
        os.path.join(os.getcwd(), "LocalData", "Models", "Scene1", base_name)
    ]
    
    for path in possible_paths:
        print(f"Trying: {path}")
        if os.path.exists(path):
            print(f"Found file at: {path}")
            return path
    
    return None


def render_model(model, width, height, output_path=None, rotation_y=30, rotation_x=15):
    """Render the model to an image file"""
    if not PIL_AVAILABLE:
        print("PIL not available. Cannot render model.")
        return False
    
    # Set default output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(model.fbx_path))[0]
        output_path = f"{base_name}_{timestamp}.png"
    
    try:
        # Create renderer
        renderer = SoftwareRenderer(width, height)
        
        # Set up camera and transformations
        camera_pos = Vector3(0, 0, 3)
        light_dir = Vector3(0.5, -1, -0.5)
        
        # Create transformation matrices
        projection = Matrix4x4.perspective(45.0, width / height, 0.1, 100.0)
        rotation = Matrix4x4.rotation_y(rotation_y).multiply(Matrix4x4.rotation_x(rotation_x))
        
        # Transform mesh
        transformed_mesh = model.mesh.transform(rotation)
        
        # Render the mesh
        renderer.clear()
        renderer.render_mesh(transformed_mesh, camera_pos, light_dir)
        
        # Save the image
        if renderer.save_image(output_path):
            return True
        return False
    except Exception as e:
        print(f"Error rendering model: {e}")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FBX Renderer')
    parser.add_argument('fbx_path', nargs='?', default="LocalData/Models/Scene1/Ham.fbx",
                        help='Path to the FBX file to render')
    parser.add_argument('--width', type=int, default=800,
                        help='Image width (default: 800)')
    parser.add_argument('--height', type=int, default=600,
                        help='Image height (default: 600)')
    parser.add_argument('--rotation-y', type=float, default=30,
                        help='Y-axis rotation in degrees (default: 30)')
    parser.add_argument('--rotation-x', type=float, default=15,
                        help='X-axis rotation in degrees (default: 15)')
    parser.add_argument('--info-only', action='store_true',
                        help='Display model information without rendering')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for rendered image (default: auto-generated)')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Find the FBX file
    fbx_path = find_fbx_file(args.fbx_path)
    if not fbx_path:
        print(f"Error: Could not find file: {args.fbx_path}")
        return 1
    
    try:
        # Load the model
        model = FbxModel(fbx_path)
        
        # Display model information
        model.print_info()
        
        # Check if we should render the model
        if args.info_only:
            print("Info mode only. Not rendering model.")
            return 0
        
        # Render the model
        print("Rendering model to image file...")
        if render_model(
            model=model,
            width=args.width,
            height=args.height,
            output_path=args.output,
            rotation_y=args.rotation_y,
            rotation_x=args.rotation_x
        ):
            return 0
        else:
            print("Failed to render model. Exiting.")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())