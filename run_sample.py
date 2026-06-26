import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory, but buggy on ROCm
os.environ["ATTN_BACKEND"] = "sdpa"
import cv2
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import trimesh
import numpy as np

print("Loading Pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

print("Loading Image...")
image = Image.open("assets/example_image/T.png")

print("Running Pipeline...")
# This will run the sparse structure generation, shape, and texturing stages
mesh = pipeline.run(image)[0]

print(f"Generated raw mesh with {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces.")

# Instead of using nvdiffrast for simplification and baking (to_glb), we just export the raw point cloud / mesh
vertices_np = mesh.vertices.cpu().numpy()
faces_np = mesh.faces.cpu().numpy()

# Flip Y and Z axis for standard 3D viewer orientation
vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()

# Extract vertex colors if available
try:
    vertex_attrs = mesh.query_attrs(mesh.vertices)
    if 'base_color' in getattr(mesh, 'layout', {}):
        idx = mesh.layout['base_color']
        # The albedo values might need normalization/clamping (0-1 -> 0-255)
        colors_float = vertex_attrs[:, idx.start:idx.stop].cpu().numpy()
        colors_float = np.clip(colors_float, 0.0, 1.0)
        vertex_colors = (colors_float * 255).astype(np.uint8)
        # Flip Y and Z axis for standard 3D viewer orientation (same as vertices)
        # Actually colors don't need flipping, only geometry does
    else:
        vertex_colors = None
except Exception as e:
    print("Warning: Could not extract vertex colors:", e)
    vertex_colors = None

print("Saving raw geometry and textures to sample_output.glb...")
out_mesh = trimesh.Trimesh(
    vertices=vertices_np, 
    faces=faces_np, 
    vertex_colors=vertex_colors,
    process=False
)
out_mesh.export("sample_output.glb")

print("Sample run complete!")
