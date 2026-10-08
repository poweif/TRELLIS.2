import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# ROCm runtime defaults (flash-attn Triton backend, MIOpen Winograd off) are set by
# trellis2/__init__.py.

# The official DINOv3 and RMBG-2.0 weights are gated on Hugging Face; use ungated substitutes.
os.environ.setdefault("TRELLIS2_UNGATED_MODELS", "1")

import argparse
import cv2
from PIL import Image
import torch
import numpy as np
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.rembg.centered import remove_background
import o_voxel
import trimesh

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run TRELLIS.2 image-to-3D and export GLB")
parser.add_argument("--image", default="./assets/example_image/T.png", help="Input image path")
parser.add_argument("--output", default="sample_output.glb", help="Output GLB path")
parser.add_argument("--texture-size", type=int, default=1024, help="Texture atlas resolution (default 1024)")
parser.add_argument("--decimation", type=int, default=500000, help="Target face count after decimation (default 500000)")
parser.add_argument("--no-glb", action="store_true", help="Skip GLB baking; export raw geometry only")
parser.add_argument("--no-remove-bg", action="store_true",
                    help="Skip background removal (use when input already has transparency)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

print("Loading Pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Image...")
image = Image.open(args.image)

if not args.no_remove_bg:
    print("Removing background...")
    image = remove_background(image, device)

print("Running Pipeline...")
mesh = pipeline.run(image)[0]
print(f"Generated raw mesh with {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces.")

if args.no_glb:
    vertices_np = mesh.vertices.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    try:
        vertex_attrs = mesh.query_attrs(mesh.vertices)
        if 'base_color' in getattr(mesh, 'layout', {}):
            idx = mesh.layout['base_color']
            colors_float = vertex_attrs[:, idx.start:idx.stop].cpu().numpy()
            vertex_colors = (np.clip(colors_float, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            vertex_colors = None
    except Exception as e:
        print("Warning: could not extract vertex colors:", e)
        vertex_colors = None
    out_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np,
                               vertex_colors=vertex_colors, process=False)
    out_mesh.export(args.output)
    print(f"Saved raw geometry to {args.output}")
else:
    print(f"Baking textures → {args.output}  (texture_size={args.texture_size}, decimation={args.decimation})")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation,
        texture_size=args.texture_size,
        remesh=False,
        verbose=True,
    )
    glb.export(args.output)
    print(f"Saved textured GLB to {args.output}")
