import torch
import trimesh
import numpy as np
from trellis2.utils.meshanything_bridge import run_inference

mesh = trimesh.creation.icosphere()
points, face_indices = trimesh.sample.sample_surface(mesh, 8192)
normals = mesh.face_normals[face_indices]
bounds = np.array([points.min(axis=0), points.max(axis=0)])
points = points - (bounds[0] + bounds[1])[None, :] / 2
points = points / np.abs(points).max() * 0.9995
pc_normal = np.concatenate([points, normals], axis=-1).astype(np.float16)
device = "cuda"
output = run_inference(pc_normal, device)
print("Output bbox:", output.min(axis=0), output.max(axis=0))
