import sys
import os
import trimesh
import numpy as np
import torch
import traceback

from trellis2.utils.meshanything_bridge import run_inference

def test_phase_1():
    print("Testing Phase 1...")
    # Create icosphere
    mesh = trimesh.creation.icosphere()
    
    # Sample 8192 points and normals
    points, face_indices = trimesh.sample.sample_surface(mesh, 8192)
    normals = mesh.face_normals[face_indices]
    
    # Format per Phase 4 / Phase 0 notes:
    # "center on bbox midpoint, scale to [-0.9995, 0.9995]³, cast to float16"
    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    points = points - (bounds[0] + bounds[1])[None, :] / 2
    points = points / np.abs(points).max() * 0.9995
    
    pc_normal = np.concatenate([points, normals], axis=-1).astype(np.float16)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        output = run_inference(pc_normal, device)
        print(f"Success! Output shape: {output.shape}")
        
        # Verify shape is (N, 9)
        assert output.ndim == 2
        assert output.shape[1] == 9
        print("Test passed.")
    except Exception as e:
        print("Failed!")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_phase_1()
