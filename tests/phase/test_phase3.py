import torch
import numpy as np
import trimesh
from PIL import Image
import os

from trellis2.utils.sds_geometry_refine import refine_geometry

def test_phase3():
    print("Testing Phase 3...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a dense mesh
    mesh = trimesh.load("samples/can.glb", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # Simplify down to 16000 to match the actual pipeline's new behavior
    if len(mesh.faces) > 16000:
        mesh = mesh.simplify_quadric_decimation(face_count=16000)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Load real reference image
    ref_image = Image.open("samples/can.png").convert("RGB")
    
    try:
        # Run 100 iterations of SDS
        print("Running 100 iterations of SDS with logging...")
        for test_seed in range(10):
            print(f"\n--- Running SDS with seed {test_seed} ---")
            refined_mesh = refine_geometry(mesh.copy(), ref_image, n_iters=100, device=device, log_every=10, log_dir=f"/tmp/phase3_logs_seed{test_seed}", seed=test_seed)
            
            print(f"Refined mesh: {len(refined_mesh.vertices)} vertices, {len(refined_mesh.faces)} faces")
            
            # Check basic metrics
            diff = np.linalg.norm(refined_mesh.vertices - mesh.vertices, axis=-1).mean()
            print(f"Mean vertex movement: {diff:.6f}")
            assert diff > 0.0, "Vertices did not move!"
            
            # The scale of the mesh should not drift massively under normal regularization
            orig_edge_lens = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=-1)
            ref_edge_lens = np.linalg.norm(refined_mesh.vertices[mesh.edges[:, 0]] - refined_mesh.vertices[mesh.edges[:, 1]], axis=-1)
            ratio = ref_edge_lens.mean() / orig_edge_lens.mean()
            print(f"Mean edge length ratio (refined/orig): {ratio:.4f}")
            
            assert 0.8 < ratio < 1.25, f"Mesh scale drifted too far from input — ratio={ratio:.4f}"

        print("\nAll 10 seeded runs passed!")
        
    except Exception as e:
        print("Failed!")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

if __name__ == "__main__":
    test_phase3()
