import torch
import trimesh
import numpy as np

from trellis2.utils.meshanything_bridge import mesh_to_conditioning_pc

def test_phase4():
    print("Testing Phase 4...")
    # Create an asymmetric mesh
    mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    mesh.apply_translation([10.0, -5.0, 4.0])
    
    print(f"Original bounding box: {mesh.bounds}")
    
    pc, inv_transform = mesh_to_conditioning_pc(mesh)
    
    print(f"Conditioning PC bounds: {pc[:, :3].min(axis=0)} to {pc[:, :3].max(axis=0)}")
    
    # Assert points are in [-0.9996, 0.9996] per Phase 4.5
    assert pc[:, :3].min() >= -0.9996
    assert pc[:, :3].max() <= 0.9996
    
    # Inverse transform
    # The generated mesh will be in [-0.5, 0.5], so we simulate the output mesh
    # by dividing the point cloud by 2 * 0.9995 before applying inverse_transform
    synthetic_model_output = pc[:, :3].astype(np.float32) / (2 * 0.9995)
    round_trip = inv_transform(synthetic_model_output)
    
    # Check if they are back to original bounding box space
    diff = np.abs(round_trip.min(axis=0) - mesh.bounds[0]).max()
    print(f"Round trip diff: {diff}")
    
    assert diff < 1e-3, f"Round trip diff too large: {diff}"
    print("Phase 4 Test passed!")

if __name__ == "__main__":
    test_phase4()
