import torch
import trimesh
import numpy as np
from PIL import Image

from trellis2.utils.meshanything_bridge import refine_mesh_coarse, mesh_to_conditioning_pc
from trellis2.utils.sds_geometry_refine import refine_geometry

import os
from unittest.mock import patch

def verify_mesh_extents(orig_bounds, out_bounds, loose=False):
    orig_extents = orig_bounds[1] - orig_bounds[0]
    out_extents = out_bounds[1] - out_bounds[0]
    
    diff = np.abs(orig_extents - out_extents)
    ratio = diff / orig_extents
    print(f"Extents diff ratio: {ratio}")
    
    if loose:
        # Phase 5 without SDS generates a simplified mesh, so it loses thin features.
        # But it should be somewhat close. We assert the diff ratio is < 0.6 per axis.
        assert (ratio < 0.6).all(), f"Control case (No SDS) bounding box heavily mismatched! Ratio: {ratio}"
    else:
        # Phase 5 with real SDS might grow slightly due to SDS refine before simplification
        # We assert it didn't explode > 1.5x (ratio diff < 0.5)
        assert (ratio < 0.5).all(), f"Real case (with SDS) bounding box heavily mismatched or exploded! Ratio: {ratio}"

def test_phase5_control():
    print("Testing Phase 5 (Control Case: No SDS)...")
    mesh = trimesh.creation.icosphere()
    orig_bounds = mesh.bounds
    
    # Monkeypatch sds refine_geometry to just return the mesh
    with patch("trellis2.utils.sds_geometry_refine.refine_geometry") as mock_sds:
        mock_sds.return_value = mesh.copy()
        out_mesh = refine_mesh_coarse(mesh, "dummy.png")
    
    assert not np.isnan(out_mesh.vertices).any()
    assert not np.isnan(out_mesh.faces).any()
    assert len(out_mesh.faces) <= 2000
    
    verify_mesh_extents(orig_bounds, out_mesh.bounds, loose=True)
    print("Control Case passed!")

def test_phase5_real():
    print("Testing Phase 5 (Real Case: With SDS)...")
    mesh = trimesh.load("tests/can.glb", force="mesh")
    # Take a small sub-part if it's a scene
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    orig_bounds = mesh.bounds
    
    out_mesh = refine_mesh_coarse(mesh, "tests/can.png") # tests/can.png isn't strictly needed for code, but let's just pass empty string if it fails
    
    assert not np.isnan(out_mesh.vertices).any()
    assert not np.isnan(out_mesh.faces).any()
    assert len(out_mesh.faces) <= 2000
    
    verify_mesh_extents(orig_bounds, out_mesh.bounds, loose=False)
    print("Real Case passed!")

if __name__ == "__main__":
    test_phase5_control()
    # We won't run the real case automatically in basic CI to save time, but it's here for manual testing.
    # test_phase5_real()
