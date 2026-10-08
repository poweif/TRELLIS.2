import pytest
import torch
import trimesh
import numpy as np
from PIL import Image

from trellis2.utils import sds_geometry_refine as sds

def test_nan_guard_raises(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    # Force a NaN into the offsets partway through, simulating a corrupted optimization step
    real_step = torch.optim.Adam.step
    call_count = {"n": 0}
    
    def poison_step(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        call_count["n"] += 1
        if call_count["n"] == 3:
            for group in self.param_groups:
                for p in group["params"]:
                    p.data[0] = float("nan")
        return result
        
    monkeypatch.setattr(torch.optim.Adam, "step", poison_step)

    ref_image = Image.new("RGB", (256, 256), color=(255, 255, 255))
    with pytest.raises(RuntimeError, match="non-finite"):
        sds.refine_geometry(mesh, ref_image, n_iters=10, device="cuda")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
