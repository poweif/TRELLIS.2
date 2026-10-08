import torch
import numpy as np
import trimesh
from trellis2.utils.mesh_rasterizer import render_mesh

def get_camera(dist=2.0, elev=0.0, azim=0.0):
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    
    x = dist * np.cos(elev_rad) * np.sin(azim_rad)
    y = dist * np.sin(elev_rad)
    z = dist * np.cos(elev_rad) * np.cos(azim_rad)
    
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    
    z_axis = target - camera_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    extrinsics = np.eye(4)
    extrinsics[:3, 0] = x_axis
    extrinsics[:3, 1] = y_axis
    extrinsics[:3, 2] = z_axis
    extrinsics[:3, 3] = camera_pos
    
    # We need world to camera (inverse)
    extrinsics = np.linalg.inv(extrinsics)
    return torch.tensor(extrinsics, dtype=torch.float32)

def test_phase2():
    print("Testing Phase 2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mesh = trimesh.creation.box()
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    
    extrinsics = get_camera(dist=3.0, elev=30, azim=45).to(device)
    
    # Simple intrinsics
    intrinsics = torch.eye(3, device=device)
    intrinsics[0, 0] = 1.0 # fx
    intrinsics[1, 1] = 1.0 # fy
    intrinsics[0, 2] = 0.5 # cx
    intrinsics[1, 2] = 0.5 # cy
    
    H, W = 256, 256
    
    out = render_mesh(vertices, faces, extrinsics, intrinsics, H, W)
    
    # Debug projection manually
    proj = torch.zeros((4, 4), device=device)
    proj[0, 0] = 2.0; proj[1, 1] = 2.0; proj[2, 2] = (1000.0 + 0.1)/(1000.0 - 0.1)
    proj[2, 3] = 2 * 0.1 * 1000.0 / (0.1 - 1000.0); proj[3, 2] = 1.0; proj[0, 2] = 0; proj[1, 2] = 0
    full_proj = proj @ extrinsics
    v_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
    v_clip = (full_proj @ v_homo.T).T
    ndc = v_clip[:, :3] / v_clip[:, 3:4]
    print(f"v_clip z: {v_clip[:, 3].min().item()} to {v_clip[:, 3].max().item()}")
    print(f"ndc x min max: {ndc[:, 0].min().item()}, {ndc[:, 0].max().item()}")
    print(f"ndc y min max: {ndc[:, 1].min().item()}, {ndc[:, 1].max().item()}")
    print(f"ndc z min max: {ndc[:, 2].min().item()}, {ndc[:, 2].max().item()}")
    
    mask = out['mask']
    pos = out['pos']
    normal = out['normal']
    
    print(f"Mask sum: {mask.sum().item()} pixels")
    assert mask.sum() > 0, "Box should be visible"
    
    from PIL import Image
    for i, (elev, azim) in enumerate([(0, 0), (0, 90), (30, 180), (-20, 270)]):
        cam = get_camera(dist=3.0, elev=elev, azim=azim).to(device)
        out = render_mesh(vertices.detach(), faces, cam, intrinsics, 256, 256)
        img = ((out['normal'] * 0.5 + 0.5) * out['mask'].unsqueeze(-1)).clamp(0, 1)
        Image.fromarray((img * 255).byte().cpu().numpy()).save(f"/tmp/phase2_view_{i}.png")
    
    # Create a dummy loss on normal
    loss = normal.sum()
    loss.backward()
    
    print(f"Gradient sum: {vertices.grad.abs().sum().item()}")
    assert vertices.grad is not None
    assert vertices.grad.abs().sum().item() > 0, "Gradients should flow back to vertices"
    
    print("Phase 2 Test passed!")

if __name__ == "__main__":
    test_phase2()
