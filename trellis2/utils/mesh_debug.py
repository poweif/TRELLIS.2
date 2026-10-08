import os
import numpy as np
import torch
import trimesh
from PIL import Image

from .mesh_rasterizer import render_mesh, intrinsics_to_projection

_DEBUG_VIEWS = [(0, 0), (0, 90), (0, 180), (30, 315)]  # (elev, azim) degrees: front, right, back, 3/4
_DIST_MARGIN = 2.5  # camera distance as a multiple of the subject's bounding radius


def _intrinsics(device):
    intrinsics = torch.eye(3, device=device)
    intrinsics[0, 0] = 0.9
    intrinsics[1, 1] = 0.9
    intrinsics[0, 2] = 0.5
    intrinsics[1, 2] = 0.5
    return intrinsics


def _camera_for_view(center, radius, elev, azim, device):
    dist = max(float(radius), 1e-4) * _DIST_MARGIN
    elev_rad, azim_rad = np.radians(elev), np.radians(azim)

    offset = dist * np.array([
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad),
        np.cos(elev_rad) * np.cos(azim_rad),
    ])
    camera_pos = center + offset
    up = np.array([0.0, 1.0, 0.0])

    z_axis = (center - camera_pos)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x_axis, y_axis, z_axis, camera_pos
    extrinsics = np.linalg.inv(c2w)
    return torch.tensor(extrinsics, dtype=torch.float32, device=device)


def _tile_2x2(images, resolution):
    grid = Image.new("RGB", (resolution * 2, resolution * 2), color=(30, 30, 30))
    for img, pos in zip(images, [(0, 0), (resolution, 0), (0, resolution), (resolution, resolution)]):
        grid.paste(img, pos)
    return grid


def render_mesh_contact_sheet(mesh: trimesh.Trimesh, out_path: str, device, resolution=256):
    """Render a mesh (normal-shaded) from 4 fixed viewpoints into one PNG contact sheet."""
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print(f"  [debug] skipping render for {out_path}: empty mesh")
        return

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    center = mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(mesh.extents)) / 2.0

    intrinsics = _intrinsics(device)
    images = []
    with torch.no_grad():
        for elev, azim in _DEBUG_VIEWS:
            extrinsics = _camera_for_view(center, radius, elev, azim, device)
            out = render_mesh(vertices, faces, extrinsics, intrinsics, resolution, resolution)
            img = (out['normal'] * 0.5 + 0.5) * out['mask'].unsqueeze(-1)
            img = (img.clamp(0, 1) * 255).byte().cpu().numpy()
            images.append(Image.fromarray(img))

    _tile_2x2(images, resolution).save(out_path)


def _project_points_screen(points: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                            resolution: int, near=0.05, far=1000.0):
    proj = intrinsics_to_projection(intrinsics, near, far)
    full_proj = proj @ extrinsics
    p_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
    p_clip = (full_proj @ p_homo.T).T
    ndc = p_clip[:, :3] / p_clip[:, 3:4]
    px = ((ndc[:, 0] + 1) * 0.5 * resolution).long()
    py = ((1 - ndc[:, 1]) * 0.5 * resolution).long()
    z_cam = p_clip[:, 3]  # camera-space z, per this codebase's projection convention (see mesh_rasterizer.py)
    valid = z_cam > near
    return px, py, z_cam, valid


def _splat(px, py, z, valid, colors, resolution, dot_size=2):
    canvas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    order = torch.argsort(-z).cpu().numpy()
    px, py, valid = px.cpu().numpy(), py.cpu().numpy(), valid.cpu().numpy()
    for i in order:  # painter's algorithm: draw far points first so near points end up on top
        if not valid[i]:
            continue
        x0, y0 = px[i], py[i]
        if x0 < 0 or x0 >= resolution or y0 < 0 or y0 >= resolution:
            continue
        x_lo, x_hi = max(0, x0 - dot_size), min(resolution, x0 + dot_size + 1)
        y_lo, y_hi = max(0, y0 - dot_size), min(resolution, y0 + dot_size + 1)
        canvas[y_lo:y_hi, x_lo:x_hi] = colors[i]
    return Image.fromarray(canvas)


def render_pointcloud_contact_sheet(points: np.ndarray, normals: np.ndarray, out_path: str, device, resolution=256):
    """Very basic normal-colored point splat render (no rasterizer, just projection + painter's algorithm)."""
    if len(points) == 0:
        print(f"  [debug] skipping render for {out_path}: empty point cloud")
        return

    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) / 2.0
    colors = ((normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    points_t = torch.tensor(points, dtype=torch.float32, device=device)
    intrinsics = _intrinsics(device)
    images = []
    for elev, azim in _DEBUG_VIEWS:
        extrinsics = _camera_for_view(center, radius, elev, azim, device)
        px, py, z, valid = _project_points_screen(points_t, extrinsics, intrinsics, resolution)
        images.append(_splat(px, py, z, valid, colors, resolution))

    _tile_2x2(images, resolution).save(out_path)


def dump_mesh_stage(mesh: trimesh.Trimesh, name: str, debug_dir: str, device):
    """Export a mesh plus its rendered contact sheet as a numbered debug stage, and log a summary line."""
    os.makedirs(debug_dir, exist_ok=True)
    mesh.export(os.path.join(debug_dir, f"{name}.glb"))
    render_mesh_contact_sheet(mesh, os.path.join(debug_dir, f"{name}.png"), device)

    bounds = mesh.bounds
    summary = (f"{name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
               f"bounds=[{bounds[0].round(4).tolist()}, {bounds[1].round(4).tolist()}]")
    print(f"  [debug] {summary}")
    with open(os.path.join(debug_dir, "manifest.txt"), "a") as f:
        f.write(summary + "\n")


def dump_pointcloud_stage(points: np.ndarray, normals: np.ndarray, name: str, debug_dir: str, device):
    """Export a point cloud plus its rendered contact sheet as a numbered debug stage, and log a summary line."""
    os.makedirs(debug_dir, exist_ok=True)
    colors = ((normals.astype(np.float32) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    trimesh.points.PointCloud(vertices=points.astype(np.float64), colors=colors).export(
        os.path.join(debug_dir, f"{name}.ply"))
    render_pointcloud_contact_sheet(points.astype(np.float32), normals.astype(np.float32),
                                     os.path.join(debug_dir, f"{name}.png"), device)

    summary = (f"{name}: {len(points)} points, "
               f"bounds=[{points.min(0).round(4).tolist()}, {points.max(0).round(4).tolist()}]")
    print(f"  [debug] {summary}")
    with open(os.path.join(debug_dir, "manifest.txt"), "a") as f:
        f.write(summary + "\n")
