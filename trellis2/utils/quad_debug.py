import numpy as np
import torch
from PIL import Image

from .mesh_rasterizer import render_mesh, intrinsics_to_projection
from .mesh_debug import _DEBUG_VIEWS, _intrinsics, _camera_for_view, _tile_2x2
from .quad_postprocess import parse_obj_raw, load_quad_obj, face_edges, find_irregular_vertices

_WIRE_COLOR = (20, 20, 20)
_IRREGULAR_COLOR = (255, 60, 60)
_DEPTH_BIAS = 2e-3


def _project_world_points(points: torch.Tensor, full_proj: torch.Tensor, H: int, W: int):
    """Perspective-project world-space points to (px, py, ndc_z, valid). Same convention
    as mesh_rasterizer.render_mesh/rasterize_forward: smaller ndc_z = nearer camera."""
    p_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
    p_clip = (full_proj @ p_homo.T).T
    ndc = p_clip[:, :3] / p_clip[:, 3:4]
    px = (ndc[:, 0] + 1) * 0.5 * W
    py = (1 - ndc[:, 1]) * 0.5 * H
    valid = p_clip[:, 3] > 1e-6
    return px, py, ndc[:, 2], valid


def _shaded_depth_buffer(shaded_pos: torch.Tensor, shaded_mask: torch.Tensor,
                          full_proj: torch.Tensor, H: int, W: int, device):
    """
    Recover the shaded pass's own ndc-z buffer by re-projecting its already-interpolated
    3D positions -- render_mesh doesn't expose depth directly, but 'pos' at each covered
    pixel already encodes exactly the surface point its internal z-buffer picked, so
    re-projecting it is equivalent to reading that z-buffer without duplicating
    rasterize_forward's logic. Uncovered pixels get depth=2.0 (outside the valid
    [-1, 1] ndc range), i.e. "no occluder, anything drawn there wins".
    """
    depth = torch.full((H, W), 2.0, device=device)
    yi, xi = shaded_mask.nonzero(as_tuple=True)
    if len(yi) == 0:
        return depth
    pos = shaded_pos[yi, xi]
    p_homo = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=-1)
    p_clip = (full_proj @ p_homo.T).T
    depth[yi, xi] = p_clip[:, 2] / p_clip[:, 3]
    return depth


def render_quad_wireframe(quad_obj_path: str, extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                           H: int, W: int, device, near: float = 0.1, far: float = 1000.0,
                           wire_color=_WIRE_COLOR, irregular_color=_IRREGULAR_COLOR,
                           depth_bias: float = _DEPTH_BIAS) -> np.ndarray:
    """
    Shaded render (via mesh_rasterizer.render_mesh on a triangulated copy) with the mesh's
    GENUINE quad edges -- never the triangulation diagonal -- overlaid as a depth-tested
    wireframe, plus irregular (valence != 4) vertices marked with a distinct dot. Returns
    an (H, W, 3) uint8 image.
    """
    raw_vertices, raw_faces = parse_obj_raw(quad_obj_path)
    tri_mesh = load_quad_obj(quad_obj_path)

    verts_t = torch.tensor(tri_mesh.vertices, dtype=torch.float32, device=device)
    tris_t = torch.tensor(tri_mesh.faces, dtype=torch.long, device=device)

    shaded = render_mesh(verts_t, tris_t, extrinsics, intrinsics, H, W, near, far)
    img = (shaded['normal'] * 0.5 + 0.5) * shaded['mask'].unsqueeze(-1)
    canvas_arr = (img.clamp(0, 1) * 255).byte().cpu().numpy().copy()

    if len(raw_vertices) == 0:
        return canvas_arr

    proj = intrinsics_to_projection(intrinsics, near, far)
    full_proj = proj @ extrinsics
    depth_buffer = _shaded_depth_buffer(shaded['pos'], shaded['mask'], full_proj, H, W, device)

    raw_verts_t = torch.tensor(raw_vertices, dtype=torch.float32, device=device)
    px_v, py_v, ndc_z_v, valid_v = _project_world_points(raw_verts_t, full_proj, H, W)

    edges = face_edges(raw_faces)
    if len(edges) > 0:
        v0, v1 = raw_verts_t[edges[:, 0]], raw_verts_t[edges[:, 1]]
        seg_len_px = torch.hypot(px_v[edges[:, 0]] - px_v[edges[:, 1]],
                                  py_v[edges[:, 0]] - py_v[edges[:, 1]])
        # ~0.7 samples/pixel of screen-space edge length -> a solid (non-dotted) line
        n_samples = (seg_len_px * 0.7 + 2).long().clamp(2, 96)
        max_n = int(n_samples.max().item())

        t = torch.linspace(0, 1, max_n, device=device).view(1, -1, 1)
        sample_pts = v0.unsqueeze(1) * (1 - t) + v1.unsqueeze(1) * t  # (E, max_n, 3)
        sample_valid = torch.arange(max_n, device=device).view(1, -1) < n_samples.view(-1, 1)

        flat_pts = sample_pts.reshape(-1, 3)
        px, py, ndc_z, valid = _project_world_points(flat_pts, full_proj, H, W)
        valid = valid & sample_valid.reshape(-1)

        px_i, py_i = px.round().long(), py.round().long()
        in_bounds = (px_i >= 0) & (px_i < W) & (py_i >= 0) & (py_i < H)
        keep = valid & in_bounds
        px_i, py_i, ndc_z = px_i[keep], py_i[keep], ndc_z[keep]

        occluder_z = depth_buffer[py_i, px_i]
        front = (ndc_z - depth_bias) <= occluder_z
        px_f, py_f = px_i[front].cpu().numpy(), py_i[front].cpu().numpy()

        wire = np.array(wire_color, dtype=np.uint8)
        # 1px dilation (draw + 4-neighbors) so thin diagonal segments stay visually
        # continuous instead of a dotted line at typical debug resolutions.
        for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            yy = np.clip(py_f + dy, 0, H - 1)
            xx = np.clip(px_f + dx, 0, W - 1)
            canvas_arr[yy, xx] = wire

    irregular = find_irregular_vertices(raw_faces, len(raw_vertices))
    if len(irregular) > 0:
        irr_px, irr_py = px_v[irregular].round().long(), py_v[irregular].round().long()
        irr_z, irr_valid = ndc_z_v[irregular], valid_v[irregular]
        in_bounds = (irr_px >= 0) & (irr_px < W) & (irr_py >= 0) & (irr_py < H)
        keep = irr_valid & in_bounds
        irr_px, irr_py, irr_z = irr_px[keep], irr_py[keep], irr_z[keep]
        if len(irr_px) > 0:
            occluder_z = depth_buffer[irr_py, irr_px]
            front = ((irr_z - depth_bias) <= occluder_z).cpu().numpy()
            irr_px_np, irr_py_np = irr_px.cpu().numpy(), irr_py.cpu().numpy()
            dot = 2
            irr = np.array(irregular_color, dtype=np.uint8)
            for x0, y0 in zip(irr_px_np[front], irr_py_np[front]):
                x_lo, x_hi = max(0, x0 - dot), min(W, x0 + dot + 1)
                y_lo, y_hi = max(0, y0 - dot), min(H, y0 + dot + 1)
                canvas_arr[y_lo:y_hi, x_lo:x_hi] = irr

    return canvas_arr


def render_quad_wireframe_contact_sheet(quad_obj_path: str, out_path: str, device, resolution: int = 256):
    """
    Multi-view (front/right/back/3-4, auto-framed to the mesh's own bounding box) contact
    sheet of render_quad_wireframe -- mirrors mesh_debug.dump_mesh_stage's convention
    rather than inventing a new visualization pattern.
    """
    tri_mesh = load_quad_obj(quad_obj_path)
    center = tri_mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(tri_mesh.extents)) / 2.0

    intrinsics = _intrinsics(device)
    images = []
    with torch.no_grad():
        for elev, azim in _DEBUG_VIEWS:
            extrinsics = _camera_for_view(center, radius, elev, azim, device)
            img = render_quad_wireframe(quad_obj_path, extrinsics, intrinsics, resolution, resolution, device)
            images.append(Image.fromarray(img))

    _tile_2x2(images, resolution).save(out_path)
