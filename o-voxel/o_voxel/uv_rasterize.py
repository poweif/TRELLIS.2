"""
UV-space rasterizer — AMD/ROCm-compatible replacement for the nvdiffrast
rasterize + interpolate calls used during GLB texture baking.

Only the two operations needed by to_glb / postprocess_mesh are provided:
  rasterize_uv   — drop-in for dr.rasterize  (UV-space, no gradients)
  interpolate_uv — drop-in for dr.interpolate (barycentric attribute lookup)
"""

from typing import Optional, Tuple
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_bucket(
    k_idx: torch.Tensor,
    xmin: torch.Tensor, xmax: torch.Tensor,
    ymin: torch.Tensor, ymax: torch.Tensor,
    a: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor,
    d00: torch.Tensor, d01: torch.Tensor, d11: torch.Tensor,
    denom: torch.Tensor, degenerate: torch.Tensor,
    max_dy: int, max_dx: int,
    H: int, W: int,
    rast_flat: torch.Tensor,
) -> None:
    """Fill rast_flat for one batch of faces (all with bbox ≤ max_dy × max_dx)."""
    device = rast_flat.device
    K = len(k_idx)
    if K == 0 or max_dx == 0 or max_dy == 0:
        return

    # Pixel grid: (K, max_dy, max_dx)
    gy = (ymin[k_idx].view(K, 1, 1)
          + torch.arange(max_dy, device=device).view(1, max_dy, 1)).expand(K, max_dy, max_dx)
    gx = (xmin[k_idx].view(K, 1, 1)
          + torch.arange(max_dx, device=device).view(1, 1, max_dx)).expand(K, max_dy, max_dx)

    valid = (
        (gx <= xmax[k_idx].view(K, 1, 1)) & (gx >= 0) & (gx < W) &
        (gy <= ymax[k_idx].view(K, 1, 1)) & (gy >= 0) & (gy < H)
    )  # (K, max_dy, max_dx)

    gx_f = gx.float()
    gy_f = gy.float()

    a_k   = a[k_idx]       # (K, 2)
    v0_k  = v0[k_idx]      # (K, 2)
    v1_k  = v1[k_idx]      # (K, 2)
    d00_k = d00[k_idx]     # (K,)
    d01_k = d01[k_idx]     # (K,)
    d11_k = d11[k_idx]     # (K,)
    denom_k = denom[k_idx] # (K,)
    deg_k   = degenerate[k_idx]  # (K,)

    v2x = gx_f - a_k[:, 0].view(K, 1, 1)
    v2y = gy_f - a_k[:, 1].view(K, 1, 1)

    d20 = v2x * v0_k[:, 0].view(K, 1, 1) + v2y * v0_k[:, 1].view(K, 1, 1)
    d21 = v2x * v1_k[:, 0].view(K, 1, 1) + v2y * v1_k[:, 1].view(K, 1, 1)

    inv_denom = 1.0 / denom_k.abs().clamp(min=1e-10)
    # bary_u: weight for face[:,1] (b); bary_v: weight for face[:,2] (c)
    bary_u = (d11_k.view(K,1,1) * d20 - d01_k.view(K,1,1) * d21) * inv_denom.view(K,1,1)
    bary_v = (d00_k.view(K,1,1) * d21 - d01_k.view(K,1,1) * d20) * inv_denom.view(K,1,1)
    bary_w = 1.0 - bary_u - bary_v  # weight for face[:,0] (a)

    inside = (
        (bary_u >= 0) & (bary_v >= 0) & (bary_w >= 0) &
        valid & ~deg_k.view(K, 1, 1)
    )  # (K, max_dy, max_dx)

    if not inside.any():
        return

    ki, yi_loc, xi_loc = inside.nonzero(as_tuple=True)
    yi_pix = gy[ki, yi_loc, xi_loc]  # (P,)
    xi_pix = gx[ki, yi_loc, xi_loc]  # (P,)
    flat_idx = (yi_pix * W + xi_pix).long()  # (P,)

    # 1-indexed face ID within the faces array passed to rasterize_uv
    tri_id = (k_idx[ki] + 1).float()
    bu_val = bary_u[ki, yi_loc, xi_loc]
    bv_val = bary_v[ki, yi_loc, xi_loc]

    rast_flat.scatter_(
        0,
        flat_idx.view(-1, 1).expand(-1, 4),
        torch.stack([bu_val, bv_val, torch.zeros_like(bu_val), tri_id], dim=1),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Bucket thresholds: (max_bbox_area, batch_size_in_faces)
# Memory per batch ≈ batch_size × max_bbox_h × max_bbox_w × 4 × 4 bytes
_BUCKETS = [
    (16 * 16,        8_000),  # small  triangles
    (64 * 64,          500),  # medium triangles
    (float("inf"),      20),  # large  triangles
]


def rasterize_uv(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    H: int,
    W: int,
) -> Tuple[torch.Tensor, None]:
    """
    UV-space rasterizer. For each texel in an H×W texture, records which
    triangle covers it and the barycentric coordinates.

    Drop-in for::

        ctx = dr.RasterizeCudaContext()
        rast, _ = dr.rasterize(ctx, uvs_clip, faces, resolution=[H, W])

    Args:
        uvs:   (N, 2) UV coordinates in [0, 1].
        faces: (M, 3) vertex indices (indexes into uvs). May be a chunk of
               the full face array — the returned tri_id is 1-indexed into
               this array, matching nvdiffrast chunk behaviour.
        H, W:  Output texture height and width.

    Returns:
        rast:  (1, H, W, 4) — channels are [bary_u, bary_v, 0, tri_id].
               tri_id is 1-indexed into *faces*; 0 = background.
        None   (no barycentric derivatives, matching nvdiffrast API).
    """
    device = uvs.device
    M = faces.shape[0]
    rast = torch.zeros(1, H, W, 4, device=device, dtype=torch.float32)
    if M == 0:
        return rast, None

    # UV → pixel space: u→x (column), v→y (row)
    face_px = torch.stack([
        uvs[faces][..., 0] * W,
        uvs[faces][..., 1] * H,
    ], dim=-1)  # (M, 3, 2)

    a_all = face_px[:, 0]  # (M, 2)
    b_all = face_px[:, 1]
    c_all = face_px[:, 2]

    v0_all = b_all - a_all   # (M, 2)
    v1_all = c_all - a_all

    d00_all = (v0_all * v0_all).sum(-1)
    d01_all = (v0_all * v1_all).sum(-1)
    d11_all = (v1_all * v1_all).sum(-1)
    denom_all = d00_all * d11_all - d01_all * d01_all
    degenerate_all = denom_all.abs() < 1e-10

    xmin_all = face_px[..., 0].min(1)[0].floor().long().clamp(0, W - 1)
    xmax_all = face_px[..., 0].max(1)[0].ceil().long().clamp(0, W - 1)
    ymin_all = face_px[..., 1].min(1)[0].floor().long().clamp(0, H - 1)
    ymax_all = face_px[..., 1].max(1)[0].ceil().long().clamp(0, H - 1)

    dx_all = (xmax_all - xmin_all + 1).clamp(min=0)
    dy_all = (ymax_all - ymin_all + 1).clamp(min=0)
    area_all = dx_all * dy_all  # (M,)

    # Sort faces by bbox area so batches have bounded memory usage
    sort_idx = area_all.argsort()
    sorted_area = area_all[sort_idx]

    rast_flat = rast[0].view(H * W, 4)

    prev_thresh = -1
    for max_area, batch_size in _BUCKETS:
        lo = (sorted_area > prev_thresh)
        hi = (sorted_area <= max_area) if max_area != float("inf") else torch.ones_like(lo)
        bucket_global = sort_idx[lo & hi]
        if len(bucket_global) == 0:
            prev_thresh = max_area
            continue

        max_dy = dy_all[bucket_global].max().item()
        max_dx = dx_all[bucket_global].max().item()

        for start in range(0, len(bucket_global), batch_size):
            k_idx = bucket_global[start:start + batch_size]
            _process_bucket(
                k_idx,
                xmin_all, xmax_all, ymin_all, ymax_all,
                a_all, v0_all, v1_all,
                d00_all, d01_all, d11_all,
                denom_all, degenerate_all,
                int(max_dy), int(max_dx),
                H, W,
                rast_flat,
            )

        prev_thresh = max_area

    return rast, None


def interpolate_uv(
    attrs: torch.Tensor,
    rast: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, None]:
    """
    Barycentric interpolation of per-vertex attributes.

    Drop-in for::

        interp, _ = dr.interpolate(attrs, rast, faces)
        result = interp[0]

    Args:
        attrs: (1, N, C) per-vertex attributes.
        rast:  (1, H, W, 4) from :func:`rasterize_uv`.
        faces: (M, 3) face vertex indices (full face array, global indices).

    Returns:
        Tuple of (interp, None) where interp is (1, H, W, C), matching
        nvdiffrast's ``(out, out_da)`` return convention.
    """
    H, W = rast.shape[1], rast.shape[2]

    tri_id  = rast[0, ..., 3].long()    # (H, W) 1-indexed; 0 = background
    bary_u  = rast[0, ..., 0]           # (H, W) weight for face[:,1]
    bary_v  = rast[0, ..., 1]           # (H, W) weight for face[:,2]
    bary_w  = 1.0 - bary_u - bary_v    # (H, W) weight for face[:,0]

    mask = tri_id > 0  # (H, W)
    tri_0idx = (tri_id - 1).clamp(min=0)  # 0-indexed, safe to index with

    face_verts = faces[tri_0idx]  # (H, W, 3)

    v0 = attrs[0][face_verts[..., 0]]  # (H, W, C)
    v1 = attrs[0][face_verts[..., 1]]
    v2 = attrs[0][face_verts[..., 2]]

    result = bary_w[..., None] * v0 + bary_u[..., None] * v1 + bary_v[..., None] * v2
    result = result * mask[..., None]  # zero out background pixels

    return result.unsqueeze(0), None
