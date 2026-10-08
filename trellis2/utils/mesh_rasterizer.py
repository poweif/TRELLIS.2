import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def intrinsics_to_projection(intrinsics: torch.Tensor, near: float, far: float) -> torch.Tensor:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = 1.0
    return ret

@torch.no_grad()
def rasterize_forward(vertices_clip, faces, H, W):
    device = vertices_clip.device
    M = faces.shape[0]
    
    ndc = vertices_clip[:, :3] / vertices_clip[:, 3:4] # (N, 3)
    # OpenCV convention for screen coordinates
    face_px = torch.zeros((M, 3, 2), device=device)
    face_px[:, :, 0] = (ndc[faces, 0] + 1) * 0.5 * W
    face_px[:, :, 1] = (1 - ndc[faces, 1]) * 0.5 * H
    face_z = ndc[faces, 2] # (M, 3)
    
    a_all = face_px[:, 0]
    b_all = face_px[:, 1]
    c_all = face_px[:, 2]
    
    v0_all = b_all - a_all
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
    area_all = dx_all * dy_all
    
    packed_buffer = torch.full((H * W,), torch.iinfo(torch.int64).max, device=device, dtype=torch.int64)
    
    _BUCKETS = [(16*16, 8000), (64*64, 500), (float("inf"), 20)]
    sort_idx = area_all.argsort()
    sorted_area = area_all[sort_idx]
    
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
            K = len(k_idx)
            if K == 0 or max_dx == 0 or max_dy == 0:
                continue
                
            gy = (ymin_all[k_idx].view(K, 1, 1) + torch.arange(max_dy, device=device).view(1, max_dy, 1)).expand(K, max_dy, max_dx)
            gx = (xmin_all[k_idx].view(K, 1, 1) + torch.arange(max_dx, device=device).view(1, 1, max_dx)).expand(K, max_dy, max_dx)
            
            valid = (gx <= xmax_all[k_idx].view(K, 1, 1)) & (gx >= 0) & (gx < W) & \
                    (gy <= ymax_all[k_idx].view(K, 1, 1)) & (gy >= 0) & (gy < H)
                    
            gx_f = gx.float() + 0.5
            gy_f = gy.float() + 0.5
            
            a_k = a_all[k_idx]
            v0_k = v0_all[k_idx]
            v1_k = v1_all[k_idx]
            d00_k = d00_all[k_idx]
            d01_k = d01_all[k_idx]
            d11_k = d11_all[k_idx]
            denom_k = denom_all[k_idx]
            deg_k = degenerate_all[k_idx]
            z_k = face_z[k_idx]
            
            v2x = gx_f - a_k[:, 0].view(K, 1, 1)
            v2y = gy_f - a_k[:, 1].view(K, 1, 1)
            
            d20 = v2x * v0_k[:, 0].view(K, 1, 1) + v2y * v0_k[:, 1].view(K, 1, 1)
            d21 = v2x * v1_k[:, 0].view(K, 1, 1) + v2y * v1_k[:, 1].view(K, 1, 1)
            
            inv_denom = 1.0 / denom_k.abs().clamp(min=1e-10)
            bary_u = (d11_k.view(K,1,1) * d20 - d01_k.view(K,1,1) * d21) * inv_denom.view(K,1,1)
            bary_v = (d00_k.view(K,1,1) * d21 - d01_k.view(K,1,1) * d20) * inv_denom.view(K,1,1)
            bary_w = 1.0 - bary_u - bary_v
            
            inside = (bary_u >= 0) & (bary_v >= 0) & (bary_w >= 0) & valid & ~deg_k.view(K, 1, 1)
            
            z_val = bary_w * z_k[:, 0].view(K, 1, 1) + \
                    bary_u * z_k[:, 1].view(K, 1, 1) + \
                    bary_v * z_k[:, 2].view(K, 1, 1)
            
            inside = inside & (z_val > -1.0) & (z_val < 1.0)
            
            if not inside.any():
                continue
                
            ki, yi_loc, xi_loc = inside.nonzero(as_tuple=True)
            yi_pix = gy[ki, yi_loc, xi_loc]
            xi_pix = gx[ki, yi_loc, xi_loc]
            flat_idx = (yi_pix * W + xi_pix).long()
            
            z_valid = z_val[ki, yi_loc, xi_loc]
            z_norm = ((z_valid + 1.0) * 0.5 * (2**31 - 1)).long().clamp(0, 2**31 - 1)
            tri_id_val = (k_idx[ki] + 1).long()
            
            packed = (z_norm << 32) | tri_id_val
            
            packed_buffer.scatter_reduce_(0, flat_idx, packed, reduce="amin", include_self=True)
            
        prev_thresh = max_area
        
    tri_id_out = (packed_buffer & 0xFFFFFFFF)
    tri_id_out[packed_buffer == torch.iinfo(torch.int64).max] = 0
    return tri_id_out.view(H, W)

def compute_vertex_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1) # (M, 3)
    
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.scatter_add_(0, faces[:, 0:1].expand(-1, 3), face_normals)
    vertex_normals.scatter_add_(0, faces[:, 1:2].expand(-1, 3), face_normals)
    vertex_normals.scatter_add_(0, faces[:, 2:3].expand(-1, 3), face_normals)
    
    return torch.nn.functional.normalize(vertex_normals, p=2, dim=1, eps=1e-8)

def render_mesh(vertices, faces, extrinsics, intrinsics, H, W, near=0.1, far=1000.0):
    """
    Differentiable mesh rasterizer.
    Returns:
        mask: (H, W) boolean tensor
        pos: (H, W, 3) interpolated 3D position
        normal: (H, W, 3) interpolated vertex normal
    """
    device = vertices.device
    
    # Forward pass (non-differentiable) to get tri_id
    proj = intrinsics_to_projection(intrinsics, near, far)
    full_proj = proj @ extrinsics
    
    v_homo = torch.cat([vertices.detach(), torch.ones_like(vertices[:, :1])], dim=-1)
    v_clip = (full_proj @ v_homo.T).T 
    
    tri_id = rasterize_forward(v_clip, faces, H, W) # (H, W)
    mask = tri_id > 0
    
    # Differentiable interpolation
    yi, xi = mask.nonzero(as_tuple=True)
    if len(yi) == 0:
        return {
            'mask': mask,
            'pos': torch.zeros((H, W, 3), device=device),
            'normal': torch.zeros((H, W, 3), device=device)
        }
        
    valid_tri_id = tri_id[yi, xi] - 1
    face_verts = faces[valid_tri_id]
    
    v0_3d = vertices[face_verts[:, 0]]
    v1_3d = vertices[face_verts[:, 1]]
    v2_3d = vertices[face_verts[:, 2]]
    
    def project(v):
        v_h = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
        v_c = (full_proj @ v_h.T).T
        ndc = v_c[:, :3] / v_c[:, 3:4]
        x = (ndc[:, 0] + 1) * 0.5 * W
        y = (1 - ndc[:, 1]) * 0.5 * H
        return torch.stack([x, y], dim=-1)
        
    a = project(v0_3d)
    b = project(v1_3d)
    c = project(v2_3d)
    
    gx = xi.float() + 0.5
    gy = yi.float() + 0.5
    
    v0 = b - a
    v1 = c - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    
    v2x = gx - a[:, 0]
    v2y = gy - a[:, 1]
    
    d20 = v2x * v0[:, 0] + v2y * v0[:, 1]
    d21 = v2x * v1[:, 0] + v2y * v1[:, 1]
    
    inv_denom = 1.0 / denom.clamp(min=1e-10)
    bary_u = (d11 * d20 - d01 * d21) * inv_denom
    bary_v = (d00 * d21 - d01 * d20) * inv_denom
    bary_w = 1.0 - bary_u - bary_v
    
    def interp(attr):
        a0 = attr[face_verts[:, 0]]
        a1 = attr[face_verts[:, 1]]
        a2 = attr[face_verts[:, 2]]
        val = bary_w.unsqueeze(1) * a0 + bary_u.unsqueeze(1) * a1 + bary_v.unsqueeze(1) * a2
        out = torch.zeros((H, W, attr.shape[1]), device=device, dtype=val.dtype)
        out[yi, xi] = val
        return out
        
    pos_map = interp(vertices)
    v_normals = compute_vertex_normals(vertices, faces)
    normal_map = interp(v_normals)
    
    return {
        'mask': mask,
        'pos': pos_map,
        'normal': normal_map
    }
