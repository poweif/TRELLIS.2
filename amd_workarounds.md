# TRELLIS.2 on AMD Strix Halo (gfx1151) - Workarounds & Patches

Running the TRELLIS.2 model on an AMD Strix Halo system (`gfx1151`) requires several deep workarounds. The model relies on custom CUDA/HIP kernels that use 64-bit `atomicCAS` instructions which silently fail on RDNA3.5 architectures. In addition, there are memory spikes due to padding in the attention layer, and PyTorch extension compile flags on AMD break C++ linking for `CuMesh`.

Below is a complete, extremely explicit instruction manual with exact code diffs and commands needed to patch and run a clean checkout of TRELLIS2 on this hardware. Another AI or developer can apply these changes blindly.

---

## 1. Environment & Basic Dependencies

You must spoof the architecture as `gfx1100` and install some missing dependencies.

**Bash Commands:**
```bash
# Spoof the architecture
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export GPU_ARCHS="gfx1100"

# Install missing system dependency for image processing
sudo apt update && sudo apt install -y libjpeg-dev

# Ensure you have pyglet installed for viewing the output later
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
pip install pyglet
```

---

## 2. Recompile Extensions with `gfx1100` target

When running `setup.sh` or manually installing extensions, you must ensure `GPU_ARCHS=gfx1100` is exported so ROCm knows what to build. Otherwise, binaries for `gfx1151` will be completely missing or malformed.

```bash
export GPU_ARCHS="gfx1100"
# Follow standard setup instructions but ensure GPU_ARCHS is exported in the same shell
```

---

## 3. `o-voxel` Hashmap Patch (Silent Failure Fix)

**File to patch:** `~/miniconda3/envs/trellis2/lib/python3.10/site-packages/o_voxel/convert/flexible_dual_grid.py`
*(Note: If you are building `o-voxel` from source in the repo, apply this to `o-voxel/o_voxel/convert/flexible_dual_grid.py` before installing).*

The `flexible_dual_grid_to_mesh` function uses a HIP kernel to hash 3D coordinates. It fails silently and returns 0 vertices on Strix Halo. We replace it with a dense tensor.

**Replace lines ~221 to ~240 (the `_C.hashmap_insert_3d_idx_as_val_cuda` and `_C.hashmap_find_3d_idx_as_val_cuda` blocks) with:**

```python
    # Extract mesh
    N = dual_vertices.shape[0]
    mesh_vertices = (coords.float() + dual_vertices) / (2 * N) - 0.5

    # USE DENSE TENSOR INSTEAD OF CUDA HASHMAP
    S_x, S_y, S_z = int(grid_size[0].item()), int(grid_size[1].item()), int(grid_size[2].item())
    dense_map = torch.full((S_x * S_y * S_z,), -1, dtype=torch.int32, device=coords.device)
    c_x, c_y, c_z = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
    
    valid_mask = (c_x >= 0) & (c_x < S_x) & (c_y >= 0) & (c_y < S_y) & (c_z >= 0) & (c_z < S_z)
    flat_indices = c_x[valid_mask] * (S_y * S_z) + c_y[valid_mask] * S_z + c_z[valid_mask]
    dense_map[flat_indices] = torch.arange(N, dtype=torch.int32, device=coords.device)[valid_mask]

    # Find connected voxels
    edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + flexible_dual_grid_to_mesh.edge_neighbor_voxel_offset
    connected_voxel = edge_neighbor_voxel[intersected_flag]
    M = connected_voxel.shape[0]
    
    cv_x, cv_y, cv_z = connected_voxel[..., 0].long(), connected_voxel[..., 1].long(), connected_voxel[..., 2].long()
    cv_valid_mask = (cv_x >= 0) & (cv_x < S_x) & (cv_y >= 0) & (cv_y < S_y) & (cv_z >= 0) & (cv_z < S_z)
    cv_flat = cv_x * (S_y * S_z) + cv_y * S_z + cv_z
    
    connected_voxel_indices = torch.full((M, 4), -1, dtype=torch.int32, device=coords.device)
    connected_voxel_indices[cv_valid_mask] = dense_map[cv_flat[cv_valid_mask]]

    connected_voxel_valid = (connected_voxel_indices != -1).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].long()
    L = quad_indices.shape[0]
```

---

## 4. `flex_gemm` Explicit PyTorch Fallback Patch (Silent Failure Fix)

`flex_gemm` uses the exact same failing HIP hashmap. However, it contains an undocumented `explicit_gemm` pure PyTorch fallback.

### Step 4a: Modify `submanifold_conv3d.py`
**File to patch:** `/tmp/extensions/FlexGEMM/flex_gemm/ops/spconv/submanifold_conv3d.py` (or wherever `FlexGEMM` source is before installing).

**Modify `_compute_neighbor_cache` function (around line 59):**
Change this block:
```python
        if spconv.ALGORITHM in [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.IMPLICIT_GEMM_SPLITK]:
            if coords.is_cuda:
                neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map(
                    hashmap_keys, hashmap_vals, coords,
                    W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    dilation[0], dilation[1], dilation[2],
                )
            else:
                raise NotImplementedError("CPU version of hashmap is not implemented")
            return SubMConv3dNeighborCache(**{
                'neighbor_map': neighbor_map,
            })
```

To just this one line:
```python
        if spconv.ALGORITHM in [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.IMPLICIT_GEMM_SPLITK]:
            return SubMConv3dFunction._compute_neighbor_cache_torch(coords, shape, kernel_size, dilation)
```

### Step 4b: Reinstall FlexGEMM
```bash
cd /tmp/extensions/FlexGEMM
export GPU_ARCHS="gfx1100"
python setup.py install
```

### Step 4c: Set the Algorithm in TRELLIS
**File to patch:** `trellis2/modules/sparse/conv/config.py`

**Change:**
```python
FLEX_GEMM_ALGO = 'implicit_gemm'
```
**To:**
```python
FLEX_GEMM_ALGO = 'explicit_gemm'
```

---

## 5. Attention Memory Bottlenecks (OOM)

Padding sparse sequences causes massive OOM spikes. We replace padding with a loop.

**File to patch:** `trellis2/modules/sparse/attention/sdpa_varlen.py`

**Replace the `sdpa_varlen` function implementation (around line 53) with:**
```python
def sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    l: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    b = l.shape[0]
    q_split = list(torch.split(q, l.tolist(), dim=0))
    k_split = list(torch.split(k, l.tolist(), dim=0))
    v_split = list(torch.split(v, l.tolist(), dim=0))
    
    out_split = []
    for i in range(b):
        if l[i] == 0: continue
        q_i = q_split[i].unsqueeze(0)
        k_i = k_split[i].unsqueeze(0)
        v_i = v_split[i].unsqueeze(0)
        
        out_i = F.scaled_dot_product_attention(
            q_i.transpose(1, 2), 
            k_i.transpose(1, 2), 
            v_i.transpose(1, 2)
        )
        out_split.append(out_i.transpose(1, 2).squeeze(0))
        
    out = torch.cat(out_split, dim=0)
    return out
```

---

## 6. `CuMesh` `fill_holes` Bypass

`CuMesh` crashes with `HIP Error 209: No binary for GPU`. Bypass it.

**File to patch:** `trellis2/pipelines/trellis2_image_to_3d.py`

**Change around line 474:**
```python
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
```
**To:**
```python
        for m, v in zip(meshes, tex_voxels):
            # m.fill_holes()
            out_mesh.append(
```

---

## Final Execution

After applying all these patches, you can run the pipeline securely:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HF_TOKEN=<your_token>
python3 run_sample.py
```
