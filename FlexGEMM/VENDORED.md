# Vendored: FlexGEMM

- **Upstream**: https://github.com/JeffreyXiang/FlexGEMM
- **Base commit**: `6dd94a859c26ee8246888502eada3dd8ad85532e` (2026-04-22, "Merge autotune cache on
  install instead of overwriting"). Upstream's HEAD when this was vendored. Upstream already
  includes ROCm support (PR #18), and `autotune_cache.json` is upstream's file, unchanged.
- Vendored as plain files (no `.git`, no submodules) so the ROCm changes live in this repo.

## Changes for ROCm (gfx1151)

| File | Change |
|---|---|
| `flex_gemm/ops/spconv/submanifold_conv3d.py` | For `EXPLICIT_GEMM` / `IMPLICIT_GEMM[_SPLITK]`, build the neighbor cache with the pure-PyTorch `_compute_neighbor_cache_torch` instead of the `kernels.cuda.hashmap_build_submanifold_conv_neighbour_map` kernel. This fork runs `explicit_gemm` on ROCm (`trellis2/modules/sparse/conv/config.py`). |

## Updating

Diff a fresh upstream checkout against this directory, re-apply the table above, and update the
base commit here.
