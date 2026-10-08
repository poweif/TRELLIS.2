# Vendored: CuMesh

- **Upstream**: https://github.com/JeffreyXiang/CuMesh
- **Base commit**: `12289e1062f0603f2f0d0771b02e1395d247f26f` (2026-05-09, "Merge pull request #29 from
  cuzelac/fix/blackwell-stream-sync-upstream"). Upstream's HEAD when this was vendored. Everything
  else (Python package, examples, `pyproject.toml`, README, LICENSE) is identical to it.
- **Submodule `third_party/cubvh`**: https://github.com/JeffreyXiang/cubvh at
  `ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70` (branch `trellis.2`), vendored unmodified. Its own
  `third_party/eigen` is Eigen 3.4.90, pruned to `Eigen/` + license files (only `<Eigen/Dense>` and
  pybind11's Eigen support are used).
- Vendored as plain files (no `.git`, no submodules) so the ROCm changes live in this repo.

## Changes for ROCm (gfx1151)

| File | Change |
|---|---|
| `src/shared.h` | On `__HIP_PLATFORM_AMD__`, include `hip/hip_runtime.h` + `hipcub/hipcub.hpp` and alias `namespace cub = hipcub`; CUDA headers otherwise. |
| `src/{atlas,clean_up,connectivity,geometry,simplify}.cu` | Drop the direct `#include <cub/cub.cuh>`; `shared.h` provides cub/hipcub. |
| `src/clean_up.cu` | `int3_decomposer` returns `hipcub::tuple` on HIP (no `::cuda::std::tuple`). |
| `src/dtypes.cuh` | `Vec3f` constructors are `__host__ __device__` (needed by host-side code under hip-clang). |
| `setup.py` | Skip the CUDA-only `nvcc` flags on HIP (upstream already maps `GPU_ARCHS` to `--offload-arch`). |
| `build.sh` | Added: build helper defaulting to `GPU_ARCHS=gfx1151`. |
| `.gitmodules` | Removed: `third_party/cubvh` is vendored as files. |

The `.hip` / `*_hip.*` files are hipify output: torch's `CUDAExtension` regenerates them from
the `.cu` sources whenever `setup.py` runs on ROCm. They're gitignored. Edit the `.cu` files only.

## Updating

Diff a fresh upstream checkout against this directory (excluding `*.hip`, `*_hip.*`), re-apply
the table above, and update the base commit here.
