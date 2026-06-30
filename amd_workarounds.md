# TRELLIS.2 on AMD Strix Halo (gfx1151) — Full Build & Run Guide

Complete instructions for building PyTorch 2.7.0 from source against system ROCm 7.1,
rebuilding the required GPU extensions, and running TRELLIS.2 natively on an AMD Radeon
8060S (Strix Halo / gfx1151 / RDNA4) without any `HSA_OVERRIDE_GFX_VERSION` workarounds.

---

## System configuration

- **GPU**: AMD Radeon 8060S Graphics (Strix Halo APU, gfx1151, RDNA4, 32-wide wavefront)
- **OS**: Ubuntu 24.04
- **ROCm**: 7.1 installed via Ubuntu packages (`/usr`) + hipblaslt/rocm-core/roctracer 7.2.3
  from AMD's upstream repo (`/opt/rocm-7.2.3`)
- **LLVM**: 21 (`llvm-21` Ubuntu package); `/usr/bin/clang++` is clang-17 — **do not use it**
- **Python**: 3.10 via conda environment `trellis2`

### Key installed packages

```
rocm 7.1.0
rocm-dev 7.1.0
rocm-core 7.2.3.70203    ← from AMD repo, installed to /opt/rocm-7.2.3
hipblaslt 1.2.2.70203    ← from AMD repo, installed to /opt/rocm-7.2.3
roctracer 4.1.70203      ← from AMD repo, installed to /opt/rocm-7.2.3
rocm-device-libs-21 7.1.1
llvm-21 1:21.1.8
```

The Ubuntu ROCm package layout differs from the standard AMD layout:
- cmake config files live under `/usr/lib/x86_64-linux-gnu/cmake/` (not `/usr/lib/cmake/`)
- HIP binaries: `/usr/bin/hipcc`, headers: `/usr/include/hip/`
- The real clang for ROCm: `/usr/lib/llvm-21/bin/clang++` (not `/usr/bin/clang++`)

---

## 1. Required system symlinks (sudo)

hipcc with `ROCM_PATH=/usr` looks for device bitcode at `/usr/amdgcn/bitcode`, but
Ubuntu installs them under the LLVM tree. The `libaotriton_v2.so` (bundled in pre-built
torch ROCm wheels) links against `libamdhip64.so.6` but ROCm 7.1 ships `.so.7`.

```bash
# ROCm device bitcode — needed for hipcc to compile GPU kernels
sudo mkdir -p /usr/amdgcn
sudo ln -sf /usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode /usr/amdgcn/bitcode

# ABI shim for aotriton (pre-built .so expects libamdhip64.so.6, ROCm 7.1 ships .so.7)
sudo ln -sf /usr/lib/x86_64-linux-gnu/libamdhip64.so.7.1.52801 \
            /usr/lib/x86_64-linux-gnu/libamdhip64.so.6
```

---

## 2. Conda environment

```bash
conda create -n trellis2 python=3.10 -y
conda activate trellis2

# setuptools must be < 80; setuptools ≥ 80 removed pkg_resources which
# torchvision's setup.py and several other packages still import
pip install "setuptools<80"
```

---

## 3. Build PyTorch 2.7.0 from source

### 3a. Clone

```bash
git clone --branch v2.7.0 --depth 1 https://github.com/pytorch/pytorch ~/pytorch-src
cd ~/pytorch-src
git submodule update --init --recursive
pip install -r requirements.txt
```

### 3b. Patch 1 — `cmake/public/LoadHIP.cmake`

Ubuntu's ROCm cmake files are under `/usr/lib/x86_64-linux-gnu/cmake/`, not the path
PyTorch expects. Also, Ubuntu does not install `rocm_version.h` (from `rocm-core`) by
default and the version header must fall back to `hip_version.h`.

Apply the following two hunks:

**Hunk 1** — add the Ubuntu multi-arch cmake search path (around the `CMAKE_MODULE_PATH`
block that already handles `UNIX`):

```cmake
if(UNIX)
  # Ubuntu/Debian multi-arch installs ROCm cmake files under
  # /usr/lib/x86_64-linux-gnu/cmake/ rather than /usr/lib/cmake/
  set(CMAKE_MODULE_PATH
    ${ROCM_PATH}/lib/cmake/hip
    ${ROCM_PATH}/lib/x86_64-linux-gnu/cmake/hip
    ${CMAKE_MODULE_PATH})
else() # Win32
  set(CMAKE_MODULE_PATH ${ROCM_PATH}/cmake/ ${CMAKE_MODULE_PATH})
endif()
```

**Hunk 2** — add the `hip_version.h` fallback for `rocm_version.h` (in the
`Find ROCM version` block):

```cmake
if(UNIX)
  set(ROCM_VERSION_HEADER_PATH ${ROCM_INCLUDE_DIRS}/rocm-core/rocm_version.h)
  if(NOT EXISTS ${ROCM_VERSION_HEADER_PATH})
    set(ROCM_VERSION_HEADER_PATH ${ROCM_INCLUDE_DIRS}/hip/hip_version.h)
    set(ROCM_LIB_NAME "HIP")
  else()
    set(ROCM_LIB_NAME "ROCM")
  endif()
else()
  set(ROCM_VERSION_HEADER_PATH ${ROCM_INCLUDE_DIRS}/hip/hip_version.h)
  set(ROCM_LIB_NAME "HIP")
endif()
```

### 3c. Patch 2 — `c10/macros/Macros.h`

gfx1151 has a **32-wide wavefront** (RDNA4), not 64. HIP's `warpSize` is a runtime
struct (not a `constexpr`), so it cannot be used in constant expressions. Replace the
existing `C10_WARP_SIZE` definition block with:

```cpp
#if defined(USE_ROCM)
// In device code, __AMDGCN_WAVEFRONT_SIZE__ is the compile-time wavefront size
// (e.g. 32 for gfx1151). In host code of .hip TUs it is not defined, so fall
// back to 64 (AMD maximum). warpSize is a runtime-only struct in HIP and cannot
// be used in constexpr expressions.
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
#define C10_WARP_SIZE __AMDGCN_WAVEFRONT_SIZE__
#else
#define C10_WARP_SIZE 64
#endif
#else
#define C10_WARP_SIZE 32
#endif
```

### 3d. Patch 3 — `cmake/Dependencies.cmake`

Three changes to the HIP flags block (search for `HIP_HIPCC_FLAGS` and `HIP_CXX_FLAGS`):

**Change A** — Remove `-std=c++17` from `HIP_CXX_FLAGS`. PyTorch applies `HIP_CXX_FLAGS`
to all targets including C source files (via `torch_compile_options()`), and passing
`-std=c++17` to a `.c` file is an error. CMake already sets the C++ standard through
`CMAKE_CXX_STANDARD`; the explicit flag is redundant and harmful.

Find the line:
```cmake
list(APPEND HIP_CXX_FLAGS -std=c++17)
```
Delete it (or comment it out).

**Change B** — Disable `--offload-compress`. This flag is AMD-clang-specific and not
supported by upstream LLVM 21:

```cmake
# --offload-compress is AMD-specific and not supported by upstream LLVM clang
# list(APPEND HIP_HIPCC_FLAGS --offload-compress)
```

**Change C** — Disable `-fclang-abi-compat=17`. This flag forces C++17 name mangling in
HIP code. `libtorch_cpu.so` is compiled with upstream clang 21 which defaults to C++20
ABI. The mismatch causes undefined symbol errors at runtime (e.g. `const_data_ptr<T>`
uses different mangling in C++17 vs C++20 for constrained templates). With upstream
LLVM both sides already use C++20 ABI so the flag is not needed and actively breaks things:

```cmake
# -fclang-abi-compat=17 would force C++17 ABI in HIP code but libtorch_cpu.so is
# built with clang 21's default C++20 ABI, causing mangling mismatch on constrained
# templates (e.g. const_data_ptr<T>). With upstream LLVM both sides already
# default to C++20 ABI so the flag is not needed and is actively harmful.
# list(APPEND HIP_HIPCC_FLAGS -fclang-abi-compat=17)
```

### 3e. Patch 4 — `aten/src/ATen/CMakeLists.txt`

The `bgemm_kernels/` and `ck*.hip` files require AMD's proprietary clang fork (they use
`CK_BUFFER_RESOURCE_3RD_DWORD` and other AMD-internal intrinsics). They fail to compile
with upstream LLVM. Exclude them on non-Windows builds.

Find the `if(WIN32)` block that calls `exclude(ATen_HIP_SRCS ...)` and extend it:

```cmake
file(GLOB native_hip_bgemm "native/hip/bgemm_kernels/*.hip")
file(GLOB native_hip_ck "native/hip/ck*.hip")
if(WIN32)
  exclude(ATen_HIP_SRCS "${ATen_HIP_SRCS}"
    ${native_hip_bgemm} ${native_hip_ck}
    ${native_transformers_hip_hip} ${native_transformers_hip_cpp})
else()
  # Exclude bgemm/CK kernels: require AMD clang fork, incompatible with upstream LLVM
  exclude(ATen_HIP_SRCS "${ATen_HIP_SRCS}" ${native_hip_bgemm} ${native_hip_ck})
endif()
```

### 3f. Patch 5 — `aten/src/ATen/native/hip/ck_gemm.h`

Excluding the `.hip` files leaves the header's `extern` declarations dangling (linker
errors). Replace the four `extern` specialisations with inline stubs:

```cpp
#pragma once

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>
namespace at::native {

template <typename Dtype>
inline void gemm_internal_ck(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas_gemm_internal_ck: not implemented");
}

template <>
inline void gemm_internal_ck<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  TORCH_CHECK(false, "gemm_internal_ck<double>: Composable Kernels not compiled");
}
template <>
inline void gemm_internal_ck<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  TORCH_CHECK(false, "gemm_internal_ck<float>: Composable Kernels not compiled");
}
template <>
inline void gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  TORCH_CHECK(false, "gemm_internal_ck<Half>: Composable Kernels not compiled");
}
template <>
inline void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  TORCH_CHECK(false, "gemm_internal_ck<BFloat16>: Composable Kernels not compiled");
}

} // namespace at::native
```

### 3g. Patch 6 — `aten/src/ATen/native/hip/ck_bgemm.h`

Same treatment for the batched gemm stub:

```cpp
#pragma once

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>

namespace at::native {

template <typename Dtype>
inline void bgemm_internal_ck(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas_bgemm_internal_ck: not implemented");
}

template <>
inline void bgemm_internal_ck<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  TORCH_CHECK(false, "bgemm_internal_ck: Composable Kernels not compiled in this build");
}

} // namespace at::native
```

### 3h. Patch 7 — `third_party/composable_kernel/include/ck/config.h`

The CK build system generates this file via cmake; when CK itself is not built (because
we excluded the CK sources) cmake never generates it, but other CK headers still `#include`
it. Create it manually:

```bash
cat > ~/pytorch-src/third_party/composable_kernel/include/ck/config.h <<'EOF'
/* Generated for PyTorch ROCm build — enables all CK dtypes */
#ifndef CK_CONFIG_H_IN
#define CK_CONFIG_H_IN
#define CK_ENABLE_ALL_DTYPES 1
#define CK_ENABLE_INT8 "ON"
#define CK_ENABLE_FP8 "ON"
#define CK_ENABLE_BF8 "ON"
#define CK_ENABLE_FP16 "ON"
#define CK_ENABLE_BF16 "ON"
#define CK_ENABLE_FP32 "ON"
#define CK_ENABLE_FP64 "ON"
#define CK_USE_WMMA 1
/* #undef CK_USE_XDL */
#endif
EOF
```

### 3i. Build script

The build script is at `build_pytorch.sh` in this repo. It assumes pytorch source is
cloned to `~/pytorch-src` and that the conda environment `trellis2` is active.

```bash
conda activate trellis2
bash build_pytorch.sh
```

Build time: approximately 3–4 hours on 24 cores.

---

## 4. Build torchvision 0.22.0 from source

The pre-built torchvision ROCm wheels are compiled against torch 2.6.0 and are ABI
incompatible with the torch 2.7.0 we just built. Build from source:

```bash
git clone --branch v0.22.0 --depth 1 https://github.com/pytorch/vision ~/torchvision-src
cd ~/torchvision-src

# setuptools < 80 required (pkg_resources used in setup.py)
pip install "setuptools<80"

PYTORCH_ROCM_ARCH=gfx1151 \
  CMAKE_CXX_COMPILER=/usr/lib/llvm-21/bin/clang++ \
  CMAKE_C_COMPILER=/usr/lib/llvm-21/bin/clang \
  pip install --no-build-isolation .
```

Build time: ~5–10 minutes.

---

## 5. Install Python dependencies

From the repo root with the `trellis2` conda environment active:

```bash
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
            trimesh transformers tensorboard pandas lpips zstandard kornia timm
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
sudo apt install -y libjpeg-dev
pip install pillow-simd
```

Notes:
- **nvdiffrast is not needed.** The UV rasterisation required for GLB texture baking
  has been replaced with a pure-PyTorch implementation at
  `trellis2/utils/uv_rasterize.py` that runs on any backend.
- `transformers` is required for the BiRefNet background removal model used by
  `run_sample.py`.
- **flash-attn must be built from the ROCm fork (see section 5b below).** There is no
  pre-built gfx1151 wheel; the ROCm fork's Triton backend is used instead.

---

## 5b. Build flash-attn from ROCm fork (Triton backend)

The upstream `flash-attn` package has no gfx1151 HIP build — its HIP backend only
supports CDNA GPUs (MI series, gfx942). The ROCm fork
(`github.com/ROCm/flash-attention`) includes a Triton-based backend that **does** work
on gfx1151 (Triton 3.7.1 confirmed working). Build and install it:

### 5b-i. Clone the ROCm fork

```bash
git clone --depth 1 https://github.com/ROCm/flash-attention.git ~/flash-attn-src
cd ~/flash-attn-src
```

The aiter library (AMD AIter) is a required dependency for the Triton backend. Clone it
as a subdirectory (not a submodule):

```bash
git clone --depth 1 https://github.com/ROCm/aiter.git third_party/aiter
```

### 5b-ii. Patch 1 — triton version pin in `setup.py`

The repo pins `triton==3.5.1` but we have 3.7.1 installed (required by PyTorch 2.7.0).
Change the pin to a minimum:

```diff
-    "triton==3.5.1",
+    "triton>=3.5.1",
```

### 5b-iii. Patch 2 — HIP version parsing in aiter

Ubuntu's `hipconfig --version` outputs `HIP version: X.Y.Z` (with a prefix and possibly
clang compiler info on subsequent lines). aiter's `get_hip_version()` in
`third_party/aiter/aiter/jit/utils/cpp_extension.py` tries to do `int(version.split(".")[0])`
and fails with `ValueError: invalid literal for int() with base 10: 'HIP version: 7'`.

Apply this fix to `third_party/aiter/aiter/jit/utils/cpp_extension.py`:

```python
def get_hip_version():
    try:
        hipconfig = executable_path("hipconfig")
        output = subprocess.check_output([hipconfig, "--version"], text=True)
        # Ubuntu's hipconfig --version prepends "HIP version: " and may append
        # compiler info on subsequent lines. Extract just the version number.
        first_line = output.strip().splitlines()[0]
        if first_line.lower().startswith("hip version:"):
            first_line = first_line.split(":", 1)[1].strip()
        return first_line
    except Exception:
        raise RuntimeError("ROCm version file not found")
```

### 5b-iv. Patch 3 — torch.distributed.Backend missing in our PyTorch build

Our custom PyTorch build was built without the distributed C10d backend, so
`torch.distributed.Backend` is not available. aiter's `dist/parallel_state.py` imports
it unconditionally. Apply this fix to
`third_party/aiter/aiter/dist/parallel_state.py` (same file will also exist in the
installed package):

```diff
-from torch.distributed import Backend, ProcessGroup
+try:
+    from torch.distributed import Backend, ProcessGroup
+except ImportError:
+    Backend = None
+    ProcessGroup = None
```

**Also apply this same patch after installation** (the installed copy is at
`$CONDA_PREFIX/lib/python3.10/site-packages/aiter/dist/parallel_state.py`).

### 5b-v. Build aiter (skip Composable Kernels)

Composable Kernels (CK) requires AMD's proprietary clang fork and cannot be built with
upstream LLVM 21. Disable it with `ENABLE_CK=0`:

```bash
cd ~/flash-attn-src/third_party/aiter
ENABLE_CK=0 pip install --no-build-isolation .
```

Apply the `parallel_state.py` patch to the installed package:

```bash
AITER_DIST="$(python -c "import site; print(site.getsitepackages()[0])")/aiter/dist/parallel_state.py"
# Edit $AITER_DIST: replace the single-line import with the try/except above
```

### 5b-vi. Build flash-attn

```bash
cd ~/flash-attn-src
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
GPU_ARCHS=gfx1151 \
pip install --no-build-isolation .
```

This builds a pure-Python wheel (`py3-none-any`) in seconds; the actual Triton kernels
are JIT-compiled at first use. To confirm it works:

```bash
python -c "
import os; os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
import torch
from flash_attn import flash_attn_varlen_func
q = torch.randn(32, 8, 64, dtype=torch.float16, device='cuda')
k = torch.randn(32, 8, 64, dtype=torch.float16, device='cuda')
v = torch.randn(32, 8, 64, dtype=torch.float16, device='cuda')
cu = torch.tensor([0, 16, 32], dtype=torch.int32, device='cuda')
out = flash_attn_varlen_func(q, k, v, cu, cu, 16, 16)
print('OK', out.shape)
"
```

`run_sample.py` sets `ATTN_BACKEND=flash_attn` and `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
automatically; no further action is needed.

---

## 6. Build GPU extensions for gfx1151

All GPU extensions must be compiled for gfx1151. Pre-built wheels target gfx1100 and
will SIGSEGV at runtime (ROCm cannot dispatch gfx1100 kernels on a gfx1151 GPU).

### 6a. CuMesh

CuMesh provides mesh extraction (marching cubes / dual contouring), mesh cleanup
(`fill_holes`, deduplication), UV unwrapping (xatlas), and BVH queries (via cubvh)
during GLB baking. It is included in this repo at `CuMesh/`. The `build.sh` script
targets gfx1151 by default:

```bash
bash CuMesh/build.sh
```

Or equivalently:

```bash
GPU_ARCHS=gfx1151 pip install --no-build-isolation CuMesh/
```

Verify:
```bash
roc-obj-ls $(python -c "import cumesh, os; print(os.path.dirname(cumesh.__file__))")/_C*.so \
  | grep hipv4
# Should show lines containing: hipv4-amdgcn-amd-amdhsa--gfx1151
```

### 6b. o-voxel

o-voxel provides sparse-voxel serialization and GPU rasterisation used by the
postprocessing pipeline. It is included at `o-voxel/` and auto-detects HIP:

```bash
GPU_ARCHS=gfx1151 pip install --no-build-isolation o-voxel/
```

### 6c. FlexGEMM

FlexGEMM provides the sparse convolution backend and `grid_sample_3d` for texture
baking. It is included at `FlexGEMM/`:

```bash
GPU_ARCHS=gfx1151 pip install --no-build-isolation FlexGEMM/
```

Verify:
```bash
roc-obj-ls $(python -c "import flex_gemm.kernels.cuda as m; import os; print(m.__file__)") \
  | grep hipv4
# Should show: hipv4-amdgcn-amd-amdhsa--gfx1151
```

---

## 7. Running the pipeline

```bash
conda activate trellis2
cd /path/to/TRELLIS.2
```

**Arbitrary input photo** (background removal runs automatically via BiRefNet):
```bash
python run_sample.py --image photo.png --output out.glb
```

**Pre-processed image that already has a transparent background:**
```bash
python run_sample.py --image assets/example_image/T.png --output out.glb --no-remove-bg
```

No environment variable overrides are needed. `run_sample.py` sets
`MIOPEN_DEBUG_CONV_WINOGRAD=0`, `ATTN_BACKEND=flash_attn`, and
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` automatically.

Expected runtime on Radeon 8060S (single-GPU inference, `1024_cascade` pipeline):

| Stage | Time |
|---|---|
| Background removal (BiRefNet on GPU) | ~0:15 |
| Sparse structure sampling (12 steps) | ~1:20 |
| Shape SLat 512 pass (12 steps) | ~1:00 |
| Shape SLat 1024 pass (12 steps) | ~23:00 |
| Texture SLat (12 steps) | ~13:00 |
| Decode + GLB bake | ~3:00 |
| **Total** | **~41 minutes** |

---

## 8. Why each change was needed

| Problem | Root cause | Fix |
|---|---|---|
| `FindHIP.cmake` not found | Ubuntu puts ROCm cmake under `/usr/lib/x86_64-linux-gnu/cmake/` | Added Ubuntu multi-arch path to `CMAKE_MODULE_PATH` in `LoadHIP.cmake` |
| `rocm_version.h` not found | Ubuntu doesn't install `rocm-core` headers by default | Fall back to `hip_version.h` in `LoadHIP.cmake` |
| `device library not found` (hipcc) | With `ROCM_PATH=/usr`, hipcc looks for bitcode at `/usr/amdgcn/bitcode` which doesn't exist | Symlink `/usr/amdgcn/bitcode` → LLVM 21 bitcode dir |
| `constexpr` compile error on `C10_WARP_SIZE` | `warpSize` in HIP is a runtime struct, not a constexpr; gfx1151 uses 32-wide wavefront | Use `__AMDGCN_WAVEFRONT_SIZE__` compile-time macro |
| `.c` files fail with `-std=c++17` | `HIP_CXX_FLAGS` (containing `-std=c++17`) is applied to all targets including C files | Remove `-std=c++17` from `HIP_CXX_FLAGS` |
| `--offload-compress` error | AMD-clang-specific flag, not in upstream LLVM 21 | Comment out |
| `undefined symbol: const_data_ptr` (ABI mismatch) | `-fclang-abi-compat=17` forced C++17 name mangling in `libtorch_hip.so`; `libtorch_cpu.so` used C++20 mangling (clang 21 default) | Remove `-fclang-abi-compat=17` |
| `CK_BUFFER_RESOURCE_3RD_DWORD` undefined | `bgemm_kernels/` and `ck*.hip` use AMD-clang-only intrinsics | Exclude those files from the build |
| `undefined symbol: bgemm_internal_ck` | Header had `extern` declarations for excluded translation units | Replace with inline `TORCH_CHECK(false, ...)` stubs |
| `ck/config.h` not found | cmake-generated file only created when CK is built as a cmake project | Manually create with all dtypes enabled |
| `libamdhip64.so.6` not found | `libaotriton_v2.so` bundled in ROCm wheels was compiled against ROCm 6.x | Symlink `.so.6` → actual ROCm 7.1 `.so.7` |
| torchvision ABI mismatch | Pre-built wheel compiled against torch 2.6.0 | Rebuild from source at v0.22.0 |
| `pkg_resources` missing (torchvision build) | `setuptools ≥ 80` removed `pkg_resources` | Pin `setuptools < 80` |
| Segfault in decode / mesh extraction | cumesh/o-voxel/FlexGEMM compiled for gfx1100, ROCm cannot dispatch on gfx1151 | Rebuild all GPU extensions with `GPU_ARCHS=gfx1151` |
| `miopenStatusUnknownError` during BiRefNet GPU inference | MIOpen probes Winograd convolution when benchmarking new shapes; the Winograd kernel assembly (`Conv_Winograd_v30_3_1_fp32_f3x2_stride1.s`) is missing for gfx1151, causing the entire forward pass to raise an exception | Set `MIOPEN_DEBUG_CONV_WINOGRAD=0` (done in `run_sample.py`) |
| nvdiffrast unavailable on ROCm | nvdiffrast has no ROCm backend | Replaced `dr.rasterize` / `dr.interpolate` calls in `postprocess.py` and `trellis2_texturing.py` with `trellis2/utils/uv_rasterize.py` (pure PyTorch) |
| `flash_attn` unavailable on gfx1151 (pre-built) | `flash-attn`'s HIP backend only supports CDNA GPUs (gfx942 / MI series); it has no pre-built gfx1151 wheel | Build the ROCm fork with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` — Triton JIT-compiles the kernels for gfx1151 at first use (see section 5b) |
| `ValueError: invalid literal for int()` during aiter build | Ubuntu's `hipconfig --version` prints `"HIP version: X.Y.Z"` with a prefix; aiter's `get_hip_version()` passes the full line to `int()` | Patch `get_hip_version()` to strip the `"HIP version: "` prefix and take only the first line |
| `ImportError: cannot import name 'Backend' from 'torch.distributed'` | Our custom PyTorch was built without the distributed C10d backend; aiter imports `Backend` unconditionally | Wrap the import in a `try/except` in `aiter/dist/parallel_state.py` (patch both source and installed copy) |
| Segfault with `HSA_OVERRIDE_GFX_VERSION=11.0.0` | After rebuilding extensions for gfx1151, the override causes ROCm to look for gfx1100 kernels that no longer exist | Remove `HSA_OVERRIDE_GFX_VERSION` entirely (not set in `run_sample.py`) |
