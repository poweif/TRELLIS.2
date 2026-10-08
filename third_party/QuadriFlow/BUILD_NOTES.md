# Building QuadriFlow in this repo

Vendored from https://github.com/hjwdzh/QuadriFlow (BSD-style license, see `LICENSE.txt`),
source only -- no `.git`, no `examples/`/`img/` (not needed to build or run).

## Why not just `apt install libeigen3-dev libboost-all-dev`

That's the normal way, but it needs `sudo`, which isn't always available (it wasn't in the
environment this was set up in). Both dependencies can be pulled in locally via conda-forge
instead, with no root access needed:

```bash
conda create -n quadriflow-build -c conda-forge eigen boost boost-cpp cmake -y
```

Eigen is header-only so this is trivial for it; conda-forge's `boost` package ships prebuilt
binaries so this avoids a from-source Boost build too.

## Build

```bash
ENV=~/miniconda3/envs/quadriflow-build   # or wherever the env above landed
cd third_party/QuadriFlow
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBoost_USE_STATIC_LIBS=OFF \
  -DEIGEN_INCLUDE_DIR_HINTS="$ENV/include/eigen3" \
  -DBOOST_ROOT="$ENV" \
  -DBoost_NO_SYSTEM_PATHS=ON
make -j$(nproc)
```

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` works around QuadriFlow's `cmake_minimum_required(VERSION 3.1)`
being older than modern CMake (>=4.0) will accept without it.

The resulting `build/quadriflow` binary is fully static w.r.t. its build-time dependencies --
`ldd` shows only `libstdc++`/`libm`/`libgcc_s`/`libc` (standard system libs), nothing from the
conda env. It does **not** need the `quadriflow-build` env (or Eigen/Boost) present at runtime,
only at build time -- safe to build once and keep the env or delete it afterward.

## In-process Python bindings (quadriflow_ext)

`bindings/quadriflow_binding.cpp` + `setup.py` build an additional entry point -- a pybind11/
torch extension callable directly from Python (`quadriflow_ext.remesh(vertices, faces,
target_faces, rho=..., adaptive=...)`), avoiding the temp-file round trip the CLI binary above
requires. It mirrors the pattern already used in this project by
`CuMesh/third_party/xatlas/binding.cpp` (a CPU-only C++ library wrapped the same way). See
`trellis2/utils/quad_remesh.py`, which prefers this binding when available and falls back to the
CLI/subprocess path above otherwise, and `docs/archive/quadriflow_postprocess_plan.md`'s adaptive-density
section for why this exists (the CLI has no way to pass a per-vertex sizing field at all -- the
binding's `rho` argument is the whole point).

This does **not** replace the CLI binary/CMake build above -- both share the same `src/*.cpp`,
built two different ways for two different purposes. Build the CLI first (its CMake build also
produces the LEMON static library this extension links against):

```bash
# 1. Build the CLI binary first (see "Build" above) -- produces
#    build/3rd/lemon-1.3.1/lemon/libemon.a, which the extension links against directly rather
#    than re-running LEMON's own CMake subproject build a second time.

# 2. Eigen/Boost headers (both header-only usage here) into the SAME conda env that has
#    torch/pybind11 (do not cross-reference the quadriflow-build env above -- keep this
#    self-contained in whichever env actually runs the Python pipeline):
conda install -c conda-forge eigen boost -y

# 3. Build the extension (in-place, importable directly from this directory):
cd third_party/QuadriFlow
python setup.py build_ext --inplace
```

One gotcha hit while building this: the vendored `src/loader.cpp` had a mis-encoded copyright
symbol (Windows-1252 `©`, not valid UTF-8) in a comment. Harmless for the CMake/g++ build above,
but `torch.utils.cpp_extension`'s HIPIFY preprocessing (this project's PyTorch is a ROCm build --
`CUDAExtension` always runs source through hipify first, even for plain CPU sources with no CUDA/
HIP code at all) reads every source file as UTF-8 and crashed on that byte. Fixed in place
(`iconv -f WINDOWS-1252 -t UTF-8`) -- a one-time, mechanical fix, not something to reintroduce if
this file is ever re-vendored from upstream.

Also note: the linked library is `-lemon` (`libraries=["emon"]` in `setup.py`), not `-llemon` --
the actual built artifact is named `libemon.a` (a quirk of the vendored `3rd/lemon-1.3.1`
CMakeLists.txt, not a typo introduced here).

## Known input requirement

QuadriFlow needs a genuinely 2-manifold input mesh (no non-manifold edges/vertices) -- it does
*not* need to be watertight/closed, open boundaries are fine, but even a handful of non-manifold
edges makes it fail outright (observed: "wrong init" error, no output file, at the "Solve index
map" stage). `trellis2/utils/mesh_repair.py`'s `repair_mesh()` (built for an unrelated reason
earlier in this project) happens to produce exactly the required cleanliness -- see
`trellis2/utils/quad_remesh.py` for the wrapper that uses it.
