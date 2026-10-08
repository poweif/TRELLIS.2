# Building the Alpha Wrap tool in this repo

A minimal standalone CLI (`src/alpha_wrap.cpp`) wrapping CGAL's `Alpha_wrap_3` package --
built the same way `third_party/QuadriFlow` is: a small self-contained binary, called via
subprocess, not a Python binding. See `quadriflow_postprocess_plan.md`'s Phase 6 section
for why this exists: real-world test assets (`tests/*.glb`) turn out to be severely
perforated (thousands of pinhole-scale gaps per connected component, most not fillable by
`trimesh.repair.fill_holes()` or MeshLab's `meshing_close_holes`), which starves
QuadriFlow of usable surface area (as little as 1-27% coverage). Alpha Wrap sidesteps this
entirely -- it takes an arbitrary triangle *soup* (no manifoldness/watertightness
required) and produces a guaranteed watertight, 2-manifold, self-intersection-free surface
that strictly contains the input, regardless of how broken the input's topology is.

## Why not a generic mesh-repair library

Two other tools were tried first and both failed to fix the actual problem (see
`quadriflow_postprocess_plan.md`'s Phase 6 section for the full account): `fill_holes()`
tools try to *patch* an existing, mostly-valid boundary polygon -- many of these assets'
"holes" are topologically degenerate (2-vertex boundary loops, too few edges to even form
a triangle), so patching-based tools have nothing valid to patch. Alpha Wrap never tries
to parse or patch the existing topology at all -- it discards it and reconstructs a
wrapping surface from the raw geometry, which is why it doesn't care that the input is
riddled with degenerate gaps.

## Getting CGAL (conda-forge, no `sudo` needed)

Same reasoning as QuadriFlow's own BUILD_NOTES.md: pull build dependencies from
conda-forge into a throwaway env rather than `apt install`.

```bash
conda create -n cgal-build -c conda-forge cgal-cpp cmake gxx=12 gcc=12 -y
```

**Use GCC 12, not the system compiler.** This environment's system GCC (15.x) fails to
compile CGAL 5.6.1's `boost::graph` iterator code (`this->base()` lookup failure in a
CRTP template, likely a genuine incompatibility between that CGAL/Boost.iterator
generation and newer GCC's stricter two-phase lookup). CGAL 6.2 does compile with the
system compiler, but installing `cgal-cpp=6.2` alongside a compatible modern Eigen pulls
in `libboost>=1.90`, and CGAL 6.2 itself was not tested for the crash below (5.6.1 was
what got debugged) -- GCC 12 from conda-forge sidesteps the whole question and is known
to work with cgal-cpp 5.6.1 (and equally with 6.2, both built successfully this way in
this session).

**Eigen version note**: if you use `cgal-cpp` versions bundled with newer Eigen, CGAL's
own bundled `FindEigen3.cmake` may fail with "found unsuitable version '..'" -- newer
Eigen (3.5+/"Eigen5") moved its `EIGEN_WORLD_VERSION`/`EIGEN_MAJOR_VERSION`/
`EIGEN_MINOR_VERSION` macros from `Eigen/src/Core/util/Macros.h` (where CGAL's Find
module looks) to a new `Eigen/Version` header. Either pin `eigen=3.4` (old macro
location, works with CGAL 5.6.1) or use a CGAL version whose conda-forge package pulls a
compatible Eigen automatically (CGAL 6.2 pulls Eigen 5.x correctly via its own
`eigen-abi` version constraint, no manual pin needed).

## Build

```bash
ENV=~/miniconda3/envs/cgal-build   # or wherever the env above landed
cd third_party/AlphaWrap
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_PREFIX_PATH="$ENV" \
  -DCMAKE_C_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-g++" \
  -DBOOST_ROOT="$ENV"
make -j$(nproc)
```

## Known pitfall already hit and fixed: `std::array` face container segfaults

The first working build **segfaulted on every input, including a trivial 320-face
icosphere**, inside `CGAL::Alpha_wrap_3::internal::...::insert_bbox_corners()` --
confirmed via `gdb` backtrace, reproduced identically across both CGAL 5.6.1 and 6.2, and
initially (wrongly) suspected to be a CGAL kernel-robustness issue (fixed by switching
`Simple_cartesian<double>` to `Exact_predicates_inexact_constructions_kernel`, which
turned out not to be the actual fix either -- same crash persisted).

**Actual root cause**: `read_polygon_soup`'s documented `PolygonRange` concept requires
the inner per-face container to be a `BackInsertionSequence` (i.e. resizable, supports
`push_back`) -- the first version of `alpha_wrap.cpp` used
`std::vector<std::array<std::size_t, 3>>` for faces, and `std::array` is fixed-size and
does not satisfy that concept. This compiled without error and even printed plausible
point/face counts after reading, but silently violated the type contract in a way that
corrupted state used later inside `alpha_wrap_3`'s own triangulation setup. **Fix**: use
`std::vector<std::vector<std::size_t>>` for the face range, exactly matching the
documented concept. This is the reason `src/alpha_wrap.cpp` uses that container type --
don't "simplify" it back to a fixed-size array, it will silently reintroduce the crash.

## In-process Python bindings (alpha_wrap_ext)

`bindings/alpha_wrap_binding.cpp` + `setup.py` build an additional entry point -- a pybind11/
torch extension (`alpha_wrap_ext.wrap(vertices, faces, alpha, offset)`) callable directly from
Python, avoiding the temp-file round trip the CLI binary above requires. Mirrors
`third_party/QuadriFlow/bindings/`'s pattern exactly. See `trellis2/utils/alpha_wrap.py`, which
prefers this binding when available and falls back to the CLI/subprocess path otherwise.

Does **not** replace the CLI binary/CMake build above -- both share `src/alpha_wrap.cpp`'s core
logic (the binding re-implements the same `read soup -> alpha_wrap_3 -> mesh` call, just against
in-memory tensors instead of files), built two different ways.

```bash
# 1. CGAL, Eigen, Boost headers + GMP/MPFR (real linked libs, not header-only) into the SAME
#    conda env that has torch/pybind11:
conda install -c conda-forge cgal-cpp gmp mpfr -y

# 2. Build (use an ABSOLUTE path for gcc12_wrapper -- ninja's build step doesn't reliably
#    resolve a relative PATH entry added this way):
cd third_party/AlphaWrap
PATH="$(pwd)/gcc12_wrapper:$PATH" CXX=g++ CC=gcc python setup.py build_ext --inplace
```

**Why the `gcc12_wrapper` prefix is needed**: conda-forge's default `cgal-cpp` is 5.6.1, which
(same as the CLI binary above) fails to compile on this environment's system GCC (15.x) --
`boost::graph` iterator CRTP code, `this->base()` lookup failure. The CLI build fixes this with a
whole separate GCC-12 conda env (`cgal-build`/`cgal-build2`, see above); for the Python extension,
`gcc12_wrapper/g++` is a thin shim that calls that same GCC 12 compiler
(`~/miniconda3/envs/cgal-build2/bin/x86_64-conda-linux-gnu-g++` -- adjust the path in the script if
that env is ever recreated elsewhere) while stripping `-I/usr/include`/`-isystem /usr/include`
from the argument list first. Without that stripping, mixing GCC 12's own bundled libstdc++
headers with the system's `/usr/include` glibc pthread headers produces a
`__gthread_cond_t`/`__GTHREAD_COND_INIT` type-conflict compile error (torch's `cpp_extension`
auto-injects `-I/usr/include`, which the CLI's CMake build never does, so this only bites the
Python-binding path). `setup.py` also appends `/usr/lib/x86_64-linux-gnu` to `library_dirs`
unconditionally -- GCC 12's linker doesn't search the system multiarch lib dir by default the way
the system toolchain does, and `libamdhip64.so` (needed by this project's ROCm PyTorch build)
lives there.

Two alternatives considered and rejected: upgrading to `cgal-cpp=6.2` (compiles fine with the
system compiler per the CLI build's own notes above) hit a dependency-solver conflict pulling in
an incompatible newer `libboost-headers` pin in this env -- not pursued further once the GCC 12
wrapper approach worked. Passing `CXX`/`CC` straight to GCC 12 without the wrapper's flag-stripping
was tried first and hit the pthread header conflict described above.

## Usage

```
./build/alpha_wrap <input.obj> <alpha> <offset> <output.obj>
```

`alpha` controls the level of detail captured (smaller = finer, follows the input more
closely, more output faces); `offset` controls how far the wrap surface sits from the
input (should be noticeably smaller than `alpha`, roughly `alpha/30` worked well in
testing). Both should scale with the mesh's own size -- as a starting point, try
`alpha = bbox_diagonal / 40` and `offset = bbox_diagonal / 1200`, then adjust based on how
much fine detail (e.g. separated fingers, thin gaps) needs to survive.

**Verified on all 4 of this project's test assets** (see `quadriflow_postprocess_plan.md`
Phase 6): produces a genuinely watertight mesh in 2-5 seconds even on ~450K-face inputs,
recovering 88.9%-97.6% of the reference's true surface area (versus 1-27% without it) --
and, critically, preserves fine separated structure correctly (hand's 5 fingers stay
distinctly separated, not fused into a mitten shape).

## Runtime dependency

Unlike QuadriFlow's binary (fully static w.r.t. build deps), this binary dynamically
links `libgmpxx`/`libmpfr`/`libgmp`/`libstdc++`/`libgcc_s` (`ldd` confirms) -- CGAL uses
GMP/MPFR for exact arithmetic internally. Rather than depending on the throwaway
`cgal-build` conda env staying around, the 5 needed `.so` files are copied into
`third_party/AlphaWrap/runtime_libs/` -- confirmed working standalone via
`env -i ... LD_LIBRARY_PATH=.../runtime_libs alpha_wrap ...` with no conda env active at
all. `trellis2/utils/alpha_wrap.py`'s `alpha_wrap_mesh()` sets `LD_LIBRARY_PATH` to this
directory automatically when invoking the binary.

`runtime_libs/` is **gitignored** (like `build/`): the libs are host-specific and useless
without the locally built binary. After building, populate it from the same env used for
the CMake build above:

```bash
ENV=~/miniconda3/envs/cgal-build   # same env as the Build step
cd third_party/AlphaWrap
mkdir -p runtime_libs
cp -L "$ENV"/lib/{libgmpxx.so.4,libgmp.so.10,libmpfr.so.6,libstdc++.so.6,libgcc_s.so.1} runtime_libs/
ldd build/alpha_wrap   # confirm these 5 sonames match what the binary actually links
```

If you rebuild with a different CGAL/GCC version, re-copy these files (check `ldd` for any
soname changes).

