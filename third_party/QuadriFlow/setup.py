# Builds the in-process pybind11/torch extension for QuadriFlow (bindings/quadriflow_binding.cpp),
# mirroring CuMesh/setup.py's CPU-only xatlas extension block exactly -- same
# CUDAExtension/BuildExtension pattern works fine for pure-CPU sources (no .cu files here).
# See BUILD_NOTES.md for how the original CLI binary (build/quadriflow, via CMakeLists.txt) is
# built -- that path is untouched; this is an additional entry point sharing the same src/*.cpp.
#
# Needs Eigen and Boost headers (both header-only usage here -- confirmed by reading
# CMakeLists.txt's target_link_libraries, which links TBB/LEMON/GUROBI but never Boost) and a
# prebuilt LEMON static library. The existing CMake build (build/3rd/lemon-1.3.1/lemon/libemon.a)
# is reused directly rather than re-implementing LEMON's own build here -- run the CMake build
# first (BUILD_NOTES.md) if that file doesn't exist yet.
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CONDA_PREFIX = os.environ.get("CONDA_PREFIX")

lemon_lib_dir = os.path.join(ROOT, "build/3rd/lemon-1.3.1/lemon")
if not os.path.isfile(os.path.join(lemon_lib_dir, "libemon.a")):
    raise RuntimeError(
        f"{lemon_lib_dir}/libemon.a not found -- build the CLI binary first via CMake "
        f"(see BUILD_NOTES.md), which also produces the LEMON static library this extension "
        f"links against.")

include_dirs = [
    os.path.join(ROOT, "src"),
    os.path.join(ROOT, "3rd/pcg32"),
    os.path.join(ROOT, "3rd/pss"),
    os.path.join(ROOT, "3rd/lemon-1.3.1"),
    os.path.join(ROOT, "build/3rd/lemon-1.3.1"),  # generated lemon/config.h
]
if CONDA_PREFIX:
    include_dirs += [
        os.path.join(CONDA_PREFIX, "include/eigen3"),
        os.path.join(CONDA_PREFIX, "include"),
    ]

sources = [
    "src/adjacent-matrix.cpp",
    "src/dedge.cpp",
    "src/hierarchy.cpp",
    "src/loader.cpp",
    "src/localsat.cpp",
    "src/merge-vertex.cpp",
    "src/optimizer.cpp",
    "src/parametrizer.cpp",
    "src/parametrizer-flip.cpp",
    "src/parametrizer-int.cpp",
    "src/parametrizer-mesh.cpp",
    "src/parametrizer-scale.cpp",
    "src/parametrizer-sing.cpp",
    "src/subdivide.cpp",
    "bindings/quadriflow_binding.cpp",
]

ext_modules = [
    CUDAExtension(
        name="quadriflow_ext",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=[lemon_lib_dir],
        libraries=["emon"],  # links libemon.a -- yes, "emon" not "lemon" (see BUILD_NOTES.md)
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-Wno-int-in-bool-context", "-Wno-sign-compare"],
        },
    ),
]

setup(
    name="quadriflow_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
