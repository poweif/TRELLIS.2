# Builds the in-process pybind11/torch extension for Alpha Wrap (bindings/alpha_wrap_binding.cpp),
# mirroring third_party/QuadriFlow/setup.py's pattern exactly. See BUILD_NOTES.md for the
# original CLI binary build (via CMakeLists.txt) -- that path is untouched.
#
# Needs CGAL (header-only), Eigen and Boost headers (both header-only usage), and GMP/MPFR
# (CGAL's exact-arithmetic backend -- real linked libraries, not header-only). Install all of
# these into the SAME conda env that has torch/pybind11 (do not cross-reference a throwaway
# build env):
#   conda install -c conda-forge cgal-cpp gmp mpfr -y
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
if not CONDA_PREFIX:
    raise RuntimeError("CONDA_PREFIX not set -- activate the conda env this builds against first.")

include_dirs = [
    os.path.join(CONDA_PREFIX, "include"),
    os.path.join(CONDA_PREFIX, "include/eigen3"),
]
library_dirs = [
    os.path.join(CONDA_PREFIX, "lib"),
    # Needed when building with the GCC 12 workaround (see BUILD_NOTES.md) -- that toolchain's
    # linker doesn't automatically search the system's multiarch lib dir the way the default
    # compiler does, and libamdhip64.so (needed by torch's ROCm build) lives there. Harmless to
    # include unconditionally.
    "/usr/lib/x86_64-linux-gnu",
]

ext_modules = [
    CUDAExtension(
        name="alpha_wrap_ext",
        sources=["bindings/alpha_wrap_binding.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["gmpxx", "gmp", "mpfr"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-frounding-math"],
        },
    ),
]

setup(
    name="alpha_wrap_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
