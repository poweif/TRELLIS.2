#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

cd ~/pytorch-src

# ROCm 7.1 is installed into /usr (Ubuntu package layout).
# hipblaslt/rocm-core/roctracer installed from ROCm 7.2.3 repo into /opt/rocm-7.2.3.
export ROCM_PATH=/usr
export HIP_PATH=/usr
export HIP_ROOT_DIR=/usr
export ROCM_HOME=/usr

# ROCM_INCLUDE_DIRS: LoadHIP.cmake reads this from ENV to find rocm_version.h and
# hipblaslt headers. Point at the 7.2.3 prefix where those files live.
export ROCM_INCLUDE_DIRS=/opt/rocm-7.2.3/include

# Only compile for gfx1151 — cuts build time significantly
export PYTORCH_ROCM_ARCH="gfx1151"

# Use ROCm, not CUDA
export USE_ROCM=1
export USE_CUDA=0
export USE_CUDNN=0
export USE_NCCL=0

# Disable components not needed for single-GPU inference
export USE_TENSORPIPE=0   # clang 21 incompatible; only needed for RPC/distributed
export USE_FBGEMM=0       # -Werror=maybe-uninitialized on clang 21
export USE_KINETO=0       # profiler; not needed
export USE_DISTRIBUTED=0
export BUILD_TEST=0        # skip test binaries; only the wheel matters  # disables gloo (clang 21 uint8_t errors); not needed

# cmake prefix path: /usr/lib/x86_64-linux-gnu for Ubuntu ROCm 7.1 cmake configs,
# /opt/rocm-7.2.3 for hipblaslt/rocm-core/roctracer cmake configs
export CMAKE_PREFIX_PATH="/opt/rocm-7.2.3:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"

# Ubuntu multi-arch: FindHIP.cmake is at /usr/lib/x86_64-linux-gnu/cmake/hip/
# LoadHIP.cmake (patched) also adds ${ROCM_PATH}/lib/x86_64-linux-gnu/cmake/hip
export CMAKE_MODULE_PATH="/usr/lib/x86_64-linux-gnu/cmake/hip"

# Fix clang 21 / Ubuntu 26.04: uint8_t not declared without explicit #include <cstdint>
export CMAKE_CXX_FLAGS="-include cstdint -include cstddef"

export MAX_JOBS=24
export HIPCC=/usr/bin/hipcc

# /usr/bin/clang++ is actually clang-17 (LLVM 17). The ROCm 7.1 HIP stack
# (hipconfig -l) and rocm-device-libs-21 both expect LLVM 21.
# Pointing cmake at the real clang 21 fixes HIP compilation.
export CMAKE_CXX_COMPILER=/usr/lib/llvm-21/bin/clang++
export CMAKE_C_COMPILER=/usr/lib/llvm-21/bin/clang

# Clear stale cmake cache that had USE_ROCM:BOOL=OFF
if [ -f build/CMakeCache.txt ]; then
    echo "[pytorch build] clearing stale cmake cache..."
    rm -f build/CMakeCache.txt build/CMakeFiles/CMakeError.log build/CMakeFiles/CMakeOutput.log
fi

# Hipify: converts CUDA source files to HIP. Must run before cmake/pip install.
# Generates c10/hip/impl/, aten/src/ATen/hip/, etc.
echo "[pytorch build] running hipify at $(date)"
python tools/amd_build/build_amd.py 2>&1 | tee ~/pytorch_hipify.log
echo "[pytorch build] hipify done at $(date)"

echo "[pytorch build] started at $(date)"
pip install --no-build-isolation . 2>&1 | tee ~/pytorch_build.log
echo "[pytorch build] finished at $(date) — exit $?"
