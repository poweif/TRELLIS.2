#!/usr/bin/env bash
# Build CuMesh for AMD Strix Halo (gfx1151 / RDNA3.5).
#
# Usage:
#   cd CuMesh && bash build.sh
#
# By default targets gfx1151 natively. If the ROCm toolchain does not
# recognise gfx1151, set GPU_ARCHS=gfx1100 and export
# HSA_OVERRIDE_GFX_VERSION=11.0.0 at runtime instead:
#   GPU_ARCHS=gfx1100 bash build.sh
#
# For CUDA machines the script is a no-op override; GPU_ARCHS is ignored
# by setup.py when IS_HIP_EXTENSION is false.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to gfx1151 (native Strix Halo target) unless caller overrides.
export GPU_ARCHS="${GPU_ARCHS:-gfx1151}"

echo "[CuMesh] Building with GPU_ARCHS=${GPU_ARCHS}"

pip install "${SCRIPT_DIR}" --no-build-isolation

echo "[CuMesh] Build complete."
