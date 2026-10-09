#!/usr/bin/env bash
# Quick local check that the fork's pieces import: the ROCm torch, the native extensions, and
# trellis2 as an installed package (docs/amd_rocm.md 6d). No GPU-heavy work, no model downloads.
#   scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== ROCm torch and native extensions"
python - <<'PY'
import torch
assert torch.version.hip, "torch is not a ROCm build"
import cumesh, flex_gemm, o_voxel, o_voxel.uv_rasterize
print(f"ok  torch {torch.__version__} (HIP {torch.version.hip}), cumesh, flex_gemm, o_voxel")
PY

echo "== trellis2 and o_voxel import from outside the repo"
(cd /tmp && python -c "import trellis2.pipelines, o_voxel.postprocess")
echo "ok  trellis2, o_voxel"
echo "smoke test passed"
