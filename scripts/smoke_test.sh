#!/usr/bin/env bash
# Quick local check that the fork's pieces still import and run. Needs the trellis2 conda env
# (ROCm stack + native extensions); does no GPU-heavy work and downloads no models.
#   scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== CPU tests"
python -m pytest tests -q

echo "== CLI --help"
for cli in tools/*.py; do
    python "$cli" --help > /dev/null
    echo "ok  $cli"
done
python run_sample.py --help > /dev/null && echo "ok  run_sample.py"

echo "== o_voxel imports without trellis2 on the path"
(cd /tmp && python -c "import o_voxel, o_voxel.uv_rasterize")
echo "ok  o_voxel"

echo "== native extensions"
python - <<'PY'
import cumesh, flex_gemm, o_voxel
print("ok  cumesh, flex_gemm, o_voxel")
from trellis2.utils.quad_remesh import quadriflow_available, quadriflow_ext_available
from trellis2.utils.alpha_wrap import alpha_wrap_available, alpha_wrap_ext_available
# Each tool needs either its in-process binding or its CLI binary (see the BUILD_NOTES).
for name, ext, any_ in (("QuadriFlow", quadriflow_ext_available(), quadriflow_available()),
                        ("Alpha Wrap", alpha_wrap_ext_available(), alpha_wrap_available())):
    print(f"{'ok ' if any_ else '-- '} {name}: binding={'yes' if ext else 'no'}, usable={'yes' if any_ else 'no'}")
PY
echo "smoke test passed"
