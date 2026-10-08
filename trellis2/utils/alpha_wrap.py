import os
import subprocess
import sys

import numpy as np
import torch
import trimesh

_ALPHA_WRAP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'AlphaWrap'))
_ALPHA_WRAP_BIN = os.path.join(_ALPHA_WRAP_DIR, 'build', 'alpha_wrap')
_ALPHA_WRAP_LIB_DIR = os.path.join(_ALPHA_WRAP_DIR, 'runtime_libs')

_alpha_wrap_ext = None
_alpha_wrap_ext_load_attempted = False


def _load_alpha_wrap_ext():
    # In-process pybind11/torch binding (third_party/AlphaWrap/bindings/alpha_wrap_binding.cpp),
    # preferred over the CLI/subprocess path below -- same pattern as quad_remesh.py's
    # _load_quadriflow_ext, avoids a temp-file round trip per call.
    global _alpha_wrap_ext, _alpha_wrap_ext_load_attempted
    if _alpha_wrap_ext_load_attempted:
        return _alpha_wrap_ext
    _alpha_wrap_ext_load_attempted = True
    if _ALPHA_WRAP_DIR not in sys.path:
        sys.path.insert(0, _ALPHA_WRAP_DIR)
    try:
        import alpha_wrap_ext as _ext
        _alpha_wrap_ext = _ext
    except ImportError:
        _alpha_wrap_ext = None
    return _alpha_wrap_ext


def alpha_wrap_ext_available() -> bool:
    return _load_alpha_wrap_ext() is not None


def alpha_wrap_available() -> bool:
    return os.path.isfile(_ALPHA_WRAP_BIN) and os.access(_ALPHA_WRAP_BIN, os.X_OK)


def alpha_wrap_mesh(mesh: trimesh.Trimesh, alpha: float = None, offset: float = None,
                     alpha_rel: float = 1 / 40, offset_rel: float = 1 / 1200) -> trimesh.Trimesh:
    """
    Reconstruct a genuinely watertight, 2-manifold, self-intersection-free surface that
    strictly contains `mesh`, via CGAL's Alpha_wrap_3 (third_party/AlphaWrap, see
    BUILD_NOTES.md there for how it's built and docs/archive/quadriflow_postprocess_plan.md's Phase 6
    section for why this exists).

    Unlike hole-filling/patching approaches (trimesh.repair.fill_holes, MeshLab's
    meshing_close_holes -- both tried and found ineffective here, see the plan doc),
    Alpha Wrap never tries to parse or patch the input's existing topology at all: it
    takes an arbitrary triangle soup (no manifoldness or watertightness required,
    however fragmented/perforated) and reconstructs a wrapping surface from scratch. This
    is the right tool for real-world scan/reconstruction meshes that turn out to be
    severely perforated (thousands of pinhole-scale, often topologically degenerate gaps
    per connected component) -- confirmed on all 4 of this project's test assets:
    recovers 88.9%-97.6% of the reference's true surface area (vs. 1-27% feeding the
    fragmented mesh directly to QuadriFlow), runs in 2-5 seconds even on ~450K-face
    inputs, and preserves fine separated structure correctly (e.g. a hand's 5 fingers
    stay distinctly separated rather than fusing into a mitten shape).

    alpha/offset: CGAL's own tuning parameters -- alpha controls the level of detail
    captured (smaller = finer, more output faces), offset controls how far the wrap
    surface sits from the input (should be noticeably smaller than alpha). If not given
    explicitly, both are derived from the mesh's own bounding-box diagonal
    (alpha_rel/offset_rel fractions of it) -- alpha_rel=1/40, offset_rel=1/1200 were
    what worked well in testing across all 4 test assets, not universally "correct"
    values; tune per-asset if fine detail is being lost or over-preserved.
    """
    if alpha is None or offset is None:
        diag = float(np.linalg.norm(mesh.extents))
        if alpha is None:
            alpha = diag * alpha_rel
        if offset is None:
            offset = diag * offset_rel

    ext = _load_alpha_wrap_ext()
    if ext is not None:
        v = torch.as_tensor(np.ascontiguousarray(mesh.vertices, dtype=np.float32))
        f = torch.as_tensor(np.ascontiguousarray(mesh.faces, dtype=np.int32))
        out_v, out_f = ext.wrap(v, f, float(alpha), float(offset))
        return trimesh.Trimesh(vertices=out_v.numpy(), faces=out_f.numpy(), process=False)

    if not alpha_wrap_available():
        raise RuntimeError(
            f"alpha_wrap binary not found at {_ALPHA_WRAP_BIN} -- build it first, "
            f"see third_party/AlphaWrap/BUILD_NOTES.md")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.obj")
        out_path = os.path.join(tmp, "out.obj")
        mesh.export(in_path)

        env = os.environ.copy()
        if os.path.isdir(_ALPHA_WRAP_LIB_DIR):
            env["LD_LIBRARY_PATH"] = _ALPHA_WRAP_LIB_DIR + os.pathsep + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(
            [_ALPHA_WRAP_BIN, in_path, str(alpha), str(offset), out_path],
            capture_output=True, text=True, env=env)
        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"alpha_wrap failed to produce output (exit {result.returncode}). "
                f"stdout: {result.stdout.strip()!r} stderr: {result.stderr.strip()!r}")

        return trimesh.load(out_path, force='mesh')
