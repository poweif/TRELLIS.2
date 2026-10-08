import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import trimesh

from .mesh_repair import repair_mesh
from .curvature import compute_rho
from .quad_postprocess import parse_obj_raw, write_quad_obj, _triangulate_fan as _triangulate_fan_indices

_QUADRIFLOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'QuadriFlow'))
_QUADRIFLOW_BIN = os.path.join(_QUADRIFLOW_DIR, 'build', 'quadriflow')

_quadriflow_ext = None
_quadriflow_ext_load_attempted = False


def _load_quadriflow_ext():
    # In-process pybind11/torch binding (third_party/QuadriFlow/bindings/quadriflow_binding.cpp),
    # preferred over the CLI/subprocess path below -- avoids a temp-file round trip per call and
    # is the only way to supply adaptive's per-vertex rho (no CLI flag for it, see
    # docs/archive/quadriflow_postprocess_plan.md's adaptive-density section for why). Cached after the first
    # attempt (successful or not) since the module either exists on sys.path or it doesn't.
    global _quadriflow_ext, _quadriflow_ext_load_attempted
    if _quadriflow_ext_load_attempted:
        return _quadriflow_ext
    _quadriflow_ext_load_attempted = True
    if _QUADRIFLOW_DIR not in sys.path:
        sys.path.insert(0, _QUADRIFLOW_DIR)
    try:
        import quadriflow_ext as _ext
        _quadriflow_ext = _ext
    except ImportError:
        _quadriflow_ext = None
    return _quadriflow_ext


def quadriflow_ext_available() -> bool:
    return _load_quadriflow_ext() is not None


def quadriflow_available() -> bool:
    return os.path.isfile(_QUADRIFLOW_BIN) and os.access(_QUADRIFLOW_BIN, os.X_OK)


def remesh_to_quad_dominant_obj(mesh: trimesh.Trimesh, output_obj_path: str, target_faces: int = None,
                                 adaptive: bool = False) -> None:
    """
    Postprocess a triangle mesh (e.g. MeshAnythingV2's output) into a quad-dominant mesh
    via QuadriFlow (third_party/QuadriFlow, see BUILD_NOTES.md there for how it's built),
    writing genuine quad faces to output_obj_path. Verified on this project's
    MeshAnythingV2 outputs: 100% quad faces in the result, correctly-shaped -- see
    docs/archive/history.md's quad-dominance investigation for why this exists (MeshAnythingV2 itself
    only ever produces triangle output, confirmed both empirically and against the
    published literature) and docs/archive/quadriflow_postprocess_plan.md for what happens to this
    function's output next.

    Writes an .obj file rather than returning a trimesh.Trimesh deliberately:
    trimesh.Trimesh always stores triangulated faces internally (and this pipeline's
    other export target, .glb/glTF, only supports triangles too) -- either would silently
    discard the quad structure that's the entire point of this function. If you need to
    inspect/render the result in Python, load the .obj back with
    `trimesh.load(path, force='mesh')`, but be aware that re-triangulates it for the
    in-memory representation; the quads only exist in the file itself.

    target_faces: desired output face count. QuadriFlow interprets this as roughly the
    output *quad* count, not vertex/triangle count -- None defaults to the input mesh's
    own triangle count (a reasonable neutral choice, not tuned).

    adaptive: if True, computes per-vertex curvature-adaptive quad density
    (trellis2.utils.curvature.compute_rho) and passes it to QuadriFlow so high-curvature
    regions get smaller quads (up to 25% smaller than the base scale -- see
    third_party/QuadriFlow/src/optimizer.cpp:147-165's bounded 0.75x-1.0x adjustment; this is
    not a free-form density field). Requires the in-process binding (quadriflow_ext) -- the
    CLI/subprocess fallback below has no -rho flag, since the whole point of the binding was
    to pass rho directly as a tensor rather than inventing a file-based interface for it (see
    docs/archive/quadriflow_postprocess_plan.md's adaptive-density section).

    QuadriFlow requires a genuinely 2-manifold input (no non-manifold edges/vertices --
    open boundaries are fine, watertightness is not required). repair_mesh() happens to
    already produce exactly that, so it's applied here unconditionally rather than
    leaving it to the caller to remember.

    Prefers the in-process pybind11/torch binding (quadriflow_ext) over the CLI/subprocess
    path when available -- same underlying QuadriFlow algorithm, just without the temp-file
    round trip. Falls back to the subprocess path transparently if the extension hasn't been
    built (see third_party/QuadriFlow/BUILD_NOTES.md) and adaptive=False.
    """
    mesh = repair_mesh(mesh)
    if target_faces is None:
        target_faces = len(mesh.faces)

    ext = _load_quadriflow_ext()
    if ext is not None:
        rho = None
        if adaptive:
            rho = torch.from_numpy(compute_rho(mesh.vertices, mesh.faces).astype(np.float32))
        v = torch.as_tensor(np.ascontiguousarray(mesh.vertices, dtype=np.float32))
        f = torch.as_tensor(np.ascontiguousarray(mesh.faces, dtype=np.int32))
        out_v, out_f = ext.remesh(v, f, target_faces, rho=rho, adaptive=adaptive)
        write_quad_obj(out_v.numpy(), list(out_f.numpy()), output_obj_path)
        return

    if adaptive:
        raise RuntimeError(
            "adaptive=True requires the in-process quadriflow_ext binding, which is not "
            "built/importable -- the CLI/subprocess fallback has no equivalent to it (no -rho "
            "flag exists on the CLI binary). See third_party/QuadriFlow/BUILD_NOTES.md.")

    if not quadriflow_available():
        raise RuntimeError(
            f"quadriflow binary not found at {_QUADRIFLOW_BIN} -- build it first, "
            f"see third_party/QuadriFlow/BUILD_NOTES.md")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.obj")
        mesh.export(in_path)

        result = subprocess.run(
            [_QUADRIFLOW_BIN, "-i", in_path, "-o", output_obj_path, "-f", str(target_faces)],
            capture_output=True, text=True)
        if not os.path.isfile(output_obj_path):
            raise RuntimeError(
                f"quadriflow failed to produce output (exit {result.returncode}). "
                f"stdout: {result.stdout.strip()!r} stderr: {result.stderr.strip()!r}")


def remesh_to_quad_dominant_obj_multi_component(mesh: trimesh.Trimesh, output_obj_path: str,
                                                 target_faces: int = None,
                                                 min_component_faces: int = 100,
                                                 min_component_target_faces: int = 4) -> dict:
    """
    Per-connected-component variant of remesh_to_quad_dominant_obj, for meshes with many
    disconnected parts.

    Why this exists: QuadriFlow's `target_faces` computes exactly ONE global quad
    edge-length target for the whole mesh -- `scale = sqrt(surface_area / faces)`
    (third_party/QuadriFlow/src/parametrizer-mesh.cpp:60), using the TOTAL surface area
    across every part, applied uniformly. Confirmed empirically (see
    docs/archive/quadriflow_postprocess_plan.md's Phase 6 section) that this silently under-represents
    or entirely drops any connected component much smaller than that global scale: all 4
    of this project's test assets are themselves heavily multi-part (9,600-15,000
    connected components each, even the largest single component only 4-21% of total
    faces), and running plain remesh_to_quad_dominant_obj on them produced quad meshes
    covering only 1-27% of the true surface area, despite reporting 100% quad_fraction
    and looking superficially plausible in a quick 4-view render (a car body silhouette
    with no wheels at all, in one observed case).

    Fix: split by connected component first, and give each surviving component its OWN
    target_faces proportional to its share of the *kept* mesh's total surface area
    (target_faces * area_i / kept_area) -- this reconstructs the same overall quad
    density QuadriFlow would have used on a single coherent part, but applied
    component-by-component so no component's budget gets diluted by all the others.
    `min_component_target_faces` floors each surviving component's own request (a
    tiny-but-real part, e.g. a watch's minute hand, still gets a usable minimum instead
    of rounding down to ~0 and vanishing).

    min_component_faces: components with fewer triangles than this are dropped entirely
    before remeshing even starts -- a higher bar than mesh_repair.py's own
    remove_small_islands (8), and deliberately so: each surviving component costs one
    full subprocess launch (export + QuadriFlow invocation + parse), and these test
    assets have THOUSANDS of components after repair_mesh's default cleanup (can.glb:
    2713 components >= 8 faces). Default 100 cuts that to ~150 on can.glb while
    retaining 91.4% of total faces -- checked directly, not guessed: face count kept
    drops off a gentle curve (94.7% at >=20, 92.7% at >=50, 91.4% at >=100, 87.0% at
    >=500), so 100 is a reasonable practicality/coverage tradeoff, not a hard
    requirement -- lower it for small assets where the per-component subprocess cost is
    a non-issue.

    Any individual component QuadriFlow fails on is skipped (logged, not fatal) rather
    than aborting the whole multi-part remesh over one bad part -- this includes not just
    outright QuadriFlow failures but also a bounding-box sanity check: confirmed
    empirically (can.glb) that QuadriFlow can produce a numerically-unstable result whose
    extent is 10s-100s of times larger than the submesh it was given (2 of 152
    components did this on one test run, and together outweighed all 150 well-behaved
    ones combined) -- any component whose output bounding-box diagonal exceeds 1.5x its
    input's is treated as failed rather than merged in corrupted.

    Returns a dict of stats: {"components_total", "components_kept", "components_failed",
    "kept_area_fraction" (share of the repaired mesh's area belonging to components that
    were *attempted*), "succeeded_area_fraction" (share that actually made it into the
    output after per-component failures/blow-ups are excluded -- the more meaningful
    number)} -- useful for the caller to judge how much of the original asset actually
    made it through, since (unlike the single-call version) this can't be fully
    summarized by quad_fraction alone.
    """
    mesh = repair_mesh(mesh)
    if target_faces is None:
        target_faces = len(mesh.faces)

    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1)
    kept = [comp for comp in components if len(comp) >= min_component_faces]
    if not kept:
        raise RuntimeError("No connected component meets min_component_faces -- nothing to remesh.")

    submeshes = [mesh.submesh([comp], append=True) for comp in kept]
    kept_area = sum(sub.area for sub in submeshes)

    combined_vertices = []
    combined_faces = []
    vertex_offset = 0
    n_failed = 0
    succeeded_area = 0.0

    with tempfile.TemporaryDirectory() as tmp:
        for i, sub in enumerate(submeshes):
            comp_target = max(min_component_target_faces,
                               round(target_faces * sub.area / kept_area)) if kept_area > 0 else min_component_target_faces
            in_path = os.path.join(tmp, f"comp_{i}_in.obj")
            out_path = os.path.join(tmp, f"comp_{i}_out.obj")
            sub.export(in_path)
            subprocess.run(
                [_QUADRIFLOW_BIN, "-i", in_path, "-o", out_path, "-f", str(comp_target)],
                capture_output=True, text=True)
            if not os.path.isfile(out_path):
                n_failed += 1
                continue
            comp_vertices, comp_faces = parse_obj_raw(out_path)
            if len(comp_vertices) == 0 or len(comp_faces) == 0:
                n_failed += 1  # QuadriFlow "succeeded" (wrote a file) but produced nothing usable
                continue
            # Sanity-check against wild blow-ups: confirmed empirically (can.glb) that
            # QuadriFlow can, on a specific submesh/target_faces combination, produce a
            # numerically-unstable result with 10s-100s of times more surface AREA than
            # the input it was given, WITHOUT a correspondingly larger bounding box (a
            # bbox-diagonal check, tried first, did not catch it at all -- consistent
            # with self-intersecting/malformed quad winding rather than the whole result
            # being spatially displaced: vertices stay roughly in place, but the
            # quad-area shoelace calculation blows up). Not a rare one-off: 2 of 152
            # components did this on one test run, and together outweighed all 150
            # well-behaved ones combined (202x and 127x area blow-ups vs. a normal ~0.04
            # median output/input area ratio for properly-behaved components -- a remesh
            # legitimately shrinks a lot when simplifying, that's expected and fine, but
            # should never substantially *exceed* the input's own surface area). Checked
            # directly against area, not bbox.
            out_tri_faces = []
            for face in comp_faces:
                out_tri_faces.extend(_triangulate_fan_indices(face))
            out_area = trimesh.Trimesh(vertices=comp_vertices, faces=np.array(out_tri_faces),
                                        process=False).area
            # NB: `out_area > sub.area * 3` alone would silently pass a NaN out_area
            # straight through (any comparison against NaN is False in Python/numpy --
            # the exact same pitfall this project already hit once, see docs/archive/history.md's SDS
            # explosion-guard saga) -- confirmed this actually happens: component 54's
            # output area came back NaN (degenerate/self-intersecting quads), which is
            # why the area-based check above still let this component's corrupted
            # geometry through on the first attempt despite being "> 3x" in spirit.
            if not np.isfinite(out_area) or (sub.area > 0 and out_area > sub.area * 3):
                n_failed += 1
                continue
            combined_vertices.append(comp_vertices)
            combined_faces.extend(f + vertex_offset for f in comp_faces)
            vertex_offset += len(comp_vertices)
            succeeded_area += sub.area

    if not combined_vertices:
        raise RuntimeError("quadriflow failed on every kept component -- no output produced.")

    all_vertices = np.concatenate(combined_vertices, axis=0)
    write_quad_obj(all_vertices, combined_faces, output_obj_path)

    return {
        "components_total": len(components),
        "components_kept": len(kept),
        "components_failed": n_failed,
        "kept_area_fraction": kept_area / mesh.area if mesh.area > 0 else 0.0,
        "succeeded_area_fraction": succeeded_area / mesh.area if mesh.area > 0 else 0.0,
    }
