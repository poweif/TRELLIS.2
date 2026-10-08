import argparse
import os
import sys
import tempfile
import time

import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trellis2.utils.mesh_repair import repair_mesh
from trellis2.utils.quad_remesh import remesh_to_quad_dominant_obj
from trellis2.utils.alpha_wrap import alpha_wrap_mesh, alpha_wrap_available
from trellis2.utils.quad_postprocess import (
    parse_obj_raw, write_quad_obj, quad_fraction, remove_small_quad_islands,
    snap_to_reference, uv_unwrap_auto, transfer_texture,
)
from trellis2.utils.subdivision import subdivide_catmull_clark


def main():
    parser = argparse.ArgumentParser(
        description="Post-process a mesh into a quad-derived, UV-unwrapped, textured GLB "
                    "(see docs/quad_pipeline.md). Standalone: takes an input GLB, "
                    "produces a final textured GLB, no manual Python required.")
    parser.add_argument("--input", required=True,
                        help="Input GLB mesh -- also used as the texture/geometry reference (Phase 2/5's reference_mesh)")
    parser.add_argument("--output", required=True, help="Output textured GLB path")
    parser.add_argument("--source", choices=["direct", "meshanything"], default="direct",
                        help="'direct' (default): QuadriFlow runs directly on the repaired original mesh -- "
                             "recommended default per Phase 1's ablation (8/8 successes, recognizable/complete "
                             "results on every test asset, see docs/quad_pipeline.md). "
                             "'meshanything': QuadriFlow runs on MeshAnythingV2's coarse retopology first -- "
                             "narrower use case (artist-style edge flow), known to fail outright or fragment badly "
                             "on some assets (Phase 1 findings), not trusted as a default.")
    parser.add_argument("--target-faces", type=int, default=None,
                        help="QuadriFlow target face count. None (default) = the input mesh's own triangle count "
                             "(remesh_to_quad_dominant_obj's own default).")
    parser.add_argument("--quad-obj", default=None,
                        help="Reuse an existing quad .obj (e.g. from mesh_refine_meshanything.py's --quad-remesh) "
                             "instead of running QuadriFlow fresh. Skips --source/--target-faces/--no-alpha-wrap "
                             "entirely.")
    parser.add_argument("--no-alpha-wrap", action="store_true",
                        help="Skip CGAL Alpha Wrap preprocessing before QuadriFlow. Alpha Wrap is on by default "
                             "because real-world test assets turn out to be severely perforated (thousands of "
                             "pinhole-scale, often topologically degenerate gaps per connected component) -- "
                             "feeding that directly to QuadriFlow starves it (QuadriFlow's target_faces sets one "
                             "global quad edge-length from TOTAL surface area, so components smaller than that "
                             "scale get little/no representation), recovering only 1-27%% of the true surface area "
                             "on this project's 4 test assets. Alpha Wrap reconstructs a genuinely watertight "
                             "surface first regardless of input topology, recovering 88.9%%-97.6%% instead (2-5s "
                             "even on ~450K-face inputs) -- see third_party/AlphaWrap/BUILD_NOTES.md and "
                             "docs/quad_pipeline.md. Only skip this for input already "
                             "known to be clean/watertight, or if the AlphaWrap binary isn't built.")
    parser.add_argument("--alpha-wrap-alpha", type=float, default=None,
                        help="Alpha Wrap detail parameter (smaller = finer). None (default) = bbox_diagonal/40.")
    parser.add_argument("--alpha-wrap-offset", type=float, default=None,
                        help="Alpha Wrap offset parameter. None (default) = bbox_diagonal/1200.")
    parser.add_argument("--no-snap", action="store_true",
                        help="Skip Phase 2 vertex snap-back onto the reference mesh.")
    parser.add_argument("--subdivide-levels", type=int, default=0,
                        help="Catmull-Clark subdivision levels (Phase 3, optional). 0 (default) = off.")
    parser.add_argument("--separatrix-max-length", type=int, default=8,
                        help="Phase 4 motorcycle-graph separatrix length cap -- see "
                             "trellis2.utils.quad_postprocess.trace_separatrices for why this exists "
                             "(without it, chart layout catastrophically over-fragments).")
    parser.add_argument("--adaptive", action="store_true",
                        help="Curvature-adaptive quad density (smaller quads in high-curvature "
                             "regions, up to 25%% smaller than the base scale -- see "
                             "trellis2.utils.curvature.compute_rho). Off by default: proven on "
                             "synthetic shapes but not yet validated across this project's real "
                             "test assets. Requires the in-process quadriflow_ext binding (see "
                             "third_party/QuadriFlow/BUILD_NOTES.md).")
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--debug-dir", default=None,
                        help="If set, dump intermediate quad meshes + wireframe contact sheets here at each stage.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file {args.input} does not exist.")
        sys.exit(1)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        def stage_path(name):
            return os.path.join(args.debug_dir, name) if args.debug_dir else os.path.join(tmp_dir, name)

        def dump_wireframe(obj_path, name):
            if args.debug_dir:
                from trellis2.utils.quad_debug import render_quad_wireframe_contact_sheet
                render_quad_wireframe_contact_sheet(obj_path, stage_path(name + ".png"), "cuda")

        t0 = time.time()
        print(f"Loading input mesh from {args.input}...")
        reference_mesh = trimesh.load(args.input, force='mesh')

        # --- Stage 1: obtain a quad .obj ---
        if args.quad_obj:
            print(f"Reusing existing quad .obj at {args.quad_obj}...")
            quad_obj_path = args.quad_obj
        else:
            quad_obj_path = stage_path("01_quad.obj")
            if args.source == "direct":
                repaired = repair_mesh(reference_mesh)
                if not args.no_alpha_wrap:
                    if not alpha_wrap_available():
                        print("Warning: --no-alpha-wrap not set but the AlphaWrap binary isn't built "
                              "(see third_party/AlphaWrap/BUILD_NOTES.md) -- proceeding without it.")
                    else:
                        print("Reconstructing a watertight surface via CGAL Alpha Wrap...")
                        repaired = alpha_wrap_mesh(repaired, alpha=args.alpha_wrap_alpha,
                                                    offset=args.alpha_wrap_offset)
                        print(f"  wrapped: {len(repaired.faces)} faces, watertight={repaired.is_watertight}")
                        if args.debug_dir:
                            from trellis2.utils.mesh_debug import dump_mesh_stage
                            dump_mesh_stage(repaired, "00b_alpha_wrapped", args.debug_dir, "cuda")
                print("Running QuadriFlow on the repaired original mesh (--source direct)...")
                remesh_to_quad_dominant_obj(repaired, quad_obj_path, target_faces=args.target_faces,
                                             adaptive=args.adaptive)
            else:
                print("Running MeshAnythingV2 (refine_mesh_direct_sample) then QuadriFlow (--source meshanything)...")
                from trellis2.utils.meshanything_bridge import refine_mesh_direct_sample
                coarse_mesh, intermediates = refine_mesh_direct_sample(reference_mesh, return_intermediates=True)
                # The true reference for detail/texture is the full-resolution repaired mesh
                # MeshAnything's conditioning was sampled from, never the MeshAnything-coarse
                # output itself (texture/UV data doesn't survive that step) -- see plan's
                # "Assumed input state" contract and the plumbing-gap note this fixes.
                reference_mesh = intermediates["original_repaired"]
                remesh_to_quad_dominant_obj(coarse_mesh, quad_obj_path, target_faces=args.target_faces,
                                             adaptive=args.adaptive)

        quad_vertices, quad_faces = parse_obj_raw(quad_obj_path)
        qf = quad_fraction(quad_obj_path)
        print(f"Quad mesh: {len(quad_vertices)} verts, {len(quad_faces)} faces (quad fraction {qf*100:.1f}%)")
        dump_wireframe(quad_obj_path, "01_quad")

        # Defensive cleanup, not the main fix (that's Alpha Wrap above): cheap and
        # harmless if the quad mesh is already clean, still useful as a safety net if
        # --no-alpha-wrap was passed or the AlphaWrap binary wasn't available.
        n_faces_before = len(quad_faces)
        quad_vertices, quad_faces = remove_small_quad_islands(quad_vertices, quad_faces)
        if len(quad_faces) != n_faces_before:
            print(f"  Removed small disconnected fragments: {n_faces_before} -> {len(quad_faces)} faces")

        # --- Stage 2: vertex snap-back ---
        if not args.no_snap:
            print("Snapping vertices onto the reference surface (Phase 2)...")
            quad_vertices = snap_to_reference(quad_vertices, reference_mesh)
            if args.debug_dir:
                snap_path = stage_path("02_snapped.obj")
                write_quad_obj(quad_vertices, quad_faces, snap_path)
                dump_wireframe(snap_path, "02_snapped")

        # --- Stage 3: optional Catmull-Clark subdivision ---
        if args.subdivide_levels > 0:
            print(f"Subdividing (Catmull-Clark, {args.subdivide_levels} level(s))...")
            pre_subdiv_path = stage_path("_pre_subdiv.obj")
            write_quad_obj(quad_vertices, quad_faces, pre_subdiv_path)
            subdiv_path = stage_path("03_subdivided.obj")
            subdivide_catmull_clark(pre_subdiv_path, subdiv_path, levels=args.subdivide_levels)
            quad_vertices, quad_faces = parse_obj_raw(subdiv_path)

            if not args.no_snap:
                print("Re-snapping after subdivision (subdivision introduces new vertices off the true surface)...")
                quad_vertices = snap_to_reference(quad_vertices, reference_mesh)

            if args.debug_dir:
                final_subdiv_path = stage_path("03_subdivided_snapped.obj")
                write_quad_obj(quad_vertices, quad_faces, final_subdiv_path)
                dump_wireframe(final_subdiv_path, "03_subdivided_snapped")

        # --- Stage 4: UV unwrap (motorcycle-graph patch layout, auto-fallback to task 1) ---
        print("Computing UV chart layout (Phase 4)...")
        uv_vertices, uv_faces, uvs = uv_unwrap_auto(
            quad_vertices, quad_faces, max_length=args.separatrix_max_length)

        # --- Stage 5: texture transfer ---
        print("Transferring texture from the reference mesh (Phase 5)...")
        final_mesh = transfer_texture(uv_vertices, uv_faces, uvs, reference_mesh, texture_size=args.texture_size)

        print(f"Saving output to {args.output}...")
        final_mesh.export(args.output)
        print(f"Done in {time.time() - t0:.1f}s. {len(final_mesh.vertices)} verts, {len(final_mesh.faces)} faces.")


if __name__ == "__main__":
    main()
