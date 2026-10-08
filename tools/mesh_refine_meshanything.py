import argparse
import trimesh
import os
import sys
from PIL import Image

# Add project root to sys.path so we can import trellis2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trellis2.utils.meshanything_bridge import refine_mesh_coarse, refine_mesh_direct_sample

def main():
    parser = argparse.ArgumentParser(description="SDS-guided geometry refinement and MeshAnything coarse mesh generation.")
    parser.add_argument("--input", required=True, help="Input GLB mesh file")
    parser.add_argument("--reference-image", default=None,
                         help="Reference image for SDS conditioning (required for --algo legacy unless --skip-sds; unused by --algo direct-sample)")
    parser.add_argument("--output", required=True, help="Output GLB mesh file")
    parser.add_argument("--debug-dir", default=None,
                         help="If set, dump intermediate meshes + rendered contact sheets at each pipeline stage here")
    parser.add_argument("--algo", choices=["legacy", "direct-sample"], default="legacy",
                         help="'legacy' (default): SDS refine -> decimated-mesh conditioning -> MeshAnything, per refine_mesh_coarse. "
                              "'direct-sample': repair-decimate-repair, no SDS, hybrid importance+uniform sampling straight off the "
                              "original repaired mesh, per refine_mesh_direct_sample. Kept side-by-side for comparison.")
    parser.add_argument("--skip-sds", action="store_true",
                         help="[legacy only] Diagnostic: bypass Phase 3 SDS geometry refinement, feed the decimated mesh straight to MeshAnything")
    parser.add_argument("--sampling", choices=["uniform", "feature", "hybrid"], default="uniform",
                         help="[legacy only] Conditioning point cloud sampling mode (direct-sample always uses hybrid)")
    parser.add_argument("--n-points", type=int, default=16384,
                         help="[direct-sample only] Conditioning point cloud size")
    parser.add_argument("--curvature-gain", type=float, default=8.0,
                         help="[direct-sample only] Feature-sampling curvature weighting strength")
    parser.add_argument("--uniform-frac", type=float, default=0.5,
                         help="[direct-sample only] Fraction of points drawn uniformly (rest drawn feature-weighted)")
    parser.add_argument("--target-faces", type=int, default=16000,
                         help="[direct-sample only] Decimation target face count")
    parser.add_argument("--quad-remesh", default=None,
                         help="If set, also postprocess the result through QuadriFlow and write a genuine "
                              "quad-dominant mesh to this .obj path (see trellis2/utils/quad_remesh.py). "
                              "--output stays triangulated (.glb doesn't support quads) -- this is a separate file.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    if args.algo == "legacy" and not args.skip_sds:
        if args.reference_image is None:
            print("Error: --reference-image is required for --algo legacy unless --skip-sds is set.")
            sys.exit(1)
        if not os.path.exists(args.reference_image):
            print(f"Error: Reference image {args.reference_image} does not exist.")
            sys.exit(1)

    print(f"Loading input mesh from {args.input}...")
    mesh = trimesh.load(args.input, force='mesh')

    ref_image = None
    if args.reference_image is not None:
        print(f"Loading reference image from {args.reference_image}...")
        ref_image = Image.open(args.reference_image).convert("RGB")

    if args.debug_dir:
        print(f"Debug mode: dumping intermediate stages to {args.debug_dir}")

    if args.algo == "direct-sample":
        print("Running refine_mesh_direct_sample pipeline (this may take a few minutes)...")
        out_mesh = refine_mesh_direct_sample(mesh, debug_dir=args.debug_dir, n_points=args.n_points,
                                              curvature_gain=args.curvature_gain, uniform_frac=args.uniform_frac,
                                              target_faces=args.target_faces)
    else:
        print("Running refine_mesh_coarse pipeline (this may take a few minutes)...")
        out_mesh = refine_mesh_coarse(mesh, ref_image, debug_dir=args.debug_dir, skip_sds=args.skip_sds,
                                       sampling=args.sampling)

    print(f"Saving output to {args.output}...")
    out_mesh.export(args.output)

    if args.quad_remesh:
        from trellis2.utils.quad_remesh import remesh_to_quad_dominant_obj
        print(f"Running QuadriFlow quad-dominant remesh, writing to {args.quad_remesh}...")
        remesh_to_quad_dominant_obj(out_mesh, args.quad_remesh)

    print("Done!")

if __name__ == "__main__":
    main()
