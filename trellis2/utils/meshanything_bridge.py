import sys
import os
import functools
import numpy as np
import torch
import trimesh

_THIRD_PARTY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'MeshAnythingV2'))
if _THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_DIR)

from MeshAnything.models.meshanything_v2 import MeshAnythingV2

@functools.lru_cache(maxsize=1)
def load_model(device):
    model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")
    model = model.to(device).half()
    return model

def run_inference(point_cloud: np.ndarray, device) -> np.ndarray:
    """
    Wraps the raw forward pass only.
    
    Args:
        point_cloud: np.ndarray of shape (8192, 6) (fp16)
        device: torch device string like "cuda" or "cpu"
        
    Returns:
        np.ndarray: shape (N, 9)
    """
    model = load_model(device)
    
    pc_tensor = torch.tensor(point_cloud, dtype=torch.float16, device=device).unsqueeze(0)
    
    with torch.no_grad():
        # Using model.eval() just to be safe
        model.eval()
        outputs = model(pc_tensor, sampling=False)
        recon_mesh = outputs[0]
        # output is shaped (N, 3, 3) where each row is a triangle with 3 vertices in 3D
        recon_mesh = recon_mesh.reshape(-1, 9)
        return recon_mesh.cpu().numpy()

def face_feature_weights(mesh: trimesh.Trimesh, curvature_gain=8.0) -> np.ndarray:
    """
    Per-face importance weight combining surface area with a discrete-curvature proxy:
    the dihedral angle across each face-adjacency edge (angle between two triangles'
    normals -- 0 for a flat/coplanar pair, up to pi for a sharp fold/crease). This is the
    standard fast, fully-vectorized local curvature proxy used in mesh
    simplification/feature-preservation work -- trimesh already computes
    face_adjacency_angles in O(faces), so this needs no O(vertices) curvature-ball query.
    """
    areas = mesh.area_faces
    if len(mesh.face_adjacency) == 0:
        return areas

    angles = mesh.face_adjacency_angles  # (E,) radians, 0=flat .. pi=folded back
    pairs = mesh.face_adjacency          # (E, 2) face index pairs sharing that edge

    # Sharpest edge touching each face (stays 0 for faces with no shared edges at all)
    face_sharpness = np.zeros(len(mesh.faces))
    np.maximum.at(face_sharpness, pairs[:, 0], angles)
    np.maximum.at(face_sharpness, pairs[:, 1], angles)

    sharpness_norm = face_sharpness / np.pi
    return areas * (1.0 + curvature_gain * sharpness_norm)


def mesh_to_conditioning_pc(mesh: trimesh.Trimesh, n_points=8192, sampling="uniform", curvature_gain=8.0,
                             uniform_frac=0.5):
    """
    Phase 4: Sample conditioning point cloud and scale to MeshAnything expected domain.

    sampling: "uniform" (default -- matches original behavior, pure area-weighted via
    trimesh.sample.sample_surface), "feature" (biases density toward high-curvature /
    sharp-edge regions via face_feature_weights), or "hybrid" (both at once: uniform_frac
    of n_points drawn uniform for a flat-area coverage floor, the rest drawn feature-
    weighted for extra density on sharp features -- guards against a pure-feature weighting
    starving large flat regions on meshes with many separate high-curvature parts). All are
    standalone alternatives -- not wired into refine_mesh_coarse's default flow, swap in
    explicitly to compare.
    """
    if sampling == "feature":
        face_weight = face_feature_weights(mesh, curvature_gain=curvature_gain)
        points, face_idx = trimesh.sample.sample_surface(mesh, n_points, face_weight=face_weight)
    elif sampling == "uniform":
        points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    elif sampling == "hybrid":
        n_uniform = int(round(n_points * uniform_frac))
        n_feature = n_points - n_uniform
        parts_pts, parts_idx = [], []
        if n_uniform > 0:
            p, i = trimesh.sample.sample_surface(mesh, n_uniform)
            parts_pts.append(p); parts_idx.append(i)
        if n_feature > 0:
            face_weight = face_feature_weights(mesh, curvature_gain=curvature_gain)
            p, i = trimesh.sample.sample_surface(mesh, n_feature, face_weight=face_weight)
            parts_pts.append(p); parts_idx.append(i)
        points = np.concatenate(parts_pts, axis=0)
        face_idx = np.concatenate(parts_idx, axis=0)
    else:
        raise ValueError(f"Unknown sampling mode: {sampling!r}, expected 'uniform', 'feature', or 'hybrid'")

    normals = mesh.face_normals[face_idx]

    bbox = mesh.bounds
    center = (bbox[0] + bbox[1]) / 2.0
    
    # MeshAnything has a strange discrepancy:
    # It expects conditioning point clouds scaled to [-0.9995, 0.9995]
    # But it trains its autoregressive token coordinates in [-0.5, 0.5]
    scale = np.max(bbox[1] - bbox[0])
    
    # Scale coordinates such that the max extent is 1.0, giving [-0.5, 0.5]
    # And multiply by 2 * 0.9995 to put the point cloud in the expected domain
    points = (points - center) / scale * (2 * 0.9995)
    
    pc = np.concatenate([points, normals], axis=-1).astype(np.float16)
    
    def inverse_transform(p):
        # The generated mesh will be in [-0.5, 0.5], so we just undo the 'scale' and 'center'
        return p * scale + center
        
    return pc, inverse_transform

def refine_mesh_coarse(mesh: trimesh.Trimesh, reference_image, debug_dir=None, skip_sds=False,
                        sampling="uniform") -> trimesh.Trimesh:
    """
    Phase 5: Full mesh generation loop including Phase 3 SDS and detokenization.

    debug_dir: if set, dump the mesh/point-cloud + a rendered contact sheet at every
    pipeline stage into this directory (00_input, 01_decimated, 02_sds_refined,
    03_conditioning_pc, 04_meshanything_raw, 05_meshanything_cleaned, 06_final),
    plus a manifest.txt summary. See trellis2/utils/mesh_debug.py.

    skip_sds: diagnostic escape hatch — bypass Phase 3 entirely and feed the decimated
    mesh straight into the conditioning point cloud. Lets you isolate whether visual
    problems come from SDS or from MeshAnything/decimation itself.

    sampling: "uniform" (default) or "feature" — passed through to mesh_to_conditioning_pc,
    see that function for what "feature" does.
    """
    from .sds_geometry_refine import refine_geometry
    from . import mesh_debug

    device = 'cuda'

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        open(os.path.join(debug_dir, "manifest.txt"), "w").close()
        mesh_debug.dump_mesh_stage(mesh, "00_input", debug_dir, device)

    # Decimate before SDS to avoid blowing up the regularization on dense meshes
    if len(mesh.faces) > 16000:
        print(f"Decimating mesh from {len(mesh.faces)} faces down to 16000 before SDS...")
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices), 
            o3d.utility.Vector3iVector(mesh.faces)
        )
        
        # 1) Topology cleanup via vertex clustering (acts like a lightweight voxelization to merge floating/disconnected faces)
        max_extent = max(mesh.extents)
        voxel_size = max_extent / 100.0
        o3d_mesh = o3d_mesh.simplify_vertex_clustering(
            voxel_size=voxel_size, 
            contraction=o3d.geometry.SimplificationContraction.Average
        )
        
        # 2) If still too large, use QEM to hit the exact target
        if len(o3d_mesh.triangles) > 16000:
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=16000)
            
        mesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles))
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(mesh, "01_decimated", debug_dir, device)

    if skip_sds:
        print("Skipping Phase 3 (SDS geometry refinement) per skip_sds=True...")
        refined_mesh = mesh
    else:
        print("Running Phase 3: SDS geometry refinement...")
        sds_log_dir = os.path.join(debug_dir, "02_sds_iters") if debug_dir is not None else None
        refined_mesh = refine_geometry(mesh, reference_image, n_iters=100, log_dir=sds_log_dir)

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(refined_mesh, "02_sds_refined", debug_dir, device)

    print("Running Phase 4: Point-cloud sampling...")
    pc, inv_transform = mesh_to_conditioning_pc(refined_mesh, sampling=sampling)

    if debug_dir is not None:
        mesh_debug.dump_pointcloud_stage(pc[:, :3].astype(np.float32), pc[:, 3:].astype(np.float32),
                                          "03_conditioning_pc", debug_dir, device)

    print("Running MeshAnything inference...")
    out_mesh_data = run_inference(pc, device='cuda')

    print("Running Phase 5: Cleanup and detokenization...")
    # Mask NaNs
    valid = ~np.isnan(out_mesh_data).any(axis=1)
    out_mesh_data = out_mesh_data[valid]

    vertices = out_mesh_data.reshape(-1, 3)
    faces = np.arange(vertices.shape[0]).reshape(-1, 3)

    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(new_mesh, "04_meshanything_raw", debug_dir, device)

    # Cleanup sequence per reference implementation
    new_mesh.merge_vertices()
    new_mesh.update_faces(new_mesh.nondegenerate_faces())
    new_mesh.update_faces(new_mesh.unique_faces())
    new_mesh.remove_unreferenced_vertices()
    new_mesh.fix_normals()

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(new_mesh, "05_meshanything_cleaned", debug_dir, device)

    # Apply inverse transform
    new_mesh.vertices = inv_transform(new_mesh.vertices)

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(new_mesh, "06_final", debug_dir, device)

    return new_mesh


def refine_mesh_direct_sample(mesh: trimesh.Trimesh, debug_dir=None, n_points=16384,
                               curvature_gain=8.0, uniform_frac=0.5, target_faces=16000,
                               return_intermediates=False):
    """
    Standalone alternative to refine_mesh_coarse, for side-by-side comparison (not wired
    in as a replacement -- refine_mesh_coarse is untouched). Differs in ways motivated by
    two things found investigating refine_mesh_coarse: decimation flipping triangle
    winding on tests/watch.glb's closed-loop band despite an existing fix_normals() call,
    and SDS being an uncertain net-positive whose hyperparameters (tuned against can.glb)
    don't generalize cleanly to other assets (also observed on watch.glb).

    - Skips SDS geometry refinement entirely -- no diffusion pipeline load, no
      SDS-hyperparameter generalization risk.
    - Samples the conditioning point cloud directly from the ORIGINAL (repaired,
      full-resolution) mesh -- no decimation involved -- using "hybrid" importance+uniform
      sampling (see mesh_to_conditioning_pc) so sharp features get extra density without
      leaving flat regions under-covered.
    - n_points is larger by default (16384 vs. 8192) and adjustable; MeshAnythingV2's
      point encoder is a Perceiver-style cross-attention encoder with a fixed set of
      learned latent queries attending into the (variable-length) point cloud, so it is
      not tied to a specific input point count.

    No decimation in the real pipeline: decimation was originally introduced (in
    refine_mesh_coarse) to keep SDS's differentiable-rasterizer optimization tractable on
    dense meshes -- a concern that doesn't apply here since this function skips SDS
    entirely, and trimesh.sample.sample_surface (what mesh_to_conditioning_pc actually
    uses) doesn't care about input face count either. target_faces is only used, and
    decimation only runs at all, when debug_dir is set -- purely as an optional visual
    reference stage (01_decimated in the debug dump), gated off by default to skip both
    the extra compute on large meshes and a whole class of decimation-specific bugs found
    while investigating this (open3d's simplify_vertex_clustering fragmenting
    tests/watch.glb into 22K+ noise islands, and an unresolved topological defect on
    tests/can.glb that survived every repair attempt tried -- see history.md's "New pipeline
    variant: refine_mesh_direct_sample" section).

    return_intermediates: if True, returns (new_mesh, {"original_repaired": ...}) instead of
    just new_mesh -- original_repaired is exactly the reference_mesh contract
    quadriflow_postprocess_plan.md's Phase 2/5 need under Option 1 (full-resolution, repaired,
    pre-MeshAnything mesh), computed here anyway but previously never exposed.
    """
    from . import mesh_debug
    from .mesh_repair import repair_mesh, repair_mesh_after_decimation, is_winding_consistent_per_component

    device = 'cuda'

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        open(os.path.join(debug_dir, "manifest.txt"), "w").close()
        mesh_debug.dump_mesh_stage(mesh, "00_input", debug_dir, device)

    print("Repairing input mesh...")
    mesh = repair_mesh(mesh)
    original_repaired = mesh.copy()

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(mesh, "00b_input_repaired", debug_dir, device)

        # Decimation is NOT part of the real pipeline anymore -- point sampling below
        # always uses original_repaired (full resolution), never this. It only exists
        # here, gated behind debug_dir, as an optional visual reference/sanity check,
        # since it was the original reason target_faces/this whole code path existed
        # (avoiding it entirely elsewhere saves real compute on large meshes and sidesteps
        # a whole class of decimation-specific bugs found this session -- see history.md:
        # open3d's simplify_vertex_clustering fragmenting tests/watch.glb into 22K+ noise
        # islands, and a genuinely unresolved topological defect on tests/can.glb that
        # survived every repair attempt tried).
        if len(mesh.faces) > target_faces:
            print(f"[debug only] Decimating mesh from {len(mesh.faces)} faces down to {target_faces}...")
            import open3d as o3d
            o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces)
            )
            max_extent = max(mesh.extents)
            voxel_size = max_extent / 100.0
            o3d_mesh = o3d_mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average
            )
            if len(o3d_mesh.triangles) > target_faces:
                o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

            decimated = repair_mesh_after_decimation(
                np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles), o3d_mesh=o3d_mesh)
        else:
            decimated = mesh

        if not is_winding_consistent_per_component(decimated):
            print("[debug only] WARNING: decimated reference mesh winding still inconsistent after repair "
                  "(does not affect the real output -- point sampling never uses this mesh)")

        mesh_debug.dump_mesh_stage(decimated, "01_decimated", debug_dir, device)

    print(f"Sampling {n_points} conditioning points directly from the original (repaired) mesh "
          f"(sampling='hybrid', uniform_frac={uniform_frac})...")
    pc, inv_transform = mesh_to_conditioning_pc(original_repaired, n_points=n_points,
                                                 sampling="hybrid", curvature_gain=curvature_gain,
                                                 uniform_frac=uniform_frac)

    if debug_dir is not None:
        mesh_debug.dump_pointcloud_stage(pc[:, :3].astype(np.float32), pc[:, 3:].astype(np.float32),
                                          "02_conditioning_pc", debug_dir, device)

    print("Running MeshAnything inference...")
    out_mesh_data = run_inference(pc, device='cuda')

    print("Cleanup and detokenization...")
    valid = ~np.isnan(out_mesh_data).any(axis=1)
    out_mesh_data = out_mesh_data[valid]

    vertices = out_mesh_data.reshape(-1, 3)
    faces = np.arange(vertices.shape[0]).reshape(-1, 3)

    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(new_mesh, "03_meshanything_raw", debug_dir, device)

    new_mesh.merge_vertices()
    new_mesh.update_faces(new_mesh.nondegenerate_faces())
    new_mesh.update_faces(new_mesh.unique_faces())
    new_mesh.remove_unreferenced_vertices()
    new_mesh.fix_normals()

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(new_mesh, "04_meshanything_cleaned", debug_dir, device)

    new_mesh.vertices = inv_transform(new_mesh.vertices)

    if debug_dir is not None:
        mesh_debug.dump_mesh_stage(new_mesh, "05_final", debug_dir, device)

    if return_intermediates:
        return new_mesh, {"original_repaired": original_repaired}
    return new_mesh
