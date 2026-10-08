from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image


def load_quad_obj(path: str) -> trimesh.Trimesh:
    """
    Load a QuadriFlow-written quad .obj as a triangulated trimesh.Trimesh (Trimesh always
    triangulates internally -- see quad_remesh.py's remesh_to_quad_dominant_obj docstring
    for why the quad structure only survives in the .obj file itself, never in memory as
    a Trimesh). Callers that need the genuine quad faces should parse the .obj directly
    (see parse_obj_raw below) rather than relying on this function's output.
    """
    return trimesh.load(path, force='mesh')


def parse_obj_raw(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Parse an .obj file's vertices and raw (un-triangulated) face vertex-index arrays
    directly from the 'v'/'f' lines, bypassing trimesh's triangulating loader entirely.
    This is the only way to recover QuadriFlow's genuine quad structure -- trimesh always
    triangulates on load (see load_quad_obj). Returns (vertices (V, 3) float64, faces:
    list of int64 arrays, one per face, length 3/4/5+ matching the file's actual
    polygon size). Ignores texture/normal indices ('v1/vt1/vn1') if present; handles
    negative (relative) OBJ indices.
    """
    vertices = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = []
                for p in parts:
                    vi = int(p.split("/")[0])
                    vi = (len(vertices) + vi) if vi < 0 else (vi - 1)
                    idx.append(vi)
                faces.append(np.array(idx, dtype=np.int64))
    if not vertices:
        return np.zeros((0, 3), dtype=np.float64), faces
    return np.array(vertices, dtype=np.float64), faces


def parse_quad_faces(path: str) -> List[int]:
    """
    Vertex count per face (e.g. 4 for a quad, 3 for a leftover triangle, 5 for a
    pentagon) -- the ground truth for "quad fraction". history.md records this lesson
    being learned the hard way once already (valence statistics on a triangulated mesh
    are not a substitute for checking the actual face structure).
    """
    _, faces = parse_obj_raw(path)
    return [len(f) for f in faces]


def write_quad_obj(vertices: np.ndarray, faces: List[np.ndarray], path: str):
    """Write vertices + arbitrary-arity (n-gon) faces back to an .obj -- the write-side
    counterpart of parse_obj_raw, used to carry snapped/subdivided positions forward onto
    the genuine quad structure (same vertex indices, only coordinates change)."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}\n")
        for face in faces:
            f.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")


def quad_fraction(path: str) -> float:
    """Fraction of faces in the raw .obj that are genuine quads (4 vertices)."""
    counts = parse_quad_faces(path)
    if len(counts) == 0:
        return 0.0
    counts = np.array(counts)
    return float((counts == 4).mean())


def face_edges(faces: List[np.ndarray]) -> np.ndarray:
    """Unique undirected edges (consecutive vertex pairs, wrapping) across all faces of
    a possibly-mixed-polygon mesh. Returns (E, 2) int64, each row sorted (a < b)."""
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            edges.add((a, b) if a < b else (b, a))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(edges), dtype=np.int64)


def snap_to_reference(vertices: np.ndarray, reference_mesh: trimesh.Trimesh, device="cuda") -> np.ndarray:
    """
    Snap each vertex in `vertices` onto its nearest point on reference_mesh's surface via
    cumesh.bvh.cuBVH.unsigned_distance(return_uvw=True). Per CuMesh/cubvh's own
    barycentric() convention (third_party/cubvh/include/gpu/triangle.cuh: returns
    Vector3f(u, v, w) with position = u*a + v*b + w*c for that face's 3 vertices in
    (a, b, c) = face order) -- confirmed by reading the CUDA source directly rather than
    assumed, per this doc's own instruction not to guess cuBVH's output layout.

    Takes/returns a plain (V, 3) array rather than a trimesh.Trimesh: this only moves
    points, never changes topology, so it applies equally to a triangulated copy (Phase
    1's load_quad_obj output) or genuine quad-mesh vertex positions directly -- same
    coordinates either way, only the caller's topology differs.
    """
    from cumesh.bvh import cuBVH

    bvh = cuBVH(reference_mesh.vertices, reference_mesh.faces)
    pts = torch.tensor(np.asarray(vertices), dtype=torch.float32, device=device)
    _, face_id, uvw = bvh.unsigned_distance(pts, return_uvw=True)

    ref_faces = torch.tensor(reference_mesh.faces, dtype=torch.long, device=device)
    ref_verts = torch.tensor(reference_mesh.vertices, dtype=torch.float32, device=device)
    tri = ref_faces[face_id]  # (V, 3) -> vertex indices of each point's nearest face
    a, b, c = ref_verts[tri[:, 0]], ref_verts[tri[:, 1]], ref_verts[tri[:, 2]]
    snapped = uvw[:, 0:1] * a + uvw[:, 1:2] * b + uvw[:, 2:3] * c
    return snapped.detach().cpu().numpy().astype(np.float64)


def measure_snap_displacement(vertices: np.ndarray, snapped: np.ndarray) -> dict:
    """Per-vertex displacement stats between pre- and post-snap positions -- Phase 2's
    acceptance check requires measuring, not assuming, how much this actually moves
    things."""
    d = np.linalg.norm(np.asarray(snapped) - np.asarray(vertices), axis=1)
    return {"mean": float(d.mean()), "max": float(d.max()), "median": float(np.median(d))}


def remove_small_quad_islands(vertices: np.ndarray, faces: List[np.ndarray], min_faces: int = 8
                               ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Drop disconnected face-adjacency components smaller than min_faces -- mirrors
    mesh_repair.py's remove_small_islands(), but for a general n-gon face list rather
    than a trimesh.Trimesh's fixed-arity triangle faces.

    Needed because QuadriFlow can itself produce a heavily fragmented quad mesh at high
    target_faces on a full-detail original mesh, not only on MeshAnything's coarse
    output: confirmed empirically -- target_faces=20000 directly on can.glb's full
    original (which has real surface relief: embossed text, rivets) produced 1608
    disconnected connected components, the largest covering only 12.4% of the mesh's
    faces. Left unfiltered, this silently corrupts everything downstream: a UV-unwrap
    over a mesh whose faces are 87% tiny disconnected fragments produces a final
    textured mesh covering only ~12% of the true surface area (confirmed: 0.243 vs.
    the reference's 1.927) even though every individual step "succeeds" with no error --
    this is not caught by quad_fraction (still 100%) or even Phase 4's own
    top5_area_frac chart-quality gate, since that only measures how well charts are
    laid out *within* whatever faces survive, not whether most of the mesh's true
    surface got lost before charting ever started.
    """
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            edge_to_faces[frozenset((a, b))].append(fi)

    adjacency = defaultdict(set)
    for e, fs in edge_to_faces.items():
        if len(fs) == 2:
            f1, f2 = fs
            adjacency[f1].add(f2)
            adjacency[f2].add(f1)

    visited = np.zeros(len(faces), dtype=bool)
    keep_face_ids = []
    for start in range(len(faces)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in adjacency[f]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) >= min_faces:
            keep_face_ids.extend(comp)

    kept_faces = [faces[fi] for fi in sorted(keep_face_ids)]
    used_verts = sorted({int(i) for face in kept_faces for i in face})
    remap = {old: new for new, old in enumerate(used_verts)}
    new_faces = [np.array([remap[int(i)] for i in face], dtype=np.int64) for face in kept_faces]
    new_vertices = vertices[used_verts]
    return new_vertices, new_faces


def find_irregular_vertices(faces: List[np.ndarray], n_vertices: int) -> np.ndarray:
    """
    Vertex indices with valence != 4 (degree in the face-edge graph), computed directly
    from face-vertex incidence -- no need to extract this from QuadriFlow's internal
    singularities map (not exposed in the output .obj anyway). Boundary vertices are
    naturally lower-valence and are flagged too by this simple check; callers that only
    care about interior singularities should cross-reference against the mesh boundary
    separately.
    """
    edges = face_edges(faces)
    valence = np.zeros(n_vertices, dtype=np.int64)
    if len(edges) > 0:
        np.add.at(valence, edges[:, 0], 1)
        np.add.at(valence, edges[:, 1], 1)
    return np.where(valence != 4)[0]


def _vertex_neighbors(faces: List[np.ndarray]) -> Dict[int, set]:
    """vertex -> set of directly-connected neighbor vertices."""
    neighbors = defaultdict(set)
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            neighbors[a].add(b)
            neighbors[b].add(a)
    return neighbors


def _vertex_face_neighbor_pairs(faces: List[np.ndarray]) -> Dict[int, List[Tuple[int, int]]]:
    """vertex -> list of (prev, next) neighbor pairs, one per face the vertex belongs to
    (prev/next being that vertex's two face-adjacent neighbors within that face's
    winding order). Used to determine which pairs of edges at a vertex are "adjacent"
    (share a face) versus "opposite" (share no face) -- the latter is the straight-ahead
    continuation for separatrix tracing."""
    vf = defaultdict(list)
    for face in faces:
        n = len(face)
        for i in range(n):
            v = int(face[i])
            prev_v = int(face[(i - 1) % n])
            next_v = int(face[(i + 1) % n])
            vf[v].append((prev_v, next_v))
    return vf


def _straight_continuation(vf_pairs: List[Tuple[int, int]], arrival: int, neighbors: set) -> Optional[int]:
    """
    At a regular (valence-4) vertex, given the neighbor we arrived from, find the unique
    "straight ahead" neighbor to continue a separatrix toward: the one neighbor that
    shares NO face with the arrival neighbor at this vertex (the 2 other neighbors are
    each "adjacent" -- co-incident in some face with the arrival edge -- and therefore a
    turn, not straight-through). Returns None if the structure is ambiguous/degenerate
    (e.g. a non-quad face at this vertex), in which case the caller should stop tracing.
    """
    adjacent_to_arrival = set()
    for (p, nx) in vf_pairs:
        if p == arrival:
            adjacent_to_arrival.add(nx)
        elif nx == arrival:
            adjacent_to_arrival.add(p)
    candidates = neighbors - {arrival} - adjacent_to_arrival
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def trace_separatrices(faces: List[np.ndarray], n_vertices: int, max_length: Optional[int] = 8) -> List[List[int]]:
    """
    Trace motorcycle-graph separatrices: straight quad-grid lines starting at each
    irregular (valence != 4, including boundary -- see find_irregular_vertices) vertex,
    continuing through regular valence-4 vertices via the unique "straight ahead"
    neighbor (_straight_continuation). A trace ends, successfully, either by reaching
    another irregular vertex (a singularity or the mesh boundary) or by crashing into an
    edge an earlier trace already claimed (genuine motorcycle-graph semantics --
    Eppstein/Goodrich/Kim/Tamstorf: motorcycles stop when they hit another motorcycle's
    trail, not only at singularities).

    `max_length` (default 8) discards any trace that does *neither* within that many
    edges. This is not optional tuning -- without it, tracing from every singularity in
    every direction and cutting along literally everything it touches is
    mathematically guaranteed to claim nearly every edge in the mesh whenever
    singularities are sparse (every regular vertex has a *unique*, deterministic
    straight-ahead pairing, so the maximal straight line through any edge always
    eventually reaches a singularity if followed far enough -- there's nothing for a
    "crash" to prevent when the whole mesh is one connected regular region). Confirmed
    empirically before adding the cap: uncapped, this claimed 671/671 edges on a
    334-face capped cylinder (334 patches, all singletons) and 1080 patches -- 1009 of
    them singletons -- on a 1349-face hand mesh. With max_length=8, the cylinder
    resolves to one dominant ~241-face body patch plus small fragments that (checked
    directly) all sit right at the two polar singularity regions -- structurally
    sensible, not noise. A trace that hits the budget without terminating is discarded
    entirely (none of its edges are committed), not truncated -- a partial "wandering"
    line isn't a meaningful cut either.

    Returns a list of polylines (each a list of vertex indices). Traces are processed in
    a fixed (sorted) order for determinism, since which trace "gets there first" and
    claims an edge affects every later trace's crash point.
    """
    irregular = set(find_irregular_vertices(faces, n_vertices).tolist())
    if len(irregular) == 0:
        return []

    neighbors = _vertex_neighbors(faces)
    vf = _vertex_face_neighbor_pairs(faces)

    consumed_edges = set()
    separatrices = []

    start_pairs = sorted((v0, v1) for v0 in irregular for v1 in neighbors[v0])
    for v0, v1 in start_pairs:
        e0 = frozenset((v0, v1))
        if e0 in consumed_edges:
            continue
        path = [v0, v1]
        local_edges = {e0}
        prev, cur = v0, v1
        exceeded_budget = False
        while cur not in irregular:
            if max_length is not None and (len(path) - 1) >= max_length:
                exceeded_budget = True
                break
            nxt = _straight_continuation(vf.get(cur, []), prev, neighbors[cur])
            if nxt is None:
                break  # graceful stop at the edge of well-defined quad structure
            e = frozenset((cur, nxt))
            if e in consumed_edges:
                break  # crashed into an existing trail -- valid termination
            local_edges.add(e)
            path.append(nxt)
            prev, cur = cur, nxt
        if exceeded_budget:
            continue  # too long to be a "local" separatrix -- discard, don't commit or cut along it
        consumed_edges |= local_edges
        separatrices.append(path)

    return separatrices


def patches_from_separatrices(faces: List[np.ndarray], separatrices: List[List[int]]) -> List[np.ndarray]:
    """
    Partition the mesh's faces into connected regions ("patches") bounded by the
    separatrices: flood-fill face adjacency, refusing to cross any edge that belongs to
    a separatrix. Returns a list of face-index arrays, one per patch.
    """
    cut_edges = set()
    for path in separatrices:
        for i in range(len(path) - 1):
            cut_edges.add(frozenset((path[i], path[i + 1])))

    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            edge_to_faces[frozenset((a, b))].append(fi)

    adjacency = defaultdict(set)
    for e, fs in edge_to_faces.items():
        if e in cut_edges or len(fs) != 2:
            continue
        f1, f2 = fs
        adjacency[f1].add(f2)
        adjacency[f2].add(f1)

    visited = np.zeros(len(faces), dtype=bool)
    patches = []
    for start in range(len(faces)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in adjacency[f]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        patches.append(np.array(comp, dtype=np.int64))
    return patches


def _triangulate_fan(face: np.ndarray) -> List[Tuple[int, int, int]]:
    """Fan-triangulate a single n-gon face (n>=3) -- same diagonal convention trimesh
    uses for a quad (0,1,2)+(0,2,3), so triangle counts/topology match Phase 1's
    triangulation check."""
    return [(int(face[0]), int(face[i]), int(face[i + 1])) for i in range(1, len(face) - 1)]


def uv_unwrap_via_patches(vertices: np.ndarray, faces: List[np.ndarray], patches: List[np.ndarray],
                           xatlas_compute_charts_kwargs: dict = None,
                           xatlas_pack_charts_kwargs: dict = None):
    """
    UV-unwrap by feeding each motorcycle-graph patch into cumesh.xatlas.Atlas as its own
    add_mesh() call, then running one shared compute_charts/pack_charts across all of
    them -- the exact same pattern CuMesh.uv_unwrap() uses for its curvature pre-clusters
    (cumesh/cumesh.py:451-460), just with patches (bounded by genuine quad-grid
    singularities) as the groups instead of curvature-based clusters. xatlas itself only
    ever sees triangles (matches CuMesh's own internal convention), so each patch's faces
    are fan-triangulated first -- the genuine quad structure only matters for finding the
    patch boundaries, not for this final parameterization step.

    Returns (vertices, faces, uvs) as numpy arrays, matching CuMesh.uv_unwrap()'s
    contract minus the vmap (not needed by callers here -- Phase 5 re-queries positions
    via interpolate_uv in UV space directly, not by tracing back to pre-unwrap indices).
    """
    from cumesh.xatlas import Atlas

    xatlas_compute_charts_kwargs = xatlas_compute_charts_kwargs or {}
    xatlas_pack_charts_kwargs = xatlas_pack_charts_kwargs or {}

    atlas = Atlas()
    patch_vmaps = []
    for patch_face_ids in patches:
        tri_faces_global = []
        for fi in patch_face_ids:
            tri_faces_global.extend(_triangulate_fan(faces[fi]))
        tri_faces_global = np.array(tri_faces_global, dtype=np.int64)

        unique_verts, local_faces_flat = np.unique(tri_faces_global.reshape(-1), return_inverse=True)
        local_faces = local_faces_flat.reshape(tri_faces_global.shape)
        local_vertices = vertices[unique_verts]

        patch_vmaps.append(unique_verts)
        atlas.add_mesh(
            torch.tensor(local_vertices, dtype=torch.float32).contiguous(),
            torch.tensor(local_faces, dtype=torch.int32).contiguous(),
        )

    atlas.compute_charts(**xatlas_compute_charts_kwargs)
    atlas.pack_charts(**xatlas_pack_charts_kwargs)

    out_vertices, out_faces, out_uvs = [], [], []
    cnt = 0
    for i, vmap in enumerate(patch_vmaps):
        xrefs, x_faces, x_uvs = atlas.get_mesh(i)
        global_idx = vmap[xrefs.numpy()]
        out_vertices.append(vertices[global_idx])
        out_faces.append(x_faces.numpy() + cnt)
        out_uvs.append(x_uvs.numpy())
        cnt += len(global_idx)

    return (np.concatenate(out_vertices, axis=0),
            np.concatenate(out_faces, axis=0),
            np.concatenate(out_uvs, axis=0))


def uv_unwrap_auto(vertices: np.ndarray, faces: List[np.ndarray], max_length: int = 8,
                    top5_frac_threshold: float = 0.3, device: str = "cuda", verbose: bool = True):
    """
    Phase 4's actual recommended entry point: try the motorcycle-graph patch layout
    (task 2) first, then fall back to tuned curvature clustering (task 1,
    area_penalty_weight=0.0) if task 2's result doesn't clear a minimum quality bar --
    exactly the comparison the plan itself calls for ("compare both, per asset, don't
    assume the structural approach always wins... fall back to that asset's task-1
    result"), automated with `top5_area_frac` (the same objective "few large charts"
    proxy used throughout this phase) as the quality gate.

    This exists because task 2's `max_length` is an absolute hop count, tuned against a
    small (334-face) coarse mesh -- it does not automatically scale to a much
    higher-resolution quad mesh. Confirmed empirically: running QuadriFlow at
    target_faces=20000 directly on can.glb's full-detail original (surface relief --
    embossed text, rivets -- genuinely needs far more singularities than a smoothed
    coarse mesh does) produced 1608 disconnected connected-components and 69% of
    vertices flagged irregular; with max_length=8, patches_from_separatrices produced
    10443 patches at top5_area_frac=0.069 -- worse than useless, and *raising* the cap
    only made it worse still (more, not fewer, patches), since more separatrices survive
    long enough to crash into each other, adding cuts rather than removing them. Task 1's
    curvature clustering has no such resolution-dependent failure mode, making it the
    right fallback rather than trying to hand-tune max_length per asset.
    """
    seps = trace_separatrices(faces, len(vertices), max_length=max_length)
    patches = patches_from_separatrices(faces, seps)
    sizes = sorted((len(p) for p in patches), reverse=True)
    top5_frac = sum(sizes[:5]) / max(sum(sizes), 1)

    if top5_frac >= top5_frac_threshold:
        if verbose:
            print(f"  motorcycle-graph patch layout: {len(patches)} charts, "
                  f"top5_area_frac={top5_frac:.3f} -- using it")
        return uv_unwrap_via_patches(vertices, faces, patches)

    if verbose:
        print(f"  motorcycle-graph patch layout only reached top5_area_frac={top5_frac:.3f} "
              f"(< {top5_frac_threshold}) -- falling back to tuned curvature clustering (task 1)")
    import cumesh

    tri_faces = []
    for face in faces:
        tri_faces.extend(_triangulate_fan(face))
    verts_t = torch.tensor(vertices, dtype=torch.float32, device=device).contiguous()
    faces_t = torch.tensor(np.array(tri_faces), dtype=torch.int32, device=device).contiguous()

    cm = cumesh.CuMesh()
    cm.init(verts_t, faces_t)
    out_v, out_f, out_uv = cm.uv_unwrap(compute_charts_kwargs={"area_penalty_weight": 0.0})
    return out_v.cpu().numpy(), out_f.cpu().numpy(), out_uv.cpu().numpy()


def _has_baked_texture(mesh: trimesh.Trimesh) -> bool:
    return (hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
            and hasattr(mesh.visual, "material")
            and getattr(mesh.visual.material, "baseColorTexture", None) is not None)


def _sample_texture_at_uv(image: Image.Image, uv: torch.Tensor) -> torch.Tensor:
    """
    Bilinear-sample a PIL texture image at (u, v) coordinates in the mesh's native UV
    convention. Mirrors trellis2_texturing.py:301's `uvs[:, 1] = 1 - uvs[:, 1]` flip --
    empirically the convention this codebase's meshes actually load with (confirmed:
    that flip is applied before treating v as a direct row index for rasterize_uv,
    i.e. v=0 is the *bottom* of the texture, not the top) -- so the same flip is applied
    here before converting to grid_sample's [-1, 1] normalized coordinates.
    """
    device = uv.device
    img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    img_t = torch.tensor(img, device=device).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    grid_x = 2 * uv[:, 0] - 1
    grid_y = 2 * (1 - uv[:, 1]) - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(img_t, grid, mode="bilinear", align_corners=False, padding_mode="border")
    return sampled[0, :, 0, :].permute(1, 0)  # (N, 3)


def transfer_texture(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray,
                      reference_mesh: trimesh.Trimesh, texture_size: int = 1024,
                      device: str = "cuda") -> trimesh.Trimesh:
    """
    Bake color onto (vertices, faces, uvs) -- typically Phase 4's UV-unwrap output --
    sourced from reference_mesh's own already-baked texture via closest-point
    projection (mesh-to-mesh transfer). This is NOT a re-run of postprocess_mesh's live
    voxel-grid sampling: this standalone tool operates on an already-exported GLB and
    has no access to the live PBR-attribute voxel tensor that only exists inside
    run_sample.py's pipeline (see quadriflow_postprocess_plan.md's Hard Constraints).

    If reference_mesh has no material/UV/baked texture at all, returns bare (untextured)
    geometry rather than fabricating texture data.

    Flips the output UV V-coordinate to match trellis2_texturing.py's own xatlas-output
    convention fix (postprocess_mesh:356) -- this is a correction for xatlas's raw UV
    layout, unrelated to vertex coordinates. Deliberately does NOT apply
    postprocess_mesh's vertex/normal axis swap (swap Y/Z, negate Y): that swap corrects
    TRELLIS2's internal (pre-export) mesh convention into GLB's, but this function's
    `reference_mesh` contract (per quadriflow_postprocess_plan.md) is always an
    *already-exported* GLB (e.g. tests/*.glb, or run_sample.py's own output) -- already
    in final GLB convention. Verified empirically: tests/can.glb's own extents already
    have Y as the tall axis (a standing can), and the output mesh's vertices are already
    in that same frame by construction (Phase 2's snap_to_reference operates directly
    against reference_mesh's coordinates) -- applying the swap again visibly rotated a
    correctly-oriented can onto its side.
    """
    if not _has_baked_texture(reference_mesh):
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    verts_t = torch.tensor(vertices, dtype=torch.float32, device=device).contiguous()
    faces_t = torch.tensor(faces, dtype=torch.long, device=device).contiguous()
    uvs_t = torch.tensor(uvs, dtype=torch.float32, device=device).contiguous()

    from o_voxel.uv_rasterize import rasterize_uv, interpolate_uv

    rast, _ = rasterize_uv(uvs_t, faces_t.int(), texture_size, texture_size)
    mask = rast[0, ..., 3] > 0
    pos = interpolate_uv(verts_t.unsqueeze(0), rast, faces_t)[0][0]  # (H, W, 3)
    texel_positions = pos[mask]  # (P, 3)

    from cumesh.bvh import cuBVH
    bvh = cuBVH(reference_mesh.vertices, reference_mesh.faces)
    _, face_id, uvw = bvh.unsigned_distance(texel_positions, return_uvw=True)

    ref_faces_t = torch.tensor(reference_mesh.faces, dtype=torch.long, device=device)
    ref_uv_t = torch.tensor(reference_mesh.visual.uv, dtype=torch.float32, device=device)
    tri = ref_faces_t[face_id]  # (P, 3) -> reference_mesh's nearest-face vertex indices
    uv_a, uv_b, uv_c = ref_uv_t[tri[:, 0]], ref_uv_t[tri[:, 1]], ref_uv_t[tri[:, 2]]
    ref_uv = uvw[:, 0:1] * uv_a + uvw[:, 1:2] * uv_b + uvw[:, 2:3] * uv_c  # (P, 2)

    sampled = _sample_texture_at_uv(reference_mesh.visual.material.baseColorTexture, ref_uv)  # (P, 3)

    texture = torch.zeros(texture_size, texture_size, 3, device=device)
    texture[mask] = sampled
    texture_np = (texture.clamp(0, 1) * 255).byte().cpu().numpy()

    # Extend into uncovered texels (background, or no valid reference UV/texture at the
    # queried point) -- same mask-extension pattern trellis2_texturing.py:337-341 uses.
    uncovered = (~mask.cpu().numpy()).astype(np.uint8)
    texture_np = cv2.inpaint(texture_np, uncovered, 3, cv2.INPAINT_TELEA)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture_np),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=True,
    )

    out_uvs = uvs.copy()
    out_uvs[:, 1] = 1 - out_uvs[:, 1]

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=out_uvs, material=material),
    )
