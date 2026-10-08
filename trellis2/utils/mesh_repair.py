import numpy as np
import trimesh
import trimesh.repair
import trimesh.graph


def remove_small_islands(mesh: trimesh.Trimesh, min_faces: int = 8) -> trimesh.Trimesh:
    """
    Drop disconnected face-adjacency components smaller than min_faces. Real-world
    asset exports (this was found on tests/watch.glb) can contain thousands of
    degenerate 1-3 face fragments -- marching-cubes noise, sub-pixel floaters -- that
    inflate the mesh's component count without contributing real geometry. Beyond being
    junk, these fragments break whole-mesh winding-consistency checks in a way that has
    nothing to do with whether the substantive geometry is correctly wound: winding
    consistency is only a meaningful concept *within* a connected patch, since
    disconnected pieces have no shared reference frame to agree or disagree on in the
    first place. Filtering them out first makes both decimation and winding-repair work
    on the geometry that actually matters instead of drowning in noise.
    """
    if len(mesh.faces) == 0:
        return mesh
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=1, nodes=np.arange(len(mesh.faces)))
    keep_mask = np.zeros(len(mesh.faces), dtype=bool)
    for comp in components:
        if len(comp) >= min_faces:
            keep_mask[comp] = True
    mesh = mesh.copy()
    mesh.update_faces(keep_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def remove_nonmanifold_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Remove faces touching any edge shared by more than 2 faces (non-manifold edges).
    These break true 2-manifold orientability, which in turn breaks
    trimesh.repair.fix_winding's algorithm: it only verifies/corrects winding along a
    BFS spanning tree of the face-adjacency graph, relying on a genuinely orientable
    manifold's spanning-tree-consistent orientation automatically satisfying every
    non-tree ("loop-closing") edge too. Non-manifold edges violate that assumption, so
    fix_winding can silently leave loop-closing edges inconsistent even after running --
    exactly what was observed on tests/can.glb's decimated mesh (one connected component
    covering nearly the whole mesh still reporting inconsistent winding despite
    fix_normals/fix_winding/orient_triangles all having already run on it). Non-manifold
    edges are typically a tiny fraction of a mesh (observed: 89 edges / 287 faces out of
    17711 on the can), so wholesale removal of the faces touching them is a fine trade.
    """
    if len(mesh.faces) == 0:
        return mesh
    edges = mesh.edges_sorted
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    nonmanifold_edge_ids = np.where(counts > 2)[0]
    if len(nonmanifold_edge_ids) == 0:
        return mesh
    bad_edge_mask = np.isin(inverse, nonmanifold_edge_ids).reshape(-1, 3)
    bad_faces = np.where(bad_edge_mask.any(axis=1))[0]
    keep_mask = np.ones(len(mesh.faces), dtype=bool)
    keep_mask[bad_faces] = False
    mesh = mesh.copy()
    mesh.update_faces(keep_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def is_winding_consistent_per_component(mesh: trimesh.Trimesh, min_faces: int = 8) -> bool:
    """
    The meaningful version of `mesh.is_winding_consistent` for a mesh that legitimately
    has multiple disconnected solid parts (e.g. a watch case + separate band links):
    checks that each component with at least min_faces faces is *internally* consistent,
    without requiring unrelated disconnected components to agree with each other (a
    requirement that doesn't correspond to anything physically meaningful). Small
    sub-min_faces fragments are ignored, matching remove_small_islands' threshold.
    """
    if len(mesh.faces) == 0:
        return True
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=1, nodes=np.arange(len(mesh.faces)))
    for comp in components:
        if len(comp) < min_faces:
            continue
        sub = mesh.submesh([comp], append=True)
        if not sub.is_winding_consistent:
            return False
    return True


def repair_mesh(mesh: trimesh.Trimesh, fill_holes: bool = True, min_island_faces: int = 8) -> trimesh.Trimesh:
    """
    Standard cleanup + winding/normal consistency repair. Meant to be called both before
    and after a decimation step, not just once -- decimation algorithms (especially
    open3d's vertex clustering / QEM) can both amplify pre-existing topology defects in
    the input and introduce new ones of their own.

    Order matters: small islands and non-manifold faces are dropped *first* (see
    remove_small_islands, remove_nonmanifold_faces), before fill_holes and fix_normals
    run. Otherwise fill_holes wastes effort capping thousands of tiny noise fragments
    (observed on tests/watch.glb: it added ~20K faces this way without fixing anything),
    and fix_normals'/fix_winding's BFS-spanning-tree algorithm can silently fail to reach
    full consistency in the presence of non-manifold edges (observed on tests/can.glb).
    """
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if min_island_faces > 0:
        mesh = remove_small_islands(mesh, min_faces=min_island_faces)
    mesh = remove_nonmanifold_faces(mesh)
    if fill_holes:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass  # best-effort -- a mesh with holes trimesh can't close is still usable
        # fill_holes can itself reintroduce non-manifold edges at the seams of the caps
        # it adds (confirmed on tests/can.glb) -- re-clean afterward rather than assuming
        # one non-manifold pass before fill_holes is enough.
        mesh = remove_nonmanifold_faces(mesh)
    mesh.fix_normals()
    return mesh


def repair_mesh_after_decimation(vertices, faces, o3d_mesh=None, min_island_faces: int = 8) -> trimesh.Trimesh:
    """
    Rebuild a trimesh.Trimesh from raw decimation output (vertices/faces arrays) and
    apply orientation repair from two independent implementations: open3d's own
    `orient_triangles()` (run on the o3d mesh before conversion, if provided) plus
    trimesh's `fix_normals()` (run after, via repair_mesh). Belt-and-suspenders rather
    than relying on either one alone.

    Verification uses is_winding_consistent_per_component, not the raw whole-mesh
    property -- a mesh with legitimately-separate solid parts (or leftover
    below-threshold noise) will never satisfy the whole-mesh check even when every real
    piece of geometry is correctly wound, so that check was firing false alarms in
    testing (tests/watch.glb: still "inconsistent" by the whole-mesh metric even though
    all 30 of its largest components were, individually, fine).
    """
    if o3d_mesh is not None:
        try:
            o3d_mesh.orient_triangles()
            vertices = np.asarray(o3d_mesh.vertices)
            faces = np.asarray(o3d_mesh.triangles)
        except Exception:
            pass

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh = repair_mesh(mesh, fill_holes=True, min_island_faces=min_island_faces)

    if not is_winding_consistent_per_component(mesh, min_faces=min_island_faces):
        # Last resort: trimesh's own explicit winding-fix pass, in case fix_normals'
        # combined winding+orientation logic missed something fix_winding alone catches.
        trimesh.repair.fix_winding(mesh)

    return mesh
