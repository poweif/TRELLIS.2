import pymeshlab


def subdivide_catmull_clark(quad_obj_path: str, output_obj_path: str, levels: int = 1):
    """
    Genuine n-gon-aware Catmull-Clark subdivision, via pymeshlab's
    meshing_surface_subdivision_catmull_clark filter.

    Verified empirically (not assumed) that this operates on the real quad/n-gon
    topology, not a plain triangulation: MeshLab/VCG's internal mesh format always
    stores triangulated faces, but groups them into their original n-gon via "faux"
    (fake/internal) diagonal edges, and the Catmull-Clark filter respects that grouping
    -- subdividing a 334-quad mesh 1 level produced exactly 334*4=1336 new quad faces
    (Catmull-Clark's "each face -> n new quads" rule), not the ~2x that number a
    naively-triangulated 668-triangle input would produce if the diagonal were treated
    as a real edge.

    Writes genuine quad-preserving output to output_obj_path (never triangulated),
    same "quad structure only survives in the .obj file" contract as
    quad_remesh.py's remesh_to_quad_dominant_obj.

    Only a single filter parameter exists (`iterations`) -- levels default to 1, not the
    2-3 an earlier (superseded) plan assumed; see quadriflow_postprocess_plan.md Phase 3
    for why (detail recovery is Phase 2's job now, not subdivision's -- this step is
    purely about silhouette/surface smoothness, evaluate visually before increasing).
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(quad_obj_path)
    ms.meshing_surface_subdivision_catmull_clark(iterations=levels)
    ms.save_current_mesh(output_obj_path, save_face_color=False, save_vertex_color=False)
