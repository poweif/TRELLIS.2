"""CPU-only checks for the quad post-processing helpers (no GPU work, no model downloads)."""
import numpy as np
import pytest
import trimesh

from trellis2.utils.curvature import compute_rho
from trellis2.utils.quad_postprocess import (
    face_edges,
    find_irregular_vertices,
    load_quad_obj,
    parse_obj_raw,
    parse_quad_faces,
    quad_fraction,
    remove_small_quad_islands,
    write_quad_obj,
)

# Unit cube as 6 quads.
CUBE_V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
CUBE_F = [np.array(f, dtype=np.int64) for f in (
    [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3])]


def grid(n):
    """n x n open quad grid in the z=0 plane: (n+1)^2 vertices, n^2 quads."""
    v = np.array([[i, j, 0.0] for i in range(n + 1) for j in range(n + 1)])
    f = [np.array([i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1])
         for i in range(n) for j in range(n)]
    return v, f


def test_obj_round_trip_keeps_ngons(tmp_path):
    faces = CUBE_F[:5] + [np.array([1, 5, 7]), np.array([1, 7, 3])]  # 5 quads + 2 triangles
    path = tmp_path / "mixed.obj"
    write_quad_obj(CUBE_V, faces, str(path))
    v, f = parse_obj_raw(str(path))
    np.testing.assert_allclose(v, CUBE_V)
    assert [list(a) for a in f] == [list(a) for a in faces]
    assert parse_quad_faces(str(path)) == [4] * 5 + [3] * 2
    assert quad_fraction(str(path)) == pytest.approx(5 / 7)


def test_parse_obj_raw_handles_slashes_and_negative_indices(tmp_path):
    path = tmp_path / "rel.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf -4/1 -3/1 -2/1 -1/1\n")
    v, f = parse_obj_raw(str(path))
    assert v.shape == (4, 3)
    assert list(f[0]) == [0, 1, 2, 3]


def test_parse_obj_raw_empty_has_shape_0_3(tmp_path):
    path = tmp_path / "empty.obj"
    path.write_text("# nothing\n")
    v, f = parse_obj_raw(str(path))
    assert v.shape == (0, 3) and f == []


def test_trimesh_triangulates_quads(tmp_path):
    path = tmp_path / "cube.obj"
    write_quad_obj(CUBE_V, CUBE_F, str(path))
    assert quad_fraction(str(path)) == 1.0
    assert len(load_quad_obj(str(path)).faces) == 2 * len(CUBE_F)


def test_face_edges_cube():
    edges = face_edges(CUBE_F)
    assert edges.shape == (12, 2)
    assert (edges[:, 0] < edges[:, 1]).all()


def test_irregular_vertices():
    # Every cube corner has valence 3.
    assert list(find_irregular_vertices(CUBE_F, len(CUBE_V))) == list(range(8))
    # In an open 4x4 grid only the 3x3 interior vertices have valence 4.
    v, f = grid(4)
    irregular = set(find_irregular_vertices(f, len(v)).tolist())
    interior = {i * 5 + j for i in range(1, 4) for j in range(1, 4)}
    assert irregular == set(range(len(v))) - interior


def test_remove_small_quad_islands_drops_fragments_and_reindexes():
    gv, gf = grid(3)                                    # 9 connected quads
    offset = len(gv)
    lone = np.array([[10, 0, 0], [11, 0, 0], [11, 1, 0], [10, 1, 0]], dtype=np.float64)
    vertices = np.vstack([gv, lone])
    faces = gf + [np.array([offset, offset + 1, offset + 2, offset + 3])]
    v, f = remove_small_quad_islands(vertices, faces, min_faces=2)
    assert len(f) == 9 and len(v) == len(gv)
    assert max(int(i) for face in f for i in face) == len(v) - 1


@pytest.mark.parametrize("radius", [0.5, 2.0])
def test_compute_rho_sphere_is_radius(radius):
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    rho = compute_rho(np.asarray(sphere.vertices), np.asarray(sphere.faces))
    assert rho.shape == (len(sphere.vertices),)
    assert np.median(rho) == pytest.approx(radius, rel=0.05)
