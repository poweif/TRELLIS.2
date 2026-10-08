import numpy as np
import torch

_TWO_PI = 2.0 * np.pi


def compute_rho(vertices: np.ndarray, faces: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-vertex curvature-adaptive "rho" (target local edge length / curvature radius) for
    QuadriFlow's adaptive quad density (see quadriflow_ext.remesh's `rho` argument, and
    docs/archive/quadriflow_postprocess_plan.md's adaptive-density section for why this exists and how it
    connects to QuadriFlow's `optimize_scale`).

    Recovers principal curvature *magnitudes* via mean curvature H (cotangent-Laplacian of
    vertex positions, Meyer/Desbrun/Schroder/Barr 2003) and Gaussian curvature K (angle-defect /
    discrete Gauss-Bonnet), then `k1, k2 = H +/- sqrt(max(H^2 - K, 0))`. Only magnitude is
    needed here -- QuadriFlow's own cross-field solve already handles orientation -- so this
    deliberately avoids libigl's quadric-fitting `principal_curvature()` (which also recovers
    principal *directions*, at the cost of a fragile git-submodule dependency upstream
    ultimately removed for exactly that fragility -- see the plan doc's investigation).

    rho = 1 / max(|k1|, |k2|), i.e. the radius of the tightest principal curvature at each
    vertex -- returned in the SAME (original, un-normalized) coordinate units as `vertices`.
    QuadriFlow's own binding divides by its internal normalize_scale before use (rho has units
    of length, exactly like vertex positions, so it needs the identical normalization) -- do not
    pre-normalize here.

    Uses the barycentric mixed area (each vertex gets 1/3 of each incident face's area) rather
    than the fully rigorous obtuse-aware Voronoi mixed area from Meyer et al. -- a common,
    robust simplification; adequate here since `optimize_scale` only ever applies rho as a
    bounded 0.75x-1.0x local scale adjustment (third_party/QuadriFlow/src/optimizer.cpp:147-165),
    not a free-form density field, so exact area precision doesn't matter much.

    vertices: [N, 3], faces: [M, 3] (triangles only).
    """
    V = torch.as_tensor(vertices, dtype=torch.float64)
    F = torch.as_tensor(faces, dtype=torch.long)
    n_v = V.shape[0]

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    def angle_and_cot(a, b):
        cross = torch.linalg.cross(a, b, dim=-1)
        cross_norm = torch.linalg.norm(cross, dim=-1)
        dot = (a * b).sum(-1)
        angle = torch.atan2(cross_norm, dot)
        cot = dot / cross_norm.clamp_min(eps)
        return angle, cot

    # angle_k / cot_k: interior angle (and its cotangent) at vertex k of each face, i.e. the
    # angle opposite the edge formed by the OTHER two vertices of that face.
    angle0, cot0 = angle_and_cot(v1 - v0, v2 - v0)  # at vertex 0, opposite edge (1,2)
    angle1, cot1 = angle_and_cot(v0 - v1, v2 - v1)  # at vertex 1, opposite edge (0,2)
    angle2, cot2 = angle_and_cot(v0 - v2, v1 - v2)  # at vertex 2, opposite edge (0,1)

    face_area = 0.5 * torch.linalg.norm(torch.linalg.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

    vertex_area = torch.zeros(n_v, dtype=torch.float64)
    third_area = face_area / 3.0
    vertex_area.index_add_(0, F[:, 0], third_area)
    vertex_area.index_add_(0, F[:, 1], third_area)
    vertex_area.index_add_(0, F[:, 2], third_area)
    vertex_area_safe = vertex_area.clamp_min(eps)

    # Gaussian curvature via angle defect (discrete Gauss-Bonnet): K_i = (2*pi - sum of
    # incident interior angles) / A_i.
    angle_sum = torch.zeros(n_v, dtype=torch.float64)
    angle_sum.index_add_(0, F[:, 0], angle0)
    angle_sum.index_add_(0, F[:, 1], angle1)
    angle_sum.index_add_(0, F[:, 2], angle2)
    K = (_TWO_PI - angle_sum) / vertex_area_safe

    # Mean curvature via cotangent-Laplacian of position: HN_i = (1/(2*A_i)) * sum_j
    # (cot_alpha + cot_beta) * (x_j - x_i). Each face contributes cot_k*(x_j - x_i) to both
    # endpoints of the edge opposite vertex k; summing both faces sharing an edge naturally
    # gives the (cot_alpha + cot_beta) term -- no extra per-face 0.5 needed, that factor
    # belongs solely to the final 1/(2*A_i) below.
    lap = torch.zeros(n_v, 3, dtype=torch.float64)
    lap.index_add_(0, F[:, 1], cot0.unsqueeze(-1) * (v2 - v1))
    lap.index_add_(0, F[:, 2], cot0.unsqueeze(-1) * (v1 - v2))
    lap.index_add_(0, F[:, 0], cot1.unsqueeze(-1) * (v2 - v0))
    lap.index_add_(0, F[:, 2], cot1.unsqueeze(-1) * (v0 - v2))
    lap.index_add_(0, F[:, 0], cot2.unsqueeze(-1) * (v1 - v0))
    lap.index_add_(0, F[:, 1], cot2.unsqueeze(-1) * (v0 - v1))

    HN = lap / (2.0 * vertex_area_safe).unsqueeze(-1)
    # H = |HN| / 2 (standard mean-curvature-normal -> scalar mean curvature relation). Sign is
    # discarded (norm is always >= 0) -- irrelevant here: negating H swaps which of k1/k2 is
    # which but leaves the set {|k1|, |k2|} -- and therefore max(|k1|,|k2|) -- unchanged.
    H = 0.5 * torch.linalg.norm(HN, dim=-1)

    disc = (H * H - K).clamp_min(0.0)
    sqrt_disc = torch.sqrt(disc)
    k1 = H + sqrt_disc
    k2 = H - sqrt_disc
    max_curv = torch.maximum(k1.abs(), k2.abs())

    rho = 1.0 / max_curv.clamp_min(eps)
    return rho.numpy()
