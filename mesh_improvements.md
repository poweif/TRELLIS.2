# Mesh Improvement Directions for TRELLIS.2

Survey of state-of-the-art methods for producing feature-preserving, quad-dominant meshes
and for optimizing texture generation, with notes on integration into the TRELLIS.2 pipeline.

---

## Current pipeline state

TRELLIS.2 produces meshes via dual contouring on a sparse voxel latent (SLat) at 1024³
resolution. The raw mesh goes through:

1. **Simplification** — QEM-based iterative edge collapse (`CuMesh/src/simplify.cu`)
2. **Cleanup** — deduplication, hole filling (`CuMesh/src/clean_up.cu`)
3. **UV unwrapping** — `cumesh.uv_unwrap` (xatlas-based angle-based flattening internally)
4. **Texture baking** — direct grid-sampling of a live sparse PBR-attribute voxel tensor at each
   UV texel's rasterized 3D position (see the correction in §2.1 below — not multi-view
   projection, corrected from this doc's original description)
5. **Export** — GLB via trimesh

The output is an all-triangle mesh. The simplification is isotropic (QEM treats all
directions equally) and the UV seam placement is purely distortion-driven with no
awareness of mesh features or edge loops.

---

## Part 1: Quad-Dominant, Feature-Preserving Remeshing

### 1.1 Background and motivation

Triangle meshes from dual contouring have no structure — edges flow in arbitrary directions,
vertices have unpredictable valences, and sharp features (creases, corners) are represented
implicitly by clusters of small triangles. Quad-dominant meshes are preferred for:

- **Animation / rigging** — edge loops follow anatomical or mechanical structure
- **Subdivision surfaces** — Catmull-Clark subdivision requires quads
- **Texture mapping** — quads produce less distortion and cleaner UV seams
- **Compression** — structured connectivity compresses far better than arbitrary triangles
- **Rendering** — tessellation shaders expect quad patches

"Quad-dominant" means the majority of faces are quads, with isolated triangles or pentagons
at irregular vertices (unavoidable for topological reasons).

### 1.2 Classical field-aligned remeshing

These methods compute a smooth **cross-field** (a 4-way symmetric direction field) over the
surface, then trace streamlines to extract a quad mesh.

#### Instant Meshes — Jakob et al., SIGGRAPH Asia 2015
- **Code**: github.com/wjakob/instant-meshes
- **Key idea**: Pose the cross-field and vertex placement as a mixed-integer optimization.
  The cross-field is computed by minimizing a Dirichlet energy subject to alignment with
  surface features. Integer variables snap field singularities and vertex positions to a
  regular grid in the parameterization domain.
- **Strengths**: Fast (interactive on 1M-face meshes), open source, produces clean quads
  in smooth regions.
- **Weaknesses**: Feature alignment requires explicit crease input. Results degrade near
  high-curvature regions without careful tuning. The mixed-integer relaxation can produce
  irregular patches.
- **Integration note**: Takes an OBJ/PLY as input; can be called from the command line or
  via Python bindings. Run after QEM simplification, before UV unwrapping.

#### QuadriFlow — Huang et al., SIGGRAPH 2018
- **Paper**: "QuadriFlow: A Scalable and Robust Method for Quadrangulation"
- **Code**: github.com/hjwdzh/QuadriFlow
- **Key idea**: Improves on Instant Meshes by casting the global consistency constraint as a
  **minimum-cost network flow** problem. This gives a more principled solver for the integer
  constraints (where to place singularities in the cross-field) and produces more regular
  meshes with fewer artifacts near singularities.
- **Strengths**: More robust than Instant Meshes on complex geometry; better handles meshes
  with topological handles.
- **Weaknesses**: Slower. Still requires crease input for feature alignment. Does not
  directly model sharp edges; creases must be provided as hard constraints.
- **Integration note**: Has a C++ library and Python bindings via `pyquadriflow` (unofficial)
  or can be called as a subprocess.

#### Mixed-Integer Quadrangulation — Bommes et al., SIGGRAPH 2009
- **Paper**: "Mixed-Integer Quadrangulation" (ACM TOG 28(3))
- The theoretical foundation for both Instant Meshes and QuadriFlow. Introduces the
  formulation of quadrangulation as optimizing a parameterization with integer transition
  functions. Required reading for understanding the field.

### 1.3 Feature detection and preservation

Feature-preserving remeshing requires first detecting sharp features and then constraining
the cross-field and remesher to align with them.

#### Sharp feature detection

The standard approach: classify edges as **sharp** if the dihedral angle between adjacent
faces exceeds a threshold (typically 30°–60°). More robust methods:

- **Angle + curvature**: combine dihedral angle with principal curvature direction alignment
- **Normal deviation**: cluster face normals; boundaries between clusters are creases
- **Fitting-based**: fit planes locally and detect deviation from planar regions (works well
  for CAD-like shapes with flat faces)

The **Geogram** library (Bruno Lévy, INRIA) has a mature implementation of sharp feature
detection and anisotropic remeshing that aligns the mesh to detected creases. The associated
**Graphite** GUI is useful for interactive exploration.

#### Frame field constrained to features

Once creases are detected, the cross-field must be constrained to have one axis aligned with
each crease curve. Key papers:

- **"N-Symmetry Direction Field Design"** — Ray et al., ACM TOG 2008. The foundational
  treatment of designing smooth N-RoSy fields with constraints.
- **"Stripe Patterns on Surfaces"** — Knöppel et al., SIGGRAPH 2015. Computes a globally
  consistent parameterization from a direction field; directly relevant to remeshing.
- Instant Meshes and QuadriFlow both support crease constraints via hard constraints on
  the cross-field at detected feature edges.

#### Anisotropic remeshing

For feature preservation without full quadrangulation, **anisotropic triangle remeshing**
aligns triangle edges with principal curvature directions, producing elongated triangles
along smooth regions and small triangles near features — a good intermediate step:

- **ACVD** (Valette & Chassery, 2004) — centroidal Voronoi diagrams on the mesh surface,
  anisotropically weighted by curvature. Available in VTK.
- **Geogram**: `vorpaline` executable does both isotropic and anisotropic remeshing with
  feature line constraints.

### 1.4 Learning-based approaches (2024 SOTA)

Classical methods require careful parameter tuning and produce variable quality on organic
shapes. Recent work trains neural networks to generate meshes that mimic artist-created
topology.

> **Correction (see `history.md`):** this section originally described MeshAnything/MeshAnything
> V2 output as "quad-dominant." That was never actually verified until this fork tried it, and
> turned out to be false — MeshAnythingV2 is fundamentally triangle-tokenized
> (`face_per_token = 9`, always 3 vertices) and never produces a quad-dominant mesh, confirmed both
> empirically (vertex valence distribution, even on a pristine synthetic cylinder) and against the
> published literature (QuadLink/QuadGPT both list it among triangle-only autoregressive
> generators). "Artist-Created Mesh" names the *training-data category* (meshes made by human
> artists), not a topology guarantee. Left the entries below otherwise as originally written for
> historical/technical accuracy about what these papers actually claim; see `history.md` for the
> full investigation and `quadriflow_postprocess_plan.md` for what this fork does instead
> (QuadriFlow, applied after MeshAnything or directly on the original mesh, actually produces
> genuine quads).

#### MeshAnything — Chen et al., 2024
- **Paper**: "MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers"
- **Code**: github.com/buaacyw/MeshAnything
- **Key idea**: Trains an autoregressive transformer on a large dataset of artist-created
  meshes. Mesh faces are tokenized using a VQ-VAE (vertex positions quantized to a discrete
  codebook, face sequences ordered by position). Given a point cloud or 3D shape as
  conditioning, the model generates a sequence of face tokens that decode to a clean mesh.
- **Output**: Meshes with ~800 faces, triangle-only (see correction above), with edge loops
  that follow the shape's semantic structure — similar to what a skilled 3D artist would produce.
- **Strengths**: Produces topology-aware meshes automatically; no parameter tuning.
- **Weaknesses**: Face count cap (~800) is low for high-detail objects; not suitable as a
  final high-poly mesh without subdivision. Inference can be slow. Pretrained on specific
  object categories — may not generalize to all shapes TRELLIS produces.

#### MeshAnything V2 — Chen et al., 2024
- **Paper**: "MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization"
- **Code**: github.com/buaacyw/MeshAnythingV2
- **Key improvement**: Changes the tokenization from a global face sequence to an
  **adjacency-based traversal** — each new face is described relative to its neighbor
  rather than in absolute coordinates. This improves local consistency, reduces sequence
  length for the same face count, and allows higher-detail outputs.
- Supports up to ~1600 faces; still constrained but better than V1. Triangle-only output
  (see correction above).
- **Caveats from the reference implementation**: expects input meshes/point clouds in
  +Y-up convention (TRELLIS output would need reorienting first). The authors also note
  that plain feed-forward generation is often suboptimal — they recommend a
  reconstruction-guided or SDS-guided variant for reliable artist-quality output rather
  than naive point-cloud-in/mesh-out inference. This fork tried the SDS-guided variant —
  see `history.md` for how that went.

#### MeshXL — 2024
- Similar autoregressive paradigm, focuses on scalability to higher polygon counts and
  better handling of open surfaces.

#### Practical note on learning-based methods

These models take a **point cloud or implicit surface** as input (not a dense triangle mesh
directly, though the point cloud can be sampled from the TRELLIS mesh). The output is a
coarse but topologically clean mesh that can then be **subdivided** (Catmull-Clark) and
**displaced** using the detail from the original TRELLIS mesh. This subdivision-plus-displacement
workflow is common in VFX pipelines.

For the TRELLIS pipeline, the most natural integration point is right after the SLat decode
step — sample a dense point cloud from the high-res dual-contour mesh (or directly from the
latent), run MeshAnything V2 to get a clean coarse mesh, then use the original dense mesh to
bake displacement and texture maps onto the coarse mesh's UV layout. (This fork built exactly
this integration — see `history.md` for the full build-out, and `quadriflow_postprocess_plan.md`
for the currently-active plan, since a QuadriFlow quad-conversion step now sits between
MeshAnything's output and the final mesh.)

---

## Part 2: Texture Generation and Optimization

### 2.1 Current approach and its limitations

> **Correction (see `history.md`):** the description below (multi-view rendering + UV projection)
> was the originally-assumed approach at survey time. The texturing pipeline actually implemented
> in `trellis2_texturing.py`'s `postprocess_mesh` / `o_voxel/postprocess.py`'s `to_glb` works
> differently: it unwraps UVs (`cumesh.uv_unwrap` if the mesh has none), rasterizes those UVs to get
> each texel's 3D position (`uv_rasterize.py`'s `rasterize_uv`/`interpolate_uv`), and directly
> **grid-samples a live sparse PBR-attribute voxel tensor** at that position
> (`flex_gemm.ops.grid_sample.grid_sample_3d`) — no multi-view rendering or per-view projection at
> all. This sidesteps the seam/blur failure modes below entirely (there's only ever one "view": the
> voxel grid itself), at the cost of only being usable while that live voxel tensor is still in
> scope (i.e. inside `run_sample.py`'s own pipeline run, before it bakes to a flat 2D atlas and
> discards it — a standalone downstream tool operating on an already-exported GLB has no access to
> it, and needs a different, mesh-to-mesh texture-transfer approach instead; see
> `quadriflow_postprocess_plan.md`).

### 2.2 UV parameterization

#### xatlas (current)
Angle-Based Flattening (ABF). Minimizes angle distortion, places seams automatically to
keep distortion below a threshold. Makes no use of mesh structure (ignores quad loops,
feature lines).

#### Better alternatives

**OptCuts — Li et al., SIGGRAPH Asia 2018**
- **Paper**: "OptCuts: Joint Optimization of Surface Cuts and Parameterization"
- **Key idea**: Jointly optimizes *where* to place UV seams and *how* to flatten. Treats
  seam placement as a continuous optimization variable rather than a greedy pre-pass. Produces
  lower distortion with fewer seams on structured meshes.
- Has a reference implementation.

**Seamless parameterization (Campen & Bommes)**
- For quad-dominant meshes specifically, **seamless maps** align UV transitions across
  seams to integer multiples of the texture period, enabling tileable textures without
  visible seam artifacts. See: "Quantized Global Parameterization" (Campen et al.,
  SIGGRAPH Asia 2015).

**Least Squares Conformal Maps (LSCM)**
- Minimizes angular distortion (conformal = angle-preserving). Better than ABF at
  preserving local shape. Available in Blender, Geogram, libigl.

**Spectral Conformal Parameterization**
- Uses eigenvectors of the Laplace-Beltrami operator for a globally smooth, low-distortion
  parameterization. Best for organic shapes without strong feature lines.

### 2.3 Differentiable rendering for texture optimization

The key insight: instead of a single-pass projection, treat texture as a differentiable
variable and optimize it by minimizing rendering loss against reference images.

#### NVDiffRec — Munkberg et al., CVPR 2022
- **Paper**: "Extracting Triangular 3D Models, Materials, and Lighting From Images"
- **Code**: github.com/NVlabs/nvdiffrec
- **Key idea**: Uses differentiable rasterization (nvdiffrast) to jointly optimize mesh
  geometry, PBR material maps (diffuse albedo, roughness, metallic), and environment
  lighting. The texture is stored in a learnable MIP-mapped texture atlas and optimized
  end-to-end using multi-view photometric loss.
- **Output**: Full PBR material decomposition, not just diffuse color. Works on real
  multi-view image inputs.
- **Integration note**: Uses nvdiffrast which is not available on ROCm. Would need the
  same `uv_rasterize.py` substitution approach, or a ROCm-compatible differentiable
  rasterizer. The optimization loop itself is pure PyTorch.

#### NVDiffRecMC — Hasselgren et al., 2022
- **Paper**: "Shape, Light, and Material Decomposition from Images using Monte Carlo Rendering and Denoising"
- Extends NVDiffRec with Monte Carlo path tracing for more accurate light transport during
  optimization. Higher quality, higher cost.

#### Fantasia3D — Chen et al., ICCV 2023
- **Paper**: "Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation"
- **Key idea**: Separates the geometry optimization phase (using DMTet) from the appearance
  optimization phase (using NVDiffRec's material model). Text-to-3D, but the material
  optimization stage is directly applicable to texture baking.

### 2.4 Diffusion-guided texture completion (for occluded regions)

When reference views don't cover all surface regions, diffusion models can hallucinate
plausible texture in unseen areas while remaining consistent with visible regions.

#### TEXTure — Richardson et al., 2023
- **Paper**: "TEXTure: Text-Guided Texturing of 3D Shapes" (SIGGRAPH 2023)
- **Key idea**: Iteratively renders the mesh from different viewpoints, uses a depth-conditioned
  diffusion model (depth-to-image) to generate texture for each view, and composites into
  the UV atlas. Each new view is "inpainted" to be consistent with previously painted
  regions by masking the diffusion model's attention to already-textured UV patches.
- **Relevance**: The iterative view-by-view approach directly addresses the occlusion
  problem in TRELLIS's current projection-based texturing. Could be applied on top of the
  existing texture as a refinement pass.

#### Text2Tex — Chen et al., 2023
- **Paper**: "Text2Tex: Text-driven Texture Synthesis via Diffusion Models"
- **Key idea**: Progressive texturing — starts with front-facing views, identifies untextured
  regions via a "texture completion score", and progressively fills them in with
  inpainting-guided diffusion while maintaining view consistency.

#### Paint3D — 2023/2024
- Focuses on generating textures that look good under varying lighting conditions (not just
  baked irradiance). Separates diffuse albedo from shading, producing textures suitable for
  real-time PBR rendering.

### 2.5 Score Distillation Sampling (SDS) refinement

SDS (Poole et al., 2022, "DreamFusion") allows using a 2D diffusion model as a prior over
3D content without multi-view consistency — by differentiating through the diffusion model's
score function.

Applied to texture refinement:
1. Start with the projected texture from TRELLIS
2. Render the textured mesh from a random viewpoint
3. Add noise to the render, denoise with a text-conditioned diffusion model
4. The gradient of the denoising loss w.r.t. the texture atlas updates the texture
5. Repeat for many viewpoints

This is how **DreamFusion** and related work (Magic3D, Fantasia3D) refine textures. The
update signal comes from the diffusion prior, not ground-truth images — useful when you
want to add detail to underspecified regions.

**Practical caveat**: SDS produces characteristic over-saturation ("deep dream" style) and
requires careful tuning of guidance weight. The variant **VSD** (Variational Score
Distillation, Wang et al., 2023 "ProlificDreamer") addresses this with a more stable
objective and produces higher quality results.

---

## Part 3: Integration Roadmap — superseded, see current docs

This section originally sketched four options (A: classical QuadriFlow remeshing, B: MeshAnything
V2 learning-based generation, C: diffusion-inpainting texture refinement, D: PBR material
decomposition via NVDiffRec-style optimization) with effort estimates for each, written before any
of them had actually been tried.

**What actually happened**: Option B was attempted first (see `history.md` for the full build-out),
and its core premise — that MeshAnything V2 produces quad-dominant output — turned out to be false.
Option A's actual mechanism (QuadriFlow) was then adopted as the fix, layered *after* MeshAnything's
geometry-improvement step rather than replacing it outright (a hybrid of A and B, not a clean either/or
the way this section originally framed it) — though running QuadriFlow directly on the original mesh
instead, without MeshAnything in the loop at all, remains an open, actively-considered alternative.
Options C and D were never attempted; they remain viable independent ideas if picked up later, and
the survey sections above (2.3-2.5) are still accurate background for them.

**For the current state and active plan, see `history.md` (what was tried and why) and
`quadriflow_postprocess_plan.md` (the live, forward-looking plan for what happens after QuadriFlow's
quad conversion — UV unwrap, texture transfer, optional subdivision, and the still-open
MeshAnything-in-the-loop-or-not decision).**

---

## Key papers to read first

| Paper | Why |
|---|---|
| Instant Meshes (Jakob et al., 2015) | Best entry point to cross-field quad remeshing |
| QuadriFlow (Huang et al., 2018) | More robust; directly usable |
| MeshAnything V2 (Chen et al., 2024) | SOTA for artist-style topology generation |
| NVDiffRec (Munkberg et al., CVPR 2022) | Gold standard for differentiable texture/material baking |
| TEXTure (Richardson et al., 2023) | Best approach for diffusion-guided texture completion |
| ProlificDreamer (Wang et al., 2023) | Best SDS variant for texture refinement quality |
