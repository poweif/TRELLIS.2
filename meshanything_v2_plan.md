# Implementation Plan: MeshAnything V2 Mesh Refinement Tool

Implements **Option B** from `mesh_improvements.md` (learning-based mesh generation),
built as a **standalone tool independent of `run_sample.py`**, but structured so wiring
it into the main pipeline later is a small, mechanical change (see "Forward-compat
contract" below). This doc is written to be handed to an LLM coding agent phase-by-phase
— each phase is self-contained with explicit files, contracts, and acceptance checks.

---

## Objective

Take a mesh/GLB (typically TRELLIS.2's own output), **refine its geometry quality via
SDS** (per MeshAnything V2's own guidance — see below), regenerate the refined shape as
an artist-style quad-dominant coarse mesh via MeshAnything V2, then subdivide +
detail-bake + re-texture it back to comparable visual fidelity. Ship as a CLI tool that
reads a mesh file and writes a mesh file — no dependency on TRELLIS.2's SLat/pipeline
internals.

## Non-goals (this iteration)

- **No `run_sample.py` / `trellis2_texturing.py` wiring yet** — user decision: keep this
  independent for now.
- **No `--input_type mesh` / marching-cubes path.** Only the point-cloud conditioning
  path into MeshAnything, which avoids building `mesh2sdf`'s compiled extension for no
  immediate benefit.
- **No general-purpose SDS texture refinement.** `mesh_improvements.md` §2.5 already
  covers SDS for *texture*; this plan's SDS phase is scoped to *geometry* refinement of
  the input shape only, per the corrected upstream guidance below. Don't conflate the two.

## Hard constraints from this environment

See `amd_workarounds.md` for full detail — the short version:

- No CUDA GPU. This repo already has a working custom PyTorch 2.7.0 (ROCm 7.1, gfx1151)
  and a custom-built flash-attn (ROCm fork, Triton backend, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`).
  **Reuse both** — do not let MeshAnything V2's pinned `torch==2.1.1+cu118` or
  `pip install flash-attn` (upstream, CUDA-only) touch this environment. Same applies to
  any diffusion library pulled in for Phase 3 — prefer PyTorch's built-in
  `scaled_dot_product_attention` or the existing flash-attn build over installing
  `xformers` (known to be a recurring source of ROCm-specific breakage; not worth the risk
  for an attention backend swap).
- **No nvdiffrast, and — newly confirmed — no working arbitrary-viewpoint differentiable
  renderer at all in this repo on this hardware.** `trellis2/renderers/mesh_renderer.py:44-48`
  and `pbr_mesh_renderer.py:200-202` both hard-require `nvdiffrast` and explicitly
  `raise RuntimeError(".. requires nvdiffrast which is not available on AMD/ROCm")`.
  `trellis2/utils/uv_rasterize.py` (the pure-PyTorch replacement mentioned in
  `amd_workarounds.md`) only rasterizes into **UV atlas space**, not camera/screen space —
  it cannot render the mesh from an arbitrary viewpoint. SDS (Phase 3) needs
  camera-viewpoint rendering, so **Phase 2 has to build one from scratch**. This is the
  single biggest net-new technical risk in this plan.
- Any new GPU extension must build for `GPU_ARCHS=gfx1151` (pre-built wheels targeting
  other archs SIGSEGV at runtime on this hardware, per `amd_workarounds.md` §6). This
  plan avoids needing any new *compiled* extension — the new rasterizer (Phase 2) and SDS
  loop (Phase 3) are plain PyTorch, same approach as `uv_rasterize.py`.

## Verified upstream facts (checked against the actual repo/README, not recalled)

- Checkpoint: `Yiwen-ntu/meshanythingv2` on Hugging Face, loaded via
  `MeshAnythingV2.from_pretrained(...)`.
- Input contract: point cloud shaped `(8192, 6)` — xyz + normals — centered on the
  bounding-box midpoint, scaled to `[-0.9995, 0.9995]³`, cast to `float16`.
- Output contract: a flat vertex tensor reshaped to `(-1, 9)` (3 vertices × 3 coords per
  face); rows containing `NaN` are padding and must be masked out. Needs
  `merge_vertices` + degenerate/duplicate-face removal afterward (this is exactly what
  the reference `main.py` does before export).
- Hard cap: 1600 faces — this is a **training-distribution** limit, not just an
  inference knob; don't expect more no matter the settings.
- Expects **+Y-up** input.
- Reference perf: ~8GB VRAM, ~45s/mesh on an A6000. Expect slower + a first-call JIT
  compile tax here (Triton kernels JIT-compile on first use, as already observed for
  flash-attn in `amd_workarounds.md` §5b-vi).
- Dependencies are pure Python (`trimesh`, `accelerate`, `einops`, `einx`, `optimum`,
  `omegaconf`, `opencv-python`, `transformers`, `numpy`, `huggingface_hub`, `matplotlib`,
  `gradio`) except `mesh2sdf` (out of scope, see Non-goals) and flash-attn (already
  solved in this repo).
- **On feed-forward input quality (this is the basis for the SDS phase — corrected from
  an earlier draft of this doc, which mischaracterized it)**: the README's exact wording
  is **"feed-forward 3D generation methods may often produce bad results due to
  insufficient shape quality. We suggest using results from 3D reconstruction, scanning,
  SDS-based method (like DreamCraft3D) or Rodin as the input."** This is a statement
  about **input shape quality**, not about MeshAnything V2's own inference mode — TRELLIS.2
  itself *is* a feed-forward 3D generation method, so this warning applies directly to
  feeding it raw TRELLIS output. It is not a recommendation to post-process MeshAnything's
  *output*. Scanning/Rodin aren't applicable here (no scan data, Rodin is an external paid
  tool); SDS-based input refinement is the applicable option, hence Phase 3 below.

## Assumptions flagged for a research spike (don't block planning on these, resolve in-phase)

- **Differentiable camera-viewpoint rasterizer (Phase 2)**: no existing implementation in
  this repo works on ROCm (see Hard Constraints above). Needs to support: perspective
  projection of a triangle mesh, z-buffered rasterization, barycentric attribute
  interpolation (position/normal at minimum), and backprop from pixel values to vertex
  positions. Scope it to only what SDS needs (Phase 3) — it does not need to match
  `MeshRenderer`'s full feature set (PBR shading, antialiasing, etc.).
- **2D diffusion prior choice for SDS (Phase 3)**: needs to be image-conditioned (TRELLIS.2
  is image-to-3D, so text-only SDS priors like the original DreamFusion setup are the
  wrong fit). Candidates to evaluate: an image-conditioned novel-view diffusion model in
  the Zero-1-to-3/Stable-Zero123 family, or a depth-conditioned ControlNet fed the
  original reference image (closer to the TEXTure-style approach already cited in
  `mesh_improvements.md` §2.4). Pick based on what's loadable via `diffusers` without
  `xformers`. This is a real design decision, not a known quantity — spike it early in
  Phase 3, don't assume a checkpoint upfront.
- **Catmull-Clark subdivision (Phase 7)**: nothing in this repo (`CuMesh`, `o-voxel`,
  `trimesh`) currently does true n-gon Catmull-Clark — `trimesh.remesh.subdivide` only
  midpoint-splits triangles. MeshAnything's output is a mix of quads/tris/pentagons, so
  Phase 7 needs to pick a library (candidate: `pyvista`, VTK-backed, CPU-only, no GPU
  build required) or write a small numpy implementation given mesh sizes are tiny
  (≤1600 faces pre-subdivision).
- **Mesh-to-mesh texture transfer quality (Phase 8)** is unverified — flagged as a risk to
  eyeball, not a solved problem.
- **Coordinate convention of the *input* GLB**: glTF/GLB spec is +Y-up, but
  `trellis2_texturing.py:353-355` explicitly swaps Y/Z and negates Y on export
  (`vertices[:,1], vertices[:,2] = vertices[:,2], -vertices[:,1]`) to satisfy that
  convention. So a GLB written by this repo's own pipeline should already be +Y-up —
  **verify this empirically in Phase 4** rather than assuming, since getting it wrong
  silently produces a sideways/upside-down conditioning point cloud with no error.

## SDS geometry-refinement design (why it's structured this way)

- **Optimize vertex offsets directly on the existing mesh, not an implicit field
  (DMTet/NeRF/occupancy).** Fantasia3D/NVDiffRec-style pipelines use DMTet because they're
  generating geometry from scratch; here we already have a reasonable coarse shape from
  TRELLIS and only need to *improve* it. Direct vertex optimization (with Laplacian
  smoothness + edge-length regularization to prevent the noisy SDS gradient from
  degenerating the mesh) keeps topology and UVs unchanged — which matters because it
  means Phase 8's texture transfer stays trivial (same connectivity, just moved vertices)
  instead of needing to re-resolve correspondence after topology changes. It also avoids
  needing to reimplement differentiable marching tetrahedra, which would be a much larger
  undertaking than the rasterizer in Phase 2.
- **Known failure modes to expect** (same family as the texture-SDS caveats already
  documented in `mesh_improvements.md` §2.5): oversaturated/degenerate gradients without
  careful guidance-weight tuning, and — specific to geometry — self-intersecting or
  spiky vertex displacement if regularization is too weak. Start with a low learning rate
  and treat this as a *refinement* of the existing shape, not a from-scratch optimization.

---

## Forward-compat contract (why this stays easy to wire in later)

All logic lives in importable functions, not the CLI script. The eventual
`run_sample.py` integration would just be:

```python
from trellis2.utils.meshanything_bridge import refine_mesh
...
mesh = decode_latent(...)          # existing call
if args.mesh_backend == "meshanything_v2":
    mesh = refine_mesh(mesh)       # new call, drop-in trimesh.Trimesh -> trimesh.Trimesh
                                    # (internally: SDS geometry refine -> MeshAnything V2 -> subdivide/bake)
mesh = to_glb(mesh, ...)           # existing call, unchanged
```

`refine_mesh()` takes and returns a `trimesh.Trimesh` and touches no pipeline/SLat state,
so this is the only integration point that would need to exist later — nothing else in
this plan should assume it's being called from a CLI.

## Directory / artifact layout

```
third_party/MeshAnythingV2/            # git submodule, pinned commit
tools/mesh_refine_meshanything.py      # standalone CLI entry point (thin argparse wrapper)
trellis2/utils/mesh_rasterizer.py      # NEW: pure-PyTorch camera-viewpoint differentiable rasterizer
trellis2/utils/sds_geometry_refine.py  # NEW: SDS geometry-refinement loop
trellis2/utils/meshanything_bridge.py  # all reusable logic; this is the forward-compat import target
```

## Architecture (data flow)

```
input.glb / input.obj  (any mesh, textured or not)
  │
  ▼
[1] load_mesh()                trimesh.load; keep original verts/faces/texture as "detail source"
  ▼
[2] refine_geometry()          render from random viewpoints (NEW rasterizer), SDS loss vs.
  │                             image-conditioned diffusion prior, optimize vertex offsets
  ▼
[3] mesh_to_conditioning_pc()  sample 8192 pts+normals from the SDS-REFINED mesh, verify/apply
  │                             +Y-up, bbox-normalize
  ▼
[4] run_meshanything_v2()      from_pretrained (cached), fp16 inference -> (N,9) raw tokens
  ▼
[5] detokenize_and_clean()     mask NaNs, build coarse Trimesh, merge/dedupe, invert Phase 3 transform
  ▼
[6] subdivide_catmull_clark()  1-2 levels -> ~6K-25K faces
  ▼
[7] bake_detail()               per-vertex nearest-point projection onto the SDS-REFINED mesh
  ▼                             (better detail source than the raw input, since Phase 2 improved it)
[8] uv_unwrap()                reuse cumesh.uv_unwrap (same call as trellis2_texturing.py:304-313)
  ▼
[9] transfer_texture()         sample the SDS-refined mesh's texture (inherited unchanged from
  │                             input, since Phase 2 preserves UVs) via closest-point lookup
  ▼
[10] export_glb()
```

---

## Phases

Each phase lists: goal, files, tasks, acceptance criteria. Phases 0–6 are the MVP
(SDS-refined coarse mesh only); ship that as milestone 1 before starting 7–8.

### Phase 0 — Environment spike

**Goal**: confirm MeshAnything V2's pure-Python deps and checkpoint work in-place inside
the existing `trellis2` conda env, without disturbing the custom torch/flash-attn build.
Also confirm a `diffusers`-based 2D diffusion checkpoint can be loaded and run a forward
pass on this ROCm stack (needed by Phase 3), without pulling in `xformers`.

**Tasks**:
1. Add MeshAnything V2 as a git submodule at `third_party/MeshAnythingV2`, pinned to a
   specific commit (record the SHA in this doc once chosen).
2. Install its deps with `--no-deps` (never let it try to reinstall torch/numpy/flash-attn):
   `pip install --no-deps -r third_party/MeshAnythingV2/requirements.txt`, then check
   each package individually against what's already installed for version conflicts;
   hand-resolve rather than blind-pinning to the repo's exact versions.
3. Re-run the flash-attn smoke test from `amd_workarounds.md` §5b-vi after any
   `transformers`/`accelerate` version change, to confirm it's still intact.
4. Load the checkpoint (`MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")`) and
   do one forward pass **on CPU first** with a synthetic point cloud, before touching
   ROCm at all — isolates "does the checkpoint/class even load" from "does it run on our
   GPU stack".
5. Retry on GPU with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` set. Expect the reference
   `main.py` / model class to hardcode `.cuda()` / `device='cuda'` in places — patch these
   to a passed-in `device` variable (matches the `self.device` pattern already used
   throughout `trellis2/pipelines/*.py`). Document every patch as a diff block, same style
   as `amd_workarounds.md`, so it's reproducible after a submodule update.
6. Install `diffusers` (pip, pure Python) and load one candidate image-conditioned
   diffusion checkpoint (see Phase 3's spike list), run a single denoising step on the
   gfx1151 GPU using default/SDPA attention — confirm no `xformers` import gets pulled in
   transitively and no CUDA-only op trips.

**Acceptance**: a bare script produces *some* mesh from a hand-built point cloud, running
on the gfx1151 GPU, using this repo's existing torch/flash-attn — not upstream pins. A
separate bare script runs one diffusion U-Net forward+backward pass on the same GPU.

### Phase 1 — Vendoring & bridge module skeleton

**Goal**: make MeshAnything V2 importable as a library.

**Files**: `third_party/MeshAnythingV2/` (patched per Phase 0), `trellis2/utils/meshanything_bridge.py` (new)

**Tasks**:
- `load_model(device) -> MeshAnythingV2`: cached singleton (module-level `functools.lru_cache`
  or a simple global), so repeated CLI invocations in a batch don't re-download/reload.
- `run_inference(point_cloud: np.ndarray[8192,6], device) -> np.ndarray` — wraps the raw
  forward pass only (raw token output), kept separate from detokenization so Phase 5 can
  unit-test against known-good raw output.

**Acceptance**: `run_inference()` on a point cloud sampled from `trimesh.creation.icosphere()`
returns a `(N, 9)` tensor with no exceptions, entirely from a fresh Python process (no
hidden state from a prior pipeline run).

### Phase 2 — Differentiable camera-viewpoint mesh rasterizer

**Goal**: the missing primitive SDS needs. Confirmed nothing in this repo does this on
ROCm (see Hard Constraints) — `MeshRenderer`/`PbrMeshRenderer` hard-require nvdiffrast,
`uv_rasterize.py` is UV-space only.

**Files**: new `trellis2/utils/mesh_rasterizer.py`

**Tasks**:
- Implement a minimal pure-PyTorch differentiable rasterizer:
  project vertices via extrinsics/intrinsics (reuse `intrinsics_to_projection` logic
  pattern from `trellis2/renderers/mesh_renderer.py:8-33` as a reference for the camera
  math, but without the `nvdiffrast` calls), rasterize triangles with a z-buffer, and
  interpolate per-vertex attributes (position, normal) via barycentric coordinates.
  Simplest correct approach: per-pixel loop is too slow — vectorize via bounding-box
  culling + `torch.searchsorted`/batched barycentric tests, or adapt the same
  vectorization approach `uv_rasterize.py` already uses for UV-space rasterization (it
  solves an analogous problem — worth reading first as a template).
- Keep the feature set deliberately narrow: only what Phase 3's SDS loop needs (RGB or
  normal/depth buffer + a differentiable path back to vertex positions). Do not try to
  match `MeshRenderer`'s antialiasing/PBR feature set.
- Unit test: render a simple textured cube from a few known viewpoints, confirm the
  silhouette matches expectations, and confirm `loss.backward()` through a pixel produces
  a nonzero gradient on vertex positions.

**Acceptance**: gradient check passes; render of a test mesh from 4 viewpoints visually
looks correct (dump PNGs, eyeball them).

### Phase 3 — SDS-guided geometry refinement

**Goal**: address the MeshAnything V2 authors' explicit guidance (see Verified Facts
above) that feed-forward-generated shapes like TRELLIS's raw output often need
higher-fidelity input before retopology — by refining the mesh's geometry via Score
Distillation Sampling before it's handed to MeshAnything V2.

**Files**: new `trellis2/utils/sds_geometry_refine.py`

**Tasks**:
- Spike + pick the diffusion prior per the "Assumptions" note above (image-conditioned;
  Stable-Zero123-family or depth-ControlNet-on-reference-image are the two candidates to
  actually try, pick whichever loads cleanly per Phase 0.6).
- `refine_geometry(mesh, reference_image, n_iters, device) -> trimesh.Trimesh`:
  - Make vertex positions an optimizable `nn.Parameter` (offsets from the original,
    initialized at zero).
  - Loop: sample a random camera viewpoint, render via Phase 2's rasterizer, add noise to
    the render, denoise one step with the diffusion prior conditioned on
    `reference_image`, compute the SDS loss (gradient of denoising loss w.r.t. the
    render), add Laplacian-smoothness + edge-length regularization on the vertex offsets,
    backprop, optimizer step.
  - Return a new `trimesh.Trimesh` with updated vertex positions, **same faces/UVs as
    input** (topology-preserving by construction).
- Log intermediate renders every N iterations for visual debugging — this loop is exactly
  the kind of thing that silently produces garbage if the guidance weight or learning
  rate is off, per the known failure modes noted above.

**Acceptance**: after refinement, the mesh's silhouette from held-out viewpoints looks
*at least as good* as the input (regression check — if SDS is making things worse, that's
a stop-and-fix signal, not something to push through) and ideally shows fixed-up detail in
regions the raw TRELLIS mesh handled poorly. This is a subjective visual check; record
before/after renders.

### Phase 4 — Point-cloud conditioning + coordinate handling

**Goal**: correct, reusable mesh→conditioning-point-cloud function matching MeshAnything's
exact preprocessing contract, with a coordinate convention that's *verified*, not assumed.
Operates on the **SDS-refined mesh from Phase 3**, not the raw input.

**Files**: `trellis2/utils/meshanything_bridge.py` (extend)

**Tasks**:
- `mesh_to_conditioning_pc(mesh, n_points=8192) -> Tuple[np.ndarray, Callable]`:
  - Sample via `trimesh.sample.sample_surface` (area-weighted) for positions; take normals
    from `mesh.face_normals[face_idx]` of the returned face indices.
  - **Verify the +Y-up assumption empirically**: load one of this repo's own exported GLBs,
    print its bounding box / an axis-aligned reference feature, confirm it matches glTF's
    +Y-up convention (per the note in "Assumptions" above) before deciding whether
    reorientation code is a no-op or actually needed. Don't skip this check.
  - Center on bbox midpoint, scale to `[-0.9995, 0.9995]³` (exact formula from Phase 0
    findings), cast to `float16`.
  - Return the point cloud **and** a closure/inverse-transform function that maps a point
    from model space back to the original mesh's frame (needed by Phase 5).

**Acceptance**: round-trip unit test — feed a mesh with a known, asymmetric bounding box
through `mesh_to_conditioning_pc`, apply the returned inverse transform to a few sampled
points, and confirm they land back within float tolerance of their pre-transform positions.

### Phase 5 — Detokenization & cleanup

**Goal**: formalize NaN-masking and mesh cleanup exactly as the reference `main.py` does,
plus apply Phase 4's inverse transform.

**Files**: `trellis2/utils/meshanything_bridge.py` (extend); add `refine_mesh_coarse(mesh, reference_image) -> trimesh.Trimesh`
tying Phases 1–5 together (including the Phase 3 SDS step).

**Tasks**:
- Mask `NaN` rows out of the `(-1, 9)` reshaped output.
- Build a `trimesh.Trimesh`, then `merge_vertices()`, remove degenerate/duplicate faces,
  fix normals — same sequence the reference implementation uses.
- Apply the inverse transform from Phase 4.

**Acceptance**: for a handful of test meshes, output has zero NaNs/degenerate faces,
face count ≤ 1600, and its bounding box matches the SDS-refined mesh's bounding box within
~1% (this is the real end-to-end sanity check that Phase 4's transform math is correct).

### Phase 6 — Standalone CLI (MVP milestone)

**Goal**: ship a usable SDS-refined-coarse-mesh tool.

**Files**: `tools/mesh_refine_meshanything.py`

**Interface**:
```
python tools/mesh_refine_meshanything.py --input out.glb --reference-image photo.png --output out_coarse.glb
```
`--reference-image` is required (Phase 3's SDS loop needs an image to condition on — for
a first pass this can be the same image originally used to generate `out.glb` via
`run_sample.py`). Thin argparse wrapper around `refine_mesh_coarse()` — all logic stays in
the bridge/SDS modules per the forward-compat contract.

**Acceptance**: running against 2–3 of this repo's existing example outputs
(`assets/example_image`, `assets/example_texturing`) produces a valid GLB that opens in a
standard viewer, with visibly quad-dominant topology (spot-check by opening in Blender or
`trimesh.Scene().show()`), exiting 0 with no manual environment tweaks beyond activating
the existing `trellis2` conda env.

**→ Stop here and evaluate before continuing.** If coarse-mesh quality/topology isn't
promising even with SDS-refined input, that's the point to revisit the SDS design
(guidance weight, diffusion prior choice, iteration count) before investing in Phases 7–8.

### Phase 7 — Catmull-Clark subdivision + detail baking

**Goal**: recover surface detail lost by MeshAnything's ≤1600-face cap, using the
SDS-refined mesh (not the raw input) as the detail source.

**Files**: new `trellis2/utils/subdivision.py` (or inline in the CLI tool if small enough)

**Tasks**:
- Spike + pick a subdivision approach per the "Assumptions" note above (`pyvista`
  recommended as lowest-integration-cost first try, given it's CPU-only and needs no new
  GPU build).
- `subdivide_catmull_clark(mesh, levels=2) -> trimesh.Trimesh`.
- `bake_detail(subdivided_mesh, detail_source_mesh) -> trimesh.Trimesh`: for each vertex in
  `subdivided_mesh`, query `detail_source_mesh.nearest.on_surface(...)` (trimesh's CPU
  proximity query — fine at these vertex counts) and replace the vertex position (try
  direct replacement before anything fancier like normal-only displacement). Pass the
  **Phase 3 output**, not the raw input, as `detail_source_mesh`.

**Acceptance**: render the detail-baked mesh from 2–3 viewpoints against the SDS-refined
mesh (Phase 2's rasterizer can be reused here); silhouette and major features should
visibly match. Document any obvious artifacts (e.g. self-intersections from naive
nearest-point projection near thin features) as known issues rather than silently
shipping them.

### Phase 8 — UV unwrap + texture transfer

**Goal**: re-texture the new mesh from whatever texture the input mesh already had.

**Files**: `tools/mesh_refine_meshanything.py`

**Tasks**:
- Reuse `cumesh.uv_unwrap` exactly as `trellis2_texturing.py:304-313` already does, on the
  final detail-baked mesh.
- `transfer_texture(new_mesh_with_uv, textured_source_mesh)`: rasterize the new UVs with
  `trellis2/utils/uv_rasterize.py::rasterize_uv` (already used in this pipeline), get each
  texel's 3D position via `interpolate_uv`, closest-point-project onto
  `textured_source_mesh` to find its UV there, bilinear-sample the source texture. Because
  Phase 3 preserves the original mesh's UVs (topology-preserving offsets), this can
  usually be a direct UV lookup rather than a full closest-point search — worth checking
  before implementing the more expensive path.
- If the input has no material (untextured OBJ), skip this step and export without one —
  don't fabricate a texture.

**Acceptance**: side-by-side render of input vs. output GLB — colors/patterns should be
recognizably in the right place. This is a human visual check, not a metric; record it as
such.

### Phase 9 — Validation pass & go/no-go note

- Run the full tool end-to-end on 3–5 existing example assets.
- Record face count before/after, wall-clock time per stage — including the SDS loop,
  which is likely the slowest new stage by far (many render+denoise+backward iterations;
  get a real number rather than assuming it's cheap).
- Write a short, explicitly subjective quality judgment comparing **with vs. without**
  the SDS refinement step (i.e. re-run Phase 6's CLI with SDS disabled as an ablation) —
  this is the actual evidence for whether Phase 3 was worth its cost, not just a hunch.
- This plan does not include a quantitative benchmark (no Chamfer distance/F-score
  harness); flag that as a possible follow-up only if the tool looks promising enough to
  invest further in.

### Phase 10 — Forward-compat documentation (no code)

- Confirm the `refine_mesh()` contract in this doc still matches what got built (function
  name/signature may have drifted across Phases 1–8 — reconcile them here).
- Note the exact call site and proposed CLI flag (`--mesh-backend {dual_contour,meshanything_v2}`)
  for a future `run_sample.py` integration, without implementing it. Note that this
  integration would need a `reference_image` available at that call site too (it already
  is, in `trellis2_image_to_3d.py`'s `run()` — the conditioning image is in scope there).

---

## Sequencing

Phase 0 blocks everything. **Phases 1 and 2 can proceed in parallel** after Phase 0
(MeshAnything vendoring and the new rasterizer are independent of each other). Phase 3
depends on Phase 2 (needs the rasterizer) but not on Phase 1. Phase 4 depends on Phase 3
(needs a refined mesh to condition on) but not on Phase 1 either. Phase 5 depends on
Phase 1 (needs MeshAnything loaded) and Phase 4 (needs the point cloud). Phase 6 (CLI)
ties everything together and needs Phases 1, 3, 4, 5.

**Phase 6 is the real milestone** — stop and evaluate SDS-refined coarse-mesh quality
before starting Phase 7. Phases 7 and 8 can be developed in parallel by different
agents/sessions once Phase 6 is done, since detail-baking and texture-transfer are
independent of each other (both just consume Phase 6's output).
