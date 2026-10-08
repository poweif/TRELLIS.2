# Repo History

A chronological/thematic record of everything done in this fork on top of upstream TRELLIS.2 —
including the detours, dead ends, and reversed decisions. This is where superseded investigation
notes now live; the still-relevant, forward-looking content that used to be spread across
`meshanything_impl_issues.md`, `meshanything_progress.md`, and `meshanything_v2_plan.md` has been
folded in here and those three files removed. `mesh_improvements.md` (survey/reference material,
still current) and `quadriflow_postprocess_plan.md` (the live forward-looking plan) remain as
separate docs — see "Where things stand now" at the bottom of this doc for how everything relates.

---

## Custom changes on top of upstream TRELLIS.2

Everything from `f7457ef`/`0a7cf5f` ("Port Trellis2 to AMD Strix Halo (gfx1151)") onward is custom
work by this fork, not part of Microsoft's original release. Summary, roughly chronological:

### 1. AMD ROCm / Strix Halo (gfx1151) port

The original repo assumes CUDA throughout (nvdiffrast, flash-attn, CuMesh/o-voxel/FlexGEMM's GPU
extensions all ship CUDA-only pre-built wheels or CUDA-only kernels). None of it runs as-is on
AMD's gfx1151. Full build/run guide: `amd_workarounds.md` (610 lines, kept as-is, still current
day-to-day reference — this section only summarizes what changed and why).

- **Custom PyTorch 2.7.0 built from source** against ROCm 7.1, with 7 source patches (CMake HIP
  module-path fixes, a `constexpr`/wavefront-size fix for gfx1151's 32-wide wavefront, a C++
  ABI-mangling mismatch fix, Composable Kernel exclusions, an `aotriton` `.so` version-symlink
  shim). See `amd_workarounds.md` §3 for exact diffs.
- **torchvision 0.22.0 rebuilt from source** (pre-built wheel was ABI-mismatched against the custom
  torch build).
- **flash-attn built from the ROCm fork** (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, Triton
  JIT-compiles kernels for gfx1151 at first use — no pre-built wheel exists for this GPU family,
  flash-attn's HIP backend otherwise only supports CDNA/MI-series). Needed 4 further patches on top
  of the ROCm fork itself: a `triton` version pin, a `hipconfig --version`-string parsing fix in
  `aiter`, a `torch.distributed.Backend` import guard (our custom PyTorch build has no distributed
  C10d backend), and an `all_gather_into_tensor`/`_all_gather_base` compatibility guard in
  `flash_attn/utils/distributed.py`.
- **CuMesh, o-voxel, and FlexGEMM's GPU extensions rebuilt with `GPU_ARCHS=gfx1151`** — pre-built
  binaries targeting other archs (e.g. gfx1100) segfault at runtime on this hardware rather than
  erroring cleanly.
- **`nvdiffrast` replaced with a pure-PyTorch UV-space rasterizer**, `trellis2/utils/uv_rasterize.py`
  (`rasterize_uv`/`interpolate_uv`, drop-in for `dr.rasterize`/`dr.interpolate`) — nvdiffrast has no
  ROCm backend at all. Wired into `postprocess.py` and `trellis2_texturing.py` in place of the
  original `dr.*` calls.
- **`MIOPEN_DEBUG_CONV_WINOGRAD=0`** set in `run_sample.py` — MIOpen's Winograd convolution kernel
  assembly is missing for gfx1151, causing a hard crash on BiRefNet background-removal inference
  otherwise.
- **`HSA_OVERRIDE_GFX_VERSION` deliberately *not* set** (some AMD guides recommend an override to a
  supported arch string) — once extensions are rebuilt natively for gfx1151, the override actively
  causes a segfault by pointing ROCm at kernels that no longer exist.

### 2. MeshAnything V2 vendored and patched for a newer `transformers`

`third_party/MeshAnythingV2/` — vendored from `buaacyw/MeshAnythingV2@461d3b6ed750ab3443281b2e4a0e30e8ee98097e`
(2025-04-29), source only, no `.git`, trimmed of one unneeded 16MB demo clip. The vendored copy
needed patching because it subclasses `transformers`' internal OPT decoder classes directly, against
an API that changed between the checkpoint's original ~4.39-era `transformers` and this repo's
installed 5.12.1:

- `ShapeOPTDecoder.forward` rewritten for the modern `Cache`/`DynamicCache` API (the old code
  assumed a legacy tuple-of-tuples `past_key_values` and a 3-tuple decoder-layer return; both
  assumptions are wrong against installed `transformers`).
- `layer_idx` added to `OPTDecoderLayer` construction (needed by the new `Cache` indexing scheme;
  silently breaks KV-caching at generation time if omitted, no construction-time error).
- Two removed-API calls dropped (`use_flash_attention_2=True` kwarg, `.to_bettertransformer()`
  method — both gone from modern `transformers`, both superseded by `attn_implementation` /
  native SDPA which the code already sets elsewhere).
- A hardcoded checkpoint-relative YAML path in `MeshAnything/miche/encode.py` made
  `__file__`-relative instead, since the bridge module imports it from an arbitrary working
  directory.
- Calling convention: checkpoint loads fp32 by default, but the reference `main.py` casts inputs to
  fp16 — call sites must `.half()` the model explicitly.

### 3. New `trellis2/utils/` modules (all custom, all built for the MeshAnything integration effort)

`mesh_repair.py`, `mesh_rasterizer.py`, `sds_geometry_refine.py`, `meshanything_bridge.py`,
`mesh_debug.py`, `quad_remesh.py` — none of these existed in upstream TRELLIS.2. See "MeshAnything
V2 integration" and "Adopting QuadriFlow" sections below for what each does and how they came to be
built. All are still live/used except where noted.

### 4. QuadriFlow vendored and built

`third_party/QuadriFlow/` — vendored from `hjwdzh/QuadriFlow` (SGP 2018, MIT license), source only.
Built as a standalone static binary (`third_party/QuadriFlow/build/quadriflow`), no runtime
dependency on the Eigen/Boost used to build it, no GPU/ROCm involvement at all — see "Adopting
QuadriFlow" below for why and the exact build steps (`third_party/QuadriFlow/BUILD_NOTES.md`, kept
as-is, still current).

### 5. New CLI tooling and test suite

`tools/mesh_refine_meshanything.py` (standalone CLI, not wired into `run_sample.py`), and
`test_phase1.py` through `test_phase5.py` plus `test_phase3_nan_guard.py`/`test_phase1_bbox.py` (a
per-phase smoke/acceptance/regression test suite built alongside the MeshAnything integration).

### 6. Planning/research docs

`mesh_improvements.md` (SOTA survey — still current, see below), `quadriflow_postprocess_plan.md`
(the live forward-looking plan), and this file.

---

## MeshAnything V2 integration: build-out, debugging, and the quad-dominance finding

### Why this was attempted

TRELLIS.2's own dual-contour mesh output is unstructured all-triangle, with no artist-like edge
flow. `mesh_improvements.md`'s survey identified MeshAnything V2 (an autoregressive transformer
trained on artist-created meshes) as 2024's SOTA candidate for regenerating a shape as a clean,
low-poly, topologically-sensible mesh — with the explicit, at-the-time-unverified assumption that
its output would be quad-dominant. Also flagged in the same survey: MeshAnything's own README
recommends feeding it higher-fidelity input than raw feed-forward generation typically provides
(the same caveat that applies to TRELLIS.2's own output), motivating a Score-Distillation-Sampling
(SDS) geometry-refinement pass *before* MeshAnything, not just a bolt-on afterward.

### Phases 0-2: vendoring, the bridge module, and a from-scratch differentiable rasterizer

Phase 0 vendored and patched MeshAnything V2 (see "Custom changes" above) and confirmed a
`diffusers`-based image-conditioned UNet could run a forward+backward pass on this ROCm stack
without pulling in `xformers`. Phase 1 built `meshanything_bridge.py`'s `load_model`/`run_inference`.
Phase 2 built `mesh_rasterizer.py` — a minimal pure-PyTorch differentiable camera-viewpoint
rasterizer (project → z-buffer rasterize → barycentric-interpolate position/normal), since nothing
else in the repo could do this on ROCm (`MeshRenderer`/`PbrMeshRenderer` hard-require nvdiffrast;
`uv_rasterize.py` only rasterizes into UV space, not from an arbitrary camera). Both phases verified
working early and never needed revisiting.

### Phase 3: SDS geometry refinement — a long bug-chasing saga

`sds_geometry_refine.py`'s `refine_geometry()` optimizes per-vertex offsets against an
image-conditioned diffusion prior (Score Distillation Sampling), rendered via Phase 2's rasterizer,
regularized to avoid degenerating the mesh. This went through many rounds of "looks done" →
disproven by actually running it, not just reading it:

1. **Regularizer had a constant shrink bias, no Laplacian term at all.** The original
   `edge_len.mean() * 0.5` penalized absolute edge length with no anchor to the input mesh — a
   constant inward force regardless of what the SDS gradient wanted. Fixed: edge-length term
   anchored to the *original* mesh's edge lengths, a real Laplacian-smoothness term added on the
   offsets (not raw vertices) via `mesh.vertex_neighbors`.
2. **Two of the phase's own acceptance tests were wrong, not the implementation** — `test_phase4.py`
   asserted a `[-0.5, 0.5]` bound when the real contract is `[-0.9995, 0.9995]³`, and applied the
   inverse-transform to data already in the wrong coordinate domain. `test_phase5.py` computed the
   real bounding-box mismatch, printed it, then explicitly chose not to assert on it.
3. **That un-asserted metric turned out to matter**: checked anyway on `tests/car.glb`, it was 71%
   off on one axis — the first real signal that Phase 3 was corrupting geometry at real-mesh scale,
   previously hidden because the regularization had only ever been validated on a 320-face
   icosphere.
4. **Root cause, confirmed**: on `tests/can.glb` (394K verts / 498K faces), the SDS loop inflated
   extents by 150-250% over 100 iterations. Fix: **decimate to ≤16,000 faces before SDS** —
   MeshAnything caps output at 1600 faces regardless of input resolution, so there was never any
   benefit to optimizing over hundreds of thousands of vertices; the regularization weights had
   simply never been tuned at real-mesh density. Also added: a seeded RNG (camera sampling had been
   unseeded, so failures weren't even reproducible), a hard per-vertex offset clamp, and an
   extents-explosion guard.
5. **The explosion guard didn't catch NaN** — `x > threshold` is always `False` when `x` is `NaN` in
   Python/NumPy/PyTorch, so a NaN-corrupted mesh sailed straight past the one guard meant to catch
   exactly this. Added an explicit `np.isfinite` check, plus defensive degenerate-face removal right
   after decimation.
6. **Retuned regularization weight (2000→6000)** after real-mesh testing still showed edge-ratios
   (1.27-1.31) just outside the `0.8-1.25` bound. Added a deterministic synthetic NaN-guard test
   (`test_phase3_nan_guard.py`, force-injects a NaN via monkeypatching `Adam.step`) rather than
   waiting on the intermittent organic failure (confirmed ~20% observed rate) to recur naturally. A
   5-seed sweep with the retuned weight: 4/5 passed, seed 4 still hit a real NaN — the retune fixed
   the "too weak" symptom but not the underlying instability.
7. **Root-caused the residual NaN via literature, not more guessing**: running a Stable-Diffusion-
   family VAE in fp16 is a well-documented overflow/NaN source (the same class of bug as the
   well-known SDXL-VAE-fp16 black-image issue), and this pipeline's input — a synthetic, hard-masked
   normal-map render — is more likely to trigger it than a natural photo the VAE was tuned on. Fix:
   upcast just the VAE to fp32, keep UNet/CLIP in fp16. Also added, per SDS-stability literature
   (arXiv:2310.12474): gradient clamping/`nan_to_num` on the SDS signal itself, and a
   skip-bad-iteration guard (mechanistic reasoning: Adam's persistent `exp_avg_sq` moment estimate
   is corrupted forever once a single NaN gradient touches it via `optimizer.step()`, which explains
   why the failure was only ever observed in the *final* output, never mid-run). **Result: 10/10
   seeded runs (seeds 0-9) passed cleanly against `tests/can.glb`, zero skip-events triggered** —
   strong evidence the fp32-VAE fix was the actual root cause, not just the other defenses papering
   over it.

### The detour: chasing a visual complaint that turned out to be upstream of SDS entirely

Separately from the numeric-stability saga above, the user reported the final CLI output on
`tests/can.glb` "looks very strange" despite every numeric check passing — prompting a debug-dump
tool (`mesh_debug.py`: dumps a mesh + 4-view normal-shaded contact sheet at every pipeline stage).
A killed mid-run already showed the smoking gun: **the input mesh (`00_input`) was clean and
smooth; immediately after decimation (`01_decimated`, *before SDS ever runs*), the surface was
visibly noisy with salt-and-pepper vertex-normal shading** — meaning the entire SDS bug-chasing
saga above, while all individually real fixes, was very likely not the actual source of the user's
complaint at all. Leading hypothesis at the time: `simplify_quadric_decimation` leaving inconsistent
face winding, with nothing in the decimation code path calling `fix_normals()` (unlike the final
MeshAnything-output cleanup, which does). This was never fully chased to ground before the
investigation moved on to a structural fix instead (see next section) — noted here as a historical
dead end, not a resolved bug in its own right, though the structural fix below coincidentally
addresses the same root cause a different way.

### New pipeline variant: `refine_mesh_direct_sample` — no SDS, and it wins

Built as a standalone alternative to `refine_mesh_coarse` (kept side by side, `--algo legacy` vs.
`--algo direct-sample`), motivated by two things: `tests/watch.glb`'s band coming out visibly
faceted/hexagonal despite `refine_mesh_coarse`'s existing repair calls, and SDS being an uncertain
net-positive whose hyperparameters (tuned against `can.glb`) don't obviously generalize. Design:
skip SDS and the diffusion pipeline entirely; sample the MeshAnything conditioning point cloud
directly from the **original, repaired, full-resolution** mesh (not a decimated one — decimation is
kept only as an optional debug-visualization stage, gated behind `--debug-dir`); use "hybrid"
sampling (half uniform, half curvature-feature-weighted) for coverage without starving flat regions;
a larger, adjustable point count (16384 vs. legacy's fixed 8192 — verified MeshAnything's
Perceiver-style point encoder isn't tied to a specific count).

Building this surfaced two more real, independent bugs, both in `mesh_repair.py`:
- **`is_winding_consistent` (whole-mesh) is the wrong metric for real assets.** `tests/watch.glb`
  has 22,237 disconnected face-adjacency components pre-processing, 87% of them 1-3-face
  marching-cubes noise fragments — disconnected pieces have no shared frame to be "consistent"
  with each other, so the whole-mesh check fires false alarms regardless of whether the substantive
  geometry is fine. Fixed with `remove_small_islands()` (drop sub-threshold components before
  repair, which also fixed a face-count blowup where `fill_holes()` was capping ~19K noise
  fragments for no benefit) and `is_winding_consistent_per_component()` (the actually-meaningful
  per-component check).
- **Non-manifold edges silently defeat `fix_winding`'s spanning-tree algorithm.** `fix_winding` only
  verifies/corrects winding along a BFS spanning tree of face adjacency — sufficient for a true
  2-manifold, but non-manifold edges (shared by 3+ faces) break the guarantee that spanning-tree
  consistency implies full consistency. Found on `tests/can.glb`: 89 non-manifold edges / 287 faces
  out of 17,711, and `fill_holes()` itself re-introduces some at the seams of the caps it adds.
  Fixed with `remove_nonmanifold_faces()`, called both before and after `fill_holes()`.

**Result across the 4 assets tested** (`can`, `watch`, `hand`, `car`): direct-sample matched legacy
on `can.glb`, won clearly on `watch.glb` (correctly round band vs. visibly hexagonal), won
dramatically on `hand.glb` (all five fingers distinctly separated vs. total finger-collapse
failure), and **neither approach worked on `car.glb`** — fragmented into disjoint chunks
(windshield-frame-like piece, wheel-like cylinder, etc.) across three independently different
conditioning strategies (uuniform sampling, feature sampling, hybrid sampling from a cleaner
higher-resolution source with 2x the points) — convergent evidence this is a genuine MeshAnything
limitation on this specific multi-part, open-cockpit vehicle shape, not a pipeline/sampling bug.
**`refine_mesh_direct_sample` was recommended as the preferred default going forward** (never lost,
won decisively twice, structurally simpler, ~4-5x faster: low single digit minutes vs. 15-20) —
`--algo legacy` was left as the CLI's hard-coded default anyway so existing callers weren't
affected, but there was no longer a strong reason to keep investing in SDS tuning.

### The actual finding that changed the plan: MeshAnything V2 never produces quad-dominant output

The plan's stated objective, from the very top of `meshanything_v2_plan.md`, was to regenerate a
shape as an "artist-style quad-dominant coarse mesh," with Phase 6's acceptance criterion explicitly
requiring "visibly quad-dominant topology." **This had never actually been checked** — every prior
verification was numeric (bounding-box/edge-length ratios) or visual-silhouette only. Checked
directly, for the first time, via vertex valence distribution (a triangulated quad-dominant mesh
shows a valence-4 majority; ordinary free triangulation clusters around valence-6, a direct
consequence of Euler's formula): **valence-6 was the plurality or majority on every real test
asset** (can 64.4%, watch 50.2%, hand 23.7%). A controlled test on a pristine, correctly-oriented
synthetic cylinder — the best-case, simplest, cleanest possible input — came back **88.3%
valence-6**, more triangle-dominant than any real asset.

Checked the vendored reference implementation itself (`third_party/MeshAnythingV2/main.py`,
unmodified upstream code): it uses the exact same triangle-only reshape/cleanup sequence this repo's
bridge does. Neither implementation ever reconstructs an explicit polygon face. Confirmed against
the literature rather than resting on empirical testing alone: neither the MeshAnythingV2 README
nor its paper abstract claims quad or quadrilateral output anywhere — "Artist-Created Mesh" (ACM)
names a *training-data category* (meshes made by human artists), not a topology guarantee. The
model architecture is fundamentally triangle-tokenized (`face_per_token = 9`, always 3 vertices × 3
coordinates — no path to emit a 4-vertex face). Independent 2025-2026 papers in the same space
confirm this isn't specific to this pipeline: QuadLink (arXiv:2605.16813) states plainly that
"existing autoregressive methods like Meshtron, DeepMesh, and MeshAnything V2 all generate pure
triangle outputs"; QuadGPT (arXiv:2509.21420) notes the field's own standard workaround is exactly a
triangle-to-quad conversion postprocess applied after generation, not something these models do
natively — and that doing this postprocess "typically produces quad meshes with poor topology,"
which is precisely why QuadGPT/QuadLink exist as native-quad alternatives in the first place. Both
were checked as possible drop-in replacements and ruled out: QuadLink has no public code/weights and
its own paper says its tri-to-quad operator's assumptions don't hold on noisy generated (rather than
clean ground-truth) triangle meshes; QuadGPT's reproducibility statement says a full model release
"is not feasible at this stage."

**Bottom line**: this was never a bug to fix in this pipeline — the plan's objective was
unachievable with MeshAnything V2 as the chosen model, full stop. See "Adopting QuadriFlow" below
for the actual resolution.

---

## Adopting QuadriFlow for genuine quad-dominant output

General triangle-to-quad remeshing has been a solved problem in geometry processing for close to a
decade — Instant Meshes (SIGGRAPH Asia 2015) and QuadriFlow (SGP 2018, builds on Instant Meshes,
~4x fewer singularities, MIT license) both have real, working, open-source releases. Went with
QuadriFlow. Vendored into `third_party/QuadriFlow/` (source only, same pattern as MeshAnythingV2).
Built via conda-forge Eigen/Boost (no `sudo` needed in this environment) into a binary with **zero
GPU/ROCm dependency at all** — confirmed via `ldd`, only standard system libs.

**One integration snag, one fix**: QuadriFlow needs a genuinely 2-manifold input (not
watertight — open boundaries are fine — but even a handful of non-manifold edges/vertices makes it
fail outright with a "wrong init" error, no output file). First attempt failed on a
`repair_mesh()`-cleaned `can.glb` output that still had 3 residual non-manifold edges — the same
`fill_holes()`-reintroduces-non-manifold-edges behavior already found and fixed once in
`repair_mesh()` itself, but this time the non-manifold removal pass had only been run once, *before*
`fill_holes()`, not again after. Fixed by re-running `remove_nonmanifold_faces()` after
`fill_holes()` too.

**Verified on two very different real assets**: `tests/can.glb` → 334/334 output faces are genuine
quads (checked directly on the raw `.obj` face structure, not inferred from valence statistics this
time); `tests/hand.glb` (five separate bone/finger parts, the hardest topology of the whole session)
→ 1349/1349 output faces are quads, with all five fingers remaining distinctly separated after
remeshing. `tests/car.glb` and `tests/watch.glb` were not yet tested with this specific step at the
time.

**Integration**: `trellis2/utils/quad_remesh.py`'s `remesh_to_quad_dominant_obj(mesh,
output_obj_path, target_faces=None)`. Deliberately writes an `.obj` file rather than returning a
`trimesh.Trimesh` — `Trimesh` always triangulates internally, and this pipeline's other export
target (`.glb`) only supports triangles too, so either representation would silently discard the
quad structure that's the entire point. Wired into `tools/mesh_refine_meshanything.py` as an opt-in
`--quad-remesh <path.obj>` flag (works with either `--algo`), producing the quad mesh as a *separate*
file alongside the normal triangulated `--output`, not a replacement for it. `target_faces` defaults
to the input mesh's own triangle count — an untuned first guess, not calibrated against what
actually looks best.

**State at handoff**: nothing downstream of that `.obj` exists — no UV unwrap, no texture, no
subdivision, only 2 of the available test assets validated with this specific step, not yet the
pipeline default. This is exactly where `quadriflow_postprocess_plan.md` picks up.

---

## Where things stand now

- **`mesh_improvements.md`** — kept, trimmed of the now-decided integration roadmap (its old "Option
  A/B/C/D" section), corrected where it stated or implied MeshAnything output is quad-dominant
  (false, see above). Still the reference for the broader SOTA survey (remeshing methods, texture
  optimization methods) that motivated all of this in the first place.
- **`quadriflow_postprocess_plan.md`** — the live, forward-looking plan: what happens after
  QuadriFlow produces a quad-dominant `.obj` (UV unwrap, texture transfer from the original mesh,
  optional subdivision, validation, and an explicit Option-1-vs-Option-2 decision on whether
  MeshAnything should still sit upstream of QuadriFlow at all, or whether QuadriFlow should run
  directly on the original mesh instead).
- **`meshanything_impl_issues.md`, `meshanything_progress.md`, `meshanything_v2_plan.md`** — removed;
  their still-relevant content is folded into this file, and their action items either completed
  (Phases 0-6.8, per above) or superseded by `quadriflow_postprocess_plan.md`'s own phases (the old
  Phase 7-10 subdivision/texture-transfer/validation work, never started, is now that doc's job
  instead, operating on QuadriFlow's output rather than MeshAnything's raw triangles).
