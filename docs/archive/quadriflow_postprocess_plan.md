# Implementation Plan: Post-Processing After QuadriFlow Quad Conversion

Picks up exactly where this fork's original MeshAnything V2 integration effort left off, after
its central finding: **MeshAnything V2 does not work as a framework for generating
quad-dominant meshes** (see "Why MeshAnything was dropped as the quad mechanism" below, and
`history.md` for the full account). QuadriFlow has already been integrated as the replacement and
is confirmed working. This doc assumes QuadriFlow (not MeshAnything, not a hand-rolled
triangle-to-quad merge) is the "initial quad conversion" step, and is entirely about **what happens
to its output next** — right now, nothing does, and that gap is this doc's scope. Written
phase-by-phase for an LLM coding agent: each phase lists goal, files, tasks, acceptance checks.

---

## Why MeshAnything was dropped as the quad mechanism (context, not this doc's problem to re-solve)

This fork checked, for the first time, whether MeshAnything V2's output is
actually quad-dominant (the stated objective of the original MeshAnything-based plan — see
`history.md` for the full investigation) — via vertex valence
distribution (a triangulated quad mesh shows a valence-4 majority; free triangulation clusters
around valence-6). **It never is**, on real assets or even a synthetic best-case cylinder (88.3%
valence-6). Confirmed against the literature too: MeshAnything V2's architecture is fundamentally
triangle-tokenized (`face_per_token = 9`, always 3 vertices — no path to emit a 4-vertex face), and
independent 2025-2026 papers (QuadLink, QuadGPT) explicitly list MeshAnythingV2 among models that
"generate pure triangle outputs," noting the field's own standard fix is exactly a triangle-to-quad
conversion postprocess. **This was never a bug in this pipeline** — the plan's objective was
unachievable with MeshAnything V2 as the chosen model.

## QuadriFlow is already integrated and confirmed working — treat it as a solved dependency

Resolved in the same investigation: QuadriFlow (SGP 2018, builds on Instant Meshes, MIT license) is
an established, released, field-aligned triangle-to-quad remesher — exactly the missing piece.
Already vendored and built:

- **`third_party/QuadriFlow/`** — source vendored (no `.git`), build steps in
  `third_party/QuadriFlow/BUILD_NOTES.md`. Built binary at `third_party/QuadriFlow/build/quadriflow`,
  statically linked against system libs only (`ldd`: `libstdc++`/`libm`/`libgcc_s`/`libc`) — **no
  GPU/ROCm dependency at all**, unlike almost everything else in this pipeline. Needs a genuinely
  2-manifold input (non-manifold edges/vertices make it fail outright with a "wrong init" error) —
  not watertightness, open boundaries are fine.
- **`trellis2/utils/quad_remesh.py`** — `remesh_to_quad_dominant_obj(mesh: trimesh.Trimesh,
  output_obj_path: str, target_faces: int = None)`. Applies `repair_mesh()` (from
  `trellis2/utils/mesh_repair.py`) unconditionally first, since that happens to already produce the
  manifoldness QuadriFlow needs, then shells out to the built binary. **Deliberately writes an
  `.obj`, not a `trimesh.Trimesh`** — `Trimesh` always triangulates internally and this pipeline's
  other export target (`.glb`) only supports triangles too, so either would silently discard the
  quad structure that's the entire point. `target_faces` defaults to the input mesh's own triangle
  count if not given (untuned guess).
- **Verified genuine quads on 2 real assets**: `tests/can.glb` → 334/334 output faces are quads;
  `tests/hand.glb` (hardest topology of the session, five separate fingers) → 1349/1349 output
  faces are quads, fingers stay correctly separated. **`tests/car.glb` and `tests/watch.glb` are
  untested with this step** — Phase 1 below covers finishing that validation.
- **Currently wired as an opt-in addendum**: `tools/mesh_refine_meshanything.py --quad-remesh
  <path.obj>` runs this on whatever `refine_mesh_coarse`/`refine_mesh_direct_sample` already
  produced, writing the quad `.obj` as a *separate* file alongside the normal triangulated
  `--output` GLB. **Nothing downstream of that `.obj` exists yet** — no UV unwrap, no texture, no
  subdivision, not validated broadly, not the default. That gap is this entire document.

## Architecture decision point: what feeds QuadriFlow matters, and is not fully settled

Two real options exist for *what mesh QuadriFlow actually remeshes*, with a real quality/effort
trade-off. This doc's phases are written to be agnostic to which one is used (every phase below
takes "a quad-dominant `.obj`" + "a reference mesh to pull detail/texture from" as its contract),
but the choice should be made deliberately, not by default, and re-evaluated once Phase 1's broader
validation lands.

**Option 1 (current wiring) — QuadriFlow on MeshAnything's coarse output.**
```
original mesh → repair_mesh() → refine_mesh_direct_sample() [MeshAnything: artist-style
  retopology + resolution reduction to MeshAnything's ~1600-face cap] → QuadriFlow
  (target_faces defaults to ~1600) → quad .obj
```
Inherits MeshAnything's known failure modes (per `history.md`: `car.glb` fragments
into disconnected chunks regardless of sampling strategy — a MeshAnything limitation, not fixable
by better conditioning) and its ~1600-face ceiling, meaning **detail recovery is not optional
here** — the quad mesh is a drastic simplification of the true asset resolution. `refine_mesh_coarse`
(the older, SDS-based variant) is also available but `history.md` records the explicit
recommendation of `refine_mesh_direct_sample` going forward (never lost to legacy on any tested
asset, wins decisively on 2/4, structurally simpler, ~4x faster).

**Option 2 — QuadriFlow directly on the (repaired, optionally simplified) original mesh, no
MeshAnything at all.**
```
original mesh → repair_mesh() → optional QEM simplification to a chosen target resolution →
  QuadriFlow (target_faces = whatever's chosen, can be the full original resolution) → quad .obj
```
This is `mesh_improvements.md`'s original "Option A" pipeline. Sidesteps MeshAnything's fragility
entirely (no car.glb-style failure mode, no SDS instability history, no ~1600-face bottleneck) —
detail loss is now *only* whatever simplification you deliberately choose, or none at all. What it
gives up: MeshAnything's aim of "artist-style" semantic edge flow (loops following joints/seams the
way a human topologized it) — QuadriFlow's edge flow is purely curvature-field-driven, which
correlates with semantic structure on many organic shapes but isn't the same guarantee.

**Recommendation for Phase 1**: run both options on the same 4-6 test assets and compare by eye
before committing either as default — this is a real, cheap, informative ablation (Phase 1, task 3)
given how much it changes whether Phase 2/3 below are load-bearing or nearly no-ops.

---

## Hard constraints from this environment

- **QuadriFlow itself needs nothing from `amd_workarounds.md`** — pure CPU/C++, no CUDA/ROCm
  dependency, confirmed via `ldd` on the built binary. This phase of the pipeline is much lower-risk
  environmentally than the SDS/rasterizer phases documented in `history.md` were.
- **UV unwrapping and texture rasterization still need this repo's existing HIP-built extensions** —
  reuse them as-is rather than reimplementing:
  - `cumesh.CuMesh().init(vertices, faces).uv_unwrap(return_vmaps=True)` — confirmed triangle-only
    (`faces` is always `(M, 3)` throughout `CuMesh/examples/uv_unwrap.py` and
    `trellis2/pipelines/trellis2_texturing.py:304-313`, the exact call this pipeline already makes).
    **This is the call that produces the fragmented "tiny pieces" charting the original pipeline is
    known for — see Phase 4 below, which changes how this gets called (and, for the preferred path,
    bypasses part of it) rather than reusing it verbatim.**
  - `trellis2/utils/uv_rasterize.py`'s `rasterize_uv`/`interpolate_uv` — pure PyTorch, drop-in for
    `nvdiffrast`'s `dr.rasterize`/`dr.interpolate`, already ROCm-safe, triangle-only.
  - `cumesh.bvh.cuBVH(vertices, triangles).unsigned_distance(positions, return_uvw=True)` — an
    existing, already-HIP-built (confirmed: `third_party/cubvh/src/bvh.hip` alongside the `.cu`,
    already compiled into `CuMesh/build/lib.../cumesh/_cubvh...so`) closest-point query: for each
    query position, returns the nearest face id **and its barycentric coordinates on that face**.
    This is the correct tool for both Phase 2 (vertex snap-back) and Phase 5 (texture transfer)
    below — much faster than CPU `trimesh.proximity.closest_point` at the face counts involved, and
    already proven in this codebase (nothing new to build).
- **QuadriFlow's output is quad-only in the `.obj` file, never in memory as a `trimesh.Trimesh`** —
  every phase that needs to run CuMesh/uv_rasterize/GLB export must work on a **triangulated copy**
  of that `.obj` (`trimesh.load(path, force='mesh')` — verify empirically in Phase 1 that this
  actually triangulates 4-vertex faces rather than erroring or silently dropping them, don't assume).
  The genuine quad structure should still be kept as a separate canonical artifact wherever a later
  step (Phase 3's subdivision) can actually use it.
- **No live PBR-voxel data available to a standalone CLI tool.** `trellis2_texturing.py`'s
  `postprocess_mesh` (and `o_voxel/postprocess.py`'s `to_glb`) bake texture by directly grid-sampling
  a live sparse PBR-attribute voxel tensor (`flex_gemm.ops.grid_sample.grid_sample_3d`) — not by
  multi-view rendering, contrary to `mesh_improvements.md`'s original (superseded) description. That
  voxel tensor only exists inside `run_sample.py`'s live pipeline run, in scope right after
  `pipeline.run(image)` and before `to_glb` bakes it into a flat 2D atlas and discards it. A
  standalone tool operating on an already-exported GLB (this doc's actual scope — see Phase 5) has
  no access to that tensor anymore, only the GLB's own already-baked 2D texture — so texture
  transfer here must be mesh-to-mesh (closest-point projection onto the original textured mesh), not
  a re-run of voxel-grid sampling. Phase 8 notes the cleaner alternative if this is ever wired
  directly into the live pipeline instead.

## Assumed input state / contract for every phase below

Everything below takes two inputs, and is agnostic to which architecture option (1 or 2 above)
produced them:

1. **`quad_obj_path`** — output of `remesh_to_quad_dominant_obj()`: genuine quad faces, no UVs, no
   material, vertex positions at or very near *some* source surface (either MeshAnything's coarse
   approximation, if Option 1, or the true original surface, if Option 2 with light/no
   simplification).
2. **`reference_mesh`** — the mesh QuadriFlow's input was derived from, **before** MeshAnything/
   simplification touched it: full original resolution, presumed textured (e.g. the GLB
   `run_sample.py` already produced, or one of `tests/*.glb`). This is the canonical source for both
   fine geometric detail (Phase 2) and appearance (Phase 5) — never the MeshAnything-coarse or
   QuadriFlow-quad mesh itself, since texture/UV data does not survive either of those steps.

**Concrete plumbing gap to fix before Phase 2 can consume this contract under Option 1**:
`refine_mesh_direct_sample` (`trellis2/utils/meshanything_bridge.py:237`) computes exactly the right
`reference_mesh` internally — the `original_repaired` local variable (line 282) — but never returns
or exposes it; only the final MeshAnything-generated mesh comes back. Add a way to get it out (e.g.
an optional `return_intermediates=True` flag returning `(new_mesh, {"original_repaired": ...})`, or
a second return value) before Phase 2/5 can be wired into that call path. Under Option 2 this gap
doesn't exist — the repaired original mesh is already the direct input to QuadriFlow, no extraction
needed.

---

## Phases

### Phase 1 — Triangulation check + broaden QuadriFlow validation + run the Option 1 vs. 2 ablation

**Goal**: confirm the basic quad-obj-to-triangle-mesh round trip works as assumed, finish validating
QuadriFlow across more of the available test assets than the 2 checked so far, and get real
evidence for the Option 1 vs. Option 2 decision instead of picking one by default.

**Files**: new `trellis2/utils/quad_postprocess.py` (home for everything in this doc, alongside
`quad_remesh.py`); no changes to `quad_remesh.py` itself expected.

**Tasks**:
1. `load_quad_obj(path) -> trimesh.Trimesh`: wraps `trimesh.load(path, force='mesh')`. Verify
   empirically (don't assume) that this triangulates QuadriFlow's 4-vertex faces rather than
   erroring — print face count before/after and confirm it's plausible (roughly 2x the quad count
   reported by QuadriFlow, not equal to it and not zero).
2. Run `remesh_to_quad_dominant_obj` on `tests/car.glb` and `tests/watch.glb` (the two assets
   `history.md` flags as untested with this step) through the existing Option 1
   path (`refine_mesh_direct_sample` → quad-remesh). Check quad-fraction directly on the `.obj`
   (parse face vertex counts, don't infer from valence stats — same lesson `history.md`
   already records learning the hard way). Note `car.glb` is expected to still hit MeshAnything's known
   fragmentation limitation *before* QuadriFlow ever runs — this validates QuadriFlow's own
   robustness on whatever fragmented input it's handed, not a fix for that upstream problem.
3. **Run the Option 2 ablation**: on the same assets (plus `can.glb`/`hand.glb` for a full
   like-for-like comparison), call `remesh_to_quad_dominant_obj` directly on `repair_mesh(original)`
   — skip `refine_mesh_direct_sample`/MeshAnything entirely — once with `target_faces` at the
   original resolution and once at a deliberately reduced target (e.g. matching Option 1's ~1600 for
   a fair comparison, and separately at a higher value to see full-detail quality). Open both
   results in a viewer side by side. Record, per asset: quad fraction, visual topology quality,
   whether edge flow looks anatomically/structurally sensible, and wall-clock time.
4. Write a short comparison note in this doc (append below this task list once done) recommending
   Option 1 or Option 2 as the new default — or documenting that it's asset-dependent if that's what
   the evidence shows.

**Acceptance**: quad-fraction numbers recorded for all of {can, hand, car, watch} under Option 1,
and for at least {can, hand, car, watch} under Option 2 at 2 target-face settings each; a written
recommendation exists, informed by actual renders, not assumed.

### Phase 1 results and recommendation (completed)

**Triangulation round-trip** (task 1): confirmed on can/hand's existing quad outputs --
`trimesh.load(path, force='mesh')` triangulates every quad into exactly 2 triangles (668/334=2.0,
2698/1349=2.0 exactly, both 100%-quad assets), never errors or drops faces.

**Option 1** (`refine_mesh_direct_sample` → QuadriFlow), broadened to car/watch:
- **can, hand**: as previously established, 100% quads, correct topology.
- **car**: MeshAnything's own output collapses to 116 faces / 107 verts (should be ~1600-ish) --
  QuadriFlow "succeeds" on this (71 quad faces, 100% quad_fraction) but the result is an
  unrecognizable shattered fragment, confirmed both by quad-fraction being meaningless here and by
  direct visual inspection. This is MeshAnything's known fragmentation limitation (see history.md),
  not a QuadriFlow problem, and QuadriFlow cannot be expected to fix garbage input.
- **watch**: QuadriFlow **fails outright** (`wrong init` -- its non-manifold-input error), despite
  `remesh_to_quad_dominant_obj`'s built-in `repair_mesh()`. Root cause (confirmed via a diagnostic
  script, not guessed): the repaired mesh has 14 connected components, and one 706/1932-face
  component has Euler characteristic -59 and fails `is_winding_consistent` even after
  `fix_normals()` -- i.e. it's not edge-non-manifold (already stripped) but intrinsically
  non-orientable/self-intersecting, which per-component winding-fix can't repair.
  Dropping any component that individually fails `is_winding_consistent` (not currently done
  anywhere in `mesh_repair.py`, which only *checks* this, never *acts* on it) lets QuadriFlow
  succeed -- but the recovered result is missing that entire component (the watch's case body),
  leaving only the band + clasp. **Recommended follow-up hardening**: add
  `remove_winding_inconsistent_components()` to `mesh_repair.py` alongside the existing
  `remove_small_islands`/`remove_nonmanifold_faces` -- out of scope to implement as part of this
  phase (no `mesh_repair.py`/`quad_remesh.py` changes were expected here), but should land before
  Option 1 is trusted as a default path for anything.

**Option 2 ablation** (repair_mesh(original) → QuadriFlow directly, no MeshAnything), at two
target-face settings per asset. Note: the plan's literal wording ("target_faces at the original
resolution") was tried first and aborted -- QuadriFlow at ~450K target faces ran 11+ minutes on
`can` alone with no sign of finishing, impractical across 4 assets and not actually required by the
plan's own "e.g." phrasing. Substituted **1600** (matches Option 1's cap, for a fair comparison) and
**20000** (a genuine high-detail setting, >10x Option 1, but tractable) as the two settings:

| asset | target | output faces | quad_fraction | time |
|---|---|---|---|---|
| can   | 1600  | 323   | 1.0 | 12.6s |
| can   | 20000 | 14175 | 1.0 | 54.5s |
| hand  | 1600  | 310   | 1.0 | 23.3s |
| hand  | 20000 | 17996 | 1.0 | 64.1s |
| car   | 1600  | 462   | 1.0 | 23.6s |
| car   | 20000 | 17115 | 1.0 | 71.2s |
| watch | 1600  | 505   | 1.0 | 21.7s |
| watch | 20000 | 17021 | 1.0 | 52.2s |

**8/8 succeeded, 100% quad fraction every time, all under 90 seconds.** Visual inspection
(wireframe renders, all four assets at the 20000 setting):
- **car**: recognizably a car (roof, hood, wheel-well silhouette all visible) -- night-and-day
  versus Option 1's unusable 71-face shattered fragment.
- **watch**: recovers the complete watch (case *and* band together) -- versus Option 1, which
  fails outright and even with the diagnostic hardening patch only recovers the band (the case
  was in the dropped bad component).
- **hand**: recovers a full connected hand (palm + all 5 fingers as one structure) -- versus
  Option 1's output, which is 5 disconnected thin finger tubes with **no palm at all** (confirmed
  independently via `mesh_debug.render_mesh_contact_sheet` on the same underlying mesh, ruling out
  a rendering bug).
- **can**: both options work; roughly comparable, unremarkable case.

One measurement caveat, not a defect: raw irregular-vertex fraction on Option 2's real-world
multi-part assets (car, watch) came out surprisingly high (58-89%) via `find_irregular_vertices`.
Root-caused: `car`'s repaired input has many disconnected parts (body, wheels, etc.), each producing
its own open-boundary quad patch; boundary vertices are naturally non-valence-4, and this function
doesn't distinguish boundary from interior irregularity (documented in its own docstring). Excluding
boundary vertices on `car`'s 20000-target output drops the figure from 58.7% to a much more normal
**14.6%**, in line with Option 1's can/hand numbers (17.4%/35.2%). Not a topology-quality problem --
a measurement-methodology footnote for whoever next reports irregular-vertex stats on a multi-part
asset.

**Recommendation: Option 2 (QuadriFlow directly on the repaired original mesh) as the default**,
not asset-dependent -- it won or tied on every one of the 4 test assets, never failed, and is
already fast (all runs well under 2 minutes, no GPU/MeshAnything inference required at all). Option
1 should be kept available (it's a real, if narrower, use case: MeshAnything's "artist-style" edge
flow on assets simple enough for it not to fragment) but not as the default, and not trusted in
production until the `mesh_repair.py` hardening noted above lands. This directly informs Phase 6's
`--source {meshanything,direct}` flag: **default to `direct`**.

### Sidetask — quad-topology wireframe visualization

**Goal**: a debug visualization utility — render the mesh normally (shaded, using the same
triangulated copy every render/export path already needs), then overlay the mesh's **genuine quad
edges** as a wireframe on top, so the actual quad topology is directly visible on the shaded surface
instead of only inferable from raw face data or a plain wireframe-only render. This is what Phase
1's "open both results in a viewer side by side" task and Phase 2/3's "spot-check by opening in
Blender/`trimesh.Scene().show()`" acceptance checks should actually use once this exists, rather
than an external DCC tool or a bare wireframe with no shading context. Also mark irregular
(valence ≠ 4) vertices distinctly — the same singularities Phase 4 task 2's separatrix tracing
depends on, so this doubles as a way to sanity-check that step's input before building it.

**Files**: new `trellis2/utils/quad_debug.py` — mirrors `mesh_debug.py`'s existing contact-sheet
convention from the earlier MeshAnything integration effort (see `history.md`: dumps a multi-view,
auto-framed PNG per stage) rather than inventing a new visualization pattern from scratch.

**Tasks**:
1. `render_quad_wireframe(quad_mesh_path_or_trimesh, camera, resolution) -> np.ndarray`:
   - Load/triangulate the quad mesh (reuse Phase 1's `load_quad_obj`), render it shaded via the
     existing `trellis2/utils/mesh_rasterizer.py`'s `render_mesh()` (built for SDS during the
     earlier MeshAnything effort — still live, no new rasterizer needed).
   - Separately project each **genuine quad edge's** two endpoints into screen space using the same
     camera/projection math `render_mesh()` already uses (`intrinsics_to_projection`), and
     rasterize them as thin (1-2px), anti-aliased lines in a color that reads clearly against the
     shading (e.g. solid black or a bright accent color). **Only the real quad edges — never the
     triangulation diagonal** introduced solely to make the mesh renderable; drawing those would
     defeat the entire point of this tool.
   - Depth-test the wireframe against the shaded render's own z-buffer (offset the line's depth
     slightly toward the camera by a small epsilon — the standard real-time-graphics
     "wireframe-over-shaded" bias trick) so lines on the visible/front-facing surface draw and lines
     on the mesh's occluded far side are correctly hidden, rather than drawing every edge regardless
     of visibility.
   - Composite the wireframe over the shaded render.
2. Mark irregular (valence ≠ 4) vertices with a small distinct marker (e.g. a filled dot in a second
   accent color) at their projected screen position — reuse `find_irregular_vertices()` from Phase 4
   task 2 if that's already built, or a standalone valence count otherwise (trivial either way; don't
   block this sidetask on Phase 4 landing first).
3. Multi-viewpoint contact sheet (front/right/back/¾, auto-framed to the mesh's own bounding box) —
   same convention `mesh_debug.py` already established, reused rather than reinvented, so this
   composes naturally if `--debug-dir` is ever extended to call it from Phase 6's CLI.

**Acceptance**: run against 2-3 QuadriFlow outputs from Phase 1 and visually confirm the overlaid
wireframe follows the mesh's actual quad grid (no triangulation diagonals visible anywhere),
correctly disappears on the mesh's occluded/far side rather than showing through, and
irregular-vertex markers land at plausible structural locations (joints, corners, tips) rather than
looking randomly scattered. Once built, treat this as the standard visual-check tool for Phase 1
(validating QuadriFlow output), Phase 2 (before/after the vertex snap-back), and Phase 3
(before/after subdivision) in place of a generic "open in Blender."

### Phase 2 — Vertex snap-back onto the reference mesh (lightweight, likely optional under Option 2)

**Goal**: correct any drift between the quad mesh's vertex positions and the true reference
surface. Note up front: this is **not** the same kind of step as the original MeshAnything-based
plan's Phase 7 `bake_detail()` (see `history.md`) — that one had to recover from MeshAnything's
drastic ~1600-face compression, a huge
correction. Here, under Option 1, QuadriFlow's own position-field mechanism already keeps vertices
close to whatever surface it was given (MeshAnything's coarse approximation), so this is a
precision/hygiene pass on top of an already-reasonable fit, not detail recovery. Under Option 2 with
no MeshAnything step, the "coarse approximation" doesn't exist at all and this step may be close to
a no-op — quantify that empirically (task 3) rather than skipping the phase outright, since it's
cheap to run and free to discover it does nothing.

**Files**: `trellis2/utils/quad_postprocess.py` (extend)

**Tasks**:
1. `snap_to_reference(mesh: trimesh.Trimesh, reference_mesh: trimesh.Trimesh) -> trimesh.Trimesh`:
   build `cumesh.bvh.cuBVH(reference_mesh.vertices, reference_mesh.faces)` once, query
   `unsigned_distance(mesh.vertices, return_uvw=True)` to get each vertex's nearest point on
   `reference_mesh` (reconstruct the actual 3D position from `face_id` + barycentric `uvw` against
   `reference_mesh`'s own triangle corners — `cuBVH` returns the barycentric weights, not the
   position directly, confirm this by reading `signed_distance`'s / `unsigned_distance`'s actual
   output shapes in `CuMesh/cumesh/bvh.py` before assuming). Replace `mesh.vertices` with these
   snapped positions; keep faces/quad-structure untouched (this only moves points, never changes
   topology, exactly like `o_voxel/postprocess.py`'s existing `project_back` parameter on its own
   `remesh_narrow_band_dc` step — a direct precedent for this same pattern already in this repo,
   worth reading for calling-convention inspiration).
2. Operate on the **triangulated** copy of the quad mesh (`load_quad_obj`'s output) for the BVH
   query itself (per-vertex positions are identical whether you think of them as triangle or quad
   corners), then write the snapped positions back onto the genuine quad structure (same vertex
   indices, only coordinates change) so Phase 3's subdivision still gets real quads.
3. **Measure, don't assume, how much this actually moves things**: report mean/max per-vertex
   displacement, separately for an Option-1-derived mesh and an Option-2-derived mesh, on 2-3 assets.
   If Option 2's displacement is negligible (e.g. sub-voxel-scale), that's the empirical basis for
   treating this phase as skippable under Option 2 — don't decide that in advance.

**Acceptance**: displacement numbers recorded for both options; a rendered before/after comparison
shows the snap either visibly improving fit (Option 1) or having negligible visible effect (Option
2, if that's what the numbers show) — no case where it visibly *degrades* the mesh (e.g. by pulling
a vertex to a wrong-sided closest point across a thin feature — check for this specifically on any
thin/close-together geometry among the test assets, e.g. `watch.glb`'s band).

### Phase 3 — Optional Catmull-Clark subdivision

**Goal**: smooth/round out the quad mesh via subdivision — now actually well-motivated, unlike
the original MeshAnything-based plan's Phase 7 (see `history.md`). That plan assumed MeshAnything's raw output was
"a mix of quads/tris/pentagons" needing n-gon-aware Catmull-Clark; we now know MeshAnything's output
is always pure triangles (see context section above), so subdividing it directly would have been
pointless. QuadriFlow's actual quad-dominant output (genuine quads, with expected occasional
triangle/pentagon irregular vertices) is the *correct* input for Catmull-Clark for the first time in
this pipeline's history.

**Files**: new `trellis2/utils/subdivision.py`, or inline in `quad_postprocess.py` if small enough.

**Tasks**:
1. Same research spike the original MeshAnything-based plan flagged and never resolved (see
   `history.md`): nothing in
   this repo (`CuMesh`, `o-voxel`, `trimesh`) does true n-gon Catmull-Clark —
   `trimesh.remesh.subdivide` only midpoint-splits triangles, which is the wrong operation for a
   quad-dominant mesh. Try `pyvista` first (CPU-only, VTK-backed, no new GPU build needed) — same
   recommendation the old plan made, still unstarted.
2. `subdivide_catmull_clark(quad_mesh_path_or_trimesh, levels=1) -> trimesh.Trimesh` (genuine
   quad-preserving output, not triangulated). Start with 1 level, not 2-3 like the old plan assumed —
   QuadriFlow's output is not coming from a ~1600-face bottleneck under Option 2, and even under
   Option 1 the old plan's "2-3 levels to recover detail" reasoning doesn't transfer directly since
   the detail-recovery job now belongs to Phase 2's snap-back, not subdivision. Subdivision here is
   about surface smoothness/silhouette quality, evaluate 1 level visually before assuming more is
   needed.
3. Re-run Phase 2's `snap_to_reference` on the subdivided mesh's (new, denser) vertex set afterward —
   subdivision introduces new vertices at edge midpoints/face centers that won't already lie on the
   reference surface, so this ordering matters: subdivide first, snap second, not the reverse.

**Acceptance**: render subdivided+snapped output next to the pre-subdivision quad mesh; confirm
visibly smoother silhouettes with no new self-intersections or obviously wrong geometry (spot-check
by opening in Blender or `trimesh.Scene().show()`). **Stop and evaluate before Phase 4** if this
phase's cost/complexity (new subdivision library integration) isn't clearly paying for itself
visually — it's the one phase in this doc that's genuinely optional, unlike Phases 4-5 which are
required for any usable output at all.

### Phase 4 — UV unwrap: few, large, semantically-bounded charts packed into one atlas

**Goal**: give the (possibly subdivided) quad mesh actual UVs — it currently has none, at any stage
up to this point — as a **small number of large charts with boundaries at meaningful locations**
(ideally one chart per natural surface region — a limb, a panel, a face of the object — the way a
human artist would cut it), not the many small, distortion-greedy fragments the *original* TRELLIS.2
texturing pipeline produces via a plain `cumesh.uv_unwrap()` call. Packing everything into a single
shared texture atlas is **already the existing behavior and doesn't need new work** (see "Why this
happens" below) — the actual problem is chart *count and size*, not packing.

**Why this happens (root-caused by reading the actual code, not assumed)**: `CuMesh.uv_unwrap()`
(`CuMesh/cumesh/cumesh.py:408-480`) runs in two stages, and the *first* stage is what sets a hard
floor on chart count that the second stage can never undo:

1. `self.compute_charts(**compute_charts_kwargs)` — CuMesh's own native, GPU-accelerated
   normal-cone chart clustering (`CuMesh/cumesh/cumesh.py:361-391`), *not* xatlas. Default
   parameters: `threshold_cone_half_angle_rad=90°`, `area_penalty_weight=0.1` (the docstring says
   this literally *penalizes* chart area — "Cost += Area \* weight... encourages larger charts if
   < 0" — so the shipped default actively discourages growing large charts), plus a small
   `perimeter_area_ratio_weight` discouraging long thin strips. This produces `num_charts`
   pre-clusters.
2. **Each pre-cluster is then added to a fresh `xatlas.Atlas()` as its own separate, disconnected
   mesh instance** (`for i in range(num_charts): xatlas.add_mesh(chart_vertices_i, chart_faces_i)`,
   `CuMesh/cumesh/cumesh.py:451-460`), and `xatlas.compute_charts()`/`pack_charts()` run once across
   all of them. Because each pre-cluster is a structurally separate mesh from xatlas's point of
   view, xatlas's own chart-growing can only **split a pre-cluster into more charts, never merge two
   pre-clusters back into one** — there's no shared connectivity left for it to grow across. **The
   final chart count can never be smaller than stage 1's `num_charts`.** `xatlas_compute_charts_kwargs`
   (already exposed by `cumesh.xatlas.Atlas.compute_charts`, full xatlas `ChartOptions` surface:
   `max_chart_area`, `max_boundary_length`, `normal_deviation_weight`, `max_cost`, etc.) only
   controls further splitting *within* that floor — it cannot fix over-fragmentation that already
   happened in stage 1.

**Files**: `trellis2/utils/quad_postprocess.py` (extend)

**Tasks**:

1. **Quick experiment first — retune stage 1's own parameters, no new code.** Sweep
   `compute_charts_kwargs={"area_penalty_weight": 0.0}` (or a small negative value, per the
   function's own docstring) and `threshold_cone_half_angle_rad` (try both tighter, e.g. 45-60°, for
   more curvature-faithful boundaries, and the 90° default) against 3-4 test assets spanning organic
   and hard-surface shapes (e.g. `hand`, `can`, `watch`, `car`). Report chart count and a chart-size
   histogram (face count per chart) before/after — this alone might close most of the gap, and it's
   a config change against an already-exposed parameter, not new engineering. Do this before
   investing in task 2.

2. **If that's not enough — bypass stage 1's generic clustering with QuadriFlow's own field
   structure**, which is a strictly better source of "semantic" boundaries than curvature clustering
   for a mesh QuadriFlow just produced. This is the same idea as **motorcycle graphs**
   (Eppstein/Goodrich/Kim/Tamstorf) / the "patch layout" extraction used in quad-mesh tools like
   QuadWild: a quad mesh's irregular vertices (valence ≠ 4) are exactly where QuadriFlow's min-cost-flow
   solve placed unavoidable singularities — tracing straight lines along the quad grid ("separatrices")
   outward from each singularity until they hit another singularity or a mesh boundary partitions the
   whole mesh into a small number of large, roughly-quadrilateral "patches" bounded at genuinely
   structural locations, not a curvature threshold's arbitrary cutoff:
   - `find_irregular_vertices(quad_mesh) -> np.ndarray`: valence ≠ 4 (computable directly from the
     quad mesh's own face-vertex incidence — no need to extract this from QuadriFlow's internal
     `singularities` map, which isn't exposed in the output `.obj` anyway; recomputing it from the
     final topology is simpler and self-contained).
   - `trace_separatrices(quad_mesh, irregular_vertices) -> List[polyline]`: from each irregular
     vertex, walk each incident quad-grid edge direction in turn, continuing straight across each
     shared quad edge (the standard quad-mesh "continue straight, exit opposite the entry edge" rule)
     until reaching another irregular vertex or a boundary edge.
   - `patches_from_separatrices(quad_mesh, separatrices) -> List[face_index_array]`: the
     separatrices, together, form a graph that partitions the quad mesh's faces into connected
     regions — flood-fill face adjacency without crossing a separatrix edge to get each patch's face
     list.
   - Feed each patch into `cumesh.xatlas.Atlas` directly as its own `add_mesh()` call (reusing the
     exact same "add each group as a separate atlas mesh, then one shared `compute_charts`/
     `pack_charts`" pattern `CuMesh.uv_unwrap()` already uses at `cumesh.py:451-460` — just replacing
     *what* the groups are: motorcycle-graph patches instead of curvature pre-clusters). Each patch
     becomes one chart (or, on a patch large/curved enough to still need internal seams, xatlas's own
     `compute_charts` can still subdivide *within* it — but the floor is now at genuinely structural
     boundaries, not an arbitrary clustering heuristic's).
   - A patch that's topologically a disk (the common case) flattens with low distortion via ordinary
     LSCM/ABF (xatlas's own default) since it's bounded by roughly-straight quad-grid lines. If a
     patch is degenerate (e.g. contains a boundary loop with no interior structure) or produces
     visibly worse distortion than task 1's tuned-parameter baseline on a specific asset, fall back
     to that asset's task-1 result rather than forcing this path everywhere — **compare both, per
     asset, don't assume the structural approach always wins.**

3. Triangulate the final geometry (post-Phase-3 if that ran, else Phase 1/2's triangulated quad mesh
   directly) for the actual UV-coordinate assignment step, regardless of which of task 1/2 produced
   the chart layout — genuine quads are not needed past this point, everything downstream is
   triangle-only per the Hard Constraints section. Note this can itself change vertex count/order —
   carry a vertex map forward if anything later needs to trace a UV-space vertex back to its
   pre-unwrap position (Phase 5 doesn't, since it re-queries positions via `interpolate_uv` in UV
   space directly, but note this for anyone extending the pipeline later).

**Acceptance**: report, per test asset and per method (task 1's tuned baseline vs. task 2's
patch-layout approach if built): total chart count, mean/median chart face-count, and what fraction
of total mesh surface area is covered by the largest 5 charts — these are the objective, checkable
proxies for "few large charts" this doc uses instead of a subjective "looks semantic" judgment.
Visually confirm (render UV chart boundaries overlaid on the mesh, e.g. color faces by chart id) that
boundaries land at plausible structural locations (limb/panel/feature boundaries) rather than
mid-surface on a flat, featureless region. Confirm the final result is still packed into one shared
texture atlas (already true of both the existing `pack_charts` call and task 2's reuse of the same
pattern — this should require no new verification beyond confirming it wasn't accidentally broken).
Resulting UVs are all within `[0, 1]`; spot-check for degenerate (near-zero-area) UV triangles by
rasterizing at a modest resolution (e.g. 256×256) via `rasterize_uv` and confirming texel coverage is
reasonably dense, not mostly background.

### Phase 4 results (completed)

**Task 1 (parameter retuning)**: negative result. Swept `area_penalty_weight` in {0.1 (default),
0.0, -0.05} and `threshold_cone_half_angle_rad` in {90° (default), 60°, 45°} on can (668 tri) and
hand (2698 tri). `area_penalty_weight` had essentially no effect at either mesh's scale (can stayed
at 15 charts across all 3 values). Tightening the cone angle made things *worse*, not better (can:
15→28→41 charts as angle tightened; hand: 177→299→392) -- the default (already the loosest setting)
was already best among those tested. Even at the best available parameters, 15 charts for a simple
capped cylinder and 177+ for a 5-finger hand is nowhere near "few large charts" — confirming task 2
is actually required, not just a fallback to try.

**Task 2 (motorcycle-graph patch layout)**: built `trace_separatrices`/`patches_from_separatrices`/
`uv_unwrap_via_patches` in `quad_postprocess.py`. First implementation (cut along literally every
traced separatrix from every singularity) catastrophically over-fragmented both assets — can: 334
patches, literally every one of the 334 faces a singleton; hand: 1080 patches, 1009 singletons.
Root cause (confirmed, not guessed): the local "straight ahead" rule is geometrically correct
(median turning angle 20.6° across ~1112 sampled continuations), but since every regular vertex has
a *unique deterministic* straight-ahead pairing, the maximal line through any edge is mathematically
guaranteed to reach a singularity eventually — and can's singularities sit only at its 2 poles, none
along the body, so "vertical" lines necessarily run the cylinder's full length (one traced example:
189 of 339 vertices, zero self-revisits — a genuinely straight line, not a spiraling bug). Tracing
from every singularity in every direction and cutting along all of it therefore claims essentially
every edge whenever singularities are sparse — the opposite of the goal.

**Fix**: the cited motorcycle-graph algorithm has motorcycles stop when they crash into *another*
motorcycle's trail, not only at singularities — my first pass only checked this for a trace's
starting edge, and it turns out that check alone never fires anyway (two distinct maximal lines
can't share an edge under the deterministic pairing rule — confirmed by testing). The actual fix is
a **length cap** (`max_length`, default 8): discard a trace's edges entirely if it exceeds the cap
without reaching a singularity or an existing trail, enforced *during* tracing (a discarded trace's
edges must never enter the shared "consumed" state, or they'd still wrongly block/shorten other
traces).

**Result with max_length=8** (swept 5/8/10/12/16/20/30 — 8 at-or-near-best on both assets):
- **can**: 85 patches, but `top5_area_frac=0.76` — one dominant 241/334-face patch that is,
  visually confirmed, *the entire cylindrical body in a single chart*. The ~80 remaining small
  (mostly 1-2 face) fragments all sit at the two polar z-extremes (checked directly) — the small
  "pie slice" pieces a singularity fan at a disk cap structurally needs, not noise. Dramatically
  better than task 1's 15 same-order-of-magnitude clusters.
- **hand**: 532 patches, `top5_area_frac=0.256` — much less clean than can (hand's singularities
  are dense, 35.2% of vertices, not sparse/polar-only, so "few large patches" is inherently harder
  here), but still enormously better than the uncapped 1009-singleton result.
- `uv_unwrap_via_patches` (xatlas per patch via the same "each pre-cluster is its own atlas mesh"
  pattern `CuMesh.uv_unwrap` already uses) ran successfully end to end on both: can → 729
  verts/668 faces, hand → 4024 verts/2698 faces, UVs in `[0, 0.999]` both times. Texel coverage at
  256×256: can 82.0%, hand 62.0% — neither close to "mostly background", passes.

**Recommendation**: use task 2 (motorcycle-graph, `max_length=8`) as the default chart layout
method, with task 1's tuned parameters (`area_penalty_weight=0.0`) as an available fallback for
assets where task 2 doesn't visibly improve on it (per the plan's own instruction to compare per
asset, not assume one always wins) — worth re-checking on a wider asset spread in Phase 7, since
`can` and `hand` sit at two very different points on the "singularity density" spectrum that
directly determines how well this method performs.

### Phase 5 — Texture transfer from the reference mesh

**Goal**: bake color onto the new UVs, sourced from `reference_mesh`'s existing baked texture (mesh
has none of its own at this point) via closest-point projection — **not** a re-run of
`postprocess_mesh`'s live voxel-grid sampling, which needs data this standalone tool doesn't have
(see Hard Constraints).

**Files**: `trellis2/utils/quad_postprocess.py` (extend)

**Tasks**:
1. `rasterize_uv(uvs, faces, texture_size, texture_size)` on the Phase 4 UVs (already-existing
   primitive, reused as-is).
2. `interpolate_uv(vertices, rast, faces)` to get each covered texel's 3D position (also
   already-existing, reused as-is) — same two-call pattern `trellis2_texturing.py:316-318` already
   uses, just swapping what happens between rasterizing and interpolating vs. what comes after.
3. Build `cumesh.bvh.cuBVH(reference_mesh.vertices, reference_mesh.faces)` once; query
   `unsigned_distance(texel_positions, return_uvw=True)` to get, per covered texel, the nearest
   `reference_mesh` face id and barycentric weights on it.
4. Look up `reference_mesh`'s own UV coordinates at those barycentric weights on that face (a plain
   per-triangle barycentric interpolation of `reference_mesh.visual.uv`, not a new BVH query), then
   bilinear-sample `reference_mesh`'s baked texture image at that UV — this is the actual color value
   for the new mesh's texel.
5. Write sampled colors into a new `texture_size × texture_size` image at the corresponding
   texel positions; for texels with no coverage (background, or where `reference_mesh` has no valid
   UV/texture at the queried point), extend via `cv2.inpaint` — same mask-extension pattern
   `trellis2_texturing.py:337-341` already uses for its own PBR channels.
6. Assemble the final `trimesh.Trimesh` + material and apply the **same coordinate-convention fixup**
   `postprocess_mesh` already applies for GLB compatibility (`trellis2_texturing.py:353-356`: swap
   Y/Z and negate Y on vertices and normals, flip the UV V-coordinate) — do not invent a different
   convention, match the existing pipeline's output exactly so downstream viewers/tools see
   consistent orientation across both code paths.
7. If `reference_mesh` has no material/UV at all (untextured input), skip texture entirely and
   export bare geometry — don't fabricate texture data.

**Acceptance**: side-by-side render of `reference_mesh` vs. the final quad-remeshed+textured output —
colors/patterns recognizably in the right place (a human visual check, record it as such, same
standard applied throughout this fork's prior work — see `history.md`). Confirm the coordinate
fixup produces a GLB that opens right-side-up and correctly oriented in a standard viewer, not just
"doesn't crash."

### Phase 5 results (completed) — task 6's fixup instruction was wrong, corrected

Built `transfer_texture()` in `quad_postprocess.py` per tasks 1-5/7 as specified, and verified
end-to-end on can and hand (both textured `tests/*.glb` assets): colors/patterns land in the right
place in both cases (can: green body/white text/red logo correctly placed; hand: bone-tan color
correctly covers the same thin finger geometry as the reference).

**Task 6 as literally written is wrong and was corrected.** Applying postprocess_mesh's full fixup
(swap Y/Z, negate Y on vertices/normals, flip UV V) visibly rotated a correctly-oriented can onto
its side. Root cause: that vertex/normal swap exists in `postprocess_mesh` to convert TRELLIS2's
*internal, pre-export* mesh convention into GLB's — but this doc's own contract (see "Assumed input
state" above) defines `reference_mesh` as always an **already-exported GLB** (`tests/*.glb`, or "the
GLB `run_sample.py` already produced" — both already in final GLB convention, never the internal
pre-export representation). Since Phase 2's `snap_to_reference` places the new mesh's vertices
directly into `reference_mesh`'s own coordinate frame, there is nothing left to fix — applying the
swap again double-transforms an already-correct mesh. Confirmed empirically:
`tests/can.glb`'s own extents already have Y as the tall axis (a standing can); re-applying the
swap flips it onto its side.

**Corrected behavior**: only flip the output UV V-coordinate (a real, separate fix for xatlas's own
raw UV-layout convention, unrelated to vertex coordinates) — do not touch vertices/normals at all.
This only matters for Phase 5's specific standalone-CLI contract (`reference_mesh` = an
already-exported GLB); if this pipeline is ever wired to run *inside* `run_sample.py` directly
against the pipeline's raw internal mesh instead (see Phase 8), that internal mesh would need the
full original fixup, same as `postprocess_mesh` — the two scenarios are different inputs, not a
contradiction.

### Phase 6 — Standalone CLI + wiring into the existing tool

**Goal**: make Phases 1-5 runnable end to end without manual Python.

**Files**: extend `tools/mesh_refine_meshanything.py` (add flags) or add a new
`tools/quad_postprocess.py` if the existing tool's argument surface is already crowded enough that
bolting on more flags hurts readability — judgment call, lean toward a new tool if in doubt, since
this doc's pipeline stage is conceptually separable from everything upstream of QuadriFlow.

**Tasks**:
1. Wire: `quad_obj_path` (from an existing `--quad-remesh` run, or produced fresh from
   `reference_mesh` per Option 2) → Phase 2 snap → Phase 3 optional subdivision (flag-gated) →
   Phase 4 UV unwrap → Phase 5 texture transfer → final GLB export.
2. Expose the Option 1 vs. Option 2 choice as a CLI flag once Phase 1's ablation has a recommendation
   (e.g. `--source {meshanything,direct}`), rather than hard-coding one path.
3. Fix the plumbing gap noted in "Assumed input state" above (`refine_mesh_direct_sample` exposing
   `original_repaired`) as part of this wiring, not as a separate phase — it only matters once
   something actually consumes it, which is here.

**Acceptance**: one CLI invocation, given an input GLB, produces a final textured, quad-derived GLB,
exiting 0, on at least the assets validated in Phase 1.

### Phase 6 results (completed) — a major, unplanned discovery: real test assets are severely perforated

Built `tools/quad_postprocess.py` per tasks 1-3 as specified. The CLI ran and exited 0 immediately
-- but the *output was wrong* in a way exit codes don't catch: rendering `can.glb`'s output looked
almost entirely black. Chasing that down uncovered something well beyond this phase's original
scope (full account in `progress.md`, condensed here):

- **The mesh's true surface area was only ~13% of the reference's**, confirmed present in the raw
  QuadriFlow output itself, not introduced by anything downstream. Root cause: `tests/*.glb` (all
  4, not just can) are severely fragmented (2,700-15,000+ connected components after repair,
  largest only 1.2%-23.2% of faces) *and* each component is riddled with pinhole-scale gaps
  (hundreds to thousands of boundary loops per component, median size 2 vertices -- too few edges
  to even form a valid triangle). Confirmed present in the **raw, unrepaired** mesh straight out
  of `trimesh.load()`, before any of this project's own code touches it -- not something
  `repair_mesh()` introduced.
- Investigated whether `run_sample.py`'s own `o_voxel.postprocess.to_glb` pipeline (which runs
  CuMesh's native `fill_holes`/`repair_non_manifold_edges`/`remove_small_connected_components`
  *twice*) should have prevented this: replaying that exact sequence on `can.glb` reduced boundary
  loops by ~87% (32,247 → 4,177) but did not reach watertight -- strong evidence this perforation
  is an inherent artifact of TRELLIS2's own dual-contouring mesh extraction at finite voxel
  resolution, not a bug anywhere in this codebase.
- Tried two hole-*patching* tools (`trimesh.repair.fill_holes`, MeshLab's `meshing_close_holes`
  via `pymeshlab`) -- neither fixed it, because patching assumes a well-formed boundary polygon to
  begin with, and many of these holes are topologically degenerate (2-vertex loops aren't valid
  polygons at all).
- **Fix: CGAL's Alpha Wrap** (`third_party/AlphaWrap/`, vendored the same way as QuadriFlow --
  see its own `BUILD_NOTES.md`, including two real build/runtime bugs found and fixed there: a
  GCC-15/CGAL-5.6.1 compile failure, and a segfault-on-every-input caused by using a fixed-size
  `std::array` where CGAL's `read_polygon_soup` requires a resizable container). Alpha Wrap never
  tries to parse the input's existing topology -- it takes an arbitrary triangle *soup* (no
  manifoldness required at all) and reconstructs a guaranteed watertight, 2-manifold surface from
  scratch. Tested on all 4 assets, full mesh, no per-component splitting needed: 2-5 seconds each,
  88.9%-97.6% of the reference's true surface area recovered, and correctly preserves fine
  separated structure (hand's 5 fingers stay distinct, not fused).
- **Wired in as the new on-by-default preprocessing step** before QuadriFlow in
  `tools/quad_postprocess.py` (`--no-alpha-wrap` to disable). Full CLI re-run on all 4 assets end
  to end: 11-22 seconds each, 100% quad fraction, 1-26 UV charts at 97.4%-100% `top5_area_frac`,
  and -- the bar that actually matters -- **visually confirmed correct and complete** on every
  asset: `car` now shows wheels/headlights/windshield (previously a wheelless shattered fragment),
  `watch` shows the complete case+band, `hand` keeps all 5 fingers separated, `can` is a clean,
  fully-textured, artifact-free can.

The earlier `remesh_to_quad_dominant_obj_multi_component()` (per-component QuadriFlow face-budget
allocation, built while chasing this before Alpha Wrap was tried) is superseded by Alpha Wrap for
the CLI's default path -- effective but far slower (90+s vs. 2-5s) and doesn't fix area coverage
as completely. Left in `quad_remesh.py` as working, tested code for meshes that are fragmented but
not helped by watertight reconstruction alone.

### Phase 7 — Validation pass across more assets

**Goal**: the same kind of broad validation `history.md` records for the MeshAnything
stages, now for this doc's stages — Phase 1 only re-validates QuadriFlow's own quad conversion, not
the full Phase 2-6 chain built on top of it.

**Tasks**:
1. Run the full Phase 6 CLI end to end on at least 5-6 of the available `tests/*.glb` assets
   (`bench`, `creature`, `creature2`, `head`, `robot`, `tiger`, `scientist` are all untouched by any
   prior investigation in this doc or in `history.md` — genuinely new coverage, not a
   re-check).
2. Record face count and wall-clock time per phase (Phase 5's BVH-based texture transfer and Phase 3's
   optional subdivision are the likely cost centers — get real numbers, don't assume either is cheap).
3. Re-run the Option 1 vs. Option 2 comparison (Phase 1, task 3) on this broader asset set, since 4
   assets is not enough to generalize a default-option recommendation from.

**Acceptance**: a short written summary (append to this doc or a new progress-log doc, matching
`history.md`'s own format) covering per-asset pass/fail, face counts, timings, and
whichever Option is recommended as default given the broader evidence.

### Phase 7 results (completed)

Ran the full CLI (Option 2/direct, Alpha Wrap on by default) on all 7 listed assets:

| asset | total time | quad faces | UV charts | top5_area_frac |
|---|---|---|---|---|
| bench     | 11.9s | 5,542 | 6  | 1.000 |
| creature  | 13.1s | 5,838 | 1  | 1.000 |
| creature2 | 12.9s | 6,005 | 1  | 1.000 |
| head      | 22.7s | 5,019 | 3  | 1.000 |
| robot     | 21.5s | 4,314 | 11 | 0.997 |
| tiger     | 12.8s | 6,329 | 1  | 1.000 |
| scientist | 20.9s | 2,766 | 27 | 0.980 |

**7/7 succeeded cleanly**, 100% quad fraction every time. Visually confirmed all 7 via textured
renders: every one is complete and artifact-free (park bench, cute creature, silver bunny, human
head correct from all angles including the back, armored robot, armored tiger-warrior, lab-coat
scientist). Two assets (`creature` 39.0%, `robot` 51.8% area-vs-reference) read as outliers on the
area metric alone but render as fully intact -- consistent with the same "reference mesh includes
hidden/internal geometry not part of the external silhouette" explanation already established on
the original 4 assets, not re-investigated further given the visual confirmation already settles
it. Combined with Phase 6's original 4, that's **11/11 test assets** producing clean, correct,
textured output via Option 2 + Alpha Wrap.

**Option 1 vs 2 re-check, spot-checked on creature/robot/head (`--source meshanything`)** rather
than all 7 (MeshAnything inference is ~330s/asset vs. Option 2's ~15-20s, and Phase 1 already
found Option 1 fragile on 2/4 assets -- a proportional re-check, not a full re-run):
**Option 1 loses badly on all 3.** `creature`: technically completes but chart layout badly
fragmented (237 charts, top5_area_frac=0.480) and the render is visibly distorted with
misapplied, splotchy texture. `robot`: chart layout collapses further (0.086, fell back to task 1
and still failed) and the render is **completely shattered into disconnected floating chunks** --
the same failure mode `history.md` already recorded for `car.glb` under Option 1, now confirmed on
a second, independent asset. `head`: area comes back **NaN** (degenerate geometry) and the render
is **entirely blank** -- total failure. All 3 of these same assets produced clean, complete,
correctly-textured output under Option 2. Combined tally across both rounds: **Option 2 succeeded
cleanly on 11/11 tested assets; Option 1 failed or badly degraded on 5/7** (car, watch originally;
creature, robot, head now), "working" only on can and hand.

**Recommendation, now backed by 11 assets instead of 4**: Option 2 (direct + Alpha Wrap) as the
default, unconditionally -- not asset-dependent. Option 1 (MeshAnything) remains available for its
narrower original use case (artist-style edge flow on assets simple enough not to fragment) but
should not be trusted as a default or fallback for anything without explicit user opt-in.

### Phase 8 — Forward-compat note: the cleaner alternative if wired into the live pipeline

**Goal**: document, without implementing, the better texture path available if this whole chain ever
runs *inside* `run_sample.py` rather than as a standalone CLI on an already-exported GLB.

**Tasks**:
- Note the call site: right after `mesh = pipeline.run(image)[0]` in `run_sample.py` (before
  `o_voxel.postprocess.to_glb(...)` bakes and discards the live PBR voxel data), the mesh's
  `attr_volume`/`coords`/`attr_layout`/`voxel_size` are still available. If QuadriFlow-based
  remeshing ran at this point instead of afterward, Phase 5's mesh-to-mesh closest-point texture
  transfer becomes unnecessary — the new quad mesh's fresh UVs could instead be textured by directly
  re-running the same `grid_sample_3d`-based voxel sampling `postprocess_mesh` already does, just
  against the new geometry/UVs. This is strictly better (no closest-point-projection approximation
  error) and should replace Phase 5 entirely if/when this gets wired in for real — flag it here so
  whoever does that wiring doesn't miss the opportunity to simplify rather than porting Phase 5
  as-is.
- `o_voxel/postprocess.py`'s existing `remesh` flag (`remesh_narrow_band_dc` +
  `project_back` parameter) is an existing, working precedent in this exact codebase for
  "retopologize, then snap vertices back to the original surface" — the same pattern Phase 2 (this
  doc) implements independently for QuadriFlow's output. Worth a look before wiring anything into
  the live pipeline, in case the two snap-back implementations can share code rather than duplicate
  it.

### Phase 8 notes (completed, documentation only)

Confirmed both call sites directly by reading the actual code (not assumed):

- **Call site**: `run_sample.py:187`, `mesh = pipeline.run(image)[0]`, followed immediately by
  `o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces,
  attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
  voxel_size=mesh.voxel_size, ...)` (`run_sample.py:211-223`). Everything Phase 5's live-voxel
  alternative would need (`attr_volume`/`coords`/`attr_layout`/`voxel_size`) is sitting right there
  on the `mesh` object at that point, before `to_glb` bakes it into a flat 2D atlas and the
  live tensor goes out of scope. If QuadriFlow-based remeshing (this doc's Phases 1-4) ran here
  instead of on an already-exported GLB, Phase 5's mesh-to-mesh closest-point texture transfer
  becomes unnecessary — bake the new quad mesh's fresh UVs the same way `to_glb` already does
  internally (`rasterize_uv`/`interpolate_uv` then `flex_gemm.ops.grid_sample.grid_sample_3d`
  against `attr_volume`, `o-voxel/o_voxel/postprocess.py:315-327`), just against the new
  geometry/UVs instead of the original dual-contoured mesh's own. Strictly better than Phase 5's
  approach: no closest-point-projection approximation error, and one fewer full mesh (this doc's
  `reference_mesh`) needing to exist as a separate already-exported, already-textured GLB at all.
  Should replace Phase 5 entirely if/when this pipeline is ever wired directly into `run_sample.py`
  rather than run as a standalone CLI on its output.
- **`remesh_narrow_band_dc` precedent**: confirmed via its actual signature
  (`cumesh.remeshing.remesh_narrow_band_dc(vertices, faces, center, scale, resolution, band=1,
  project_back=0, ...)`) — narrow-band UDF + dual-contouring retopology with a `project_back`
  *ratio* parameter that blends new vertices toward the original surface. This is a different
  mechanism than Phase 2's `snap_to_reference` (a hard 100%-snap closest-point BVH projection, not
  a blend ratio) but the same underlying idea ("retopologize, then correct drift against the
  original surface"). Worth a look for anyone wiring Phase 2 into the live pipeline in case the
  two can share the BVH-query plumbing, but they're different enough in their actual snapping
  behavior (ratio-blend vs. hard snap) that this isn't a drop-in replacement either direction —
  flagged here as a starting point for that investigation, not a finished answer.

Not implemented (by design — this phase is documentation only, per its own stated goal).

---

## Sequencing

The quad-topology visualization sidetask is not part of this dependency chain — it only needs
Phase 1's `load_quad_obj` and can be built any time after that, in parallel with everything else,
since its whole purpose is supporting the visual-inspection steps already called for throughout
Phases 1-3.

Phase 1 blocks everything (need validated QuadriFlow output and the Option 1/2 decision before
building on top of either). Phase 2 depends on Phase 1's `reference_mesh` contract; Phase 3 is
independent of Phase 2 but should run *before* Phase 2's snap if both are used (subdivide, then snap
the new vertices — see Phase 3 task 3), or after Phase 1 directly if Phase 2 is skipped. Phases 4-5
depend on whatever Phase 2/3 produced, in that order (UVs before texture). Phase 6 ties everything
into a CLI once 2-5 exist. **Phase 6 is the real milestone** — stop and evaluate on a handful of
assets before investing in Phase 7's broader validation. Phase 8 is documentation only, no code, and
can happen any time after Phase 5's design is settled.
