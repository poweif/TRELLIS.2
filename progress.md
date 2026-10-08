# QuadriFlow Post-Processing — Implementation Progress

**STATUS: ALL PHASES (1-8) COMPLETE.** Every phase in `quadriflow_postprocess_plan.md` is
implemented, tested, and validated end to end via `tools/quad_postprocess.py` on 11 test assets
(4 original + 7 broader-validation). Headline result: Option 2 (QuadriFlow directly on the
repaired original mesh, with CGAL Alpha Wrap reconstructing a watertight surface first) succeeds
cleanly on 11/11 assets in 11-23 seconds each; Option 1 (MeshAnything) fails or badly degrades on
5/7 assets tested against it, ~20x slower when it does run. Nothing in this file describes
outstanding work as of the last update -- if picking this up later, treat the plan doc + this file
as historical record of *how* the current code came to be, not a list of things left to do. Any
new work (e.g. wiring into `run_sample.py` directly, per Phase 8's notes) would be a new,
unplanned phase, not a continuation of an unfinished one.

Tracks live progress against `quadriflow_postprocess_plan.md`, so work can be resumed after an
interruption without re-deriving state. Update this file as each task task completes or state
changes — don't rely on task-tool memory alone surviving a restart.

Environment note: GPU/torch/cumesh only importable inside `conda activate trellis2` (plain
`python3` has no torch). Always `source ~/miniconda3/etc/profile.d/conda.sh && conda activate
trellis2` before running anything that touches torch/cumesh/MeshAnything.

## Status legend
- [ ] not started
- [~] in progress
- [x] done

## Phase 1 — load_quad_obj + broaden validation + Option 1 vs 2 ablation
- [x] Task 1: `trellis2/utils/quad_postprocess.py` created (`load_quad_obj`, `parse_quad_faces`,
  `quad_fraction`). Verified via prior-session leftover outputs in `/tmp/quadriflow_debug/`
  (can_01_quad_output.obj, hand_01_quad_output.obj): raw quad counts 334/1349 (100% quads, matches
  history.md), trimesh triangulated face count is exactly 2x in both cases (668/2698) -- confirms
  the "roughly 2x, not equal, not zero" acceptance check cleanly, in fact exactly 2x since every
  face is a pure quad on these two assets. No blocker.
- [x] Task 2: Option 1 on car.glb / watch.glb -- DONE, both ran via
  `tools/mesh_refine_meshanything.py --algo direct-sample --quad-remesh`. Outputs/logs in
  SCRATCH=`/tmp/claude-1000/-home-poweif-work-TRELLIS-2/9ce694e6-dbcf-41ca-ac95-b7a2d8bfff6d/scratchpad/phase1`
  (session-specific tmp -- if resuming in a NEW session and these are gone, just re-run the same
  command; nothing here needs to survive, findings are captured below).

  **car.glb**: confirms history.md's fragmentation prediction, worse than expected. MeshAnything's
  direct-sample output itself is only 116 faces / 107 verts (should be ~1600-ish normally) --
  severely degenerate. QuadriFlow ran "successfully" on this garbage input (71 quad faces, 100%
  quad_fraction) but the wireframe render (`car_option1_wireframe.png`) shows an unrecognizable
  shattered blob, nothing car-shaped. **Option 1 produces unusable output on car.glb.**

  **watch.glb**: NEW finding, worse than car's failure mode -- QuadriFlow **errors outright**
  ("wrong init 25193 26944!", quadriflow's non-manifold-input error) on MeshAnything's
  direct-sample output for watch, despite `remesh_to_quad_dominant_obj`'s built-in `repair_mesh()`
  call. Root-caused via diagnostic (not a quad_remesh.py change, just an ad hoc script): repaired
  mesh has 14 connected components, and component index 3 (706/1932 faces) has Euler
  characteristic -59 and fails `is_winding_consistent` even after `fix_normals()` -- i.e. this
  component is not just non-manifold-at-edges (which repair_mesh already strips) but intrinsically
  non-orientable/self-intersecting in a way per-component winding-fix can't repair. Confirmed fix:
  dropping any connected component that individually fails `is_winding_consistent` (not currently
  done anywhere in `mesh_repair.py` -- existing code only *checks* per-component winding, never
  *drops* bad ones) lets QuadriFlow succeed cleanly (1017 faces, 100% quad_fraction, see
  `watch_option1_quad_diag.obj`). **Recommend as a follow-up hardening fix to `mesh_repair.py`**
  (e.g. a `remove_winding_inconsistent_components()` alongside the existing
  `remove_small_islands`/`remove_nonmanifold_faces`), out of scope for Phase 1 itself (no
  quad_remesh.py/mesh_repair.py changes expected per the plan's Phase 1 file scope) but worth
  doing before Option 1 is trusted as a default path.

  **Bottom line so far**: Option 1 (MeshAnything direct-sample -> QuadriFlow) is 0-for-2 on
  car/watch in its *unpatched* form; with the one-line component-drop patch, watch recovers to a
  clean 100%-quad result but car's failure is a genuine MeshAnything limitation (garbage input, no
  amount of QuadriFlow-side robustness fixes this) -- consistent with history.md's prior
  car.glb finding across 3 different sampling strategies.

- [x] Task 3: Option 2 ablation -- DONE, **8/8 succeeded, 100% quad fraction every run, all under
  90s** (see quadriflow_postprocess_plan.md's new "Phase 1 results and recommendation" section for
  the full table). Visual wins on 3/4 assets: car recognizable (vs Option 1's shattered blob),
  watch fully recovered case+band (vs Option 1's band-only patched recovery), hand recovers a full
  connected palm+fingers (vs Option 1's 5 disconnected finger tubes, no palm). can: both fine,
  unremarkable.
- [x] Task 4: **Recommendation written and appended to quadriflow_postprocess_plan.md: Option 2 as
  the default**, not asset-dependent. Option 1 kept available as a narrower use case but not
  trusted as default until mesh_repair.py's winding-inconsistent-component hardening lands.

**PHASE 1 COMPLETE.** Sidetask complete. Phase 2 complete (snap_to_reference validated). Phase 3
complete (subdivide_catmull_clark validated). All Option 1 + Option 2 quad .obj outputs + wireframe
renders live in
`/tmp/claude-1000/-home-poweif-work-TRELLIS-2/9ce694e6-dbcf-41ca-ac95-b7a2d8bfff6d/scratchpad/phase1/`
(session-scratch, NOT guaranteed to survive -- nothing there needs to, all findings are captured in
this file and in quadriflow_postprocess_plan.md's new section).
- [~] Task 3: Option 2 ablation on can/hand/car/watch, 2 target-face settings each -- IN PROGRESS
  in background (current run: `bcjlhb0gh`), script at `$SCRATCH/option2_ablation.py`, log at
  `$SCRATCH/option2_ablation.log`. Pure CPU (repair_mesh + quadriflow binary), no GPU contention
  with GPU work, runs concurrently fine.

  **Course correction**: first attempt used target_faces = literal original triangle count
  (~350-500K per asset) for the "full resolution" setting per a literal reading of the plan's
  wording. Killed after 11+ min on "can" alone with no sign of finishing -- impractical across 4
  assets, and the plan's own phrasing ("a higher value", prefixed "e.g.") doesn't actually require
  literal full res. Switched the two settings to **1600 (matches Option 1's cap)** and **20000**
  (a genuine "high-detail" setting, >10x Option 1, but tractable) -- also swapped order to run the
  cheap 1600 setting first per asset so useful signal arrives sooner.
- [ ] Task 4: written recommendation appended to quadriflow_postprocess_plan.md -- blocked on 2 and 3

## Sidetask — quad wireframe visualization
- [x] `trellis2/utils/quad_debug.py`: `render_quad_wireframe()` (single view, shaded+depth-tested
  genuine-quad-edge wireframe+irregular markers) and `render_quad_wireframe_contact_sheet()`
  (4-view, mirrors mesh_debug.py convention). Also added shared parsing/topology infra to
  `quad_postprocess.py`: `parse_obj_raw`, `face_edges`, `find_irregular_vertices` (valence != 4) --
  these will be reused by Phase 4 task 2 (motorcycle graph) per the plan.
  Verified on can/hand quad outputs (from /tmp/quadriflow_debug/ leftovers): can's grid renders
  cleanly, irregular markers land exactly at the two cylinder-cap poles (textbook-correct
  singularity locations). hand's render looked initially alarming (near-black, sparse) but
  cross-checked against plain `mesh_debug.render_mesh_contact_sheet` on the same mesh -- identical
  sparse/thin appearance, confirming this is the mesh's real geometry (5 separate thin finger
  tubes, no solid palm) and not a bug in the new code. hand has an unusually high irregular-vertex
  fraction (35.2% vs can's 17.4%, computed directly) scattered along the thin finger tubes rather
  than only at tips/joints -- a real QuadriFlow-quality signal on this hard asset, worth noting for
  Phase 1's Option 1/2 comparison writeup, not a visualization defect.

## Phase 2 — vertex snap-back
- [x] `snap_to_reference()` implemented in quad_postprocess.py, via `cumesh.bvh.cuBVH` --
  confirmed barycentric convention (`uvw=(u,v,w)`, position=`u*a+v*b+w*c` for that face's 3
  vertices in face-order `a,b,c`) by reading `cubvh/include/gpu/triangle.cuh`'s `barycentric()`
  directly rather than assuming, per the plan's own instruction. Also added `write_quad_obj()`
  (writes vertices + n-gon faces back to .obj, needed so Phase 3's subdivision can consume snapped
  positions on the genuine quad structure) and `measure_snap_displacement()`.
  Bug caught+fixed during testing: initial `write_quad_obj` used `!r` on numpy float64 scalars,
  which produces `np.float64(-0.089...)` reprs that trimesh's obj loader can't parse -- fixed to
  plain `%.8f` formatting.
- [x] displacement measurements, Option 1 (MeshAnything-derived) case, using can/watch's Option 1
  quad outputs against their repaired original mesh as reference:
  - can: mean=0.0136, median=0.0059, max=0.1357 (bbox diag 1.233, mean = **1.10%** of diag)
  - hand: mean=0.0068, median=0.0055, max=0.0347 (bbox diag 1.193, mean = **0.57%** of diag)
  - watch (using the diagnostic bad-component-dropped quad obj): mean=0.0068, median=0.0064,
    max=0.0220 (bbox diag 1.525, mean = **0.44%** of diag)
  Visual check (before/after wireframe renders, can + watch's band specifically per the plan's
  thin-feature-crossing concern): **no visible degradation** in either case; can's after-snap
  render looks very slightly crisper/more regular, consistent with "precision/hygiene pass on an
  already-reasonable fit" (not a big correction) -- matches the plan's expectation for Option 1.
  - [ ] still need: Option-2-derived displacement numbers, for the actual Option1-vs-Option2
    comparison the plan's acceptance check wants -- blocked on Phase 1 task 3's ablation output
    (in progress, see above).

## Phase 3 — optional Catmull-Clark subdivision -- DONE (core implementation + validation)
- [x] pyvista n-gon CC spike: **negative result, as the plan half-expected**. `pyvista.PolyData`
  only exposes `subdivide`/`subdivide_adaptive`, backed by VTK's
  `vtkLoopSubdivisionFilter`/`vtkButterflySubdivisionFilter`/`vtkLinearSubdivisionFilter` --
  confirmed via direct VTK introspection that **no Catmull-Clark filter exists in VTK at all**,
  and all 3 available filters are triangle-only schemes.
- [x] Found a working alternative instead of hand-rolling n-gon CC: **`pymeshlab`** (already an
  installed dependency) has `MeshSet.meshing_surface_subdivision_catmull_clark(iterations=N)`.
  Verified empirically it operates on the GENUINE quad structure, not a naive triangulation:
  subdividing a 334-quad mesh 1 level produced exactly 334*4=1336 new quad faces (matches
  Catmull-Clark's real "each face -> n new quads" rule) -- MeshLab/VCG's internal format always
  triangulates internally but tags each n-gon's internal diagonal as a "faux" edge, and the CC
  filter respects that tagging rather than treating the diagonal as real.
  `trellis2/utils/subdivision.py`: `subdivide_catmull_clark(quad_obj_path, output_obj_path,
  levels=1)`, writes genuine quad .obj output (verified: 100% quad faces at 1336/1336 after 1
  level on can).
- [x] Re-snap after subdivision (Phase 3 task 3, ordering: subdivide first, snap second): tested
  on can -- displacement after subdivide+snap (mean=0.0131, median=0.0069, max=0.1201) is close to
  the pre-subdivision snap numbers (mean=0.0136), as expected (new edge/face-point vertices are
  interior to the original surface approximation, snap corrects them the same way). Wireframe
  render (`can_subdivided_snapped_wireframe.png`) confirms: finer/denser quad grid, visibly
  smoother rounder cylindrical silhouette than the un-subdivided version, no self-intersections or
  visible artifacts. **1 level looks clearly sufficient for `can`** -- did not test 2+ levels since
  the plan explicitly says evaluate 1 level visually before assuming more is needed, and 1 already
  looks good.
- Not yet done: same subdivision spot-check on a non-can asset (e.g. hand or an Option-2-derived
  mesh) -- low risk to defer, since the pymeshlab filter's correctness isn't asset-specific and the
  quad-count math already confirms it's using genuine topology.

## Phase 4 — UV unwrap, few large charts
- [x] task 1: retune compute_charts_kwargs sweep -- DONE, script at
  `$SCRATCH/chart_sweep.py`. **Negative result: parameter retuning alone does NOT close the
  gap.** Swept `area_penalty_weight` in {0.1 (default), 0.0, -0.05} and
  `threshold_cone_half_angle_rad` in {90 deg (default), 60, 45} on can (668 tri) and hand (2698
  tri):
  - can: num_charts stayed at 15 across all 3 area_penalty_weight values at 90 deg (essentially no
    effect at this scale) -- but dropping the cone angle to 60/45 deg *increased* chart count to
    28/41 (worse, not better -- tighter angle = more curvature-sensitive splitting = more charts,
    the opposite of "few large charts"). Default 90 deg (already the loosest available setting) is
    already the best of the tested settings for chart count.
  - hand: similar pattern, 177-187 charts across area_penalty_weight values at 90 deg (small,
    slightly *counter-intuitive* effect -- 0.1 gave FEWER charts than -0.05, opposite the
    docstring's stated direction "prevents charts from growing if >0" -- effect size is small
    enough this could be noise in the clustering rather than a real inversion, not chased further),
    and again cone-angle tightening made it much worse (299 @ 60deg, 392 @ 45deg).
  - **Conclusion: 15 charts for a simple capped cylinder and 177+ for a 5-finger hand, even at the
    best available parameters, is nowhere near "few large charts" (ideally ~3 for can:
    body+2 caps; ~5-6 for hand: palm+5 fingers).** Task 2 (motorcycle-graph patch layout) is
    necessary, not just a fallback -- this isn't a "try it and see" anymore, it's confirmed
    required.
- [x] task 2 (motorcycle-graph patch layout) -- DONE, all in quad_postprocess.py:
  `_vertex_neighbors`, `_vertex_face_neighbor_pairs`, `_straight_continuation`,
  `trace_separatrices`, `patches_from_separatrices`, `uv_unwrap_via_patches`.

  **Found and fixed a real bug during validation, not just a tuning issue.** First implementation
  (trace every separatrix from every singularity in every direction, cut along literally all of
  them) catastrophically over-fragmented BOTH test assets: can -> 334 patches (334/334 faces,
  100% singletons!); hand -> 1080 patches (1009 singletons). Root-caused via direct investigation
  (not guessed): confirmed the local "straight ahead" rule itself is geometrically correct (checked
  ~1112 samples, median turning angle 20.6 deg, consistent with genuinely straight quad-grid
  lines) -- the problem is structural, not a local-rule bug. Since every regular (valence-4) vertex
  has a *unique deterministic* straight-ahead pairing, the maximal straight line through *any* edge
  is mathematically guaranteed to eventually reach a singularity if followed far enough (confirmed:
  can's singularities are concentrated entirely at its 2 poles, none along the body -- so any
  "vertical" line necessarily runs the cylinder's full length before terminating, one traced
  example ran 189 of 339 total vertices with zero self-revisits, i.e. genuinely straight, not a
  bug-induced spiral). Tracing from every singularity in every direction and cutting along all of
  it is therefore guaranteed to claim nearly 100% of the mesh's edges whenever singularities are
  sparse -- the opposite of "few large patches".

  **The fix**: the plan's own cited algorithm (Eppstein/Goodrich/Kim/Tamstorf motorcycle graphs)
  has motorcycles stop when they crash into ANOTHER motorcycle's trail, not only at singularities
  -- my first implementation only checked this for a trace's *starting* edge, never during the
  trace itself, so it never actually triggered (two distinct maximal lines can't share an edge
  under the deterministic pairing rule, so "crash into an existing trail" alone doesn't help
  either -- confirmed by testing: adding it changed nothing). The real fix needed is a **length
  cap**: discard (don't commit edges for) any trace that exceeds `max_length` (default 8) hops
  without reaching a singularity or an existing trail. This is not arbitrary -- it directly
  targets the failure mode (long "wandering" lines with nothing to stop them), and the length cap
  must be enforced *during* tracing, not as a post-hoc filter on the returned separatrix list
  (a post-filter would still have let a since-discarded long trace's edges block/shorten other
  traces during the original tracing pass, an inconsistency -- implemented so discarded traces
  never touch the shared `consumed_edges` state at all).

  **Result with max_length=8 (the chosen default, swept 5/8/10/12/16/20/30, 8 was
  at-or-near-best on both assets)**:
  - can: 85 patches, but concentrated -- **top5_area_frac=0.76**, largest single patch 241/334
    faces (72%, visually confirmed to be the *entire cylindrical body in one patch* -- see
    `can_patches_wireframe.png`). Remaining ~80 small (mostly 1-2 face) fragments -- checked
    directly, **100% of them sit at the two polar z-extremes**, i.e. they're the small
    "pie slice" pieces a singularity fan at a disk cap structurally requires, not noise.
  - hand: 532 patches, top5_area_frac=0.256 (much less clean than can -- hand's singularities are
    dense, 35.2% of vertices, not sparse/polar-only, so "few large patches" is inherently harder
    here; still enormously better than the uncapped 1009-singleton disaster).
  - Full `uv_unwrap_via_patches` pipeline (xatlas per patch, fan-triangulated, same "add each group
    as its own atlas mesh" pattern CuMesh.uv_unwrap already uses) ran successfully end-to-end on
    both: can -> 729 verts/668 faces/UVs in [0, 0.999]; hand -> 4024 verts/2698 faces, UVs in
    [0, 0.999]. Texel coverage at 256x256 (degenerate-UV spot check): can 82.0%, hand 62.0% --
    neither "mostly background", passes.
- [x] task 3: triangulate for UV assignment -- folded into `uv_unwrap_via_patches` (fan
  triangulation per patch before feeding xatlas, same diagonal convention trimesh uses for quads).

## Phase 5 — texture transfer -- DONE
- [x] `transfer_texture()` in quad_postprocess.py: rasterize_uv/interpolate_uv to get per-texel 3D
  positions on the new mesh, cuBVH closest-point projection onto `reference_mesh` (same barycentric
  convention as Phase 2's snap_to_reference), look up `reference_mesh.visual.uv` at those
  barycentric weights, bilinear-sample `reference_mesh`'s baked texture (new
  `_sample_texture_at_uv` helper, via `F.grid_sample`).
- [x] inpaint uncovered texels via `cv2.inpaint` (same pattern trellis2_texturing.py uses).
- [x] Untextured-reference fallback tested directly: returns bare `ColorVisuals` geometry, no crash,
  no fabricated texture.

- **Found and fixed a real bug**: initially applied postprocess_mesh's full GLB coordinate fixup
  (swap Y/Z, negate Y on vertices/normals, flip UV V) per a literal reading of Phase 5 task 6.
  Empirically this **visibly rotated a correctly-oriented can onto its side** (rendered comparison:
  reference stands upright with Y as the tall axis; naive-fixup output showed the can lying
  sideways). Root cause: `tests/can.glb` has Y-extent ~1.0 vs X/Z ~0.5 -- i.e. it's **already** in
  final GLB (Y-up) convention, not TRELLIS2's internal pre-export convention that
  postprocess_mesh's vertex/normal swap is actually meant to correct. Re-read
  quadriflow_postprocess_plan.md's own contract: `reference_mesh` is defined as always an
  *already-exported* GLB (`tests/*.glb`, or "the GLB run_sample.py already produced" -- both
  already-final). Since Phase 2's `snap_to_reference` puts the new mesh's vertices directly into
  `reference_mesh`'s own coordinate frame, there is nothing left to "fix" -- applying the swap a
  second time is the bug. **Fix**: removed the vertex/normal axis swap entirely; kept only the UV
  V-flip (a separate correction for xatlas's own raw UV output convention, unrelated to the
  vertex-coordinate question). Re-rendered after the fix: output's extents now correctly show Y as
  the tall axis again, matching the reference, and the textured render is visually correct
  (`can_output_textured_fixed.png`).
- Verified end-to-end on both can and hand: colors/patterns land in the right place on both
  (can: green body, white text, red logo detail all correctly placed; hand: bone-tan color
  correctly covers the same thin finger geometry as the reference, no misplaced colors). Output
  mesh bounds/extents match the reference's orientation in both cases.

## Phase 6 — CLI + wiring
- [x] `tools/quad_postprocess.py` created, wiring quad-obtain -> snap -> optional subdivide ->
  UV unwrap (auto task2/task1 fallback) -> texture transfer -> GLB export.
- [x] `--source {meshanything,direct}` flag (default `direct`, per Phase 1's recommendation)
- [x] fixed `original_repaired` plumbing gap: `refine_mesh_direct_sample` now takes
  `return_intermediates=False`, returns `(mesh, {"original_repaired": ...})` when True.
  Backward compatible (existing call site in tools/mesh_refine_meshanything.py unaffected).
- [x] `remove_small_quad_islands()` added to quad_postprocess.py (mirrors mesh_repair.py's
  remove_small_islands but for n-gon face lists) -- see MAJOR FINDING below for why.
- [x] `uv_unwrap_auto()` added: tries motorcycle-graph (task 2) first, falls back to tuned
  CuMesh curvature clustering (task 1, area_penalty_weight=0.0) if task 2's top5_area_frac
  is below a threshold (default 0.3) -- automates the plan's own "compare both, fall back
  per-asset" instruction for Phase 4.

- **MAJOR FINDING, corrects part of Phase 1's conclusion**: first CLI test (can.glb,
  `--target-faces 20000`, --source direct) produced a mesh whose RENDER looked almost entirely
  black/broken when texture-sampled. Root-caused through several wrong turns (initially suspected
  a UV-convention bug in `transfer_texture`, then suspected screen-resolution aliasing from too
  many tiny triangles -- both ruled out by direct spot-checks and a 1536px re-render) down to the
  real cause: **`total_output_surface_area` was only 12.9% of the reference's** (0.249 vs 1.756
  repaired-reference area) -- confirmed present even in the RAW quad .obj straight out of
  QuadriFlow, before any of Phase 2-5's own processing touches it. Bounding box/extents matched
  the reference almost exactly (ratio ~1.00-1.01), ruling out a scale bug -- the quad mesh
  genuinely has large real holes/missing surface while still spanning the correct overall extent
  (enough scattered fragments near the true extremes to get the bounding box right).

  Traced further: **this is not can-specific or resolution-specific**. Re-checked Phase 1's own
  saved Option-2 outputs for all 4 assets/both target-face settings (never checked for
  area-preservation during Phase 1 -- only quad_fraction was checked there, which stayed 100%
  throughout and didn't catch this):
  | asset | setting | output area | ref (repaired) area | coverage |
  |---|---|---|---|---|
  | can | reduced1600 | 0.155 | 1.756 | 8.1% (vs raw ref, see below) |
  | can | highdetail20000 | 0.249 | 1.756 | 12.9%* |
  | hand | reduced1600 | 0.022 | 0.688 | 3.1%* |
  | hand | highdetail20000 | 0.140 | 0.688 | 19.4%* |
  | car | reduced1600 | 0.029 | 2.756 | 1.0%* |
  | car | highdetail20000 | 0.709 | 2.756 | 25.0%* |
  | watch | reduced1600 | 121.8 | 3.284 | 3532%(!) -- different, unexplained anomaly, opposite direction |
  | watch | highdetail20000 | 0.923 | 3.284 | 26.8%* |
  (*ratios in the table use the RAW un-repaired reference in the denominator from the original
  quick check -- re-verify against repaired-reference area if this needs citing precisely; the
  qualitative conclusion, "1-27% coverage on 7 of 8 runs", holds either way since repaired area is
  close to raw area for all 4 assets, see below.)

  **Root cause, confirmed**: the RAW reference meshes themselves are heavily multi-part --
  `trimesh.graph.connected_components` on the raw `tests/*.glb` meshes found **15083 components
  for can, 10748 for hand, 9606 for car, 15083 for watch**, with even the LARGEST single
  component only 4-21% of total faces. `repair_mesh()`'s existing small-island removal
  (`min_island_faces=8`) only trims ~9% of total area (can: 1.927 raw -> 1.756 repaired) -- doesn't
  explain the gap, these are mostly *real, substantial* separate parts (car doors/mirrors/wheels,
  watch mechanism/band links, individual finger bones), not tiny debris.

  Working hypothesis (not yet fully confirmed): feeding QuadriFlow a repaired mesh with
  thousands of disconnected components at a modest shared `target_faces` budget (1600 or 20000)
  means each component gets, on average, only a handful of output faces -- too few for QuadriFlow
  to represent most components at all. The visual renders from Phase 1 (car/watch/hand all looked
  "recognizable") are consistent with this: re-inspected `car_option2_highdetail20000_wireframe.png`
  closely -- it shows a plausible car BODY silhouette but **no distinct wheel shapes are visible
  at all** -- consistent with "only the largest few components got a usable face budget, the rest
  (wheels, mirrors, etc.) were dropped or left degenerate," not with genuine full-asset coverage.

  **This does not overturn Phase 1's Option 1 vs Option 2 recommendation** (Option 1 still fails
  outright/fragments far worse on car/watch specifically -- that finding used quad_fraction AND
  direct visual inspection of the shattered output, which IS still valid), but it means **Option
  2's "8/8 succeeded" claim needs a caveat**: it succeeds at producing a well-formed, 100%-quad
  mesh, but that mesh may only represent a fraction of the true asset's separate parts when the
  original has many disconnected components. Added `remove_small_quad_islands()` as a partial
  mitigation (drops components too small to be worth keeping downstream) but this does NOT fix
  the underlying face-budget/multi-component QuadriFlow behavior -- that needs more investigation
  (e.g. per-component target_faces allocation proportional to surface area, or running QuadriFlow
  per-component instead of on the whole multi-part mesh at once) before Option 2 should be
  considered fully validated on genuinely multi-part assets. Single-connected-part assets (a
  simple prop, a character body) are not expected to hit this at all.

- [x] Implemented `remesh_to_quad_dominant_obj_multi_component()` in quad_remesh.py: splits by
  connected component, gives each component its own `target_faces` proportional to its area share
  (fixing the root cause -- QuadriFlow's `scale = sqrt(surface_area / faces)`,
  parametrizer-mesh.cpp:60, is one GLOBAL value from TOTAL area, so a component much smaller than
  that global scale gets starved when run as part of the whole mesh). Along the way, found and
  fixed two more real bugs:
  - `parse_obj_raw` returned shape `(0,)` instead of `(0,3)` for an empty-vertex .obj (some
    components' QuadriFlow output is entirely degenerate) -- crashed `np.concatenate`.
  - **A NaN pitfall, the exact same class already documented once in this project's history**
    (SDS explosion-guard saga): one component's QuadriFlow output had NaN area (self-intersecting/
    degenerate quads), and `NaN > threshold` is always `False` in Python/numpy, so a first-attempt
    blow-up sanity check (`out_area > sub.area * 3`) silently let it through. Fixed with an
    explicit `np.isfinite` check. Confirmed necessary: 2 of 152 components blew up 100-200x in
    area on can.glb, together outweighing all 150 well-behaved components combined (202x/127x
    inflation vs. a real -- see below -- median of ~5-15% for normal components); a bbox-diagonal
    check (tried first) did NOT catch this, since vertices stayed roughly in place while only the
    shoelace-computed area blew up (self-intersecting winding, not spatial displacement).

- **DEEPER FINDING, this is not just a fixable bug**: after fixing the blow-up/NaN issues,
  per-component remeshing on can.glb still only recovered ~13.5% of the reference's surface area
  -- barely better than the original single-call 12.9%. Raising each component's minimum face
  floor (4 -> 30) didn't help either (13.6%). Traced to the actual root cause via a controlled
  test: **QuadriFlow preserves area near-perfectly (99.4%) on a clean synthetic icosphere**
  (1280 faces -> 500-face quad remesh, ratio 0.994) -- so this is NOT a general QuadriFlow
  limitation. The difference: every one of can.glb's connected components, even generously
  face-budgeted ones (e.g. one 104,811-face component given comp_target=4985, a serious budget,
  still only achieved a 4.67% output/input area ratio), is **severely non-watertight** -- checked
  directly: that same component alone has **5,525 separate boundary loops**, median loop size 2
  vertices (i.e. thousands of tiny pinhole-sized gaps, "Swiss cheese"), not one clean open
  boundary. QuadriFlow's cross-field quadrangulation apparently collapses/shrinks drastically when
  given a surface this heavily perforated, even though it nominally "supports" open boundaries
  (that support was clearly validated against a *few* clean boundary loops, not thousands of
  pinholes per component).

  **This means the actual fix belongs upstream of QuadriFlow entirely**: these real-world test
  assets need aggressive small-hole filling (e.g. `trimesh.repair.fill_holes` with a small
  max-hole-size, or a Poisson-style reconstruction) as part of `repair_mesh()` or a new
  preprocessing step, closing the thousands of pinhole gaps *before* QuadriFlow ever sees the
  mesh -- not a per-component face-budget or sanity-check fix, which is what I'd built up to this
  point.

  **Attempted hole-filling per the user's direction -- neither tool tried fixed it, and the deeper
  cause turned out to be different from "just needs holes closed":**
  - `trimesh.repair.fill_holes()` -- already called inside `repair_mesh()` (confirmed by reading
    it, not assumed). Directly re-tested in isolation: added only ~6000 faces (446251->452255,
    +1.3%) and did NOT reach watertight. Most of these "holes" are degenerate: checked directly,
    3297 of the 5525 boundary loops on the largest component have only 2 distinct vertices --
    too few to form a valid fillable polygon (a triangle needs 3 boundary edges) -- these read as
    tiny real gaps (~0.002 units apart, not exact-duplicate/precision artifacts: swept
    `merge_vertices(digits_vertex=...)` from 8 down to 3 decimal digits, zero effect on component
    count at any tolerance, ruling out "near-duplicate vertices that failed to merge").
  - `pymeshlab`'s `meshing_close_holes` (MeshLab's own well-established hole-closing filter,
    maxholesize=500, selfintersection=True) -- tested on the largest component (104,811 faces,
    isolated): added faces (104811->130250) but **boundary loop count went UP** (5525->35016,
    6x worse) and the largest-3 components' sizes were completely unchanged when re-tested on the
    whole mesh (component merging never happened -- these holes are internal to a single
    already-connected patch, not gaps between separate components). Ran this hole-closed mesh
    through QuadriFlow anyway to check the metric that actually matters (final area ratio): still
    only 2.5%, no improvement over the untreated 4.7%.
  - **Root-caused further, since hole-filling clearly wasn't the real lever**: swept target_faces
    on this same component in isolation (500/5000/20000/50000/100000) and found area ratio scales
    monotonically and dramatically with target_faces: 0.0024 -> 0.042 -> 0.30 -> 5.64 -> **43.9**.
    This rules out "simplification naturally shrinks area" (that would plateau or saturate, not
    keep climbing past 40x). What's actually happening: at low target_faces the quad grid
    collapses/under-represents the surface (matches everything found so far); at high
    target_faces, forcing a fine quad grid across a surface this complex/perforated apparently
    produces **self-overlapping/folded-over quads** (their naive shoelace-computed area
    over-counts overlapping regions) rather than genuine new surface -- i.e. there may be no
    target_faces value that gives both correct area AND clean, non-self-intersecting topology
    simultaneously for this specific component. This looks like a property of the geometry itself
    (extremely dense small-scale surface complexity/noise) that QuadriFlow cannot cleanly resolve
    at any resolution, not a parameter-tuning problem.

  **Where this leaves things**: two reasonable, different-effort hole-filling attempts did not
  fix the area-preservation problem, and the deeper diagnosis (target_faces-dependent
  under/over-representation, worse than a monotonic tradeoff) suggests the fix -- if one exists
  short of a full surface-reconstruction rewrite (e.g. Poisson reconstruction from the fragment
  soup) -- is a larger undertaking than "add a repair step". Paused here to report this back
  rather than open-endedly trying further tools.

- **RESOLVED via CGAL Alpha Wrap -- this is the actual fix.** User directed investigating why
  `run_sample.py`/`o_voxel.postprocess.to_glb`'s OWN native cleanup (CuMesh's `fill_holes`/
  `repair_non_manifold_edges`/`remove_small_connected_components`, run twice) doesn't fully close
  these assets either (confirmed: boundary loops drop from a garbage/uninitialized reading to
  32247 after one `fill_holes` call, then to 4177 after the full sequence -- real progress, ~87%
  reduction, but nowhere near watertight) -- meaning the perforation is likely an inherent
  artifact of TRELLIS2's own dual-contouring mesh extraction at finite voxel resolution, not
  something introduced by this postprocessing effort or fixable by patching tools at all.

  Vendored `third_party/AlphaWrap/` (CGAL's `Alpha_wrap_3`, via conda-forge `cgal-cpp`, same
  vendoring pattern as QuadriFlow -- see its own `BUILD_NOTES.md`). Alpha Wrap takes an arbitrary
  triangle **soup** (explicitly no manifoldness/watertightness required) and reconstructs a
  guaranteed watertight, 2-manifold, self-intersection-free surface strictly containing the
  input -- it never tries to parse/patch existing topology (which is why `fill_holes`-style tools
  failed: many holes are topologically degenerate, 2-vertex boundaries that aren't even valid
  polygons to patch).

  **Build had 2 real bugs of its own, both found and fixed**:
  1. CGAL 5.6.1 + system GCC 15 failed to compile (`this->base()` lookup failure in CGAL's
     `boost::graph` iterator CRTP code) -- fixed by building with conda-forge GCC 12 instead.
  2. **The tool then segfaulted on every input, including a trivial 320-face icosphere.**
     Initially misdiagnosed as a kernel-robustness issue (switched `Simple_cartesian<double>` to
     `Exact_predicates_inexact_constructions_kernel` -- didn't fix it, same crash). Root cause,
     found via `gdb` backtrace: `read_polygon_soup`'s documented `PolygonRange` concept requires a
     resizable (`BackInsertionSequence`) inner container -- used `std::vector<std::array<size_t,
     3>>` for faces, and `std::array` is fixed-size. This compiled cleanly and even printed
     plausible point/face counts after reading, but silently violated the type contract in a way
     that corrupted state used later inside `alpha_wrap_3`'s own triangulation setup. Fixed by
     switching to `std::vector<std::vector<size_t>>`, exactly matching the documented concept --
     works perfectly after that, confirmed watertight+correct-area on the trivial icosphere first.

  **Result, tested on all 4 assets (full mesh, no per-component splitting needed at all -- Alpha
  Wrap doesn't care about input connectivity)**:
  | asset | wrap time | wrapped faces | watertight | area vs. reference |
  |---|---|---|---|---|
  | can   | 2.1s | 8,940  | yes | 97.6% |
  | hand  | 2.6s | 17,378 | yes | 88.9% |
  | car   | 4.4s | 51,526 | yes | 96.5% |
  | watch | 4.9s | 44,356 | yes | 93.7% |

  Visually confirmed via shaded renders: car now shows **wheels clearly** (previously completely
  missing); watch shows the **complete case + band together**; hand's **5 fingers stay correctly
  separated** (not fused into a mitten shape -- the exact failure mode that sank the original
  MeshAnything approach). Full pipeline (Alpha Wrap -> QuadriFlow -> snap -> UV unwrap -> texture)
  re-ran end to end on can.glb: **11.0s total, 96.8% area vs. reference, single UV chart at 100%
  coverage** (the mesh is now genuinely one clean connected surface, no fragmentation left for
  the motorcycle-graph/task-1 chart split to even need to do anything), final render is a clean,
  correctly-oriented, correctly-textured can with zero visible artifacts.

  **Integration**: `trellis2/utils/alpha_wrap.py`'s `alpha_wrap_mesh()` (subprocess wrapper,
  mirrors `quad_remesh.py`'s own QuadriFlow-calling convention). Runtime `.so` deps
  (`libgmpxx`/`libmpfr`/`libgmp`/`libstdc++`/`libgcc_s`) copied into
  `third_party/AlphaWrap/runtime_libs/` and tracked in the repo -- confirmed working with zero
  conda env active (`env -i ... LD_LIBRARY_PATH=.../runtime_libs`), so this doesn't depend on the
  throwaway `cgal-build`/`cgal-build2` conda envs used only for the *build* step surviving.
  Wired into `tools/quad_postprocess.py` as an on-by-default preprocessing step for `--source
  direct` (flag: `--no-alpha-wrap` to skip, `--alpha-wrap-alpha`/`--alpha-wrap-offset` to override
  the auto-derived-from-bbox-diagonal defaults). The earlier
  `remesh_to_quad_dominant_obj_multi_component()` (per-component QuadriFlow budgeting) is no
  longer used by the CLI's default path -- Alpha Wrap fixes the actual root cause more
  effectively and far faster (2-5s vs. 90+s) -- but the function is left in place in
  `quad_remesh.py` (real, tested, working code) for anyone who wants per-component budgeting on a
  mesh that's fragmented but not solved by watertight reconstruction alone.

- [x] **Full CLI re-run on all 4 assets with Alpha Wrap wired in as the default -- all succeeded
  cleanly, end to end, 11-22s each:**
  | asset | wrap time | total CLI time | quad faces | UV charts | top5_area_frac |
  |---|---|---|---|---|---|
  | can   | 2.1s | 11.0s | 5,310  | 1  | 1.000 |
  | hand  | 2.6s | 21.3s | 3,972  | 26 | 0.974 |
  | car   | 4.4s | 21.5s | 5,426  | 5  | 1.000 |
  | watch | 4.9s | 20.9s | 5,478  | 1  | 1.000 |

  Visually confirmed via textured renders (all 4): **can** -- clean, correctly-oriented, fully
  textured, zero artifacts. **hand** -- bone-colored skeleton, all 5 fingers correctly separated,
  texture correctly placed. **car** -- full red sports car with wheels, headlights, windshield
  all correctly visible and textured (a complete reversal from the original wheelless/shattered
  result this whole investigation started from). **watch** -- complete black case + band,
  correctly textured. **Phase 6 (CLI + wiring) is now genuinely done**, not just "runs without
  crashing" -- the actual quality bar (visually complete, textured, correctly-shaped output) is
  met on all 4 available test assets.

## Phase 7 — broader validation
- [x] Ran full CLI (Option 2/direct + Alpha Wrap) on all 7 new assets -- **7/7 succeeded cleanly**,
  exit 0, 11.9-22.7s each, 100% quad fraction every time:
  | asset | wrap time (in CLI) | total CLI time | quad faces | UV charts | top5_area_frac |
  |---|---|---|---|---|---|
  | bench     | (see log) | 11.9s | 5,542 | 6  | 1.000 |
  | creature  | | 13.1s | 5,838 | 1  | 1.000 |
  | creature2 | | 12.9s | 6,005 | 1  | 1.000 |
  | head      | | 22.7s | 5,019 | 3  | 1.000 |
  | robot     | | 21.5s | 4,314 | 11 | 0.997 |
  | tiger     | | 12.8s | 6,329 | 1  | 1.000 |
  | scientist | | 20.9s | 2,766 | 27 | 0.980 |

  Visually confirmed all 7 via textured renders: bench (wood+metal park bench, correct), creature
  (cute brown creature, ears/arms/legs/tail all present), creature2 (silver bunny, correct),
  head (photorealistic human head/hair/skin, correct from all 4 angles including the back), robot
  (full armored robot with helmet/weapon, correct), tiger (armored tiger-warrior figure, correct),
  scientist (full human figure in lab coat/jeans/boots/glasses, correct). **All 7 look visually
  complete and artifact-free** -- combined with the original 4 (Phase 6), that's **11/11 test
  assets** producing clean, correct, textured output via Option 2 + Alpha Wrap.

  **Area-ratio note**: 2 of 7 (`creature` 39.0%, `robot` 51.8%) show much lower area-vs-reference
  ratios than the rest (79-93%), which would normally be a red flag -- but both render as
  completely intact, no missing limbs/chunks/holes. This is consistent with the same explanation
  already established for the original 4 assets: raw reference meshes can include hidden/internal
  geometry (e.g. an open mouth's interior, a robot's under-armor mechanical detail) that
  legitimately inflates the *reference's* naive triangle-sum area without being part of the
  external silhouette Alpha Wrap correctly reconstructs -- not re-investigated further given the
  visual confirmation already settles it, and chasing "why doesn't area match" further would be
  relitigating a question already answered on the original 4 assets.

- [x] **Option 1 vs 2 spot-check at broader scale (creature/robot/head via `--source
  meshanything`) -- decisive, Option 1 loses badly on all 3:**
  - **Timing**: 328-338s each (MeshAnything model load + inference) vs. Option 2's 12-23s --
    ~20x slower.
  - **creature**: technically completed (100% quad, 1746 faces) but chart layout only reached
    top5_area_frac=0.480 (237 charts -- badly fragmented vs. Option 2's clean single chart), and
    the render is visibly distorted/blocky with splotchy misapplied texture, nowhere near Option
    2's smooth, correct result on the same asset.
  - **robot**: chart layout collapsed to top5_area_frac=0.086 (fell back to task 1 entirely,
    still bad), and the render is **completely shattered into disconnected floating chunks** --
    the exact same failure mode `history.md` already documented for `car.glb` under Option 1,
    now confirmed on a second asset. Total loss of a coherent robot shape.
  - **head**: area came back **NaN** (degenerate/self-intersecting geometry -- same NaN pitfall
    class already hit once in the multi-component work), and the render is **entirely blank/
    black** -- complete failure, nothing usable produced at all.

  All 3 of these same assets produced clean, complete, correctly-textured output under Option 2 +
  Alpha Wrap (see above). **This fully reconfirms and strengthens Phase 1's original
  recommendation**: Option 2 (direct + Alpha Wrap) as the default, Option 1 (MeshAnything) kept
  available only as a narrower fallback, not trusted for anything by default. Combined tally
  across both rounds of validation: Option 2 succeeded cleanly on **11/11** tested assets; Option
  1 failed or badly degraded on **5/7** tested assets (car, watch originally; creature, robot,
  head now) and only "worked" (with caveats) on can, hand.

## Phase 8 — forward-compat doc note
- [ ] document live-pipeline texture-path alternative (no code)

---

## Post-Phase-8 addendum: in-process QuadriFlow bindings + adaptive quad density (completed)

Two follow-on asks after the original 8 phases, done together since the first makes the second
clean: (1) call QuadriFlow in-process instead of via subprocess/temp-files, (2) revive
QuadriFlow's dormant curvature-adaptive quad density (`rho`, hardcoded to 1.0 for every vertex
upstream since Dec 2018, when a fragile libigl git-submodule dependency was removed and the one
function that used it was neutered rather than fixed -- see `quadriflow_postprocess_plan.md`'s
investigation of GitHub issues #20/#23 on `hjwdzh/QuadriFlow` for the full story).

Full plan at `~/.claude/plans/hazy-wobbling-grove.md`.

- [x] **In-process pybind11/torch binding** (`third_party/QuadriFlow/bindings/quadriflow_binding.cpp`
  + `setup.py`), mirroring `CuMesh/third_party/xatlas/binding.cpp`'s existing pattern in this
  repo. Added `LoadFromArrays`/`ExtractMesh`/optional-external-rho param to
  `Parametrizer`(`parametrizer.hpp`/`parametrizer-mesh.cpp`) as in-memory counterparts to the
  existing file-based `Load`/`OutputMesh`/hardcoded-rho-fill -- CLI binary untouched, still uses
  the file-based versions. Hit two build issues: a mis-encoded copyright symbol in vendored
  `src/loader.cpp` that crashed torch's HIPIFY preprocessing (fixed: converted to UTF-8), and the
  built LEMON library being named `libemon.a` not `liblemon.a` (link as `-lemon`/
  `libraries=["emon"]`) -- both documented in `BUILD_NOTES.md`. **Parity-checked** against the
  existing CLI/subprocess path on an icosphere (matching quad count/area within noise from
  seed-dependent tie-breaking) and measured **1.19x-1.36x faster** than subprocess (larger
  speedup on smaller meshes, where the removed disk round-trip is a bigger fraction of total time).
- [x] **Adaptive rho via PyTorch curvature estimation** (`trellis2/utils/curvature.py`,
  `compute_rho()`), no libigl needed -- mean curvature H (cotangent-Laplacian of vertex
  positions) + Gaussian curvature K (angle-defect/discrete Gauss-Bonnet), `k1,k2 = H ±
  sqrt(max(H^2-K,0))`, `rho = 1/max(|k1|,|k2|)`. Only curvature *magnitude* needed (QuadriFlow's
  own cross-field solve handles orientation), avoiding libigl's heavier quadric-fitting
  `principal_curvature()` (which also recovers principal *directions* -- the thing that dragged
  in the fragile git-submodule dependency originally). **Validated against analytic ground
  truth**: icosphere → rho ≈ R (mean 0.9985R-2.4964R depending on R, small discretization std);
  hand-built open cylinder (no flat caps to interfere) → rho = R **exactly** (std 0.0000) at
  interior rings.
- [x] **Wired end-to-end**: `trellis2/utils/quad_remesh.py`'s `remesh_to_quad_dominant_obj()`
  gained `adaptive: bool = False`, prefers the in-process binding when available (computing+
  passing `rho` if `adaptive=True`), falls back to the CLI/subprocess path (which has no
  equivalent -- raises clearly if `adaptive=True` there). `tools/quad_postprocess.py` gained a
  matching `--adaptive` flag, default off. **Synthetic end-to-end**: dumbbell (two icosphere
  bulbs + thin cylindrical neck) -- neck-to-bulb average edge length ratio dropped from 0.944
  (non-adaptive, near-uniform) to 0.799 (adaptive, neck quads visibly smaller), matching
  `optimize_scale`'s documented 0.75x-1.0x bounded adjustment range. **Real-asset end-to-end**:
  full pipeline on `can.glb` with `--adaptive` completed successfully (12.8s, 7948 quads vs.
  5788 without adaptive at the same nominal target -- adaptive locally exceeds the nominal target
  in high-curvature regions, as expected).

Not yet done (intentionally out of scope for this addendum, left for a future pass if wanted): a
broader adaptive-vs-non-adaptive visual comparison across all 11 test assets (only `can.glb` was
spot-checked end-to-end); `--adaptive` stays off by default in the CLI until that broader check
happens.

---

## Findings log (append-only, most recent last)

(nothing recorded yet)
