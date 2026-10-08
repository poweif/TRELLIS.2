# Quad post-processing pipeline

TRELLIS.2 emits an all-triangle, dual-contoured mesh (hundreds of thousands of faces, unstructured
edge flow, xatlas charts that shatter into many small pieces). This fork adds a standalone
post-processing tool that turns that GLB into a **quad-derived, cleanly charted, textured GLB**,
plus a separate MeshAnything V2 retopology tool. Both take an already-exported GLB, so they work
on any `run_sample.py` output without touching the generation pipeline.

Script-by-script map: [`scripts.md`](scripts.md). How this design was arrived at (including dead
ends): [`archive/`](archive/). Background survey of remeshing / texturing methods:
[`mesh_survey.md`](mesh_survey.md).

## Quick start

```bash
python run_sample.py --image photo.png --output out.glb               # generate
python tools/quad_postprocess.py --input out.glb --output out_quad.glb  # quad-derived GLB
```

Useful flags (`--help` for all):

| Flag | Default | Effect |
|---|---|---|
| `--target-faces N` | input triangle count | QuadriFlow target face count |
| `--no-alpha-wrap` | wrap on | Skip watertight reconstruction (only for already-clean input) |
| `--alpha-wrap-alpha / --alpha-wrap-offset` | bbox diag / 40, / 1200 | Wrap detail / offset |
| `--subdivide-levels N` | 0 | Catmull-Clark levels on the genuine quad structure |
| `--no-snap` | snap on | Skip closest-point snap onto the input surface |
| `--adaptive` | off | Curvature-adaptive quad density (see below) |
| `--source meshanything` | `direct` | Run QuadriFlow on MeshAnything's retopology instead (not recommended) |
| `--quad-obj PATH` | — | Reuse an existing quad `.obj` and skip remeshing |
| `--debug-dir DIR` | — | Dump every stage's mesh + 4-view wireframe contact sheets |

GLB can't store quads, so the output GLB is the quad mesh fan-triangulated. With `--debug-dir`,
the genuine quad `.obj` for each stage is kept.

## Stages

```mermaid
flowchart LR
    A[input GLB] --> R[repair_mesh] --> W[Alpha Wrap] --> Q[QuadriFlow] --> I[drop small islands]
    I --> S[snap to input] --> C["Catmull-Clark (opt.)"] --> S2[re-snap] --> U[UV: separatrix patches] --> T[texture transfer] --> O[output GLB]
```

| Stage | Module | Notes |
|---|---|---|
| Repair | `mesh_repair.repair_mesh` | Drop sub-8-face islands, drop non-manifold faces before *and* after `fill_holes`, fix winding |
| Watertight wrap | `alpha_wrap.alpha_wrap_mesh` | CGAL `Alpha_wrap_3`; in-process `alpha_wrap_ext`, else the CLI binary |
| Quad remesh | `quad_remesh.remesh_to_quad_dominant_obj` | QuadriFlow; in-process `quadriflow_ext`, else the CLI binary. Writes `.obj` (trimesh would triangulate) |
| Snap | `quad_postprocess.snap_to_reference` | Hard closest-point projection via `cumesh.cuBVH` (barycentric `u*a+v*b+w*c`) |
| Subdivide | `subdivision.subdivide_catmull_clark` | pymeshlab CC; respects the quad structure (334 quads → 1336) |
| UV unwrap | `quad_postprocess.uv_unwrap_auto` | Separatrix patch layout, xatlas per patch; falls back to CuMesh clustering if top-5 patch area < 30% |
| Texture | `quad_postprocess.transfer_texture` | Per-texel position via `o_voxel.uv_rasterize`, BVH closest point on the input, bilinear-sample the input's baked texture, `cv2.inpaint` gaps |

## Design decisions and why

**Quads come from QuadriFlow, not MeshAnything.** The original plan expected MeshAnything V2
("artist-created meshes") to be quad-dominant. It never is: the model is triangle-tokenized
(`face_per_token = 9`), and valence-6 dominates even on a clean synthetic cylinder (88%). QuadLink and QuadGPT
list it among triangle-only generators. QuadriFlow (SGP 2018, network-flow cross-field
quantization) gives 100% quads.

**QuadriFlow runs directly on the input (`--source direct`).** Feeding it MeshAnything's
~1,600-face retopology instead was ~20× slower (≈5 min of model load and inference) and failed or badly
degraded on 5 of 7 assets (car and robot shatter into floating chunks, head comes out NaN/blank).
Direct + Alpha Wrap succeeded on 11/11.

**Alpha Wrap before QuadriFlow is the key step.** TRELLIS's dual-contoured output is badly
perforated: the largest component of `can.glb` alone has 5,525 boundary loops, most with only 2
vertices, so there's no polygon for hole-fillers to patch (`trimesh.fill_holes` and MeshLab's
`close_holes` both failed). QuadriFlow sets one global edge length from total area, and on that
input it recovered only 1–27% of the surface area. A watertight wrap fixes the
input instead of the mesher. Coverage went to 89–98%, and thin parts survive
(wheels appear, the five fingers stay separate). The wrap takes 2–5 s.

**Separatrix patches need a length cap.** Tracing every separatrix from every singularity cuts
almost every edge when singularities are sparse (can: 334/334 single-face patches): on a regular
grid every straight line eventually hits a singularity. Traces longer than
`--separatrix-max-length` (8, swept 5–30) are discarded *during* tracing. Result on `can`: the whole
cylindrical body is one patch, and the leftovers are the pie slices at the cap poles.

**Texture transfer doesn't re-orient.** The input is an already-exported (Y-up) GLB, and snapping
puts the new mesh in its frame. Applying `postprocess_mesh`'s Y/Z swap again rotated the can onto
its side. Only the xatlas UV V-flip is applied.

**Adaptive density is off by default.** `--adaptive` feeds QuadriFlow a per-vertex `rho = 1/max(|k1|,|k2|)`
from a cotangent-Laplacian mean curvature and an angle-defect Gaussian curvature
(`curvature.compute_rho`). That revives upstream's input, which has been hardcoded to 1.0 since it
dropped libigl. It was validated analytically (sphere and cylinder: rho ≈ R) and on a dumbbell (neck
quads are 0.80× the bulb size, vs 0.94× without it), but end to end only on `can`.

## Results (Option 2 + Alpha Wrap, CPU QuadriFlow, gfx1151)

| Asset | Total time | Quad faces | UV charts | Top-5 chart area |
|---|---|---|---|---|
| can | 11.0 s | 5,310 | 1 | 1.000 |
| hand | 21.3 s | 3,972 | 26 | 0.974 |
| car | 21.5 s | 5,426 | 5 | 1.000 |
| watch | 20.9 s | 5,478 | 1 | 1.000 |
| bench | 11.9 s | 5,542 | 6 | 1.000 |
| creature | 13.1 s | 5,838 | 1 | 1.000 |
| creature2 | 12.9 s | 6,005 | 1 | 1.000 |
| head | 22.7 s | 5,019 | 3 | 1.000 |
| robot | 21.5 s | 4,314 | 11 | 0.997 |
| tiger | 12.8 s | 6,329 | 1 | 1.000 |
| scientist | 20.9 s | 2,766 | 27 | 0.980 |

All 11 renders were checked visually. `creature` and `robot` report 39% and 52% area against the
input, but render intact: the input's naive area includes interior geometry (mouth interior,
under-armour detail) that a wrap correctly drops.

## MeshAnything refinement (`tools/mesh_refine_meshanything.py`)

Separate tool that regenerates a ≤1,600-face artist-style triangle mesh with MeshAnything V2
(`third_party/MeshAnythingV2`, patched for transformers ≥ 5). Optionally it also writes a quad
`.obj` (`--quad-remesh`) that `quad_postprocess.py --quad-obj` can pick up.

- `--algo direct-sample`: repair → sample a 16K-point conditioning cloud (half uniform, half
  curvature-weighted) from the full-res input → MeshAnything. It never lost to `legacy`, won clearly
  on watch and hand, and is ~4× faster.
- `--algo legacy` (CLI default): decimate to ≤16K faces → SDS geometry refinement against a
  reference image (SD image-variations UNet) → MeshAnything. The SDS loop was stabilized
  (fp32 VAE, which was the NaN root cause; gradient clamping; a skip-bad-iteration guard; offset
  clamp) and passes 10/10 seeds on `can`, but it isn't worth further tuning.
- Known limitation: multi-part shapes (car, robot) fragment no matter how conditioning is sampled.

## Open items

- `--adaptive` needs a visual comparison across the 11 assets before it can become the default.
- `mesh_repair` only *checks* per-component winding consistency. Dropping non-orientable
  components (`remove_winding_inconsistent_components`) is what made `--source meshanything` work on
  `watch`, and it isn't implemented.
- `quad_remesh.remesh_to_quad_dominant_obj_multi_component` (per-component face budgets) is
  tested but unused since Alpha Wrap fixed the underlying problem.
- **Live-pipeline variant**: inside `run_sample.py`, after `pipeline.run(image)` and before
  `to_glb`, `mesh.attrs/coords/layout/voxel_size` are still live. Remeshing there could texture
  the quad mesh by re-sampling the PBR voxels with `grid_sample_3d`, the way `to_glb` does. That
  would remove the closest-point transfer and its approximation error.

## Building the native pieces

QuadriFlow and Alpha Wrap are CPU-only and built locally (binaries and `runtime_libs/` are
gitignored): see [`third_party/QuadriFlow/BUILD_NOTES.md`](../third_party/QuadriFlow/BUILD_NOTES.md)
and [`third_party/AlphaWrap/BUILD_NOTES.md`](../third_party/AlphaWrap/BUILD_NOTES.md). Without
them, `quad_postprocess.py` warns and skips Alpha Wrap, and QuadriFlow remeshing raises a
`RuntimeError`.
