# Archived development logs

Working notes kept while building this fork's mesh post-processing. They record *how* the
current code came to be, including dead ends, and are not maintained. For the current state, read
[`../quad_pipeline.md`](../quad_pipeline.md) and [`../scripts.md`](../scripts.md).

| File | What it is |
|---|---|
| `history.md` | Chronological account: ROCm port summary, MeshAnything V2 integration and its SDS debugging, the finding that MeshAnything never emits quads, adopting QuadriFlow. |
| `quadriflow_postprocess_plan.md` | The phase-by-phase plan (Phases 1–8) for post-processing QuadriFlow output, with per-phase results appended. |
| `quadriflow_progress.md` | Task-level progress log for that plan, plus the in-process QuadriFlow binding / adaptive-density addendum. |

Paths in these files are as they were when written: `tests/*.glb` is now `samples/*.glb`,
`amd_workarounds.md` is now `docs/amd_rocm.md`, `trellis2/utils/uv_rasterize.py` is now
`o-voxel/o_voxel/uv_rasterize.py`, `test_phase*.py` is now `tests/phase/`, and `/tmp/claude-…`
scratch paths no longer exist.
