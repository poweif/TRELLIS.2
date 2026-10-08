"""Every fork-added script/module must be listed in docs/scripts.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# trellis2/utils modules that ship with upstream TRELLIS.2.
UPSTREAM_UTILS = {
    "__init__", "data_utils", "dist_utils", "elastic_utils", "general_utils", "grad_clip_utils",
    "loss_utils", "mesh_utils", "random_utils", "render_utils", "vis_utils",
}
# Root scripts that ship with upstream TRELLIS.2.
UPSTREAM_ROOT = {"app", "app_texturing", "example", "example_texturing", "train"}


def test_scripts_md_lists_every_fork_script():
    doc = (ROOT / "docs" / "scripts.md").read_text()
    expected = [p.relative_to(ROOT).as_posix() for p in (ROOT / "tools").glob("*.py")]
    expected += [p.name for p in ROOT.glob("*.py") if p.stem not in UPSTREAM_ROOT]
    expected += [p.relative_to(ROOT).as_posix() for p in (ROOT / "trellis2" / "utils").glob("*.py")
                 if p.stem not in UPSTREAM_UTILS]
    missing = [e for e in expected if e not in doc]
    assert not missing, f"add these to docs/scripts.md: {missing}"
