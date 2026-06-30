import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["ATTN_BACKEND"] = "flash_attn"
# Required for the ROCm Triton-based flash-attn backend (no HIP binary for gfx1151).
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
# MIOpen on gfx1151 fails to load the Winograd kernel assembly when benchmarking
# new convolution shapes, raising miopenStatusUnknownError. Disable Winograd so
# MIOpen only considers GEMM/Direct algorithms that work on this GPU.
os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")

import argparse
import cv2
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel
import trimesh

# ---------------------------------------------------------------------------
# Background removal (BiRefNet)
# ---------------------------------------------------------------------------

_birefnet_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MASK_THRESHOLD = 0.5
OUT_SIZE = (1024, 1024)
PADDING_FRAC = 0.05


def _load_birefnet(device):
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    model.float().eval().to(device)
    return model


def _segment(img_rgb: Image.Image, model, device) -> np.ndarray:
    """Return a float32 mask (H×W, 0–1) in the original image resolution."""
    orig_w, orig_h = img_rgb.size
    tensor_cpu = _birefnet_transform(img_rgb).unsqueeze(0)

    preds = None
    for dev in ([device] if device.type == "cpu" else [device, torch.device("cpu")]):
        try:
            model.to(dev)
            with torch.no_grad():
                preds = model(tensor_cpu.to(dev))
            break
        except Exception as e:
            if dev.type != "cpu":
                print(f"  GPU inference failed ({e}); retrying on CPU...")
            else:
                raise

    raw = preds[-1].sigmoid().squeeze().cpu().numpy()
    mask_pil = Image.fromarray((raw * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BILINEAR
    )
    return np.array(mask_pil).astype(np.float32) / 255.0


def _flood_fill_mask(arr, seed_points, tolerance=15):
    from collections import deque
    h, w = arr.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    for sy, sx in seed_points:
        if not visited[sy, sx]:
            visited[sy, sx] = True
            queue.append((sy, sx))
    bg = arr[seed_points[0][0], seed_points[0][1], :3].astype(int)
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if np.abs(arr[ny, nx, :3].astype(int) - bg).max() <= tolerance:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return visited.astype(np.float32)


def _flood_fill_segment(img_rgb: Image.Image) -> np.ndarray:
    arr = np.array(img_rgb.convert("RGBA"))
    h, w = arr.shape[:2]
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return 1.0 - _flood_fill_mask(arr, corners)


def _tight_bbox(mask_binary: np.ndarray):
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def _center_on_canvas(img_rgba: Image.Image, mask_binary: np.ndarray) -> Image.Image:
    bbox = _tight_bbox(mask_binary)
    if bbox is None:
        return img_rgba.resize(OUT_SIZE, Image.LANCZOS).convert("RGBA")
    rmin, rmax, cmin, cmax = bbox
    subject = img_rgba.crop((cmin, rmin, cmax + 1, rmax + 1))
    pad_px = int(min(OUT_SIZE) * PADDING_FRAC)
    max_w = OUT_SIZE[0] - 2 * pad_px
    max_h = OUT_SIZE[1] - 2 * pad_px
    scale = min(max_w / subject.width, max_h / subject.height, 1.0)
    new_w = max(1, int(subject.width * scale))
    new_h = max(1, int(subject.height * scale))
    subject = subject.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    canvas.paste(subject, ((OUT_SIZE[0] - new_w) // 2, (OUT_SIZE[1] - new_h) // 2), subject)
    return canvas


def remove_background(img: Image.Image, device: torch.device) -> Image.Image:
    """Remove the background from img and return a centered RGBA image."""
    img_rgba = img.convert("RGBA")
    img_rgb = img.convert("RGB")

    mask_f32 = None
    try:
        print("Loading BiRefNet segmentation model...")
        model = _load_birefnet(device)
        print("Running semantic segmentation...")
        mask_f32 = _segment(img_rgb, model, device)
        coverage = mask_f32.mean()
        print(f"  Foreground coverage: {coverage:.1%}")
        if coverage < 0.01 or coverage > 0.99:
            print("  Mask looks degenerate; falling back to flood-fill.")
            mask_f32 = None
    except Exception as e:
        print(f"  Segmentation failed ({e}); falling back to flood-fill.")

    if mask_f32 is None:
        print("Running flood-fill background removal...")
        mask_f32 = _flood_fill_segment(img_rgb)

    arr = np.array(img_rgba).astype(np.float32)
    arr[:, :, 3] = arr[:, :, 3] * mask_f32
    masked = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return _center_on_canvas(masked, mask_f32 > MASK_THRESHOLD)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run TRELLIS.2 image-to-3D and export GLB")
parser.add_argument("--image", default="./assets/example_image/T.png", help="Input image path")
parser.add_argument("--output", default="sample_output.glb", help="Output GLB path")
parser.add_argument("--texture-size", type=int, default=1024, help="Texture atlas resolution (default 1024)")
parser.add_argument("--decimation", type=int, default=500000, help="Target face count after decimation (default 500000)")
parser.add_argument("--no-glb", action="store_true", help="Skip GLB baking; export raw geometry only")
parser.add_argument("--no-remove-bg", action="store_true",
                    help="Skip background removal (use when input already has transparency)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

print("Loading Pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Image...")
image = Image.open(args.image)

if not args.no_remove_bg:
    print("Removing background...")
    image = remove_background(image, device)

print("Running Pipeline...")
mesh = pipeline.run(image)[0]
print(f"Generated raw mesh with {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces.")

if args.no_glb:
    vertices_np = mesh.vertices.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    try:
        vertex_attrs = mesh.query_attrs(mesh.vertices)
        if 'base_color' in getattr(mesh, 'layout', {}):
            idx = mesh.layout['base_color']
            colors_float = vertex_attrs[:, idx.start:idx.stop].cpu().numpy()
            vertex_colors = (np.clip(colors_float, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            vertex_colors = None
    except Exception as e:
        print("Warning: could not extract vertex colors:", e)
        vertex_colors = None
    out_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np,
                               vertex_colors=vertex_colors, process=False)
    out_mesh.export(args.output)
    print(f"Saved raw geometry to {args.output}")
else:
    print(f"Baking textures → {args.output}  (texture_size={args.texture_size}, decimation={args.decimation})")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation,
        texture_size=args.texture_size,
        remesh=False,
        verbose=True,
    )
    glb.export(args.output)
    print(f"Saved textured GLB to {args.output}")
