#!/usr/bin/env python3
"""
Background removal via BiRefNet semantic segmentation, followed by
subject centering on a transparent 1024×1024 canvas.

Usage:
    python remove_bg.py [input.png] [output.png]

Defaults: test.png → test_nobg.png
"""

import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

INPUT = "test.png"
OUTPUT = "test_nobg.png"
OUT_SIZE = (1024, 1024)
PADDING_FRAC = 0.05   # fraction of canvas to keep as margin around subject
MASK_THRESHOLD = 0.5  # sigmoid cutoff for foreground


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_birefnet(device):
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    # Keep in float32 for broadest GPU/driver compatibility
    model.float().eval().to(device)
    return model


_birefnet_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _run_model(model, tensor):
    with torch.no_grad():
        return model(tensor)


def segment(img_rgb: Image.Image, model, device) -> np.ndarray:
    """Return a float32 mask (H×W, 0-1) in the original image resolution."""
    orig_w, orig_h = img_rgb.size
    tensor_cpu = _birefnet_transform(img_rgb).unsqueeze(0)

    preds = None
    for dev in ([device] if device.type == "cpu" else [device, torch.device("cpu")]):
        try:
            model.to(dev)
            t = tensor_cpu.to(dev)
            preds = _run_model(model, t)
            break
        except Exception as e:
            if dev.type != "cpu":
                print(f"  GPU inference failed ({e}); retrying on CPU...")
            else:
                raise

    # BiRefNet returns a list of predictions; last is the finest-scale output
    raw = preds[-1].sigmoid().squeeze().cpu().numpy()   # (1024, 1024)
    # Resize mask back to original image size
    mask_pil = Image.fromarray((raw * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BILINEAR
    )
    return np.array(mask_pil).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def apply_mask(img_rgba: Image.Image, mask_f32: np.ndarray) -> Image.Image:
    """Apply soft alpha mask to RGBA image."""
    arr = np.array(img_rgba).astype(np.float32)
    # Multiply existing alpha by mask (handles source images that already have
    # partial transparency)
    arr[:, :, 3] = arr[:, :, 3] * mask_f32
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def tight_bbox(mask_binary: np.ndarray):
    """Return (rmin, rmax, cmin, cmax) for the non-zero region, or None."""
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def center_on_canvas(img_rgba: Image.Image, mask_binary: np.ndarray,
                     out_size=OUT_SIZE, padding_frac=PADDING_FRAC) -> Image.Image:
    """
    Crop the subject tightly, scale to fit within the canvas (with padding),
    and paste it centered on a transparent canvas.
    """
    bbox = tight_bbox(mask_binary)
    if bbox is None:
        # Nothing to center — just resize the whole image
        return img_rgba.resize(out_size, Image.LANCZOS).convert("RGBA")

    rmin, rmax, cmin, cmax = bbox
    subject = img_rgba.crop((cmin, rmin, cmax + 1, rmax + 1))

    pad_px = int(min(out_size) * padding_frac)
    max_w = out_size[0] - 2 * pad_px
    max_h = out_size[1] - 2 * pad_px
    scale = min(max_w / subject.width, max_h / subject.height, 1.0)
    new_w = max(1, int(subject.width  * scale))
    new_h = max(1, int(subject.height * scale))
    subject = subject.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", out_size, (0, 0, 0, 0))
    x = (out_size[0] - new_w) // 2
    y = (out_size[1] - new_h) // 2
    canvas.paste(subject, (x, y), subject)
    return canvas


# ---------------------------------------------------------------------------
# Flood-fill fallback (solid-colour background)
# ---------------------------------------------------------------------------

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


def flood_fill_segment(img_rgb: Image.Image) -> np.ndarray:
    arr = np.array(img_rgb.convert("RGBA"))
    h, w = arr.shape[:2]
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    bg_mask = _flood_fill_mask(arr, corners)
    return 1.0 - bg_mask   # foreground = inverse of background


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(input_path: str, output_path: str):
    print(f"Loading image: {input_path}")
    img = Image.open(input_path).convert("RGBA")
    img_rgb = img.convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mask_f32 = None
    try:
        print("Loading BiRefNet segmentation model...")
        model = load_birefnet(device)
        print("Running semantic segmentation...")
        mask_f32 = segment(img_rgb, model, device)
        coverage = mask_f32.mean()
        print(f"  Foreground coverage: {coverage:.1%}")
        # Sanity check: if mask covers almost nothing or almost everything,
        # it likely failed — fall back.
        if coverage < 0.01 or coverage > 0.99:
            print("  Mask looks degenerate; falling back to flood-fill.")
            mask_f32 = None
    except Exception as e:
        print(f"  Segmentation failed ({e}); falling back to flood-fill.")

    if mask_f32 is None:
        print("Running flood-fill background removal...")
        mask_f32 = flood_fill_segment(img_rgb)

    mask_binary = mask_f32 > MASK_THRESHOLD

    print("Applying mask...")
    masked = apply_mask(img, mask_f32)

    print("Centering subject on canvas...")
    result = center_on_canvas(masked, mask_binary)

    result.save(output_path)
    print(f"Saved: {output_path}  ({result.size[0]}×{result.size[1]})")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT
    process(inp, out)
