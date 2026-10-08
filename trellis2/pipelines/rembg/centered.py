"""
Background removal via BiRefNet semantic segmentation (with a flood-fill fallback for
solid-colour backgrounds), followed by subject centering on a transparent canvas.

Used by run_sample.py and tools/remove_bg.py ahead of the pipeline's own preprocessing.
"""
from collections import deque
import numpy as np
from PIL import Image
import torch

from .BiRefNet import BiRefNet

__all__ = ["remove_background"]

OUT_SIZE = (1024, 1024)
PADDING_FRAC = 0.05   # fraction of canvas to keep as margin around subject
MASK_THRESHOLD = 0.5  # sigmoid cutoff for foreground


def load_birefnet(device: torch.device) -> BiRefNet:
    rembg = BiRefNet("ZhengPeng7/BiRefNet")
    # Keep in float32 for broadest GPU/driver compatibility
    rembg.model.float().to(device)
    return rembg


def segment(img_rgb: Image.Image, rembg: BiRefNet, device: torch.device) -> np.ndarray:
    """Return a float32 mask (H×W, 0-1) in the original image resolution."""
    orig_w, orig_h = img_rgb.size
    tensor_cpu = rembg.transform_image(img_rgb).unsqueeze(0)

    preds = None
    for dev in ([device] if device.type == "cpu" else [device, torch.device("cpu")]):
        try:
            rembg.model.to(dev)
            with torch.no_grad():
                preds = rembg.model(tensor_cpu.to(dev))
            break
        except Exception as e:
            if dev.type != "cpu":
                print(f"  GPU inference failed ({e}); retrying on CPU...")
            else:
                raise

    # BiRefNet returns a list of predictions; last is the finest-scale output
    raw = preds[-1].sigmoid().squeeze().cpu().numpy()
    mask_pil = Image.fromarray((raw * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BILINEAR
    )
    return np.array(mask_pil).astype(np.float32) / 255.0


def _flood_fill_mask(arr, seed_points, tolerance=15):
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
    return 1.0 - _flood_fill_mask(arr, corners)   # foreground = inverse of background


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
    new_w = max(1, int(subject.width * scale))
    new_h = max(1, int(subject.height * scale))
    subject = subject.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", out_size, (0, 0, 0, 0))
    canvas.paste(subject, ((out_size[0] - new_w) // 2, (out_size[1] - new_h) // 2), subject)
    return canvas


def remove_background(img: Image.Image, device: torch.device = None) -> Image.Image:
    """Remove the background from img and return the subject centered on an RGBA canvas."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_rgba = img.convert("RGBA")
    img_rgb = img.convert("RGB")

    mask_f32 = None
    try:
        print("Loading BiRefNet segmentation model...")
        rembg = load_birefnet(device)
        print("Running semantic segmentation...")
        mask_f32 = segment(img_rgb, rembg, device)
        coverage = mask_f32.mean()
        print(f"  Foreground coverage: {coverage:.1%}")
        # If the mask covers almost nothing or almost everything, it likely failed.
        if coverage < 0.01 or coverage > 0.99:
            print("  Mask looks degenerate; falling back to flood-fill.")
            mask_f32 = None
    except Exception as e:
        print(f"  Segmentation failed ({e}); falling back to flood-fill.")

    if mask_f32 is None:
        print("Running flood-fill background removal...")
        mask_f32 = flood_fill_segment(img_rgb)

    # Multiply existing alpha by the mask (keeps any partial transparency in the source)
    arr = np.array(img_rgba).astype(np.float32)
    arr[:, :, 3] = arr[:, :, 3] * mask_f32
    masked = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return center_on_canvas(masked, mask_f32 > MASK_THRESHOLD)
