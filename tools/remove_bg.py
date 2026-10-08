"""
Background removal via BiRefNet semantic segmentation, followed by
subject centering on a transparent 1024×1024 canvas.

Usage:
    python tools/remove_bg.py input.png output.png
"""
import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trellis2.pipelines.rembg.centered import remove_background


def main():
    parser = argparse.ArgumentParser(description="Remove the background from an image and center the subject.")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output RGBA PNG path")
    args = parser.parse_args()

    print(f"Loading image: {args.input}")
    result = remove_background(Image.open(args.input))
    result.save(args.output)
    print(f"Saved: {args.output}  ({result.size[0]}×{result.size[1]})")


if __name__ == "__main__":
    main()
