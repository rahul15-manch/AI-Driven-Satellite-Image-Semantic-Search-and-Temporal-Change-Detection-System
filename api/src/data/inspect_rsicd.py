"""Visual sanity check utility for researcher inspection of supplied RSICD imagery.

Selects a deterministic sample of images across categories, prints their properties,
and optionally generates a contact sheet grid so researchers can confirm that actual
satellite imagery is present rather than random RGB noise.

Usage:
    python3 -m src.data.inspect_rsicd
    python3 -m src.data.inspect_rsicd --data-dir data/raw/rsicd --num-samples 16 --save-grid
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def inspect_rsicd_samples(
    data_dir: Path,
    num_samples: int = 16,
    save_grid: bool = True,
    output_grid_path: Path | None = None,
) -> None:
    """Inspects a sample of supplied RSICD images and optionally outputs a contact sheet."""
    target_img_dir = data_dir / "images" if (data_dir / "images").is_dir() else data_dir

    print("=" * 80)
    print("RSICD VISUAL SANITY CHECK")
    print(f"Scanning Directory: {target_img_dir.resolve()}")
    print("=" * 80)

    if not target_img_dir.exists():
        print(f"\n[ERROR] Directory not found: {target_img_dir}", file=sys.stderr)
        print("Please place the RSICD images inside data/raw/rsicd/ and try again.")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([p for p in target_img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

    if not image_files:
        print("\n[INFO] No image files found in target directory.")
        print("Once you have placed the dataset files into data/raw/rsicd/, run this command again.")
        return

    print(f"\nTotal image candidates found: {len(image_files)}")

    # Sample uniformly across the sorted list to capture multiple categories
    step = max(1, len(image_files) // num_samples)
    selected_files = image_files[::step][:num_samples]

    print(f"\nInspecting {len(selected_files)} sampled images:")
    print("-" * 80)
    print(f"{'Filename':<35} | {'Dimensions':<12} | {'Mode':<6} | {'Size (KB)':<10} | {'Status'}")
    print("-" * 80)

    loaded_images: List[Tuple[Image.Image, str]] = []

    for p in selected_files:
        try:
            sz_kb = p.stat().st_size / 1024.0
            with Image.open(p) as img:
                w, h = img.size
                mode = img.mode
                # Load pixel data to verify decode
                img.load()
                loaded = img.copy().convert("RGB")
                loaded_images.append((loaded, p.name))
                print(f"{p.name:<35} | {f'{w}x{h}':<12} | {mode:<6} | {sz_kb:<10.1f} | Decoded OK")
        except Exception as exc:
            print(f"{p.name:<35} | {'ERROR':<12} | {'ERR':<6} | {'0.0':<10} | Failed: {exc}")

    # Create contact sheet grid if requested
    if save_grid and loaded_images:
        grid_cols = 4
        grid_rows = (len(loaded_images) + grid_cols - 1) // grid_cols
        tile_w, tile_h = 224, 224
        label_h = 24

        grid_img = Image.new("RGB", (grid_cols * tile_w, grid_rows * (tile_h + label_h)), color=(30, 30, 30))
        draw = ImageDraw.Draw(grid_img)

        for idx, (im, name) in enumerate(loaded_images):
            col = idx % grid_cols
            row = idx // grid_cols
            x = col * tile_w
            y = row * (tile_h + label_h)

            im_resized = im.resize((tile_w, tile_h))
            grid_img.paste(im_resized, (x, y))

            # Draw label box
            draw.rectangle([(x, y + tile_h), (x + tile_w, y + tile_h + label_h)], fill=(15, 15, 15))
            label_text = name if len(name) <= 24 else f"{name[:21]}..."
            draw.text((x + 4, y + tile_h + 4), label_text, fill=(220, 220, 220))

        grid_path = output_grid_path or Path("data/processed/rsicd/sanity_check_grid.png")
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(grid_path)
        print(f"\n[✓] Contact sheet saved to: {grid_path.resolve()}")
        print("Please visually inspect this image to confirm valid satellite scenes rather than noise.")

    print("=" * 80)


def main():
    default_dir = os.environ.get("RSICD_ROOT", "data/raw/rsicd")
    parser = argparse.ArgumentParser(description="Visual sanity check for RSICD images")
    parser.add_argument("--data-dir", type=str, default=default_dir, help=f"RSICD root (default: {default_dir})")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to inspect (default: 16)")
    parser.add_argument("--no-grid", action="store_true", help="Skip contact sheet generation")
    parser.add_argument("--output-grid", type=str, default=None, help="Custom output path for contact sheet")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    grid_path = Path(args.output_grid) if args.output_grid else None

    inspect_rsicd_samples(
        data_dir=data_dir,
        num_samples=args.num_samples,
        save_grid=not args.no_grid,
        output_grid_path=grid_path,
    )


if __name__ == "__main__":
    main()
