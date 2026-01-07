"""
prepare_data.py
- Reads HR images from data/DIV2K/HR/
- Creates bicubic-downsampled LR images for scales x2, x3, x4
- Saves LR images to data/DIV2K/LR_x{scale}/ with matching filenames
Usage:
  python scripts/prepare_data.py --hr_dir data/DIV2K/HR --scales 2 3 4 --out_dir data/DIV2K
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def bicubic_downsample(img: Image.Image, scale: int):
    w, h = img.size
    new_size = (w // scale, h // scale)
    return img.resize(new_size, resample=Image.BICUBIC)

def process(hr_dir, out_dir, scales, exts=(".png", ".jpg", ".jpeg")):
    hr_dir = Path(hr_dir)
    out_dir = Path(out_dir)
    hr_files = [p for p in sorted(hr_dir.iterdir()) if p.suffix.lower() in exts]
    if len(hr_files) == 0:
        raise RuntimeError(f"No HR images found in {hr_dir}")

    print(f"Found {len(hr_files)} HR images. Scales: {scales}")

    # Ensure folders exist
    for s in scales:
        ensure_dir(out_dir / f"LR_x{s}")

    for idx, hr_path in enumerate(tqdm(hr_files, desc="Processing HR images"), 1):
        try:
            img = Image.open(hr_path).convert("RGB")
        except Exception as e:
            print("Failed to open", hr_path, e)
            continue

        for s in scales:
            lr = bicubic_downsample(img, s)
            lr_folder = out_dir / f"LR_x{s}"
            lr_name = lr_folder / hr_path.name
            lr.save(lr_name, format="PNG")

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(hr_files)} images")

    print("Done creating LR images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, default="data/DIV2K/HR")
    parser.add_argument("--out_dir", type=str, default="data/DIV2K")
    parser.add_argument("--scales", type=int, nargs="+", default=[2,3,4], help="scales to generate, e.g. 2 3 4")
    args = parser.parse_args()
    process(args.hr_dir, args.out_dir, args.scales)