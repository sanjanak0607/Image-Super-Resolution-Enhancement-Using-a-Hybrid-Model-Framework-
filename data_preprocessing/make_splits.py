# scripts/make_splits.py
import random
from pathlib import Path

# Set the root folder where HR images are stored
root = Path("data/DIV2K/HR")  # change if your HR images are elsewhere

# Collect all image files in the HR folder
files = sorted([p.name for p in root.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

total = len(files)
print(f"Total HR images found: {total}")

# Shuffle files with a fixed seed for reproducibility
random.seed(42)
random.shuffle(files)

# Split into train/val/test: 640 / 80 / 80
train = files[:640]      # first 640 images
val   = files[640:720]   # next 80 images
test  = files[720:]      # remaining 80 images

# Make sure output folder exists
output_dir = Path("data/DIV2K")
output_dir.mkdir(parents=True, exist_ok=True)

# Save the splits to text files
(train_list := output_dir / "train_list.txt").write_text("\n".join(train))
(val_list := output_dir / "val_list.txt").write_text("\n".join(val))
(test_list := output_dir / "test_list.txt").write_text("\n".join(test))

print("Created splits:")
print(f"Train: {len(train)} images -> {train_list}")
print(f"Val:   {len(val)} images -> {val_list}")
print(f"Test:  {len(test)} images -> {test_list}")
