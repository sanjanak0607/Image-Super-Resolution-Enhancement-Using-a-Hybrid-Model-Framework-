# src/datasets.py
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def open_img(path):
    """Open an image and convert to RGB"""
    return Image.open(path).convert("RGB")


class SRDataset(Dataset):
    """
    PyTorch dataset for Super-Resolution training.
    Expects folder structure:
      <root>/HR/        (high-res images)
      <root>/LR_x{scale}/  (low-res images, optional)
    Returns: lr_tensor, hr_tensor (both in range [0,1], dtype=torch.float32)
    """

    def __init__(self, root_dir, scale=4, hr_patch_size=96, augment=True, split="train", file_list=None):
        super().__init__()
        self.root = Path(root_dir)
        self.scale = scale
        self.lr_dir = self.root / f"LR_x{scale}"
        self.hr_dir = self.root / "HR"
        self.augment = augment
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale

        # collect HR files
        if file_list is not None:
            self.hr_files = [self.hr_dir / f for f in file_list]
        else:
            self.hr_files = sorted([p for p in self.hr_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

        if len(self.hr_files) == 0:
            raise RuntimeError(f"No HR images found in {self.hr_dir}")

    def __len__(self):
        return len(self.hr_files)

    # ---------------- helper methods -----------------
    def _ensure_min_size(self, img, size):
        """Resize image if smaller than required patch size"""
        w, h = img.size
        if w < size or h < size:
            scale = max(size / w, size / h)
            new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
            img = img.resize((new_w, new_h), Image.BICUBIC)
        return img

    def _random_patch_with_coords(self, img, patch_size):
        """Return random patch and coordinates on the original image"""
        w, h = img.size
        if w < patch_size or h < patch_size:
            patch = TF.center_crop(img, (min(h, w), min(h, w)))
            return patch, 0, 0
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        return img.crop((x, y, x + patch_size, y + patch_size)), x, y

    def _to_tensor(self, img: Image.Image):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1)).copy()  # HWC -> CHW, ensure contiguous
        return torch.from_numpy(arr)

    # ---------------- main method -----------------
    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_dir / hr_path.name

        hr = open_img(hr_path)
        hr = self._ensure_min_size(hr, self.hr_patch_size)

        # sample HR patch
        hr_patch, x, y = self._random_patch_with_coords(hr, self.hr_patch_size)

        # get LR patch
        use_disk_lr = lr_path.exists()
        if use_disk_lr:
            disk_lr = open_img(lr_path)
            lr_x = x // self.scale
            lr_y = y // self.scale
            lr_patch = disk_lr.crop((lr_x, lr_y, lr_x + self.lr_patch_size, lr_y + self.lr_patch_size))
        else:
            # fallback: synthesize LR by downsampling HR patch
            lr_patch = hr_patch.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)

        # data augmentation
        if self.augment:
            if random.random() < 0.5:
                hr_patch = TF.hflip(hr_patch)
                lr_patch = TF.hflip(lr_patch)
            if random.random() < 0.5:
                hr_patch = TF.vflip(hr_patch)
                lr_patch = TF.vflip(lr_patch)
            k = random.randint(0, 3)
            if k:
                if k == 1:
                    hr_patch = hr_patch.transpose(Image.ROTATE_90)
                    lr_patch = lr_patch.transpose(Image.ROTATE_90)
                elif k == 2:
                    hr_patch = hr_patch.transpose(Image.ROTATE_180)
                    lr_patch = lr_patch.transpose(Image.ROTATE_180)
                elif k == 3:
                    hr_patch = hr_patch.transpose(Image.ROTATE_270)
                    lr_patch = lr_patch.transpose(Image.ROTATE_270)

        hr_t = self._to_tensor(hr_patch)
        lr_t = self._to_tensor(lr_patch)
        return {"lr": lr_t, "hr": hr_t, "fname": hr_path.name}


# ------------------- test block -------------------
if __name__ == "__main__":
    from pathlib import Path

    # Change this path to your dataset folder
    root = Path(r"C:\Users\sanja\OneDrive\Desktop\hybrid-sr\data\DIV2K")
    ds = SRDataset(root, scale=4, hr_patch_size=96, augment=False)

    print("Total images:", len(ds))

    if len(ds):
        sample = ds[0]
        print("Filename:", sample["fname"])
        print("HR shape:", sample["hr"].shape)
        print("LR shape:", sample["lr"].shape)

        # optional: show images using PIL
        # Image.fromarray((sample["hr"].numpy().transpose(1,2,0)*255).astype(np.uint8)).show()
        # Image.fromarray((sample["lr"].numpy().transpose(1,2,0)*255).astype(np.uint8)).show()
