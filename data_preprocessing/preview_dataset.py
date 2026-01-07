# notebooks/01_preview.py
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root folder to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.datasets import SRDataset
import torchvision

dataset = SRDataset(root_dir="data/DIV2K", scale=4, hr_patch_size=128, augment=False)
sample = dataset[0]
lr = sample["lr"]  # CHW tensor
hr = sample["hr"]

def imshow_tensor(t):
    img = t.numpy().transpose(1,2,0)
    plt.imshow(img)
    plt.axis('off')

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("LR patch (upsampled for display)")
# upsample lr for visualization
lr_up = torchvision.transforms.functional.resize(torchvision.transforms.functional.to_pil_image(lr), hr.shape[1:])
plt.imshow(lr_up)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("HR patch")
imshow_tensor(hr)
plt.show()