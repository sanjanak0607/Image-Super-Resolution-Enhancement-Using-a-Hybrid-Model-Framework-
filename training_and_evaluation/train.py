import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os, time
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# =============================================
# ============== Device ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =============================================
# ============== Hyperparameters ==============
BATCH_SIZE = 8       # Adjust based on your GPU memory
LR = 1e-4
EPOCHS_INDIV = 75    # CNN, Transformer, GAN
EPOCHS_FUSION = 350  # Fusion network
CHECKPOINT_EVERY = 5 # save every N epochs

# =============================================
# ============== Helper Metrics ==============
def compute_metrics_batch(pred_tensor, hr_tensor):
    batch_size = pred_tensor.shape[0]
    psnr_list, ssim_list = [], []

    for i in range(batch_size):
        pred_np = pred_tensor[i].permute(1,2,0).cpu().numpy()
        hr_np = hr_tensor[i].permute(1,2,0).cpu().numpy()

        if pred_np.shape != hr_np.shape:
            hr_np = resize(hr_np, pred_np.shape, preserve_range=True, anti_aliasing=True)

        pred_np = np.clip(pred_np, 0, 1)
        hr_np = np.clip(hr_np, 0, 1)
        h, w = pred_np.shape[:2]
        win_size = min(7, h, w)
        psnr_val = psnr(hr_np, pred_np, data_range=1.0)
        ssim_val = ssim(hr_np, pred_np, data_range=1.0, channel_axis=2, win_size=win_size)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    return np.mean(psnr_list), np.mean(ssim_list)

# =============================================
# ============== Training Function ===========
def train_model(model, train_loader, val_loader, epochs, name, save_dir):
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler()  # mixed precision

    train_losses, val_psnrs, val_ssims = [], [], []

    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"{name} Epoch {ep}")
        for lr_imgs, hr_imgs in loop:
            lr_imgs = lr_imgs.to(DEVICE).float()
            hr_imgs = hr_imgs.to(DEVICE).float()

            opt.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                out = model(lr_imgs)
                loss = criterion(out, hr_imgs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            loop.set_postfix(loss=running_loss/(loop.n+1))

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ===== Validation =====
        model.eval()
        psnr_sum, ssim_sum = 0.0, 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(DEVICE).float()
                hr_imgs = hr_imgs.to(DEVICE).float()
                out = model(lr_imgs)
                p, s = compute_metrics_batch(out, hr_imgs)
                psnr_sum += p
                ssim_sum += s
        val_psnr = psnr_sum / len(val_loader)
        val_ssim = ssim_sum / len(val_loader)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)

        print(f"[{name}] Epoch {ep}: Train Loss={avg_loss:.4f}, Val PSNR={val_psnr:.4f}, Val SSIM={val_ssim:.4f}")

        # ===== Checkpoint =====
        if ep % CHECKPOINT_EVERY == 0 or ep == epochs:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"{name}_epoch{ep}.pth")
            torch.save(model.state_dict(), ckpt_path)

    return train_losses, val_psnrs, val_ssims

# =============================================
# ============== Dataset & Loaders ===========
# Replace these with your actual Dataset objects
# train_dataset, val_dataset, test_dataset = YourDataset(...)
# Example:
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================================
# ============== Train Individual Models =====
SAVE_DIR = "/kaggle/working/sr_checkpoints"

cnn_losses, cnn_psnrs, cnn_ssims = train_model(cnn, train_loader, val_loader, EPOCHS_INDIV, "CNN", SAVE_DIR)
tr_losses, tr_psnrs, tr_ssims = train_model(tr, train_loader, val_loader, EPOCHS_INDIV, "Transformer", SAVE_DIR)
gen_losses, gen_psnrs, gen_ssims = train_model(gen, train_loader, val_loader, EPOCHS_INDIV, "GAN", SAVE_DIR)

# =============================================
# ============== Train Fusion Model =========
# Freeze pretrained models
for m in [cnn, tr, gen]:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

fusion.to(DEVICE)
opt_f = torch.optim.Adam(fusion.parameters(), lr=LR)
criterion_f = nn.L1Loss()
scaler_f = torch.amp.GradScaler()

fusion_train_losses, fusion_val_psnrs, fusion_val_ssims = [], [], []

for ep in range(1, EPOCHS_FUSION+1):
    fusion.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Fusion Epoch {ep}")
    for lr_imgs, hr_imgs in loop:
        lr_imgs = lr_imgs.to(DEVICE).float()
        hr_imgs = hr_imgs.to(DEVICE).float()

        with torch.no_grad():
            out_c = cnn(lr_imgs)
            out_t = tr(lr_imgs)
            out_g = gen(lr_imgs)
        inp = torch.cat([out_c, out_t, out_g], dim=1)

        opt_f.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            out_f = fusion(inp)
            loss = criterion_f(out_f, hr_imgs)
        scaler_f.scale(loss).backward()
        scaler_f.step(opt_f)
        scaler_f.update()
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss/(loop.n+1))

    avg_loss = running_loss / len(train_loader)
    fusion_train_losses.append(avg_loss)

    # Validation
    fusion.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(DEVICE).float()
            hr_imgs = hr_imgs.to(DEVICE).float()
            out_c = cnn(lr_imgs)
            out_t = tr(lr_imgs)
            out_g = gen(lr_imgs)
            inp = torch.cat([out_c, out_t, out_g], dim=1)
            out_f = fusion(inp)
            p, s = compute_metrics_batch(out_f, hr_imgs)
            psnr_sum += p
            ssim_sum += s
    val_psnr = psnr_sum / len(val_loader)
    val_ssim = ssim_sum / len(val_loader)
    fusion_val_psnrs.append(val_psnr)
    fusion_val_ssims.append(val_ssim)

    print(f"[Fusion] Epoch {ep}: Train Loss={avg_loss:.4f}, Val PSNR={val_psnr:.4f}, Val SSIM={val_ssim:.4f}")

    if ep % CHECKPOINT_EVERY == 0 or ep == EPOCHS_FUSION:
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(fusion.state_dict(), os.path.join(SAVE_DIR, f"Fusion_epoch{ep}.pth"))

# =============================================
# ============== Comparison Graph ===========
avg_psnrs = [
    np.mean(cnn_psnrs),
    np.mean(tr_psnrs),
    np.mean(gen_psnrs),
    np.mean(fusion_val_psnrs)
]
avg_ssims = [
    np.mean(cnn_ssims),
    np.mean(tr_ssims),
    np.mean(gen_ssims),
    np.mean(fusion_val_ssims)
]
models_names = ["CNN", "Transformer", "GAN", "Fusion"]

plt.figure(figsize=(10,5))
plt.bar(np.arange(len(models_names)) - 0.2, avg_psnrs, width=0.4, label='PSNR')
plt.bar(np.arange(len(models_names)) + 0.2, avg_ssims, width=0.4, label='SSIM')
plt.xticks(np.arange(len(models_names)), models_names)
plt.ylabel("Value")
plt.title("Average PSNR and SSIM Comparison")
plt.legend()
plt.show()  