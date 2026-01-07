import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os, time, json
from collections import defaultdict

def psnr_per_image(pred, target, data_range=1.0):
    # pred/target: [B,C,H,W] tensors in [0,1]
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    ps = 20 * torch.log10(torch.tensor(data_range).to(pred.device)) - 10 * torch.log10(mse + 1e-8)
    return ps.cpu().numpy()  # array length B

# basic SSIM per image (y-channel approximate)
def ssim_per_image(img1, img2, window_size=11):
    # img1/img2 in [B,3,H,W]
    device = img1.device
    # convert to Y
    coef = torch.tensor([0.2989,0.5870,0.1140]).to(device).view(1,3,1,1)
    y1 = (img1 * coef).sum(dim=1, keepdim=True)
    y2 = (img2 * coef).sum(dim=1, keepdim=True)
    C1 = (0.01)**2
    C2 = (0.03)**2
    # gaussian kernel
    def gauss(window_size, sigma=1.5):
        coords = torch.arange(window_size).float() - window_size//2
        g = torch.exp(-(coords**2)/(2*sigma**2))
        g = g / g.sum()
        kernel = g[:,None] * g[None,:]
        return kernel.unsqueeze(0).unsqueeze(0).to(device)
    kernel = gauss(window_size)
    mu1 = F.conv2d(y1, kernel, padding=window_size//2)
    mu2 = F.conv2d(y2, kernel, padding=window_size//2)
    mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(y1*y1, kernel, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(y2*y2, kernel, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(y1*y2, kernel, padding=window_size//2) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_vals = ssim_map.view(ssim_map.size(0), -1).mean(dim=1)
    return ssim_vals.cpu().numpy()

# save outputs per-batch (LR, HR, cnn, transformer, gan, fusion)
def save_batch_outputs(out_dir, basename, batch_idx, lr, hr, outs_dict):
    os.makedirs(out_dir, exist_ok=True)
    B = lr.size(0)
    for i in range(B):
        idx = f"{batch_idx:04d}_{i:02d}"
        save_image(lr[i], os.path.join(out_dir, f"{basename}_{idx}_lr.png"))
        save_image(hr[i], os.path.join(out_dir, f"{basename}_{idx}_hr.png"))
        for k,v in outs_dict.items():
            save_image(v[i], os.path.join(out_dir, f"{basename}_{idx}_{k}.png"))

# batch-wise metric aggregator (group every group_size samples)
def evaluate_models_batchwise(models_dict, dataloader, device, group_size=20, save_dir=None, basename="eval"):
    import torch
    import os
    from skimage.metrics import structural_similarity as ssim
    import numpy as np

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    results = {k: {'psnr_list': [], 'ssim_list': []} for k in models_dict}

    def PSNR(pred, gt):
        mse = torch.mean((pred - gt) ** 2).item()
        return 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))

    # ✅ Ensure eval mode
    for model in models_dict.values():
        model.eval()

    with torch.no_grad():
        for lr, hr in dataloader:   # ✅ dataset returns only (LR, HR)
            lr = lr.to(device)
            hr = hr.to(device)

            for name, model in models_dict.items():

                # ✅ Special handling for Fusion model
                if name == "fusion":
                    out_c = models_dict["cnn"](lr)
                    out_t = models_dict["transformer"](lr)
                    out_g = models_dict["gan"](lr)
                    fusion_input = torch.cat([out_c, out_t, out_g], dim=1)  # (B,9,H,W)
                    out = model(fusion_input)

                # ✅ Regular models run directly on LR input
                else:
                    out = model(lr)

                # ✅ Per-image metric computation
                for b in range(out.shape[0]):
                    pred_img = out[b].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    hr_img   = hr[b].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

                    psnr_val = PSNR(out[b], hr[b])
                    ssim_val = ssim(hr_img, pred_img, channel_axis=2, data_range=1.0)

                    results[name]['psnr_list'].append(psnr_val)
                    results[name]['ssim_list'].append(ssim_val)

    # ✅ Mean summary
    summary = {}
    for name, vals in results.items():
        summary[name] = {
            'overall_psnr_mean': np.mean(vals['psnr_list']),
            'overall_ssim_mean': np.mean(vals['ssim_list'])
        }

    return summary