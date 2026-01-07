import os, time
from tqdm import tqdm

SAVE_DIR = "/kaggle/working/hybrid_sr_checkpoints"
BACKUP_DIR = "/kaggle/working/hybrid_sr_backups"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

def checkpoint_and_backup(model, name, epoch):
    fname = os.path.join(SAVE_DIR, f"{name}_epoch{epoch}.pth")
    torch.save(model.state_dict(), fname)
    # zip a copy to backup dir (smaller metadata zip)
    ts = int(time.time())
    zname = os.path.join(BACKUP_DIR, f"{name}_ep{epoch}_backup_{ts}.zip")
    os.system(f"zip -rq {zname} {fname}")
    print("Saved & backed up:", fname, "->", zname)

# generic supervised training with AMP
def train_supervised_amp(model, dataloader, val_loader, device, epochs, name, lr=1e-4, save_every=1):
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    from torch.optim import Adam
    from tqdm import tqdm

    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler()   # ✅ Works everywhere

    for ep in range(1, epochs+1):
        model.train()
        loop = tqdm(dataloader, desc=f"{name} Ep{ep}")
        running = 0.0

        for lr_imgs, hr_imgs in loop:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            with autocast():     # ✅ Mixed precision
                pred = model(lr_imgs)
                loss = F.l1_loss(pred, hr_imgs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            loop.set_postfix(loss=running / (loop.n + 1))

        print(f"Epoch {ep}: Train Loss = {running/len(dataloader):.4f}")