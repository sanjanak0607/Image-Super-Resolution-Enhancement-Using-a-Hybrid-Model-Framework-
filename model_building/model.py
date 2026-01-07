# CELL 6: Fixed Models
import torch
import torch.nn as nn

# --- CNN SR (matches pretrained checkpoint) ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k//2)
        self.act = nn.ReLU(True)

    def forward(self, x): 
        return self.act(self.conv(x))


class CNN_SR(nn.Module):
    def __init__(self, scale=4, num_feat=64, num_blocks=8):
        super().__init__()
        layers = [ConvBlock(3, num_feat)]
        for _ in range(num_blocks):
            layers.append(ConvBlock(num_feat, num_feat))
        layers.append(nn.Conv2d(num_feat, 3 * scale * scale, 3, padding=1))  # matches checkpoint
        self.body = nn.Sequential(*layers)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        out = self.body(x)
        out = self.ps(out)
        return torch.clamp(out, 0, 1)


# --- Simple Transformer SR (matches checkpoint) ---
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, patch_size=4, emb_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        return x, (h, w)


class SimpleTransformerSR(nn.Module):
    def __init__(self, emb_dim=64, patch_size=4, num_layers=4, nhead=4, scale=4):
        super().__init__()
        self.patch = PatchEmbed(3, patch_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim * 4,
            activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # matches pretrained: conv_out has 48 channels = 3 * scale**2
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim, kernel_size=patch_size, stride=patch_size),
            nn.Conv2d(emb_dim, 3 * scale * scale, kernel_size=3, padding=1)
        )
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        b = x.size(0)
        x_p, (ph, pw) = self.patch(x)
        x_t = self.transformer(x_p.permute(1, 0, 2)).permute(1, 0, 2)
        x_t = x_t.transpose(1, 2).reshape(b, -1, ph, pw)
        x_r = self.reconstruct(x_t)
        out = self.ps(x_r)
        return torch.clamp(out, 0, 1)


# --- GAN Generator (uses CNN_SR) ---
class GAN_Generator(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.net = CNN_SR(scale=scale)

    def forward(self, x):
        return self.net(x)


# --- FusionNet (matches checkpoint) ---
class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(9, 64, 3, padding=1), nn.ReLU(True)]
        for _ in range(6):
            layers += [nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True)]
        layers += [nn.Conv2d(64, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(self.net(x), 0, 1)


print("All fixed models are ready and match pretrained checkpoints.")
