"""
Phase 2 Part B1: lightweight 2D U-Net for tumor segmentation.

4 encoder blocks (channel widths 32/64/128/256) + a bottleneck + 4 decoder
blocks with skip connections, standard U-Net topology. Input: (B,1,128,128)
grayscale in [0,1]. Output: (B,1,128,128) sigmoid tumor-probability map.
"""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, widths=(32, 64, 128, 256)):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.enc1 = conv_block(in_channels, w1)   # 128x128
        self.enc2 = conv_block(w1, w2)             # 64x64
        self.enc3 = conv_block(w2, w3)             # 32x32
        self.enc4 = conv_block(w3, w4)              # 16x16
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(w4, w4)         # 8x8

        self.up4 = nn.ConvTranspose2d(w4, w4, 2, stride=2)
        self.dec4 = conv_block(w4 + w4, w4)
        self.up3 = nn.ConvTranspose2d(w4, w3, 2, stride=2)
        self.dec3 = conv_block(w3 + w3, w3)
        self.up2 = nn.ConvTranspose2d(w3, w2, 2, stride=2)
        self.dec2 = conv_block(w2 + w2, w2)
        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1 = conv_block(w1 + w1, w1)

        self.out_conv = nn.Conv2d(w1, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)                # (B,32,128,128)
        e2 = self.enc2(self.pool(e1))     # (B,64,64,64)
        e3 = self.enc3(self.pool(e2))     # (B,128,32,32)
        e4 = self.enc4(self.pool(e3))     # (B,256,16,16)
        b = self.bottleneck(self.pool(e4))  # (B,256,8,8)

        d4 = self.up4(b)                  # (B,256,16,16)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)                 # (B,128,32,32)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)                 # (B,64,64,64)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)                 # (B,32,128,128)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


def dice_loss(pred, target, eps=1e-6):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def dice_coefficient(pred_binary, target, eps=1e-6):
    """pred_binary, target: numpy or torch, same shape, {0,1}."""
    pred_flat = pred_binary.reshape(pred_binary.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(axis=1)
    union = pred_flat.sum(axis=1) + target_flat.sum(axis=1)
    return (2 * intersection + eps) / (union + eps)
