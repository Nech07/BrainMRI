from collections import OrderedDict

import torch
import torch.nn as nn


# class UNet(nn.Module):
#     ############################################################################
#     # TODO: Put your code here

#     # Implement UNet and try more models
#     # Hint: don't forget the last sigmoid layer
#     ##########################################################################
#     pass

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two consecutive 3×3 convs each followed by ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """Downsampling: max‑pool then DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """Upsampling (bilinear or transposed conv) then DoubleConv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in = in_ch
        else:
            # in_ch → in_ch//2 after convtranspose, so convtranspose maps half channels
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
            conv_in = in_ch
        self.conv = DoubleConv(conv_in, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad x1 to match x2's size (if needed)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1×1 convolution to map to num_classes"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, bilinear=True):
        super().__init__()
        self.inc    = DoubleConv(in_channels, 64)
        self.down1  = Down(64, 128)
        self.down2  = Down(128, 256)
        self.down3  = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4  = Down(512, 1024 // factor)
        self.up1    = Up(1024, 512 // factor, bilinear)
        self.up2    = Up(512, 256 // factor, bilinear)
        self.up3    = Up(256, 128 // factor, bilinear)
        self.up4    = Up(128, 64, bilinear)
        self.outc   = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)           # →64×H×W
        x2 = self.down1(x1)        # →128×H/2×W/2
        x3 = self.down2(x2)        # →256×H/4×W/4
        x4 = self.down3(x3)        # →512×H/8×W/8
        x5 = self.down4(x4)        # →1024/factor×H/16×W/16
        x  = self.up1(x5, x4)      # →512/factor×H/8×W/8
        x  = self.up2(x,  x3)      # →256/factor×H/4×W/4
        x  = self.up3(x,  x2)      # →128/factor×H/2×W/2
        x  = self.up4(x,  x1)      # →64×H×W
        return self.outc(x)        # →num_classes×H×W
