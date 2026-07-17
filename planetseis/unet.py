"""Spectrogram U-Net for event/noise mask prediction (MQNet-style, PyTorch).

MarsQuakeNet (Dahmen et al. 2022) predicts per-pixel event/noise fractions on
STFTs with an encoder-decoder U-Net; this is the same design on the project's
2x128x128 (real/imag) grid. The output is a single event-mask logit map —
the noise mask is its complement, so one channel carries the same
information as MQNet's normalized pair. Dropout2d in the two deepest levels
gives MC-Dropout epistemic uncertainty, consistent with the 1D CNN pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin, cout):
    # MQNet uses no batch norm (batch_norm=False in the released training
    # notebook); GroupNorm(1) = LayerNorm-over-HW keeps small-batch training
    # stable without the train/eval statistics mismatch of BatchNorm.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(1, cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(1, cout),
        nn.ReLU(inplace=True),
    )


# Width variants, mirroring model.ARCHS naming.
UNET_ARCHS = {
    "tiny": 8,
    "base": 16,
    "large": 32,
}


class SpecUNet(nn.Module):
    """Input [B, 2, 128, 128] normalized STFT -> event-mask logits [B, 1, 128, 128]."""

    def __init__(self, base: int = 16, dropout: float = 0.25):
        super().__init__()
        w = [base, base * 2, base * 4, base * 8]
        self.enc1 = _conv_block(2, w[0])
        self.enc2 = _conv_block(w[0], w[1])
        self.enc3 = _conv_block(w[1], w[2])
        self.enc4 = _conv_block(w[2], w[3])
        self.pool = nn.MaxPool2d(2)
        self.drop_enc = nn.Dropout2d(dropout)

        self.bott = _conv_block(w[3], w[3] * 2)
        self.drop_bott = nn.Dropout2d(dropout)

        self.up4 = nn.ConvTranspose2d(w[3] * 2, w[3], 2, stride=2)
        self.dec4 = _conv_block(w[3] * 2, w[3])
        self.up3 = nn.ConvTranspose2d(w[3], w[2], 2, stride=2)
        self.dec3 = _conv_block(w[2] * 2, w[2])
        self.up2 = nn.ConvTranspose2d(w[2], w[1], 2, stride=2)
        self.dec2 = _conv_block(w[1] * 2, w[1])
        self.up1 = nn.ConvTranspose2d(w[1], w[0], 2, stride=2)
        self.dec1 = _conv_block(w[0] * 2, w[0])
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bott(self.drop_enc(self.pool(s4)))
        b = self.drop_bott(b)
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.head(d1)


if __name__ == "__main__":
    from .model import count_params

    for name, base in UNET_ARCHS.items():
        m = SpecUNet(base=base)
        y = m(torch.randn(2, 2, 128, 128))
        print(f"{name:5s} base={base:2d} params={count_params(m):,} out={tuple(y.shape)}")
