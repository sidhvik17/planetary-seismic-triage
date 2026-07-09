"""Lightweight dual-head 1D CNN (FR-6, NFR-1).

Backbone: 6 stride-2 conv blocks, 4096 samples -> 64 timesteps.
Head A (detection): pooled features -> logit.
Head B (arrival): per-timestep score -> softmax over time -> soft-argmax,
giving a differentiable arrival offset in [0,1] plus an interpretable
localization heatmap for the web app.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=2, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# Width variants for the efficiency-vs-accuracy Pareto study.
ARCHS = {
    "tiny": (8, 16, 24, 32, 48, 48),
    "base": (16, 32, 48, 64, 96, 96),
    "large": (32, 64, 96, 128, 192, 192),
}


class SeisCNN(nn.Module):
    def __init__(self, channels=ARCHS["base"], dropout=0.3):
        super().__init__()
        blocks, cin = [], 1
        kernels = (9, 9, 7, 7, 5, 3)
        for cout, k in zip(channels, kernels):
            blocks.append(ConvBlock(cin, cout, k))
            cin = cout
        self.backbone = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(2 * cin, 64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.loc_head = nn.Sequential(
            nn.Conv1d(cin, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv1d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        """x: [B, 1, N] normalized window.

        Returns (cls_logit [B], offset [B] in [0,1], heatmap [B, T]).
        """
        f = self.backbone(x)                       # [B, C, T]
        f = self.drop(f)
        pooled = torch.cat([f.mean(dim=2), f.amax(dim=2)], dim=1)
        logit = self.cls_head(pooled).squeeze(-1)
        heat = self.loc_head(f).squeeze(1)         # [B, T]
        attn = torch.softmax(heat, dim=1)
        t = torch.linspace(0, 1, heat.shape[1], device=x.device)
        offset = (attn * t).sum(dim=1)             # soft-argmax
        return logit, offset, heat


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = SeisCNN()
    x = torch.randn(2, 1, 4096)
    logit, offset, heat = m(x)
    print("params:", count_params(m))
    print("logit", logit.shape, "offset", offset.shape, "heat", heat.shape)
