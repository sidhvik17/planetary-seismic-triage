"""Self-supervised masked-reconstruction pretraining (MAE-style, conv variant).

The labeled catalog covers <1% of the archive; the continuous stream itself
teaches the noise structure of an airless body (multi-hour scattering coda,
no oceanic/cultural noise). We mask contiguous patches of the normalized
window and train an encoder-decoder to reconstruct the masked samples; the
encoder is the SeisCNN backbone, so pretrained weights drop straight into
the detector for few-shot fine-tuning.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import ARCHS, SeisCNN

PATCH = 64  # samples per mask patch; 8192-sample window -> 128 patches


class MaskedAutoencoder(nn.Module):
    def __init__(self, arch: str = "base", mask_ratio: float = 0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        base = SeisCNN(channels=ARCHS[arch])
        self.backbone = base.backbone            # [B,1,8192] -> [B,C,128]
        c = ARCHS[arch][-1]
        self.decoder = nn.Sequential(            # [B,C,128] -> [B,1,8192]
            nn.Conv1d(c, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, 4, stride=4), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, 4, stride=4), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, 4, stride=4),
        )

    def random_mask(self, x: torch.Tensor):
        """Zero out a random subset of PATCH-sized spans. Returns (x_masked, mask)
        where mask is 1 on masked samples."""
        B, _, N = x.shape
        n_patches = N // PATCH
        n_mask = int(round(n_patches * self.mask_ratio))
        mask = torch.zeros(B, 1, N, device=x.device)
        for b in range(B):
            idx = torch.randperm(n_patches, device=x.device)[:n_mask]
            for i in idx:
                mask[b, :, i * PATCH:(i + 1) * PATCH] = 1.0
        return x * (1 - mask), mask

    def forward(self, x):
        x_masked, mask = self.random_mask(x)
        recon = self.decoder(self.backbone(x_masked))
        # loss only where the model could not see the input
        loss = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        return loss, recon, mask


def transfer_backbone(encoder_ckpt: str, model: SeisCNN) -> SeisCNN:
    """Load pretrained backbone weights into a fresh detector."""
    state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(state["backbone"])
    return model


class UnlabeledWindows(torch.utils.data.Dataset):
    """Random normalized windows drawn from cached continuous traces."""

    def __init__(self, npz_paths, n_samples: int, windows_per_epoch: int, seed=0):
        self.paths = list(npz_paths)
        self.n = n_samples
        self.count = windows_per_epoch
        self.rng = np.random.default_rng(seed)
        self.traces = []
        for p in self.paths:
            z = np.load(p)
            tr = z["trace"]
            if len(tr) > self.n:
                self.traces.append(tr)

    def __len__(self):
        return self.count

    def __getitem__(self, _):
        from .preprocessing import normalize_window
        tr = self.traces[self.rng.integers(len(self.traces))]
        s = self.rng.integers(0, len(tr) - self.n)
        return torch.from_numpy(normalize_window(tr[s:s + self.n])).unsqueeze(0)
