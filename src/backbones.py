"""Backbones and preprocessing extracted unchanged from the evaluated harness."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_csv(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("date", "rv1", "rv2") if c in df.columns], errors="ignore")
    return df.select_dtypes("number").values.astype("float32")

class DLinear(nn.Module):
    def __init__(self, L, H, C, kernel=25):
        super().__init__()
        self.k = kernel
        self.lin_season = nn.Linear(L, H)
        self.lin_trend = nn.Linear(L, H)

    def forward(self, x):                                              # [B,L,C]->[B,H,C]
        pad = self.k // 2
        xp = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
        trend = F.avg_pool1d(xp, self.k, stride=1).transpose(1, 2)
        s = self.lin_season((x - trend).transpose(1, 2)).transpose(1, 2)
        t = self.lin_trend(trend.transpose(1, 2)).transpose(1, 2)
        return s + t

class PatchTST(nn.Module):
    """Compact channel-independent PatchTST (Nie et al. 2023). in_affine = a per-channel
    calibration used by the parameter-efficient 'calib' strategy."""
    def __init__(self, L, H, C, P=16, S=8, d=64, nhead=4, nlayers=2):
        super().__init__()
        self.P, self.S = P, S
        self.np = (L - P) // S + 1
        self.in_affine_w = nn.Parameter(torch.ones(C))
        self.in_affine_b = nn.Parameter(torch.zeros(C))
        self.embed = nn.Linear(P, d)
        self.pos = nn.Parameter(torch.zeros(1, self.np, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 2 * d, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Linear(self.np * d, H)

    def forward(self, x):                                             # [B,L,C]->[B,H,C]
        B, L, C = x.shape
        x = x * self.in_affine_w + self.in_affine_b
        xci = x.permute(0, 2, 1).reshape(B * C, L)
        patches = xci.unfold(1, self.P, self.S)                       # [BC,np,P]
        z = self.embed(patches) + self.pos
        z = self.encoder(z)
        out = self.head(z.reshape(B * C, -1))                        # [BC,H]
        return out.reshape(B, C, -1).permute(0, 2, 1)

def build_model(backbone, L, H, C):
    return DLinear(L, H, C) if backbone == "dlinear" else PatchTST(L, H, C)

def prep(data, warmup_frac=0.5, device="cuda"):
    T, C = data.shape
    n_warm = int(T * warmup_frac)
    mean, std = data[:n_warm].mean(0), data[:n_warm].std(0) + 1e-8
    return torch.tensor((data - mean) / std, device=device), n_warm, C
