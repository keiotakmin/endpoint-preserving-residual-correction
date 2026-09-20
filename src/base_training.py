"""Training functions extracted unchanged from the evaluated harness."""
import numpy as np
import torch
import torch.nn.functional as F
from backbones import build_model
WARM_GRID = [200,500,1000,2000,4000,8000,20000]

def warmup_model(backbone, L, H, C, d, n_warm, warmup_steps, lr=1e-3, device="cuda", bs=32):
    model = build_model(backbone, L, H, C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for _ in range(warmup_steps):
        ii = np.random.randint(L, n_warm - H, size=bs)
        x = torch.stack([d[i - L:i] for i in ii]); y = torch.stack([d[i:i + H] for i in ii])
        opt.zero_grad(); F.mse_loss(model(x), y).backward(); opt.step()
    return model

def val_mse(model, d, a, b, L, H):
    """Static MSE over non-overlapping windows in [a, b] (the held-out pre-drift validation slice)."""
    errs, t = [], a + L
    while t + H <= b:
        with torch.no_grad():
            errs.append(F.mse_loss(model(d[t - L:t].unsqueeze(0)), d[t:t + H].unsqueeze(0)).item())
        t += H
    return float(np.mean(errs))

def warm_and_select(backbone, L, H, C, d, n_train, n_warm, seed, warm_grid=None, lr=1e-3, bs=32):
    """FAIR warmup = the C1 deployable protocol: train the base model on the TRAIN region only,
    checkpoint at each grid milestone, return the min-held-out-validation checkpoint
    (+ its warmup step and val MSE). Shared by combined_grid / frontier / staleness so every
    downstream figure reads the baseline at the same fair warmup."""
    import copy
    warm_grid = WARM_GRID if warm_grid is None else warm_grid
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(backbone, L, H, C).to(d.device); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, best_step = float("inf"), None, warm_grid[0]
    for step in range(1, max(warm_grid) + 1):
        ii = np.random.randint(L, n_train - H, size=bs)
        x = torch.stack([d[i - L:i] for i in ii]); y = torch.stack([d[i:i + H] for i in ii])
        opt.zero_grad(); F.mse_loss(model(x), y).backward(); opt.step()
        if step in warm_grid:
            v = val_mse(model, d, n_train, n_warm, L, H)
            if v < best_val:
                best_val, best_state, best_step = v, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best_state)
    return model, best_step, best_val
