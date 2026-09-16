"""Retrain CNN with Balanced SGKFold.
Distributes dominant group across all folds for unbiased OOF estimates.
Uses same Net3dR architecture, seeds, and training protocol as v26.
Reports per-fold and overall OOF metrics."""
import os, sys, time, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder

# ============ CONFIG ============
SEEDS = [42, 777]
EPOCHS = 30
BATCH = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ DATA ============
X = np.load(r'E:\DaT\cnn3d\X_reg.npy')[:, None].astype(np.float32)
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
le = LabelEncoder()
groups = le.fit_transform(groups_raw.astype(str))

print(f"X: {X.shape}, y: {y.shape}, groups: {groups.shape}")
print(f"Class balance: {y.mean():.3f} ({y.sum()}/{len(y)})")
for g in np.unique(groups):
    gm = groups == g
    print(f"  Group {g} ({le.inverse_transform([g])[0]}): n={gm.sum()}, abnormal={y[gm].sum()}")

# ============ MODEL ============
class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.BatchNorm3d(c))
    def forward(self, x):
        return F.relu(x + self.b2(self.b1(x)))

class Net3dR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            ResBlock3D(32), ResBlock3D(32),
            nn.Conv3d(32,64,3,padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 192), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)

class ScanDS(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else (self.X[i],)

# ============ BALANCED SPLIT ============
def balanced_sgkf_split(y, groups, n_folds=5, seed=42):
    """Create balanced folds by distributing each group evenly."""
    rng = np.random.RandomState(seed)
    all_idx = np.arange(len(y))
    fold_idx = np.full(len(y), -1, dtype=int)
    
    for g in np.unique(groups):
        g_mask = groups == g
        g_idx = all_idx[g_mask]
        rng.shuffle(g_idx)
        parts = np.array_split(g_idx, n_folds)
        for fi, part in enumerate(parts):
            fold_idx[part] = fi
    
    # Create (trn, val) tuples
    folds = []
    for fi in range(n_folds):
        val_mask = fold_idx == fi
        trn_mask = ~val_mask
        folds.append((all_idx[trn_mask], all_idx[val_mask]))
    
    return folds

# ============ TRAINING ============
def train_fold(fold_idx, trn_idx, val_idx, seed):
    set_seed(seed)
    model = Net3dR().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    
    trn_ds = ScanDS(X[trn_idx], y[trn_idx])
    val_ds = ScanDS(X[val_idx])
    trn_dl = DataLoader(trn_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH*2, shuffle=False, num_workers=0)
    
    best_loss = float('inf')
    best_state = None
    
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in trn_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        
        model.eval()
        vl = []
        with torch.no_grad():
            for xb, in val_dl:
                with torch.amp.autocast('cuda'):
                    vl.append(model(xb.to(DEVICE)).cpu().numpy())
        vl = np.concatenate(vl)
        vp = expit(vl)
        vloss = -np.mean(y[val_idx]*np.log(np.clip(vp,1e-7,1-1e-7)) + (1-y[val_idx])*np.log(np.clip(1-vp,1e-7,1-1e-7)))
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    model.eval()
    vl = []
    with torch.no_grad():
        for xb, in val_dl:
            with torch.amp.autocast('cuda'):
                vl.append(model(xb.to(DEVICE)).cpu().numpy())
    probs = expit(np.concatenate(vl))
    return probs

def brier_score(y_true, y_prob):
    return np.mean((y_true - y_prob) ** 2)

def expected_calibration_error(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_true) * abs(acc - conf)
    return ece

# ============ MAIN ============
print(f"\n{'='*70}")
print(f"BALANCED SGKFOLD RETRAIN")
print(f"Seeds: {SEEDS}, Epochs: {EPOCHS}, Folds: {N_FOLDS}")
print(f"{'='*70}")

folds = balanced_sgkf_split(y, groups, N_FOLDS, seed=42)
for fi, (tri, vai) in enumerate(folds):
    print(f"Fold {fi}: trn={len(tri)}, val={len(vai)}, val_groups={np.unique(groups[vai])}, "
          f"val_abnormal={y[vai].sum()}/{len(y[vai])}")

all_oof = np.zeros((len(y), len(SEEDS)))
fold_info = []

for si, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({si+1}/{len(SEEDS)}) ---")
    seed_oof = np.zeros(len(y))
    t0 = time.time()
    
    for fi, (trn_idx, val_idx) in enumerate(folds):
        probs = train_fold(fi, trn_idx, val_idx, seed)
        seed_oof[val_idx] = probs
        fold_auc = roc_auc_score(y[val_idx], probs)
        fold_ll = log_loss(y[val_idx], np.clip(probs, 1e-7, 1-1e-7))
        fold_info.append({
            'seed': seed, 'fold': fi, 'n_trn': len(trn_idx), 'n_val': len(val_idx),
            'auc': fold_auc, 'll': fold_ll,
            'groups': [le.inverse_transform([g])[0] for g in np.unique(groups[val_idx])]
        })
        print(f"  Fold {fi}: AUC={fold_auc:.4f}, LL={fold_ll:.4f}")
    
    all_oof[:, si] = seed_oof
    seed_auc = roc_auc_score(y, seed_oof)
    seed_ll = log_loss(y, np.clip(seed_oof, 1e-7, 1-1e-7))
    elapsed = time.time() - t0
    print(f"  Seed {seed} overall: AUC={seed_auc:.4f}, LL={seed_ll:.4f}, Time={elapsed:.0f}s")
    # Incremental save
    np.save(r'E:\DaT\balanced_oof_partial.npy', all_oof[:, :si+1])

# ============ ENSEMBLE ============
print(f"\n{'='*70}")
print(f"ENSEMBLE RESULTS")
print(f"{'='*70}")

# Individual seeds
for si, seed in enumerate(SEEDS):
    auc = roc_auc_score(y, all_oof[:, si])
    ll = log_loss(y, np.clip(all_oof[:, si], 1e-7, 1-1e-7))
    print(f"  Seed {seed}: AUC={auc:.4f}, LL={ll:.4f}")

# Mean ensemble
mean_oof = all_oof.mean(axis=1)
mean_auc = roc_auc_score(y, mean_oof)
mean_ll = log_loss(y, np.clip(mean_oof, 1e-7, 1-1e-7))
mean_brier = brier_score(y, mean_oof)
mean_ece = expected_calibration_error(y, mean_oof)
print(f"\n  Mean ensemble ({len(SEEDS)} seeds):")
print(f"    AUC:              {mean_auc:.4f}")
print(f"    Log Loss:         {mean_ll:.4f}")
print(f"    Brier Score:      {mean_brier:.4f}")
print(f"    ECE:              {mean_ece:.4f}")
print(f"    Mean Prediction:  {mean_oof.mean():.4f}")
print(f"    Pred Std:         {mean_oof.std():.4f}")
print(f"    Fraction > 0.5:   {(mean_oof > 0.5).mean():.4f}")

# Compare with v26
print(f"\n  Comparison with v26:")
print(f"    v26 (60 models + SBR): AUC=0.9382, LL=0.3077")
print(f"    Balanced ({len(SEEDS)} seeds): AUC={mean_auc:.4f}, LL={mean_ll:.4f}")
if mean_ll < 0.3077:
    print(f"    >>> IMPROVEMENT: {0.3077 - mean_ll:.4f} LL reduction <<<")

# ============ PER-GROUP ANALYSIS ============
print(f"\n  Per-group OOF AUC:")
for g in np.unique(groups):
    gm = groups == g
    if gm.sum() > 10 and len(np.unique(y[gm])) > 1:
        g_auc = roc_auc_score(y[gm], mean_oof[gm])
        g_ll = log_loss(y[gm], np.clip(mean_oof[gm], 1e-7, 1-1e-7))
        g_name = le.inverse_transform([g])[0]
        print(f"    {g_name:>15s} (n={gm.sum():>4d}): AUC={g_auc:.4f}, LL={g_ll:.4f}")

# ============ BLEND WITH SBR ============
print(f"\n{'='*70}")
print(f"BLEND WITH SBR")
print(f"{'='*70}")

try:
    with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
        A = pickle.load(f)
    names_arr, oofs_arr = A['names'], np.column_stack(A['oof'])
    sbr_idx = [i for i, n in enumerate(names_arr) if 'sbr' in n.lower()]
    if sbr_idx:
        sbr_oof = oofs_arr[:, sbr_idx].mean(axis=1)
        
        from scipy.special import logit
        from scipy.optimize import minimize
        
        def blend_loss(params):
            wd, T = params
            b = np.clip(wd * mean_oof + (1-wd) * sbr_oof, 1e-7, 1-1e-7)
            p = np.clip(expit(logit(b) / T), 1e-7, 1-1e-7)
            return log_loss(y, p)
        
        best = None
        for init in ([0.8,0.7],[0.85,0.8],[0.75,0.6],[0.9,0.5],[0.82,0.65]):
            r = minimize(blend_loss, init, method='Nelder-Mead', options={'maxiter':500})
            if best is None or r.fun < best.fun: best = r
        
        wd, T = best.x
        bp = np.clip(expit(logit(np.clip(wd*mean_oof+(1-wd)*sbr_oof,1e-7,1-1e-7))/T), 1e-7, 1-1e-7)
        blend_auc = roc_auc_score(y, bp)
        blend_ll = log_loss(y, bp)
        print(f"  w_deep={wd:.4f}, w_sbr={1-wd:.4f}, T={T:.4f}")
        print(f"  Blend AUC={blend_auc:.4f}, LL={blend_ll:.4f}")
        print(f"  v26 blend:         AUC=0.9382, LL=0.3077")
        if blend_ll < 0.3077:
            print(f"  >>> IMPROVEMENT: {0.3077 - blend_ll:.4f} LL reduction <<<")
except Exception as e:
    print(f"  Could not load SBR OOF: {e}")

import pickle

# Save results
result = {
    'mean_auc': mean_auc, 'mean_ll': mean_ll, 'mean_brier': mean_brier, 'mean_ece': mean_ece,
    'n_seeds': len(SEEDS), 'n_folds': N_FOLDS, 'epochs': EPOCHS,
    'seeds': SEEDS, 'fold_info': fold_info,
}
with open(r'E:\DaT\balanced_retrain_results.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
np.save(r'E:\DaT\balanced_oof.npy', mean_oof)
np.save(r'E:\DaT\balanced_oof_all_seeds.npy', all_oof)
print(f"\nSaved balanced_retrain_results.json, balanced_oof.npy, balanced_oof_all_seeds.npy")
