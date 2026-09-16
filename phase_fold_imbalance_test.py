"""CRITICAL: Test impact of fold imbalance.
The SGKFold creates 874/318/57/56/57 split.
This experiment tests if this imbalance biases OOF estimates."""
import os, json, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import expit

SEED = 42
EPOCHS = 30
BATCH = 8
LR = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(s):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

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

X = np.load(r'E:\DaT\cnn3d\X_reg.npy')[:, None].astype(np.float32)
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
from sklearn.preprocessing import LabelEncoder
groups = LabelEncoder().fit_transform(groups_raw.astype(str))

print(f"X: {X.shape}, y: {y.shape}, groups: {groups.shape}")
print(f"Class balance: {y.mean():.3f}")
print(f"Group sizes: {np.bincount(groups)}")

def train_and_eval(split_name, train_idx, val_idx):
    set_seed(SEED)
    model = Net3dR().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    
    trn_ds = ScanDS(X[train_idx], y[train_idx])
    val_ds = ScanDS(X[val_idx])
    trn_dl = DataLoader(trn_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    best_loss = float('inf')
    best_state = None
    
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in trn_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        
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
    
    auc = roc_auc_score(y[val_idx], probs)
    ll = log_loss(y[val_idx], np.clip(probs, 1e-7, 1-1e-7))
    
    # Per-group breakdown
    val_groups = groups[val_idx]
    per_group = {}
    for g in np.unique(val_groups):
        gm = val_groups == g
        if gm.sum() > 5 and len(np.unique(y[val_idx][gm])) > 1:
            g_auc = roc_auc_score(y[val_idx][gm], probs[gm])
            per_group[int(g)] = {'auc': g_auc, 'n': int(gm.sum())}
    
    return auc, ll, per_group

print(f"\n{'='*60}")
print(f"EXPERIMENT: Fold Imbalance Impact (30 epochs, 1 fold each)")
print(f"{'='*60}")

# Strategy A: SGKFold (current - imbalanced)
print(f"\n--- A: StratifiedGroupKFold (current) ---")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
sgkf_results = []
for fi, (tri, vai) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
    auc, ll, pg = train_and_eval(f"SGKF-{fi}", tri, vai)
    sgkf_results.append({'fold': fi, 'n_trn': len(tri), 'n_val': len(vai), 'auc': auc, 'll': ll, 'groups': pg})
    print(f"  Fold {fi}: n_val={len(vai)}, groups={np.unique(groups[vai])}, AUC={auc:.4f}, LL={ll:.4f}")
    for g, info in pg.items():
        print(f"    Group {g}: AUC={info['auc']:.4f} (n={info['n']})")

sgkf_mean_auc = np.mean([r['auc'] for r in sgkf_results])
sgkf_mean_ll = np.mean([r['ll'] for r in sgkf_results])
# Weighted by fold size
sgkf_weighted_ll = sum(r['ll'] * (r['n_val']/len(y)) for r in sgkf_results)
print(f"\n  SGKFold Mean AUC: {sgkf_mean_auc:.4f}")
print(f"  SGKFold Weighted LL: {sgkf_weighted_ll:.4f}")

# Strategy B: StratifiedKFold (no grouping)
print(f"\n--- B: StratifiedKFold (no grouping, potential leakage) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_results = []
for fi, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y)):
    auc, ll, pg = train_and_eval(f"SKF-{fi}", tri, vai)
    skf_results.append({'fold': fi, 'n_trn': len(tri), 'n_val': len(vai), 'auc': auc, 'll': ll})
    print(f"  Fold {fi}: n_val={len(vai)}, AUC={auc:.4f}, LL={ll:.4f}")

skf_mean_auc = np.mean([r['auc'] for r in skf_results])
skf_weighted_ll = sum(r['ll'] * (r['n_val']/len(y)) for r in skf_results)
print(f"\n  SKFold Mean AUC: {skf_mean_auc:.4f}")
print(f"  SKFold Weighted LL: {skf_weighted_ll:.4f}")

# Strategy C: SGKFold with equal group distribution
# Manually ensure each fold gets some from dominant group
print(f"\n--- C: Balanced SGKFold ---")
# Split dominant group (7) into 5 parts
dom_mask = groups == 7
dom_idx = np.where(dom_mask)[0]
np.random.seed(42)
np.random.shuffle(dom_idx)
dom_parts = np.array_split(dom_idx, 5)

other_idx = np.where(~dom_mask)[0]
np.random.seed(42)
np.random.shuffle(other_idx)
other_parts = np.array_split(other_idx, 5)

bal_results = []
for fi in range(5):
    val_idx = np.concatenate([dom_parts[fi], other_parts[fi]])
    trn_idx = np.concatenate([np.concatenate([dom_parts[i] for i in range(5) if i != fi]),
                               np.concatenate([other_parts[i] for i in range(5) if i != fi])])
    auc, ll, pg = train_and_eval(f"BAL-{fi}", trn_idx, val_idx)
    bal_results.append({'fold': fi, 'n_trn': len(trn_idx), 'n_val': len(val_idx), 'auc': auc, 'll': ll, 'groups': pg})
    print(f"  Fold {fi}: n_val={len(val_idx)}, AUC={auc:.4f}, LL={ll:.4f}")

bal_mean_auc = np.mean([r['auc'] for r in bal_results])
bal_weighted_ll = sum(r['ll'] * (r['n_val']/len(y)) for r in bal_results)
print(f"\n  Balanced SGKFold Mean AUC: {bal_mean_auc:.4f}")
print(f"  Balanced SGKFold Weighted LL: {bal_weighted_ll:.4f}")

print(f"\n{'='*60}")
print(f"COMPARISON:")
print(f"  SGKFold (current):  AUC={sgkf_mean_auc:.4f}, Weighted LL={sgkf_weighted_ll:.4f}")
print(f"  SKFold (leakage):   AUC={skf_mean_auc:.4f}, Weighted LL={skf_weighted_ll:.4f}")
print(f"  Balanced SGKFold:   AUC={bal_mean_auc:.4f}, Weighted LL={bal_weighted_ll:.4f}")
print(f"  Current v26 (60 models): AUC=0.9382, LL=0.3077")
print(f"{'='*60}")

result = {
    'sgkf': {'mean_auc': sgkf_mean_auc, 'weighted_ll': sgkf_weighted_ll, 'folds': sgkf_results},
    'skf': {'mean_auc': skf_mean_auc, 'weighted_ll': skf_weighted_ll, 'folds': skf_results},
    'balanced': {'mean_auc': bal_mean_auc, 'weighted_ll': bal_weighted_ll, 'folds': bal_results},
}
with open(r'E:\DaT\phase_fold_imbalance.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"Saved to phase_fold_imbalance.json")
