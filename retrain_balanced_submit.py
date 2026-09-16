"""Retrain with balanced folds and SAVE fold weights for submission.
Uses 4 seeds × 5 folds = 20 fold weights + 4 full-data weights = 24 total."""
import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder

SEEDS = [42, 777, 2024, 100]
EPOCHS = 30
BATCH = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = r'E:\DaT\submission_v27\weights'

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

X = np.load(r'E:\DaT\cnn3d\X_reg.npy')[:, None].astype(np.float32)
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
le = LabelEncoder()
groups = le.fit_transform(groups_raw.astype(str))

class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.BatchNorm3d(c))
    def forward(self, x): return F.relu(x + self.b2(self.b1(x)))

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
    def forward(self, x): return self.head(self.net(x)).squeeze(1)

class ScanDS(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else (self.X[i],)

def balanced_sgkf_split(y, groups, n_folds=5, seed=42):
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
    folds = []
    for fi in range(n_folds):
        val_mask = fold_idx == fi
        trn_mask = ~val_mask
        folds.append((all_idx[trn_mask], all_idx[val_mask]))
    return folds

def train_and_save(trn_idx, val_idx, seed, fold, save_path):
    set_seed(seed)
    model = Net3dR().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    trn_ds = ScanDS(X[trn_idx], y[trn_idx])
    trn_dl = DataLoader(trn_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in trn_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        scheduler.step()
    
    # Save fold weight
    torch.save({'arch': 'Net3dR', 'seed': seed, 'fold': fold,
                'state': model.state_dict()}, save_path)
    
    # Also generate OOF prediction
    if val_idx is not None:
        model.eval()
        val_ds = ScanDS(X[val_idx])
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        vl = []
        with torch.no_grad():
            for xb, in val_dl:
                with torch.amp.autocast('cuda'):
                    vl.append(model(xb.to(DEVICE)).cpu().numpy())
        return expit(np.concatenate(vl))
    return None

def train_full_data(seed, save_path):
    """Train on full data for submission (no OOF)."""
    return train_and_save(np.arange(len(y)), None, seed, 'full', save_path)

folds = balanced_sgkf_split(y, groups, N_FOLDS, seed=42)

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Training {len(SEEDS)} seeds × {N_FOLDS} folds + {len(SEEDS)} full-data models")
print(f"Save dir: {SAVE_DIR}")

all_oof = np.zeros((len(y), len(SEEDS)))

for si, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({si+1}/{len(SEEDS)}) ---")
    seed_oof = np.zeros(len(y))
    t0 = time.time()
    
    for fi, (tri, vai) in enumerate(folds):
        # Save fold weight
        fold_path = os.path.join(SAVE_DIR, f'deep_r_{seed}_f{fi}.pt')
        probs = train_and_save(tri, vai, seed, fi, fold_path)
        seed_oof[vai] = probs
        fold_auc = roc_auc_score(y[vai], probs)
        fold_ll = log_loss(y[vai], np.clip(probs, 1e-7, 1-1e-7))
        print(f"  Fold {fi}: AUC={fold_auc:.4f}, LL={fold_ll:.4f}")
    
    # Save full-data model
    full_path = os.path.join(SAVE_DIR, f'deep_r_{seed}_full.pt')
    train_full_data(seed, full_path)
    
    all_oof[:, si] = seed_oof
    seed_auc = roc_auc_score(y, seed_oof)
    seed_ll = log_loss(y, np.clip(seed_oof, 1e-7, 1-1e-7))
    print(f"  Seed {seed}: AUC={seed_auc:.4f}, LL={seed_ll:.4f}, Time={time.time()-t0:.0f}s")

# Save OOF
mean_oof = all_oof.mean(axis=1)
np.save(os.path.join(SAVE_DIR, 'fold_oof.npy'), mean_oof)
for si, seed in enumerate(SEEDS):
    np.save(os.path.join(SAVE_DIR, f'fold_oof_r_{seed}.npy'), all_oof[:, si])

# Blend with SBR
import pickle
from scipy.special import logit
from scipy.optimize import minimize

with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names_arr, oofs_arr = A['names'], np.column_stack(A['oof'])
sbr_idx = [i for i, n in enumerate(names_arr) if 'sbr' in n.lower()]
sbr_oof = oofs_arr[:, sbr_idx].mean(axis=1)

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

ship = {
    'w_deep': round(float(wd), 4),
    'w_sbr': round(float(1-wd), 4),
    'T': round(float(T), 4),
    'deep_auc': float(roc_auc_score(y, mean_oof)),
    'deep_ll': float(log_loss(y, np.clip(mean_oof, 1e-7, 1-1e-7))),
    'sbr_auc': float(roc_auc_score(y, sbr_oof)),
    'sbr_ll': float(log_loss(y, np.clip(sbr_oof, 1e-7, 1-1e-7))),
    'final_auc': float(blend_auc),
    'final_ll': float(blend_ll),
    'n_seeds': len(SEEDS),
    'n_folds': N_FOLDS,
    'epochs': EPOCHS,
    'balanced': True,
}
with open(os.path.join(SAVE_DIR, 'ship_final.json'), 'w') as f:
    json.dump(ship, f, indent=2)

# Copy SBR weights
import shutil
for f in ['sbr_full_lr.pkl', 'sbr_full_ridge.pkl', 'sbr_full_et.pkl', 'sbr_full_xgb.pkl', 'sbr_full_lgb.pkl',
          'sbr_full_scaler.pkl', 'sbr_full_sel.pkl', 'sbr_full_cols.json']:
    src = os.path.join(r'E:\DaT\submission_v26\weights', f)
    dst = os.path.join(SAVE_DIR, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)

print(f"\n{'='*70}")
print(f"FINAL RESULTS:")
print(f"  Deep OOF: AUC={roc_auc_score(y, mean_oof):.4f}, LL={log_loss(y, np.clip(mean_oof,1e-7,1-1e-7)):.4f}")
print(f"  Blend:    AUC={blend_auc:.4f}, LL={blend_ll:.4f}")
print(f"  v26:      AUC=0.9382, LL=0.3077")
print(f"  Improvement: {0.3077 - blend_ll:.4f} LL")
print(f"\n  Saved {len(SEEDS)*N_FOLDS + len(SEEDS)} weights to {SAVE_DIR}")
print(f"{'='*70}")
