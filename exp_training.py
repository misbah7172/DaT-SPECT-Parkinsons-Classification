"""Train with mixup augmentation + more epochs (60ep) + focal loss on balanced folds.
Tests whether training improvements help beyond calibration."""
import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder

SEED = 42
EPOCHS = 60
BATCH = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

X = np.load(r'E:\DaT\cnn3d\X_reg.npy')[:, None].astype(np.float32)
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
groups = LabelEncoder().fit_transform(groups_raw.astype(str))

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
        self.y = torch.from_numpy(y).float() if y is not None else None
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
        folds.append((all_idx[~val_mask], all_idx[val_mask]))
    return folds

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def focal_bce_loss(logits, targets, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    focal = ((1 - pt) ** gamma) * bce
    return focal.mean()

folds = balanced_sgkf_split(y, groups, 5, seed=42)

configs = [
    {'name': '60ep_bce', 'loss': 'bce', 'mixup': False, 'epochs': 60},
    {'name': '60ep_mixup', 'loss': 'bce', 'mixup': True, 'epochs': 60},
    {'name': '60ep_focal', 'loss': 'focal', 'mixup': False, 'epochs': 60},
    {'name': '60ep_mixup_focal', 'loss': 'focal', 'mixup': True, 'epochs': 60},
]

results = {}
for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Config: {cfg['name']}")
    print(f"{'='*60}")
    
    oof = np.zeros(len(y))
    for fi, (tri, vai) in enumerate(folds):
        set_seed(SEED)
        model = Net3dR().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
        scaler = torch.amp.GradScaler('cuda')
        
        trn_dl = DataLoader(ScanDS(X[tri], y[tri]), batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
        val_dl = DataLoader(ScanDS(X[vai]), batch_size=32, shuffle=False, num_workers=0)
        
        best_loss = float('inf')
        best_state = None
        
        for ep in range(cfg['epochs']):
            model.train()
            for xb, yb in trn_dl:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                
                if cfg['mixup']:
                    xb_m, yb_a, yb_b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        logits = model(xb_m)
                        if cfg['loss'] == 'focal':
                            loss = lam * focal_bce_loss(logits, yb_a) + (1-lam) * focal_bce_loss(logits, yb_b)
                        else:
                            loss = lam * F.binary_cross_entropy_with_logits(logits, yb_a) + (1-lam) * F.binary_cross_entropy_with_logits(logits, yb_b)
                else:
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        logits = model(xb)
                        if cfg['loss'] == 'focal':
                            loss = focal_bce_loss(logits, yb)
                        else:
                            loss = F.binary_cross_entropy_with_logits(logits, yb)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            
            # Validate
            model.eval()
            vl = []
            with torch.no_grad():
                for xb, in val_dl:
                    with torch.amp.autocast('cuda'):
                        vl.append(model(xb.to(DEVICE)).cpu().numpy())
            vp = expit(np.concatenate(vl))
            vloss = -np.mean(y[vai]*np.log(np.clip(vp,1e-7,1-1e-7)) + (1-y[vai])*np.log(np.clip(1-vp,1e-7,1-1e-7)))
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
        oof[vai] = probs
        
        fold_auc = roc_auc_score(y[vai], probs)
        fold_ll = log_loss(y[vai], np.clip(probs, 1e-7, 1-1e-7))
        print(f"  Fold {fi}: AUC={fold_auc:.4f}, LL={fold_ll:.4f}")
    
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
    results[cfg['name']] = {'auc': auc, 'll': ll}
    print(f"  {cfg['name']}: AUC={auc:.4f}, LL={ll:.4f}")
    
    # Save for ensemble
    np.save(rf'E:\DaT\exp_oof_{cfg["name"]}.npy', oof)

# Compare with baseline
print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
baseline_ll = log_loss(y, np.clip(np.load(r'E:\DaT\balanced_oof.npy'), 1e-7, 1-1e-7))
print(f"  Baseline (30ep, no mixup, bce): LL={baseline_ll:.4f}")
for name, res in results.items():
    print(f"  {name:25s}: LL={res['ll']:.4f}, AUC={res['auc']:.4f}")
