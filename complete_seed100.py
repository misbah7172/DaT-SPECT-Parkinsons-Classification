"""Complete missing weights for seed 100."""
import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder

SEED = 100
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

folds = balanced_sgkf_split(y, groups, N_FOLDS, seed=42)

# Train missing: f3, f4, full
for fold_idx in [3, 4]:
    save_path = os.path.join(SAVE_DIR, f'deep_r_{SEED}_f{fold_idx}.pt')
    if os.path.exists(save_path):
        print(f"Skipping fold {fold_idx} (already exists)")
        continue
    
    tri, vai = folds[fold_idx]
    set_seed(SEED)
    model = Net3dR().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    trn_dl = DataLoader(ScanDS(X[tri], y[tri]), batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    val_dl = DataLoader(ScanDS(X[vai]), batch_size=32, shuffle=False, num_workers=0)
    
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in trn_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        scheduler.step()
    
    torch.save({'arch': 'Net3dR', 'seed': SEED, 'fold': fold_idx, 'state': model.state_dict()}, save_path)
    
    # OOF prediction
    model.eval(); vl = []
    with torch.no_grad():
        for xb, in val_dl:
            with torch.amp.autocast('cuda'):
                vl.append(model(xb.to(DEVICE)).cpu().numpy())
    probs = expit(np.concatenate(vl))
    auc = roc_auc_score(y[vai], probs)
    ll = log_loss(y[vai], np.clip(probs, 1e-7, 1-1e-7))
    print(f"  Fold {fold_idx}: AUC={auc:.4f}, LL={ll:.4f}, Time={time.time()-t0:.0f}s")

# Full data model
full_path = os.path.join(SAVE_DIR, f'deep_r_{SEED}_full.pt')
if not os.path.exists(full_path):
    set_seed(SEED)
    model = Net3dR().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    trn_dl = DataLoader(ScanDS(X, y), batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in trn_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        scheduler.step()
    
    torch.save({'arch': 'Net3dR', 'seed': SEED, 'fold': 'full', 'state': model.state_dict()}, full_path)
    print(f"  Full model: Time={time.time()-t0:.0f}s")
else:
    print(f"  Full model already exists")

print("Done!")
