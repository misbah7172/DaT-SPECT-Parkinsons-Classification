"""PHASE 1: Establish a trustworthy OOF baseline.
ResNet10_SE architecture with clean 5-fold StratifiedKFold.
Reports: OOF log loss, AUROC, Brier, ECE, mean prediction, std, fraction>0.5.
No EMA, No DropPath, No caching, AMP enabled, BCEWithLogitsLoss, pos_weight=1.0."""
import os, sys, time, json, csv, random, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from scipy.special import expit

# ============ CONFIG ============
SEED = 42
N_FOLDS = 5
EPOCHS = 60
BATCH = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
AMP = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============ DATA ============
X = np.load(r'E:\DaT\cnn3d\X_reg.npy')
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
uids = np.load(r'E:\DaT\cnn3d\uids.npy', allow_pickle=True).ravel() if os.path.exists(r'E:\DaT\cnn3d\uids.npy') else None

print(f"X: {X.shape}, y: {y.shape}, group: {groups.shape}")
print(f"Class balance: {y.mean():.3f} ({y.sum()}/{len(y)})")
print(f"Unique groups: {len(np.unique(groups))}")

# Add channel dim
X = X[:, None].astype(np.float32)

class ScanDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        if self.y is not None:
            return self.X[i], self.y[i]
        return self.X[i]

# ============ MODEL: ResNet10_SE ============
class SEBlock3D(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
    def forward(self, x):
        return F.relu(x + self.b2(self.b1(x)))

class ResNet10_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            ResBlock3D(32),
            SEBlock3D(32),
            ResBlock3D(32),
            SEBlock3D(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 192), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(192, 1)
        )
    def forward(self, x):
        return self.head(self.enc(x)).squeeze(1)

# ============ TRAINING ============
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

def train_fold(fold_idx, trn_idx, val_idx):
    set_seed(SEED + fold_idx)
    model = ResNet10_SE().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    
    train_ds = ScanDataset(X[trn_idx], y[trn_idx])
    val_ds = ScanDataset(X[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH*2, shuffle=False, num_workers=0)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batch = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP):
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()
        
        # Validation
        model.eval()
        val_logits = []
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=AMP):
                    val_logits.append(model(xb).cpu().numpy())
        val_logits = np.concatenate(val_logits)
        val_probs = expit(val_logits)
        val_loss = -np.mean(y[val_idx] * np.log(np.clip(val_probs, 1e-7, 1-1e-7)) + 
                           (1-y[val_idx]) * np.log(np.clip(1-val_probs, 1e-7, 1-1e-7)))
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Load best and generate OOF predictions
    model.load_state_dict(best_state)
    model.eval()
    val_logits = []
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=AMP):
                val_logits.append(model(xb).cpu().numpy())
    val_logits = np.concatenate(val_logits)
    val_probs = expit(val_logits)
    return val_probs, val_logits

print(f"\n{'='*60}")
print(f"PHASE 1: OOF Baseline - ResNet10_SE, {N_FOLDS}-fold, {EPOCHS} epochs")
print(f"{'='*60}")

# Use StratifiedGroupKFold to be consistent with v26
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros(len(y))
oof_logits = np.zeros(len(y))
fold_info = []

for fold_idx, (trn_idx, val_idx) in enumerate(sgkf.split(X[:, 0], y, groups)):
    print(f"\n--- Fold {fold_idx} ---")
    t0 = time.time()
    probs, logits = train_fold(fold_idx, trn_idx, val_idx)
    elapsed = time.time() - t0
    oof_probs[val_idx] = probs
    oof_logits[val_idx] = logits
    
    fold_auc = roc_auc_score(y[val_idx], probs)
    fold_ll = log_loss(y[val_idx], np.clip(probs, 1e-7, 1-1e-7))
    fold_brier = brier_score(y[val_idx], probs)
    fold_info.append({
        'fold': fold_idx, 'n_trn': len(trn_idx), 'n_val': len(val_idx),
        'auc': fold_auc, 'll': fold_ll, 'brier': fold_brier, 'time': elapsed
    })
    print(f"  AUC={fold_auc:.4f}, LL={fold_ll:.4f}, Brier={fold_brier:.4f}, Time={elapsed:.0f}s")

# Overall metrics
oof_auc = roc_auc_score(y, oof_probs)
oof_ll = log_loss(y, np.clip(oof_probs, 1e-7, 1-1e-7))
oof_brier = brier_score(y, oof_probs)
oof_ece = expected_calibration_error(y, oof_probs)
mean_pred = oof_probs.mean()
std_pred = oof_probs.std()
frac_above = (oof_probs > 0.5).mean()

print(f"\n{'='*60}")
print(f"OOF RESULTS:")
print(f"  AUROC:            {oof_auc:.4f}")
print(f"  Log Loss:         {oof_ll:.4f}")
print(f"  Brier Score:      {oof_brier:.4f}")
print(f"  ECE:              {oof_ece:.4f}")
print(f"  Mean Prediction:  {mean_pred:.4f}")
print(f"  Pred Std:         {std_pred:.4f}")
print(f"  Fraction > 0.5:   {frac_above:.4f}")
print(f"{'='*60}")

# Save
result = {
    'auroc': oof_auc, 'log_loss': oof_ll, 'brier': oof_brier, 'ece': oof_ece,
    'mean_pred': mean_pred, 'std_pred': std_pred, 'frac_above_05': frac_above,
    'n_folds': N_FOLDS, 'epochs': EPOCHS, 'lr': LR, 'batch': BATCH,
    'amp': AMP, 'model': 'ResNet10_SE', 'seed': SEED,
    'fold_results': fold_info
}
with open(r'E:\DaT\phase1_baseline_oof.json', 'w') as f:
    json.dump(result, f, indent=2)
np.save(r'E:\DaT\phase1_oof_probs.npy', oof_probs)
np.save(r'E:\DaT\phase1_oof_logits.npy', oof_logits)
print(f"\nSaved to phase1_baseline_oof.json, phase1_oof_probs.npy, phase1_oof_logits.npy")
