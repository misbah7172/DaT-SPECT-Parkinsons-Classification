"""Train 2 more seeds (2024, 100) and combine with existing 2 seeds."""
import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder

SEEDS = [2024, 100]
EPOCHS = 30
BATCH = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    best_loss = float('inf'); best_state = None
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in trn_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        scheduler.step()
        model.eval(); vl = []
        with torch.no_grad():
            for xb, in val_dl:
                with torch.amp.autocast('cuda'):
                    vl.append(model(xb.to(DEVICE)).cpu().numpy())
        vl = np.concatenate(vl); vp = expit(vl)
        vloss = -np.mean(y[val_idx]*np.log(np.clip(vp,1e-7,1-1e-7)) + (1-y[val_idx])*np.log(np.clip(1-vp,1e-7,1-1e-7)))
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval(); vl = []
    with torch.no_grad():
        for xb, in val_dl:
            with torch.amp.autocast('cuda'):
                vl.append(model(xb.to(DEVICE)).cpu().numpy())
    return expit(np.concatenate(vl))

def brier_score(y_true, y_prob):
    return np.mean((y_true - y_prob) ** 2)
def expected_calibration_error(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece

folds = balanced_sgkf_split(y, groups, N_FOLDS, seed=42)

# Load existing seeds
existing_oof = np.load(r'E:\DaT\balanced_oof_all_seeds.npy')  # (1362, 2)
print(f"Existing OOF: {existing_oof.shape}")

new_oof = np.zeros((len(y), len(SEEDS)))

for si, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ---")
    seed_oof = np.zeros(len(y))
    t0 = time.time()
    for fi, (tri, vai) in enumerate(folds):
        probs = train_fold(fi, tri, vai, seed)
        seed_oof[vai] = probs
        print(f"  Fold {fi}: AUC={roc_auc_score(y[vai], probs):.4f}, LL={log_loss(y[vai], np.clip(probs,1e-7,1-1e-7)):.4f}")
    new_oof[:, si] = seed_oof
    print(f"  Seed {seed} overall: AUC={roc_auc_score(y, seed_oof):.4f}, LL={log_loss(y, np.clip(seed_oof,1e-7,1-1e-7)):.4f}, Time={time.time()-t0:.0f}s")

# Combine all 4 seeds
all_oof = np.concatenate([existing_oof, new_oof], axis=1)
mean_oof = all_oof.mean(axis=1)
print(f"\n{'='*70}")
print(f"4-SEED BALANCED ENSEMBLE:")
print(f"  AUC:     {roc_auc_score(y, mean_oof):.4f}")
print(f"  LL:      {log_loss(y, np.clip(mean_oof,1e-7,1-1e-7)):.4f}")
print(f"  Brier:   {brier_score(y, mean_oof):.4f}")
print(f"  ECE:     {expected_calibration_error(y, mean_oof):.4f}")

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
for init in ([0.8,0.7],[0.85,0.8],[0.75,0.6],[0.9,0.5],[0.82,0.65],[0.95,0.3],[0.7,0.9]):
    r = minimize(blend_loss, init, method='Nelder-Mead', options={'maxiter':500})
    if best is None or r.fun < best.fun: best = r

wd, T = best.x
bp = np.clip(expit(logit(np.clip(wd*mean_oof+(1-wd)*sbr_oof,1e-7,1-1e-7))/T), 1e-7, 1-1e-7)
print(f"\n  Blend: w_deep={wd:.4f}, w_sbr={1-wd:.4f}, T={T:.4f}")
print(f"  Blend AUC={roc_auc_score(y, bp):.4f}, LL={log_loss(y, bp):.4f}")
print(f"\n  v26:       AUC=0.9382, LL=0.3077")
print(f"  2-seed:    AUC=0.9456, LL=0.2926")
print(f"  4-seed:    AUC={roc_auc_score(y, bp):.4f}, LL={log_loss(y, bp):.4f}")

# Save
np.save(r'E:\DaT\balanced_oof_all_seeds.npy', all_oof)
np.save(r'E:\DaT\balanced_oof.npy', mean_oof)
result = {
    'auroc': float(roc_auc_score(y, mean_oof)),
    'log_loss': float(log_loss(y, np.clip(mean_oof,1e-7,1-1e-7))),
    'blend_auroc': float(roc_auc_score(y, bp)),
    'blend_log_loss': float(log_loss(y, bp)),
    'w_deep': float(wd), 'w_sbr': float(1-wd), 'T': float(T),
    'n_seeds': 4, 'epochs': EPOCHS, 'balanced': True,
}
with open(r'E:\DaT\balanced_retrain_results.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved balanced_retrain_results.json, balanced_oof.npy, balanced_oof_all_seeds.npy")
