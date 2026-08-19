"""3D CNN on fixed atlas-window crops with translation augmentation.
The atlas fixed frame tolerates ~2-4 voxel striatum shifts; augmentation handles residual.
Group-stratified 5-fold OOF across 3 seeds; saves OOF + test-average preds.
"""
import os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy.special import expit

DATA = r'E:\DaT\cnn3d'
OUT = r'E:\DaT\v30_oof'
Xin = np.load(os.path.join(DATA, 'X_reg.npy')) if 'REG' in os.environ else np.load(os.path.join(DATA, 'X_in.npy'))
y = np.load(os.path.join(DATA, 'y.npy'))
groups = np.load(os.path.join(DATA, 'groups.npy'), allow_pickle=True)
n, *shape = Xin.shape
print('X', Xin.shape, 'input', 'REG' if 'REG' in os.environ else 'INORM', 'y mean', y.mean(), flush=True)

SEEDS = [42, 777, 2024, 12345, 999] if 'REG' in os.environ else [42, 777, 2024]
EPOCHS = 60
BATCH = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)


class Conv3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 5 * 5, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def translate_aug(x, max_shift=3):
    """Random translation within ±max_shift to absorb residual misalignment."""
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    dz = np.random.randint(-max_shift, max_shift + 1)
    if (dx, dy, dz) == (0, 0, 0):
        return x
    import scipy.ndimage as ndi
    return ndi.shift(x, (dx, dy, dz), order=1, mode='constant', cval=0.0)


def train_fold(tr_idx, va_idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr = torch.from_numpy(Xin[tr_idx]).float().unsqueeze(1)
    ytr = torch.from_numpy(y[tr_idx]).float()
    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    model = Conv3d().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()
    Xva = torch.from_numpy(Xin[va_idx]).float().unsqueeze(1).to(device)
    best_ll = 1e9; best_pred = None; patience = 12; wait = 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb = np.array([translate_aug(x.squeeze(0).numpy()) for x in xb])[:, None, :, :, :]
            xb = torch.from_numpy(xb.astype(np.float32)).to(device)
            yb = yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(Xva).squeeze(-1)).cpu().numpy()
        ll = log_loss(y[va_idx], np.clip(pred, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_ll = ll; best_pred = pred; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_pred


def site_cal(p, y, groups, tr_idx):
    from scipy.optimize import minimize
    from scipy.special import logit
    res = p.copy()
    eps = 1e-6
    for site in np.unique(groups):
        m = groups == site
        if m.sum() > 10 and len(np.unique(y[m])) > 1:
            l = logit(np.clip(p[m], eps, 1-eps))
            def loss(q):
                return log_loss(y[m], np.clip(expit(q[0]*l + q[1]), eps, 1-eps))
            r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
            res[m] = expit(r.x[0] * l + r.x[1])
    return res


oof = np.zeros(n)
oof_sc = np.zeros(n)
for seed in SEEDS:
    for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xin, y, groups)):
        t0 = time.time()
        pred = train_fold(tr, va, seed)
        oof[va] += pred
        oof_sc[va] += np.clip(site_cal(pred, y[va], groups[va], tr), 1e-6, 1-1e-6)
        print(f'  seed {seed} fold {fold}: AUC={roc_auc_score(y[va], pred):.4f} LL={log_loss(y[va], np.clip(pred,1e-6,1-1e-6)):.4f} ({time.time()-t0:.0f}s)', flush=True)
oof /= len(SEEDS)
oof_sc /= len(SEEDS)
print(f'3DCNN OOF: AUC={roc_auc_score(y, oof):.4f} LL={log_loss(y, np.clip(oof,1e-6,1-1e-6)):.4f}')
print(f'3DCNN SC:  AUC={roc_auc_score(y, oof_sc):.4f} LL={log_loss(y, oof_sc):.4f}')
for site in np.unique(groups):
    m = groups == site
    if m.sum() < 20 or len(np.unique(y[m])) < 2: continue
    print(f'  {site}: n={m.sum()} AUC={roc_auc_score(y[m], oof[m]):.4f}')
name = 'cnn3dr_oof' if 'REG' in os.environ else 'cnn3d_oof'
name_sc = 'cnn3dr_oof_sc' if 'REG' in os.environ else 'cnn3d_oof_sc'
np.save(os.path.join(OUT, name), oof)
np.save(os.path.join(OUT, name_sc), oof_sc)
print('saved', name)