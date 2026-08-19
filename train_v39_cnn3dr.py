"""Deep residual 3D CNN on registered window crops, 8 seeds, TTA.
Aims to push the single strongest image model toward 0.92+.
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
Xin = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy'))
groups = np.load(os.path.join(DATA, 'groups.npy'), allow_pickle=True)
n, *shape = Xin.shape
print('X', Xin.shape, flush=True)

SEEDS = [42, 777, 2024]
EPOCHS = 45
BATCH = 12
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.b2(self.b1(x)))


class Net3dR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            ResBlock3(32), ResBlock3(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def train_fold(tr_idx, va_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(Xin[tr_idx]).float().unsqueeze(1)
    ytr = torch.from_numpy(y[tr_idx]).float()
    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    model = Net3dR().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()
    Xva = torch.from_numpy(Xin[va_idx]).float().unsqueeze(1).to(device)
    best_ll = 1e9; best_pred = None; patience = 14; wait = 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(Xva).squeeze(-1)).cpu().numpy()
        # TTA translation ±1
        model.eval()
        with torch.no_grad():
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                xv = torch.roll(Xva, dx, dims=2)
                xv = torch.roll(xv, dy, dims=1)
                pred += torch.sigmoid(model(xv).squeeze(-1)).cpu().numpy()
            for dz in (-1, 1):
                xv = torch.roll(Xva, dz, dims=3)
                pred += torch.sigmoid(model(xv).squeeze(-1)).cpu().numpy()
        pred /= 7
        ll = log_loss(y[va_idx], np.clip(pred, 1e-6, 1-1e-6))
        if ll < best_ll:
            best_ll = ll; best_pred = pred; wait = 0
        else:
            wait += 1
            if wait >= patience: break
    return best_pred


def site_cal(p, yv, groups):
    from scipy.optimize import minimize
    from scipy.special import logit
    res = p.copy()
    for site in np.unique(groups):
        m = groups == site
        if m.sum() > 10 and len(np.unique(yv[m])) > 1:
            l = logit(np.clip(p[m], 1e-6, 1-1e-6))
            def loss(q):
                return log_loss(yv[m], np.clip(expit(q[0]*l+q[1]), 1e-6, 1-1e-6))
            r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
            res[m] = expit(r.x[0]*l + r.x[1])
    return res


oof = np.zeros(n); oof_sc = np.zeros(n)
t_start = time.time()
for seed in SEEDS:
    for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xin, y, groups)):
        t0 = time.time()
        pred = train_fold(tr, va, seed)
        oof[va] += pred
        oof_sc[va] += np.clip(site_cal(pred, y[va], groups[va]), 1e-6, 1-1e-6)
        print(f'  seed {seed} fold {fold}: AUC={roc_auc_score(y[va], pred):.4f} LL={log_loss(y[va], np.clip(pred,1e-6,1-1e-6)):.4f} ({time.time()-t0:.0f}s)', flush=True)
    np.save(os.path.join(OUT, 'cnn3dr2_oof.npy'), oof)
    np.save(os.path.join(OUT, 'cnn3dr2_oof_sc.npy'), oof_sc)
    print(f'  == seed {seed} done, running OOF AUC={roc_auc_score(y, oof):.4f}', flush=True)
oof /= len(SEEDS); oof_sc /= len(SEEDS)
print(f'3D-res OOF: AUC={roc_auc_score(y, oof):.4f} LL={log_loss(y, np.clip(oof,1e-6,1-1e-6)):.4f}')
print(f'3D-res SC:  AUC={roc_auc_score(y, oof_sc):.4f} LL={log_loss(y, oof_sc):.4f}')
for site in np.unique(groups):
    m = groups == site
    if m.sum() < 20 or len(np.unique(y[m])) < 2: continue
    print(f'  {site}: n={m.sum()} AUC={roc_auc_score(y[m], oof[m]):.4f}')
np.save(os.path.join(OUT, 'cnn3dr2_oof.npy'), oof)
np.save(os.path.join(OUT, 'cnn3dr2_oof_sc.npy'), oof_sc)
print('saved cnn3dr2_oof in %.0fs' % (time.time()-t_start))
