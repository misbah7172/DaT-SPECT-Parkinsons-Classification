"""Variant of strong 2.5D with DIFFERENT slice offsets (decorrelated views) on registered
crops, 8 seeds. Adds diversity to the CNN ensemble.
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
n = len(y)

SEEDS = [42, 777, 2024, 12345, 999, 100, 200, 300]
EPOCHS = 60
BATCH = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_slices(X):
    rez = []
    axial = np.stack([X[:, :, :, i:i+3].sum(3) for i in range(1, 19, 3)])
    rez.append(axial.transpose(1, 2, 3, 0))
    cor = np.stack([X[:, :, i:i+3, :].sum(2) for i in range(8, 32, 4)])
    rez.append(cor.transpose(1, 2, 3, 0))
    sag = np.stack([X[:, i:i+3, :, :].sum(1) for i in range(4, 28, 4)])
    rez.append(sag.transpose(1, 2, 3, 0))
    return rez


slices = make_slices(Xin)
shapes = [s.shape[1:] for s in slices]
print('branch shapes:', shapes, flush=True)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.b2(self.b1(x)))


class Net25s(nn.Module):
    def __init__(self, shape):
        super().__init__()
        c, h, w = shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
            ResBlock(48), ResBlock(48),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96*3*3, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def train_fold(tr_idx, va_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dl_data = [(torch.from_numpy(s[tr_idx]).float(), torch.from_numpy(s[va_idx]).float().to(device)) for s in slices]
    ytr = torch.from_numpy(y[tr_idx]).float()
    ds = TensorDataset(torch.arange(len(tr_idx)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    models = []
    for sh in shapes:
        m = Net25s(sh).to(device)
        o = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(o, T_max=EPOCHS)
        models.append((m, o, sched))
    crit = nn.BCEWithLogitsLoss()
    Xva_list = [d[1] for d in dl_data]
    best_ll = 1e9; best_pred = None; patience = 14; wait = 0
    for ep in range(EPOCHS):
        for (m, o, sch), (Xtr, Xva) in zip(models, dl_data):
            m.train()
            for idx in dl:
                xb = Xtr[idx].to(device)
                yb = ytr[idx].to(device)
                o.zero_grad()
                crit(m(xb), yb).backward()
                o.step()
        for (m, o, sch) in models:
            sch.step()
        preds = []
        for (m, _, _), Xva in zip(models, Xva_list):
            m.eval()
            with torch.no_grad():
                base = torch.sigmoid(m(Xva).squeeze(-1)).cpu().numpy()
            tta = [base]
            m.eval()
            with torch.no_grad():
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    xv = torch.roll(Xva, dx, dims=2)
                    xv = torch.roll(xv, dy, dims=1)
                    tta.append(torch.sigmoid(m(xv).squeeze(-1)).cpu().numpy())
            preds.append(np.mean(tta, axis=0))
        pred = np.mean(preds, axis=0)
        ll = log_loss(y[va_idx], np.clip(pred, 1e-6, 1-1e-6))
        if ll < best_ll:
            best_ll = ll; best_pred = pred; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
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
    for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(slices[0], y, groups)):
        t0 = time.time()
        pred = train_fold(tr, va, seed)
        oof[va] += pred
        oof_sc[va] += np.clip(site_cal(pred, y[va], groups[va]), 1e-6, 1-1e-6)
        print(f'  seed {seed} fold {fold}: AUC={roc_auc_score(y[va], pred):.4f} LL={log_loss(y[va], np.clip(pred,1e-6,1-1e-6)):.4f} ({time.time()-t0:.0f}s)', flush=True)
oof /= len(SEEDS); oof_sc /= len(SEEDS)
print(f'2.5D-v2 OOF: AUC={roc_auc_score(y, oof):.4f} LL={log_loss(y, np.clip(oof,1e-6,1-1e-6)):.4f}')
print(f'2.5D-v2 SC:  AUC={roc_auc_score(y, oof_sc):.4f} LL={log_loss(y, oof_sc):.4f}')
np.save(os.path.join(OUT, 'cnn25v2_oof.npy'), oof)
np.save(os.path.join(OUT, 'cnn25v2_oof_sc.npy'), oof_sc)
print('saved cnn25v2_oof in %.0fs' % (time.time()-t_start))