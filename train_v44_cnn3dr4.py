"""Variant A arch + physically-plausible small in-plane rotation (±6°) + translation TTA.
Killed the 4-way 90/180/270 rotation (destroys L/R asymmetry). Resume-capable."""
import os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from scipy.ndimage import rotate
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DATA = r'E:\DaT\cnn3d'
OUT = r'E:\DaT\v30_oof'
Xin = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy'))
groups = np.load(os.path.join(DATA, 'groups.npy'), allow_pickle=True)
n = len(y)

SEEDS = [666, 999, 313]
EPOCHS = 40
BATCH = 12
device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROG = os.path.join(OUT, 'cnn3dr4_progress.txt')


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
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            ResBlock3(32), ResBlock3(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, 1))

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def train_fold(tr_idx, va_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = Xin[tr_idx]
    rng = np.random.default_rng(seed + 1)
    ang = rng.uniform(-8, 8, size=Xtr.shape[0])
    Xaug = np.stack([rotate(x, float(a), axes=(1, 2), reshape=False, order=1) for x, a in zip(Xtr, ang)], 0)
    Xaug = np.concatenate([Xtr, Xaug], 0)
    yaug = np.concatenate([y[tr_idx], y[tr_idx]])
    da = np.concatenate([np.zeros(len(tr_idx)), rng.uniform(-2, 2, size=len(tr_idx))], 0)
    db = np.concatenate([np.zeros(len(tr_idx)), rng.uniform(-2, 2, size=len(tr_idx))], 0)
    Xaug = np.stack([ndi.shift(x, (0.0, float(a), float(b)), order=1) for x, a, b in zip(Xaug, da, db)], 0)
    Xt = torch.from_numpy(Xaug).float().unsqueeze(1)
    yt = torch.from_numpy(yaug).float()
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True, num_workers=0)
    model = Net3dR().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    Xva = torch.from_numpy(Xin[va_idx]).float().unsqueeze(1).to(device)
    best_ll = 1e9; best_pred = None; patience = 16; wait = 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            with torch.autocast('cuda'):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(Xva).squeeze(-1)).float().cpu().numpy()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                xv = torch.roll(Xva, dx, dims=2)
                xv = torch.roll(xv, dy, dims=1)
                pred += torch.sigmoid(model(xv).squeeze(-1)).float().cpu().numpy()
        pred /= 5
        ll = log_loss(y[va_idx], np.clip(pred, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_ll = ll; best_pred = pred; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_pred


def main():
    done = set()
    if os.path.exists(PROG):
        done = set(l.strip() for l in open(PROG))
    for seed in SEEDS:
        if str(seed) in done:
            print('skip seed', seed, flush=True)
            continue
        oof_seed = np.zeros(n)
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xin, y, groups)):
            t0 = time.time()
            pred = train_fold(tr, va, seed)
            oof_seed[va] = pred
            print(f'  seed {seed} fold {fold}: AUC={roc_auc_score(y[va], pred):.4f} LL={log_loss(y[va], np.clip(pred,1e-6,1-1e-6)):.4f} ({time.time()-t0:.0f}s)', flush=True)
        print(f'== seed {seed} seed-OOF AUC={roc_auc_score(y, oof_seed):.4f} LL={log_loss(y, np.clip(oof_seed,1e-6,1-1e-6)):.4f}', flush=True)
        with open(PROG, 'a') as f:
            f.write(str(seed) + '\n')
        np.save(os.path.join(OUT, f'cnn3dr4_seed{seed}.npy'), oof_seed)
        print(f'saved cnn3dr4_seed{seed}', flush=True)


if __name__ == '__main__':
    main()