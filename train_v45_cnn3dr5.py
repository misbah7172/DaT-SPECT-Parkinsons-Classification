"""Scaled-up variant A: more ResBlocks, wider intermediate channels, same pooling.
Registered crops, Inorm, TTA translations, AMP, save per-seed OOF. Resume-capable.
No rotation aug (hurts L/R asymmetry signal)."""
import os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DATA = r'E:\DaT\cnn3d'
OUT = r'E:\DaT\v30_oof'
Xin = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy'))
groups = np.load(os.path.join(DATA, 'groups.npy'), allow_pickle=True)
n = len(y)

SEEDS = [11, 22, 33, 44, 55, 66]
EPOCHS = 40
BATCH = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROG = os.path.join(OUT, 'cnn3dr5_progress.txt')


class ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.b2(self.b1(x)))


class Net3dBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 20, 3, padding=1), nn.BatchNorm3d(20), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(20, 36, 3, padding=1), nn.BatchNorm3d(36), nn.ReLU(), nn.MaxPool3d(2),
            ResBlock3(36), ResBlock3(36),
            nn.Conv3d(36, 48, 3, padding=1), nn.BatchNorm3d(48), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 8, 224), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(224, 1))

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def train_fold(tr_idx, va_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(Xin[tr_idx]).float().unsqueeze(1)
    ytr = torch.from_numpy(y[tr_idx]).float()
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, num_workers=0)
    model = Net3dBig().to(device)
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
            ev = []
            for i in range(0, len(Xva), 32):
                xb = Xva[i:i + 32]
                p = torch.sigmoid(model(xb).squeeze(-1)).float().cpu().numpy()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    xv = torch.roll(xb, dx, dims=2)
                    xv = torch.roll(xv, dy, dims=1)
                    p += torch.sigmoid(model(xv).squeeze(-1)).float().cpu().numpy()
                for dz in (-1, 1):
                    xv = torch.roll(xb, dz, dims=3)
                    p += torch.sigmoid(model(xv).squeeze(-1)).float().cpu().numpy()
                ev.append(p / 7)
            pred = np.concatenate(ev, 0)
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
        np.save(os.path.join(OUT, f'cnn3dr5_seed{seed}.npy'), oof_seed)
        print(f'saved cnn3dr5_seed{seed}', flush=True)


if __name__ == '__main__':
    main()