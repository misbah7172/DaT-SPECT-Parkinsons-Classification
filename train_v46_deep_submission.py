"""Train deep-3D CNN models on FULL data for submission, saving state_dicts.
Covered seeds/archs:
  cnn3dr2 (Net3dR):   42, 777, 2024, 100
  cnn3dr5 (Net3dBig): 2025, 1984, 11, 22, 33, 44, 55, 66
Saves E:/DaT/submission_v23/weights/deep_{arch}_{seed}.pt with 'arch','seed','state'.
Also dumps final validation LL/AUC on full data (no split) for sanity."""
import os, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score

DATA = r'E:\DaT\cnn3d'
OUT = r'E:\DaT\submission_v23\weights'
os.makedirs(OUT, exist_ok=True)
Xin = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy'))

EPOCHS = 40
BATCH = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROG = os.path.join(OUT, 'deep_progress.txt')


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


def make_model(arch):
    return Net3dR() if arch == 'r' else Net3dBig()


def train_full(arch, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.from_numpy(Xin).float().unsqueeze(1)
    yt = torch.from_numpy(y).float()
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True, num_workers=0)
    model = make_model(arch).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    Xva = Xt
    N = len(Xva); step = 48
    last_state = None
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
            pred = np.zeros(N)
            for s in range(0, N, step):
                e = min(s + step, N)
                xv = Xva[s:e].to(device)
                p = torch.sigmoid(model(xv).squeeze(-1)).float().cpu().numpy()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    tmp = torch.roll(xv, dx, dims=2)
                    tmp = torch.roll(tmp, dy, dims=1)
                    p += torch.sigmoid(model(tmp).squeeze(-1)).float().cpu().numpy()
                for dz in (-1, 1):
                    tmp = torch.roll(xv, dz, dims=3)
                    p += torch.sigmoid(model(tmp).squeeze(-1)).float().cpu().numpy()
                pred[s:e] = p
        pred /= 7
        ll = log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6))
        last_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        print(f'  {arch}_s{seed} ep{ep}: LL={ll:.4f} AUC={roc_auc_score(y, pred):.4f}', flush=True)
    torch.save({'arch': 'r' if arch == 'r' else 'big', 'seed': seed, 'state': last_state},
               os.path.join(OUT, f'deep_{arch}_{seed}.pt'))
    with open(PROG, 'a') as f:
        f.write(f'{arch},{seed}\n')
    print(f'saved deep_{arch}_{seed}.pt (end of schedule)', flush=True)


if __name__ == '__main__':
    jobs = [('r', s) for s in [42, 777, 2024, 100]] + [('big', s) for s in [2025, 1984, 11, 22, 33, 44, 55, 66]]
    done = set()
    if os.path.exists(PROG):
        for l in open(PROG):
            a, s = l.strip().split(',')
            done.add((a, int(s)))
    for arch, seed in jobs:
        if (arch, seed) in done:
            print('skip', arch, seed, flush=True)
            continue
        print(f'== train deep_{arch}_{seed}', flush=True)
        train_full(arch, seed)