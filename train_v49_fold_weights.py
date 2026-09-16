"""V49: train fold-specific deep-3D CNN weights for fold-averaged submission.
Same arch/pipeline as v46 (Net3dR/Net3dBig on X_reg, BCE, AMP, cosine) but trained
per StratifiedGroupKFold(5, random_state=42) outer group fold, so submission can
average fold models (test prediction mimics OOF ensemble instead of full-data overfit).

Saves:
  submission_v24/weights/deep_{arch}_{seed}_f{fold}.pt   (state at end of schedule)
  submission_v24/weights/fold_oof.npy                    (OOF sigmoid preds, 1362)
  submission_v24/weights/fold_oof_{arch}_{seed}.npy      (per-combo OOF, 1362)
Logs to submission_v24/train_v49.log
"""
import os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

DATA = r'E:\DaT\cnn3d'
OUT = r'E:\DaT\submission_v25\weights'
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(os.path.dirname(OUT), 'train_v49.log')

X = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy')).ravel().astype(np.int64)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()

EPOCHS = 45
BATCH = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
JOBS = [('r', s) for s in [42, 777, 2024, 100]] + [('big', s) for s in [2025, 1984]]
SPLIT = StratifiedGroupKFold(5, shuffle=True, random_state=42)
FOLDS = list(SPLIT.split(np.zeros(len(y)), y, groups))


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')
    print(msg, flush=True)


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


def train_fold(arch, seed, fold, tr, va):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.from_numpy(X[tr]).float().unsqueeze(1)
    yt = torch.from_numpy(y[tr]).float()
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True, num_workers=0)
    model = make_model(arch).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    Nv = len(va); step = 48
    Xva = torch.from_numpy(X[va]).float().unsqueeze(1)
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
    pred = np.zeros(Nv)
    with torch.no_grad():
        for s in range(0, Nv, step):
            e = min(s + step, Nv)
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
    state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    torch.save({'arch': arch, 'seed': seed, 'fold': fold, 'state': state},
               os.path.join(OUT, f'deep_{arch}_{seed}_f{fold}.pt'))
    ll = log_loss(y[va], np.clip(pred, 1e-6, 1 - 1e-6))
    return pred, ll, roc_auc_score(y[va], pred)


if __name__ == '__main__':
    if not os.path.exists(LOG):
        open(LOG, 'w').close()
    log(f'v49 start device={device} jobs={len(JOBS)} folds={len(FOLDS)}')
    import pickle
    oof_maps = {}
    done_k = set()
    total_combos = len(JOBS) * len(FOLDS)
    done_i = 0
    for arch, seed in JOBS:
        oof_arr = np.zeros(len(y))
        counts = np.zeros(len(y))
        for fold, (tr, va) in enumerate(FOLDS):
            key = (arch, seed, fold)
            if done_k and key in done_k:
                continue
            t0 = time.time()
            pred, ll, auc = train_fold(arch, seed, fold, tr, va)
            oof_arr[va] += pred
            counts[va] += 1
            el = time.time() - t0
            done_i += 1
            log(f'{arch}_{seed} f{fold}: LL={ll:.4f} AUC={auc:.4f} ({el/60:.1f}m, {done_i}/{total_combos})')
        oof_arr /= np.maximum(counts, 1)
        np.save(os.path.join(OUT, f'fold_oof_{arch}_{seed}.npy'), oof_arr)
        ll = log_loss(y, np.clip(oof_arr, 1e-6, 1 - 1e-6))
        log(f'{arch}_{seed} OOF: LL={ll:.4f} AUC={roc_auc_score(y, oof_arr):.4f}')
    # ensemble OOF = mean across combos
    files = [os.path.join(OUT, f'fold_oof_{a}_{s}.npy') for a, s in JOBS]
    ens = np.mean([np.load(f) for f in files], axis=0)
    np.save(os.path.join(OUT, 'fold_oof.npy'), ens)
    log(f'ENSEMBLE OOF: LL={log_loss(y, np.clip(ens, 1e-6, 1 - 1e-6)):.4f} AUC={roc_auc_score(y, ens):.4f}')
    np.save(os.path.join(OUT, 'fold_oof_meta.npy'), {'y': y, 'groups': groups})
    log('v49 done')
