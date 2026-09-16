"""V52: Retrain STRONG-protocol CNNs (Net3dR/Net3dBig, 40ep early-stop, TTA rolls) as
fold-weights under the SAME SGKF42 split used for v24, so test-time fold-averaging
reproduces the strong per-seed streams (cnn3dr2/cnn3dr5 family) instead of the weak
v24 variants. Saves per-combo fold OOF + weights; computes ensemble OOF.

Strong protocols (matching train_v45/v46 era):
  - Net3dR seeds: 42, 777, 2024, 100
  - Net3dBig seeds: 2025, 1984, 11, 22, 33, 44, 55, 66
  - EPOCHS=40, patience early-stop by val LL, cosine, AMP, 7x translation TTA eval
"""
import os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

DATA = r'E:\DaT\cnn3d'
OUT = r'E:\DaT\submission_v26\weights'
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(r'E:\DaT\submission_v26', 'train_v52.log')

X = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy')).ravel().astype(np.int64)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()

EPOCHS = 40
BATCH = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
JOBS_R = [42, 777, 2024, 100]
JOBS_B = [2025, 1984, 11, 22, 33, 44, 55, 66]
SPLIT = StratifiedGroupKFold(5, shuffle=True, random_state=42)  # same split as v24
FOLDS = list(SPLIT.split(np.zeros(len(y)), y, groups))


def log(m):
    with open(LOG, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] {m}\n')
    print(m, flush=True)


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
    Xva = torch.from_numpy(X[va]).float().unsqueeze(1)
    best_ll = 1e9; best_state = None; patience = 16; wait = 0
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
            pred = np.zeros(len(va))
            for s in range(0, len(va), 48):
                e = min(s + 48, len(va))
                xv = Xva[s:e].to(device)
                p = torch.sigmoid(model(xv).squeeze(-1)).float().cpu().numpy()
                pred[s:e] = p
        ll = log_loss(y[va], np.clip(pred, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_ll = ll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    pred = np.zeros(len(va))
    with torch.no_grad():
        for s in range(0, len(va), 48):
            e = min(s + 48, len(va))
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
    torch.save({'arch': arch, 'seed': seed, 'fold': fold, 'state': best_state},
               os.path.join(OUT, f'deep_{arch}_{seed}_f{fold}.pt'))
    ll = log_loss(y[va], np.clip(pred, 1e-6, 1 - 1e-6))
    return pred, ll, roc_auc_score(y[va], pred)


if __name__ == '__main__':
    if not os.path.exists(LOG):
        open(LOG, 'w').close()
    log(f'v52 start device={device}')
    JOBS = [('r', s) for s in JOBS_R] + [('big', s) for s in JOBS_B]
    total = len(JOBS) * len(FOLDS)
    done = 0
    oof_maps = {}
    for arch, seed in JOBS:
        oof_arr = np.zeros(len(y)); cnt = np.zeros(len(y))
        for fold, (tr, va) in enumerate(FOLDS):
            t0 = time.time()
            pred, ll, auc = train_fold(arch, seed, fold, tr, va)
            oof_arr[va] += pred; cnt[va] += 1
            done += 1
            log(f'{arch}_{seed} f{fold}: LL={ll:.4f} AUC={auc:.4f} ({time.time()-t0:.0f}s, {done}/{total})')
        oof_arr /= np.maximum(cnt, 1)
        np.save(os.path.join(OUT, f'fold_oof_{arch}_{seed}.npy'), oof_arr)
        log(f'{arch}_{seed} OOF: LL={log_loss(y, np.clip(oof_arr,1e-6,1-1e-6)):.4f} AUC={roc_auc_score(y, oof_arr):.4f}')
    files = [os.path.join(OUT, f'fold_oof_{a}_{s}.npy') for a, s in JOBS]
    ens = np.mean([np.load(f) for f in files], axis=0)
    np.save(os.path.join(OUT, 'fold_oof.npy'), ens)
    log(f'ENSEMBLE OOF (12 combos): LL={log_loss(y, np.clip(ens,1e-6,1-1e-6)):.4f} AUC={roc_auc_score(y, ens):.4f}')
    log('v52 done')