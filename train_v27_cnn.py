"""V27: 2.5D CNN on atlas-aligned striatal crops (Inorm, central slices).
Group-stratified OOF, several seeds, checkpoint best by val LL.
"""
import os, time, copy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
N_SLICES = 7
BATCH = 32
EPOCHS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = r'E:\DaT\v27_cnn_oof'


class ResNet2_5D(nn.Module):
    def __init__(self, n_slices=7, channels_out=3):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.m = models.resnet18(weights=weights)
        # replicate the initial conv to n_slices channels
        orig = self.m.conv1
        new_conv = nn.Conv2d(n_slices, 64, kernel_size=orig.kernel_size,
                             stride=orig.stride, padding=orig.padding, bias=False)
        with torch.no_grad():
            mean_w = orig.weight.mean(dim=1, keepdim=True)
            for c in range(n_slices):
                new_conv.weight[:, c] = mean_w[:, 0]
        self.m.conv1 = new_conv
        in_f = self.m.fc.in_features
        self.m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, 1))

    def forward(self, x):
        return self.m(x).squeeze(-1)


def train_fold(model, Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=1e-4, wd=1e-4):
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    best_state, best_vl = None, float('inf')
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(Xva_t)).cpu().numpy()
        vl = log_loss(yva, np.clip(vp, 1e-6, 1 - 1e-6))
        if vl < best_vl:
            best_vl = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(r'E:\DaT\cnn25_input.npz', allow_pickle=True)
    X, y, groups = d['X'], d['y'], d['groups']
    n = len(y)
    print('X', X.shape, 'y', y.sum(), 'dev', DEVICE)

    oof = np.zeros(n)
    cnt = np.zeros(n)
    fold_metrics = []
    for seed in SEEDS:
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups)):
            model = ResNet2_5D(n_slices=N_SLICES).to(DEVICE)
            # augment: mild intensity + flips
            Xtr = X[tr].copy()
            if np.random.RandomState(seed + fold).rand() < 0.8:
                flip = np.random.RandomState(seed + fold).rand() < 0.5
                Xtr = np.ascontiguousarray(np.flip(Xtr, axis=2)) if flip else Xtr
            model = train_fold(model, Xtr, y[tr], X[va], y[va])
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(X[va]).to(DEVICE))).cpu().numpy()
            oof[va] += p
            cnt[va] += 1
            auc = roc_auc_score(y[va], p)
            ll = log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6))
            fold_metrics.append((seed, fold, auc, ll))
            print(f'  seed {seed} fold {fold}: val AUC={auc:.4f} LL={ll:.4f}', flush=True)
            del model; torch.cuda.empty_cache()
    oof = np.where(cnt > 0, oof / cnt, 0.5)
    np.save(os.path.join(OUT, 'oof.npy'), oof)
    np.save(os.path.join(OUT, 'y.npy'), y)
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))
    print(f'\nCNN 2.5D OOF: AUC={auc:.4f} LL={ll:.4f}')
    # per-site breakdown
    for s in np.unique(groups):
        m = groups == s
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        print(f'  {s}: n={m.sum()} AUC={roc_auc_score(y[m], oof[m]):.4f} LL={log_loss(y[m], np.clip(oof[m],1e-6,1-1e-6)):.4f}')


if __name__ == '__main__':
    main()