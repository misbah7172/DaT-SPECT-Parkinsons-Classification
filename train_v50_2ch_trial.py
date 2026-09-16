"""Trial: resolution-aware CNN. Input = 2 channels: [registered crop, broadcast(log voxel_vol)].
Quick honest trial on 1 seed x 5 folds (SGKF42, same folds as v24) to measure gain before
committing to full 30-fold retrain. Saves nothing permanent; prints per-fold and combined OOF.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, os, time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

X = np.load(r'E:\DaT\cnn3d\X_reg.npy')
y = np.load(r'E:\DaT\cnn3d\y.npy').ravel().astype(np.int64)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
uids = np.load(r'E:\DaT\cnn3d\uids.npy', allow_pickle=True)
vg = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
vvol = np.array([dict(zip(vg['uid'], vg['voxel_vol'])).get(u, 30.0) for u in uids], dtype=np.float32)
lv = np.log(vvol)
lv = (lv - lv.mean()) / (lv.std() + 1e-9)

H, Wd, D = X.shape[1:]
XC = np.zeros((len(X), 2, H, Wd, D), dtype=np.float32)
XC[:, 0] = X
XC[:, 1] = lv[:, None, None, None] * np.ones((1, H, Wd, D), dtype=np.float32)
print('XC', XC.shape, 'chan1 range', XC[:,1].min(), XC[:,1].max())

EPOCHS = 35
BATCH = 8
device = 'cuda'
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups))

class ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.b2(self.b1(x)))

class Net2ch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            ResBlock3(32), ResBlock3(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

def run_seed(seed):
    oof = np.zeros(len(y)); 
    for fold, (tr, va) in enumerate(folds):
        torch.manual_seed(seed); np.random.seed(seed)
        Xt = torch.from_numpy(XC[tr]).float(); yt = torch.from_numpy(y[tr]).float()
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True, num_workers=0)
        m = Net2ch().to(device)
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler('cuda')
        for ep in range(EPOCHS):
            m.train()
            for xb, yb in dl:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                with torch.autocast('cuda'):
                    loss = crit(m(xb), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
        m.eval()
        Xva = torch.from_numpy(XC[va]).float()
        pred = np.zeros(len(va)); ST = 24
        with torch.no_grad():
            for s in range(0, len(va), ST):
                e = min(s+ST, len(va)); xv = Xva[s:e].to(device)
                p = torch.sigmoid(m(xv).squeeze(-1)).float().cpu().numpy()
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    tmp = torch.roll(xv, dx, dims=2); tmp = torch.roll(tmp, dy, dims=1)
                    p += torch.sigmoid(m(tmp).squeeze(-1)).float().cpu().numpy()
                for dz in (-1,1):
                    tmp = torch.roll(xv, dz, dims=3)
                    p += torch.sigmoid(m(tmp).squeeze(-1)).float().cpu().numpy()
                pred[s:e] = p
        oof[va] = pred/7
        ll = log_loss(y[va], np.clip(pred/7, 1e-6, 1-1e-6))
        print(f'  seed{seed} fold{fold}: LL={ll:.4f} AUC={roc_auc_score(y[va], pred/7):.4f}', flush=True)
    ll = log_loss(y, np.clip(oof, 1e-6, 1-1e-6))
    print(f'seed{seed} OOF (2ch): AUC={roc_auc_score(y, oof):.4f} LL={ll:.4f}', flush=True)
    return oof

if __name__ == '__main__':
    o42 = run_seed(42)