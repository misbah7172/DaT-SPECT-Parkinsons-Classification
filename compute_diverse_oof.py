"""Compute OOF for v29 diverse models + optimize deep/SBR blend + T."""
import numpy as np, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DATA = r'E:\DaT\cnn3d'
X = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy')).ravel().astype(np.float64)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"X: {X.shape}, y mean: {y.mean():.3f}, device: {device}")

EPOCHS = 35
BATCH = 8

class ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
    def forward(self, x):
        return F.relu(x + self.b2(self.b1(x)))

class Net3dVarB(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=2, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            ResBlock3(32), ResBlock3(32), ResBlock3(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 192), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

class SEBlock3D(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                nn.Linear(ch, ch//r), nn.ReLU(), nn.Linear(ch//r, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

class ResNet10_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            ResBlock3(32), SEBlock3D(32), ResBlock3(32), SEBlock3D(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 192), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)

class NetMS(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = nn.Sequential(nn.Conv3d(1, 12, 3, padding=1), nn.BatchNorm3d(12), nn.ReLU(), nn.MaxPool3d(2))
        self.fine = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
                                  nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2))
        self.merge = nn.Sequential(nn.Conv3d(44, 48, 3, padding=1), nn.BatchNorm3d(48), nn.ReLU(),
                                   ResBlock3(48), ResBlock3(48),
                                   nn.Conv3d(48, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
                                   nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 224), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(224, 1))
    def forward(self, x):
        cf = self.fine(x)
        cx = F.avg_pool3d(x, kernel_size=2)
        c = self.ctx(cx)
        m = torch.cat([c, cf], dim=1)
        return self.head(self.merge(m)).squeeze(-1)

ARCHS = [('varb', Net3dVarB), ('se', ResNet10_SE), ('ms', NetMS)]

def tta7(model, x):
    """7-view TTA: base + 4 XY + 2 Z."""
    with torch.no_grad():
        xin = torch.from_numpy(x[None, None]).to(device)
        total = torch.sigmoid(model(xin)).float()
        for dx in (-1, 1):
            xv = torch.from_numpy(np.roll(x, dx, axis=0)[None, None]).to(device)
            total += torch.sigmoid(model(xv)).float()
        for dy in (-1, 1):
            xv = torch.from_numpy(np.roll(x, dy, axis=1)[None, None]).to(device)
            total += torch.sigmoid(model(xv)).float()
        for dz in (-1, 1):
            xv = torch.from_numpy(np.roll(x, dz, axis=2)[None, None]).to(device)
            total += torch.sigmoid(model(xv)).float()
    return float((total / 7).cpu().numpy()[0])

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Compute OOF for each architecture (2 seeds for speed)
for arch_name, ArchCls in ARCHS:
    print(f"\n=== OOF for {arch_name} ===")
    for seed in [42, 777]:
        oof = np.zeros(len(y))
        for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = ArchCls().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
            crit = nn.BCEWithLogitsLoss()
            scaler = torch.amp.GradScaler('cuda')

            Xtr = torch.from_numpy(X[trn_idx]).float().unsqueeze(1)
            ytr = torch.from_numpy(y[trn_idx]).float()
            dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, num_workers=0)

            for ep in range(EPOCHS):
                model.train()
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    with torch.amp.autocast('cuda'):
                        loss = crit(model(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                sched.step()

            model.eval()
            for i in val_idx:
                oof[i] = tta7(model, X[i])

        auc = roc_auc_score(y, oof)
        ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
        print(f"  {arch_name} seed{seed}: AUC={auc:.4f}, LL={ll:.4f}")
        np.save(f'oof_{arch_name}_s{seed}.npy', oof)

print("\nDone!")
