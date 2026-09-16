"""Train 3 diverse CNN architectures on full data for v29 submission.
Net3dVarB (strided conv), ResNet10_SE (squeeze-excitation), NetMS (multi-scale).
Saves full-data weights compatible with v23's cnn_infer.py."""
import os, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA = r'E:\DaT\cnn3d'
OUT = r'E:\DaT\submission_v29\weights'
os.makedirs(OUT, exist_ok=True)

X = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy')).ravel().astype(np.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"X: {X.shape}, y: {y.mean():.3f}, device: {device}")

EPOCHS = 35
BATCH = 8
SEEDS = [42, 777, 2024, 100]

# ==================== ARCHITECTURES ====================

class ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
    def forward(self, x):
        return F.relu(x + self.b2(self.b1(x)))

class Net3dVarB(nn.Module):
    """Strided conv downsampling (no MaxPool), 3 ResBlocks."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=2, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            ResBlock3(32), ResBlock3(32), ResBlock3(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8, 192), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

class SEBlock3D(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

class ResNet10_SE(nn.Module):
    """ResBlock + Squeeze-Excitation blocks."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            ResBlock3(32), SEBlock3D(32),
            ResBlock3(32), SEBlock3D(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 192), nn.ReLU(inplace=True),
                                  nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.enc(x)).squeeze(1)

class NetMS(nn.Module):
    """Multi-scale: coarse context + fine detail branches."""
    def __init__(self):
        super().__init__()
        self.ctx = nn.Sequential(
            nn.Conv3d(1, 12, 3, padding=1), nn.BatchNorm3d(12), nn.ReLU(), nn.MaxPool3d(2))
        self.fine = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2))
        self.merge = nn.Sequential(
            nn.Conv3d(12+32, 48, 3, padding=1), nn.BatchNorm3d(48), nn.ReLU(),
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

# ==================== TRAINING ====================

def train_full_data(model_cls, arch_name, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    Xtr = torch.from_numpy(X).float().unsqueeze(1)
    ytr = torch.from_numpy(y).float()
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, num_workers=0)
    
    model = model_cls().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    t0 = time.time()
    for ep in range(epochs):
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
        if (ep + 1) % 10 == 0:
            print(f"  {arch_name} seed{seed} ep{ep+1}: {time.time()-t0:.0f}s")
    
    # Save with arch tag compatible with v23 cnn_infer
    fname = f'deep_{arch_name}_{seed}.pt'
    torch.save({'arch': arch_name, 'seed': seed, 'state': model.state_dict()},
               os.path.join(OUT, fname))
    print(f"  Saved {fname} ({time.time()-t0:.0f}s)")
    return fname

# Train all architectures
models_trained = []
for arch_name, model_cls in [('varb', Net3dVarB), ('se', ResNet10_SE), ('ms', NetMS)]:
    print(f"\n=== Training {arch_name} ===")
    for seed in SEEDS:
        fname = train_full_data(model_cls, arch_name, seed)
        models_trained.append(fname)

print(f"\nTrained {len(models_trained)} models total")
print("Files in weights/:", sorted(os.listdir(OUT)))
