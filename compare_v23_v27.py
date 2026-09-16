"""Compare v23 original weights vs our retrained weights on OOF.
This tests whether the original CNN weights are better."""
import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder
import pickle

X = np.load(r'E:\DaT\cnn3d\X_reg.npy')[:, None].astype(np.float32)
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
groups = LabelEncoder().fit_transform(groups_raw.astype(str))

class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.BatchNorm3d(c))
    def forward(self, x): return F.relu(x + self.b2(self.b1(x)))

class Net3dR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            ResBlock3D(32), ResBlock3D(32),
            nn.Conv3d(32,64,3,padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 192), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x): return self.head(self.net(x)).squeeze(1)

class Net3dBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1,20,3,padding=1), nn.BatchNorm3d(20), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(20,36,3,padding=1), nn.BatchNorm3d(36), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            ResBlock3D(36), ResBlock3D(36),
            nn.Conv3d(36,48,3,padding=1), nn.BatchNorm3d(48), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48*8, 224), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(224, 1))
    def forward(self, x): return self.head(self.net(x)).squeeze(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path, arch):
    if arch == 'r':
        m = Net3dR()
    else:
        m = Net3dBig()
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    if 'state' in ck:
        m.load_state_dict(ck['state'])
    else:
        m.load_state_dict(ck)
    m.eval().to(DEVICE)
    return m

def predict_batch(model, X_batch):
    with torch.no_grad():
        xb = torch.from_numpy(X_batch).float().to(DEVICE)
        return expit(model(xb).cpu().numpy())

# TTA
def predict_tta(model, x_np):
    """x_np: (34,40,42). Returns mean of base + 6 translation views."""
    views = [x_np]
    for ax in [0, 1, 2]:
        for d in [-1, 1]:
            v = np.roll(x_np, d, axis=ax)
            views.append(v)
    batch = np.stack(views)[:, None]  # (7, 1, 34, 40, 42)
    with torch.no_grad():
        xb = torch.from_numpy(batch).float().to(DEVICE)
        probs = expit(model(xb).cpu().numpy())
    return probs.mean()

print("Loading and evaluating v23 weights on full training data...")
print("This tests each model individually (no OOF split - just raw performance).\n")

# v23 weights (single full-data)
v23_models = {
    'r_42': ('r', r'E:\DaT\submission_v23\weights\deep_r_42.pt'),
    'r_777': ('r', r'E:\DaT\submission_v23\weights\deep_r_777.pt'),
    'r_2024': ('r', r'E:\DaT\submission_v23\weights\deep_r_2024.pt'),
    'r_100': ('r', r'E:\DaT\submission_v23\weights\deep_r_100.pt'),
    'big_2025': ('big', r'E:\DaT\submission_v23\weights\deep_big_2025.pt'),
    'big_1984': ('big', r'E:\DaT\submission_v23\weights\deep_big_1984.pt'),
}

# v27 weights (fold-averaged, 1 per fold)
v27_models = {}
for seed in [42, 777, 2024, 100]:
    for fold in range(5):
        path = rf'E:\DaT\submission_v27\weights\deep_r_{seed}_f{fold}.pt'
        if os.path.exists(path):
            v27_models[f'r_{seed}_f{fold}'] = ('r', path)

# Also v27 full-data
for seed in [42, 777, 2024, 100]:
    path = rf'E:\DaT\submission_v27\weights\deep_r_{seed}_full.pt'
    if os.path.exists(path):
        v27_models[f'r_{seed}_full'] = ('r', path)

# Evaluate each model individually
print(f"{'Model':<20s} {'AUC':>8s} {'LL':>8s}")
print("-" * 40)

v23_preds = {}
for name, (arch, path) in v23_models.items():
    model = load_model(path, arch)
    preds = np.zeros(len(y))
    for i in range(len(y)):
        preds[i] = predict_tta(model, X[i, 0])
    auc = roc_auc_score(y, preds)
    ll = log_loss(y, np.clip(preds, 1e-7, 1-1e-7))
    v23_preds[name] = preds
    print(f"  v23_{name:<15s} {auc:>8.4f} {ll:>8.4f}")

# v27 full-data models
v27_full_preds = {}
for name, (arch, path) in v27_models.items():
    if '_full' in name:
        model = load_model(path, arch)
        preds = np.zeros(len(y))
        for i in range(len(y)):
            preds[i] = predict_tta(model, X[i, 0])
        auc = roc_auc_score(y, preds)
        ll = log_loss(y, np.clip(preds, 1e-7, 1-1e-7))
        v27_full_preds[name] = preds
        print(f"  v27_{name:<15s} {auc:>8.4f} {ll:>8.4f}")

# v27 fold models (average per-seed)
v27_seed_preds = {}
for seed in [42, 777, 2024, 100]:
    seed_preds = []
    for fold in range(5):
        name = f'r_{seed}_f{fold}'
        if name in v27_models:
            model = load_model(v27_models[name][1], 'r')
            preds = np.zeros(len(y))
            for i in range(len(y)):
                preds[i] = predict_tta(model, X[i, 0])
            seed_preds.append(preds)
    if seed_preds:
        avg = np.mean(seed_preds, axis=1) if len(seed_preds) > 1 else seed_preds[0]
        auc = roc_auc_score(y, avg)
        ll = log_loss(y, np.clip(avg, 1e-7, 1-1e-7))
        v27_seed_preds[seed] = avg
        print(f"  v27_seed{seed:<11s} {auc:>8.4f} {ll:>8.4f}")

# Ensembles
print(f"\n{'Ensemble':<20s} {'AUC':>8s} {'LL':>8s}")
print("-" * 40)

# v23: all 6 models
v23_all = np.mean(list(v23_preds.values()), axis=0)
print(f"  v23_all_6           {roc_auc_score(y, v23_all):>8.4f} {log_loss(y, np.clip(v23_all,1e-7,1-1e-7)):>8.4f}")

# v27: all 4 seeds
if v27_seed_preds:
    v27_all = np.mean(list(v27_seed_preds.values()), axis=0)
    print(f"  v27_all_4seeds      {roc_auc_score(y, v27_all):>8.4f} {log_loss(y, np.clip(v27_all,1e-7,1-1e-7)):>8.4f}")

# v27: all 4 full-data
if v27_full_preds:
    v27_full_all = np.mean(list(v27_full_preds.values()), axis=0)
    print(f"  v27_full_4seeds     {roc_auc_score(y, v27_full_all):>8.4f} {log_loss(y, np.clip(v27_full_all,1e-7,1-1e-7)):>8.4f}")

# Blend with SBR
with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names_arr, oofs_arr = A['names'], np.column_stack(A['oof'])
sbr_idx = [i for i, n in enumerate(names_arr) if 'sbr' in n.lower()]
sbr_oof = oofs_arr[:, sbr_idx].mean(axis=1)

from scipy.special import logit
from scipy.optimize import minimize

for deep_name, deep_pred in [('v23_all', v23_all), ('v27_all', v27_all), ('v27_full', v27_full_all)]:
    def blend_loss(params):
        wd, T = params
        b = np.clip(wd * deep_pred + (1-wd) * sbr_oof, 1e-7, 1-1e-7)
        p = np.clip(expit(logit(b) / T), 1e-7, 1-1e-7)
        return log_loss(y, p)
    best = None
    for init in ([0.8,0.7],[0.9,0.5],[0.7,0.8],[0.95,0.3]):
        r = minimize(blend_loss, init, method='Nelder-Mead', options={'maxiter':500})
        if best is None or r.fun < best.fun: best = r
    wd, T = best.x
    bp = np.clip(expit(logit(np.clip(wd*deep_pred+(1-wd)*sbr_oof,1e-7,1-1e-7))/T), 1e-7, 1-1e-7)
    print(f"  {deep_name}+SBR       {roc_auc_score(y, bp):>8.4f} {log_loss(y, bp):>8.4f}  (w_d={wd:.2f})")

print(f"\n  Current v27 test:     0.0941")
print(f"  v23 test:             0.0685")
