"""Compute full OOF from all fold-specific models."""
import os, sys, json
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CACHE_DIR = "D:/DaT_cache/volumes_2mm"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"
SITE_PATH = "E:/DaT/Dataset/site_labels.csv"
MODEL_DIR = "D:/DaT_cache/models"

class _ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c),nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c))
    def forward(self, x):
        return torch.relu(x + self.b2(self.b1(x)))

class _DeepNet(nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net; self.head = head
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

def build_net(arch):
    if arch == "r":
        net = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1),nn.BatchNorm3d(16),nn.ReLU(),nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1),nn.BatchNorm3d(32),nn.ReLU(),nn.MaxPool3d(2),
            _ResBlock3(32),_ResBlock3(32),nn.Conv3d(32,64,3,padding=1),nn.BatchNorm3d(64),nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,2,2)))
        head = nn.Sequential(nn.Flatten(),nn.Linear(64*8,192),nn.ReLU(),nn.Dropout(0.4),nn.Linear(192,1))
    else:
        net = nn.Sequential(
            nn.Conv3d(1,20,3,padding=1),nn.BatchNorm3d(20),nn.ReLU(),nn.MaxPool3d(2),
            nn.Conv3d(20,36,3,padding=1),nn.BatchNorm3d(36),nn.ReLU(),nn.MaxPool3d(2),
            _ResBlock3(36),_ResBlock3(36),nn.Conv3d(36,48,3,padding=1),nn.BatchNorm3d(48),nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,2,2)))
        head = nn.Sequential(nn.Flatten(),nn.Linear(48*8,224),nn.ReLU(),nn.Dropout(0.4),nn.Linear(224,1))
    return _DeepNet(net, head)

class VolumeDataset(Dataset):
    def __init__(self, cache_dir, uids):
        self.cache_dir = cache_dir; self.uids = uids
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        vol = np.load(os.path.join(self.cache_dir, f"{self.uids[idx]}.npy"))
        vol = (vol - vol.mean())/(vol.std()+1e-8)
        return torch.from_numpy(vol[np.newaxis].astype(np.float32))

def main():
    device = torch.device("cuda")
    labels_df = pd.read_csv(LABELS_PATH)
    available = [f.replace(".npy","") for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    labels_df = labels_df[labels_df["uid"].isin(available)].reset_index(drop=True)
    try:
        site_df = pd.read_csv(SITE_PATH)
        labels_df = labels_df.merge(site_df, on="uid", how="left")
        if "sx" in labels_df.columns:
            labels_df["scanner_group"] = labels_df.apply(lambda r: f"{r['sx']:.1f}x{r['sy']:.1f}x{r['sz']:.1f}", axis=1)
        else: labels_df["scanner_group"] = "unknown"
    except: labels_df["scanner_group"] = "unknown"

    uids = labels_df["uid"].tolist()
    labels_arr = labels_df["is_pathologic"].values.astype(float)
    groups = labels_df["scanner_group"].values

    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(labels_arr, labels_arr, groups))

    model_names = ["deep_r_42","deep_r_777","deep_r_2024","deep_big_2025","deep_big_1984","deep_big_100"]
    archs = ["r","r","r","big","big","big"]

    oof = np.zeros((len(uids), 6))

    for fold in range(5):
        _, val_idx = splits[fold]
        val_uids = [uids[i] for i in val_idx]
        ds = VolumeDataset(CACHE_DIR, val_uids)
        ld = DataLoader(ds, batch_size=12, shuffle=False, num_workers=0)
        print(f"Fold {fold} ({len(val_idx)} scans):", flush=True)

        for mi, (arch, name) in enumerate(zip(archs, model_names)):
            ckpt = os.path.join(MODEL_DIR, f"{name}_fold{fold}.pt")
            if not os.path.exists(ckpt):
                print(f"  {name}: MISSING", flush=True)
                continue
            ck = torch.load(ckpt, map_location=device)
            model = build_net(ck["arch"]).to(device)
            model.load_state_dict(ck["state"])
            model.eval()
            bs = 0
            with torch.no_grad():
                for vol in ld:
                    vol = vol.to(device)
                    preds = torch.sigmoid(model(vol)).cpu().numpy()
                    be = bs + len(preds)
                    oof[val_idx[bs:be], mi] = preds
                    bs = be
            del model; torch.cuda.empty_cache()
            print(f"  {name}: done", flush=True)

    # Save
    np.save(os.path.join(MODEL_DIR, "oof_preds.npy"), oof)
    np.save(os.path.join(MODEL_DIR, "oof_labels.npy"), labels_arr)
    mask = np.ones(len(uids), dtype=bool)
    np.save(os.path.join(MODEL_DIR, "oof_mask.npy"), mask)

    # Evaluate
    from sklearn.metrics import log_loss, roc_auc_score
    avg = oof.mean(axis=1)
    ll = log_loss(labels_arr, np.clip(avg,1e-7,1-1e-7))
    auc = roc_auc_score(labels_arr, avg)
    print(f"\nOVERALL: LL={ll:.4f} AUC={auc:.4f}", flush=True)

    # Per-fold
    for fold in range(5):
        _, val_idx = splits[fold]
        fp = oof[val_idx].mean(axis=1)
        fl = log_loss(labels_arr[val_idx], np.clip(fp,1e-7,1-1e-7))
        fa = roc_auc_score(labels_arr[val_idx], fp)
        print(f"  Fold {fold}: LL={fl:.4f} AUC={fa:.4f}", flush=True)

    with open(os.path.join(MODEL_DIR, "oof_metrics.json"), "w") as f:
        json.dump({"logloss": float(ll), "auc": float(auc)}, f, indent=2)

if __name__ == "__main__":
    main()
