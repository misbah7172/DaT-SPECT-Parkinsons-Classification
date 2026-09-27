"""V25 training: deeper models + mixup + label smoothing + more seeds.

Improvements over v24:
1. Deeper ResNet (3 blocks instead of 2) + wider filters
2. Mixup augmentation (alpha=0.2)
3. Label smoothing (0.05)
4. 5 seeds per architecture (10 total models)
5. Cosine annealing LR
6. Gradient accumulation (effective batch 24)
"""
import os, sys, time, json, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

sys.stdout.reconfigure(line_buffering=True)

CACHE_DIR = "D:/DaT_cache/volumes_2mm"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"
OUTPUT_DIR = "D:/DaT_cache/v25_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── v25 Architecture: Deeper + Wider ────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
    def forward(self, x):
        return torch.relu(x + self.b2(self.b1(x)))

class _DownBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ci, co, 3, stride=2, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True))
        self.res = _ResBlock(co)
    def forward(self, x):
        return self.res(self.conv(x))

class _DeepNet(nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net; self.head = head
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

def build_net_v25(arch):
    """v25: deeper ResNet with 3 residual blocks + downsampling."""
    if arch == "r":
        # Net3dR-v25: 16->32->64->128, 3 blocks
        net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            _ResBlock(16),
            _DownBlock(16, 32),
            _DownBlock(32, 64),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2)))
        head = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 8, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 1))
    else:
        # Net3dBig-v25: 20->40->80->160, 3 blocks
        net = nn.Sequential(
            nn.Conv3d(1, 20, 3, padding=1), nn.BatchNorm3d(20), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            _ResBlock(20),
            _DownBlock(20, 40),
            _DownBlock(40, 80),
            nn.Conv3d(80, 160, 3, padding=1), nn.BatchNorm3d(160), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2)))
        head = nn.Sequential(
            nn.Flatten(), nn.Linear(160 * 8, 320), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(320, 1))
    return _DeepNet(net, head)

# ─── Dataset with Mixup ──────────────────────────────────────────────────

class VolumeDataset(Dataset):
    def __init__(self, cache_dir, uids, labels=None, augment=False):
        self.cache_dir = cache_dir
        self.uids = uids
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        vol = np.load(os.path.join(self.cache_dir, f"{self.uids[idx]}.npy"))
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        vol = vol[np.newaxis]
        if self.augment:
            if np.random.random() > 0.5:
                vol = vol[:, :, :, ::-1].copy()
            if np.random.random() > 0.5:
                vol = vol[:, ::-1, :, :].copy()
            if np.random.random() > 0.5:
                vol = vol[:, :, ::-1, :].copy()
            # Random affine-ish: scale + shift
            vol = vol * np.random.uniform(0.93, 1.07) + np.random.uniform(-0.02, 0.02)
        vol = torch.from_numpy(vol.astype(np.float32))
        if self.labels is not None:
            return vol, torch.tensor(self.labels[idx], dtype=torch.float32)
        return vol

# ─── Mixup ───────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ─── Label Smoothing BCE ─────────────────────────────────────────────────

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, targets):
        # Smooth labels
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)
        return loss

# ─── Training ────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-4, accum_steps=2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingBCE(smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0

        for vol, lbl in train_loader:
            vol, lbl = vol.to(device), lbl.to(device)

            # Mixup
            vol_mixed, lbl_a, lbl_b, lam = mixup_data(vol, lbl, alpha=0.2)

            with torch.amp.autocast("cuda"):
                logits = model(vol_mixed)
                loss = (lam * criterion(logits, lbl_a) + (1 - lam) * criterion(logits, lbl_b)) / accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_steps
            n_batches += 1

            if n_batches % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for vol, lbl in val_loader:
                vol, lbl = vol.to(device), lbl.to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(vol)
                    loss = criterion(logits, lbl)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss_sum += loss.item() * len(lbl)
                    val_count += len(lbl)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(lbl.cpu().numpy())

        avg_val_loss = val_loss_sum / max(val_count, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            try:
                auc = roc_auc_score(np.array(val_true), np.array(val_preds))
            except:
                auc = 0.0
            print(f"  Ep {epoch}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} best={best_val_loss:.4f} auc={auc:.4f} pat={patience_counter}", flush=True)

        if patience_counter >= patience:
            print(f"  Early stop at ep {epoch}", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss

# ─── Main ────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    labels_df = pd.read_csv(LABELS_PATH)
    available = [f.replace(".npy", "") for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    labels_df = labels_df[labels_df["uid"].isin(available)].reset_index(drop=True)
    uids = labels_df["uid"].tolist()
    labels_arr = labels_df["is_pathologic"].values.astype(float)
    groups = labels_df["uid"].values
    print(f"Data: {len(uids)} scans, pos={labels_arr.mean():.3f}", flush=True)

    # v25 config: 2 archs x 5 seeds = 10 models
    archs = ["r", "r", "r", "r", "r", "big", "big", "big", "big", "big"]
    seeds = [42, 777, 2024, 1984, 100, 2025, 314, 271, 1337, 999]
    names = [f"v25_{'r' if a=='r' else 'big'}_{s}" for a, s in zip(archs, seeds)]

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(labels_arr, labels_arr, groups))

    oof_preds = np.zeros((len(labels_arr), len(names)))
    oof_mask = np.zeros(len(labels_arr), dtype=bool)

    for fold in range(5):
        train_idx, val_idx = splits[fold]
        print(f"\n{'='*60} FOLD {fold} train={len(train_idx)} val={len(val_idx)} {'='*60}", flush=True)

        t_uids = [uids[i] for i in train_idx]
        t_labels = labels_arr[train_idx].tolist()
        v_uids = [uids[i] for i in val_idx]
        v_labels = labels_arr[val_idx]

        for mi, (arch, seed, name) in enumerate(zip(archs, seeds, names)):
            print(f"\n  {name} (arch={arch}, seed={seed})", flush=True)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            model = build_net_v25(arch).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Params: {n_params:,}", flush=True)

            ds_tr = VolumeDataset(CACHE_DIR, t_uids, t_labels, augment=True)
            ds_va = VolumeDataset(CACHE_DIR, v_uids, v_labels.tolist(), augment=False)
            ld_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
            ld_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

            t0 = time.time()
            model, vl = train_model(model, ld_tr, ld_va, device, epochs=100, lr=1e-4, accum_steps=2)
            print(f"    Done: loss={vl:.4f} time={time.time()-t0:.0f}s", flush=True)

            # Save
            torch.save({"arch": arch, "state": model.state_dict()},
                       os.path.join(OUTPUT_DIR, f"{name}_fold{fold}.pt"))

            # OOF predictions
            model.eval()
            bs = 0
            with torch.no_grad():
                for vol, _ in ld_va:
                    vol = vol.to(device)
                    with torch.amp.autocast("cuda"):
                        preds = torch.sigmoid(model(vol)).cpu().numpy()
                    be = bs + len(preds)
                    oof_preds[val_idx[bs:be], mi] = preds
                    bs = be

            del model
            torch.cuda.empty_cache()
            gc.collect()

        oof_mask[val_idx] = True
        fold_preds = oof_preds[val_idx].mean(axis=1)
        fold_ll = log_loss(v_labels, np.clip(fold_preds, 1e-7, 1-1e-7))
        fold_auc = roc_auc_score(v_labels, fold_preds)
        print(f"\n  Fold {fold} ENSEMBLE: LL={fold_ll:.4f} AUC={fold_auc:.4f}", flush=True)

    # Overall
    overall_preds = oof_preds[oof_mask].mean(axis=1)
    overall_true = labels_arr[oof_mask]
    overall_ll = log_loss(overall_true, np.clip(overall_preds, 1e-7, 1-1e-7))
    overall_auc = roc_auc_score(overall_true, overall_preds)
    print(f"\n{'='*60} OVERALL {'='*60}", flush=True)
    print(f"LL={overall_ll:.4f} AUC={overall_auc:.4f}", flush=True)

    # Per-fold
    for fold in range(5):
        _, vi = splits[fold]
        fp = oof_preds[vi].mean(axis=1)
        fl = log_loss(labels_arr[vi], np.clip(fp, 1e-7, 1-1e-7))
        fa = roc_auc_score(labels_arr[vi], fp)
        print(f"  Fold {fold}: LL={fl:.4f} AUC={fa:.4f}", flush=True)

    # Save
    np.save(os.path.join(OUTPUT_DIR, "oof_preds.npy"), oof_preds)
    np.save(os.path.join(OUTPUT_DIR, "oof_labels.npy"), labels_arr)
    np.save(os.path.join(OUTPUT_DIR, "oof_mask.npy"), oof_mask)
    with open(os.path.join(OUTPUT_DIR, "oof_metrics.json"), "w") as f:
        json.dump({"logloss": float(overall_ll), "auc": float(overall_auc)}, f, indent=2)
    print("Saved", flush=True)

if __name__ == "__main__":
    main()
