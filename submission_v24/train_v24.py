"""V24 training script: 3/3 architecture split + joint optimization.

Trains 3 Net3dR + 3 Net3dBig models (instead of v23's 4+2 split)
for better ensemble diversity.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Architecture definitions (same as v23) ─────────────────────────────

class _ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.b2(self.b1(x)))


class _DeepNet(nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net
        self.head = head

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def build_net(arch):
    if arch == "r":
        net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            _ResBlock3(32), _ResBlock3(32),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)))
        head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8, 192), nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, 1))
    else:  # "big"
        net = nn.Sequential(
            nn.Conv3d(1, 20, 3, padding=1), nn.BatchNorm3d(20), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(20, 36, 3, padding=1), nn.BatchNorm3d(36), nn.ReLU(), nn.MaxPool3d(2),
            _ResBlock3(36), _ResBlock3(36),
            nn.Conv3d(36, 48, 3, padding=1), nn.BatchNorm3d(48), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)))
        head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 8, 224), nn.ReLU(), nn.Dropout(0.4), nn.Linear(224, 1))
    return _DeepNet(net, head)


# ─── Dataset ────────────────────────────────────────────────────────────

class VolumeDataset(Dataset):
    def __init__(self, cache_dir, uids, labels=None, augment=False):
        self.cache_dir = cache_dir
        self.uids = uids
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        vol = np.load(os.path.join(self.cache_dir, f"{uid}.npy"))
        vol = vol[np.newaxis, ...]
        if self.augment:
            if np.random.random() > 0.5:
                vol = vol[:, :, :, ::-1].copy()
            if np.random.random() > 0.5:
                vol = vol[:, ::-1, :, :].copy()
            vol = vol * np.random.uniform(0.9, 1.1)
        vol = torch.from_numpy(vol.astype(np.float32))
        if self.labels is not None:
            return vol, torch.tensor(self.labels[idx], dtype=torch.float32)
        return vol, uid


# ─── Training ───────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, epochs=80, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=5/epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 15
    warmup_done = False

    for epoch in range(epochs):
        model.train()
        for volumes, labels in train_loader:
            volumes, labels = volumes.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(volumes)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        if epoch >= 5 and not warmup_done:
            warmup_done = True

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for volumes, labels in val_loader:
                volumes, labels = volumes.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(volumes)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)

        if warmup_done and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif warmup_done:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch}: val_loss={avg_val_loss:.4f} best={best_val_loss:.4f} patience={patience_counter}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    cache_dir = "pipeline_output/cache"
    import pandas as pd
    labels_df = pd.read_csv("Dataset/train_labels.csv")

    available = [f.replace(".npy", "") for f in os.listdir(cache_dir) if f.endswith(".npy")]
    labels_df = labels_df[labels_df["uid"].isin(available)].reset_index(drop=True)

    # Scanner groups
    try:
        site_df = pd.read_csv("Dataset/site_labels.csv")
        labels_df = labels_df.merge(site_df, on="uid", how="left")
        if "sx" in labels_df.columns:
            labels_df["scanner_group"] = labels_df.apply(
                lambda r: f"{r['sx']:.1f}x{r['sy']:.1f}x{r['sz']:.1f}", axis=1)
    except Exception:
        labels_df["scanner_group"] = "unknown"

    uids = labels_df["uid"].tolist()
    labels = labels_df["is_pathologic"].tolist()
    groups = labels_df["scanner_group"].values

    # V24 config: 3/3 architecture split
    seeds = [42, 777, 2024]
    archs = ["r", "r", "r", "big", "big", "big"]
    model_names = ["deep_r_42", "deep_r_777", "deep_r_2024", "deep_big_2025", "deep_big_1984", "deep_big_100"]
    model_seeds = [42, 777, 2024, 2025, 1984, 100]

    output_dir = "submission_v24/weights"
    os.makedirs(output_dir, exist_ok=True)

    # 5-fold CV
    n_folds = 5
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(labels))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(labels, labels, groups)):
        print(f"\n{'='*50} FOLD {fold} {'='*50}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_uids = [uids[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_uids = [uids[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        for model_idx, (arch, name, seed) in enumerate(zip(archs, model_names, model_seeds)):
            print(f"\n  --- {name} (arch={arch}, seed={seed}) ---")
            torch.manual_seed(seed)

            model = build_net(arch).to(device)
            print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")

            train_ds = VolumeDataset(cache_dir, train_uids, train_labels, augment=True)
            val_ds = VolumeDataset(cache_dir, val_uids, val_labels, augment=False)

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

            model, val_loss = train_model(model, train_loader, val_loader, device, epochs=80)

            # Save model
            ckpt_path = os.path.join(output_dir, f"{name}.pt")
            torch.save({"arch": arch, "state": model.state_dict()}, ckpt_path)
            print(f"    Saved: {ckpt_path} (val_loss={val_loss:.4f})")

            # Get OOF predictions
            model.eval()
            with torch.no_grad():
                for i, (volumes, _) in enumerate(val_loader):
                    volumes = volumes.to(device)
                    logits = model(volumes)
                    preds = torch.sigmoid(logits).cpu().numpy()
                    start = i * val_loader.batch_size
                    end = start + len(preds)
                    # Average across seeds for this architecture
                    if model_idx < 3:  # First 3 are Net3dR
                        oof_preds[val_idx[start:end]] += preds / 3
                    else:  # Last 3 are Net3dBig
                        oof_preds[val_idx[start:end]] += preds / 3

    # Evaluate
    oof_preds_clipped = np.clip(oof_preds, 1e-7, 1 - 1e-7)
    ll = log_loss(labels, oof_preds_clipped)
    auc = roc_auc_score(labels, oof_preds_clipped)
    print(f"\nOOF LogLoss: {ll:.4f}, AUROC: {auc:.4f}")

    # Save OOF
    np.save(os.path.join(output_dir, "oof_preds.npy"), oof_preds)
    with open(os.path.join(output_dir, "oof_metrics.json"), "w") as f:
        json.dump({"logloss": float(ll), "auc": float(auc)}, f, indent=2)


if __name__ == "__main__":
    main()
