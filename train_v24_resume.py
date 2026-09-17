"""Resume V24 CNN training from fold 2, model index 2."""
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

CACHE_DIR = "D:/DaT_cache/volumes_2mm"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"
SITE_PATH = "E:/DaT/Dataset/site_labels.csv"
OUTPUT_DIR = "D:/DaT_cache/models"
LOG_DIR = "D:/DaT_cache/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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
    else:
        net = nn.Sequential(
            nn.Conv3d(1, 20, 3, padding=1), nn.BatchNorm3d(20), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(20, 36, 3, padding=1), nn.BatchNorm3d(36), nn.ReLU(), nn.MaxPool3d(2),
            _ResBlock3(36), _ResBlock3(36),
            nn.Conv3d(36, 48, 3, padding=1), nn.BatchNorm3d(48), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)))
        head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 8, 224), nn.ReLU(), nn.Dropout(0.4), nn.Linear(224, 1))
    return _DeepNet(net, head)


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
        vol_mean = vol.mean()
        vol_std = vol.std() + 1e-8
        vol = (vol - vol_mean) / vol_std
        vol = vol[np.newaxis, ...]
        if self.augment:
            if np.random.random() > 0.5:
                vol = vol[:, :, :, ::-1].copy()
            if np.random.random() > 0.5:
                vol = vol[:, ::-1, :, :].copy()
            vol = vol * np.random.uniform(0.95, 1.05)
        vol = torch.from_numpy(vol.astype(np.float32))
        if self.labels is not None:
            return vol, torch.tensor(self.labels[idx], dtype=torch.float32)
        return vol, uid


def train_model(model, train_loader, val_loader, device, epochs=80, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=5/epochs)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
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
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(volumes)
                loss = criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        if epoch >= 5:
            warmup_done = True
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for volumes, labels in val_loader:
                volumes, labels = volumes.to(device), labels.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(volumes)
                    loss = criterion(logits, labels)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss_sum += loss.item() * len(labels)
                    val_count += len(labels)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        avg_val_loss = val_loss_sum / max(val_count, 1)
        if warmup_done and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif warmup_done:
            patience_counter += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            try:
                auc = roc_auc_score(np.array(val_true), np.array(val_preds))
            except Exception:
                auc = 0.0
            print(f"  Epoch {epoch}: loss={avg_val_loss:.4f} best={best_val_loss:.4f} auc={auc:.4f} pat={patience_counter}")
        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels_df = pd.read_csv(LABELS_PATH)
    available = [f.replace(".npy", "") for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    labels_df = labels_df[labels_df["uid"].isin(available)].reset_index(drop=True)

    try:
        site_df = pd.read_csv(SITE_PATH)
        labels_df = labels_df.merge(site_df, on="uid", how="left")
        if "sx" in labels_df.columns:
            labels_df["scanner_group"] = labels_df.apply(
                lambda r: f"{r['sx']:.1f}x{r['sy']:.1f}x{r['sz']:.1f}", axis=1)
        else:
            labels_df["scanner_group"] = "unknown"
    except Exception:
        labels_df["scanner_group"] = "unknown"

    uids = labels_df["uid"].tolist()
    labels_arr = labels_df["is_pathologic"].values.astype(float)
    groups = labels_df["scanner_group"].values

    archs = ["r", "r", "r", "big", "big", "big"]
    model_names = ["deep_r_42", "deep_r_777", "deep_r_2024", "deep_big_2025", "deep_big_1984", "deep_big_100"]
    model_seeds = [42, 777, 2024, 2025, 1984, 100]

    n_folds = 5
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(labels_arr, labels_arr, groups))

    oof_preds = np.zeros((len(labels_arr), len(model_names)))
    oof_mask = np.zeros(len(labels_arr), dtype=bool)

    # Resume from fold 2, model 2
    START_FOLD = 2
    START_MODEL = 2

    for fold in range(START_FOLD, n_folds):
        train_idx, val_idx = splits[fold]
        print(f"\n{'='*60} FOLD {fold} {'='*60}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_uids = [uids[i] for i in train_idx]
        train_labels = labels_arr[train_idx].tolist()
        val_uids = [uids[i] for i in val_idx]
        val_labels_np = labels_arr[val_idx]

        for model_idx in range(START_MODEL if fold == START_FOLD else 0, len(model_names)):
            arch = archs[model_idx]
            name = model_names[model_idx]
            seed = model_seeds[model_idx]
            print(f"\n  --- {name} (arch={arch}, seed={seed}) ---")
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            model = build_net(arch).to(device)
            print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")

            train_ds = VolumeDataset(CACHE_DIR, train_uids, train_labels, augment=True)
            val_ds = VolumeDataset(CACHE_DIR, val_uids, val_labels_np.tolist(), augment=False)
            train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=0, pin_memory=True)

            t0 = time.time()
            model, val_loss = train_model(model, train_loader, val_loader, device, epochs=80)
            print(f"    Done: val_loss={val_loss:.4f}, time={time.time()-t0:.0f}s")

            ckpt_path = os.path.join(OUTPUT_DIR, f"{name}_fold{fold}.pt")
            torch.save({"arch": arch, "state": model.state_dict()}, ckpt_path)

            model.eval()
            batch_start = 0
            with torch.no_grad():
                for volumes, _ in val_loader:
                    volumes = volumes.to(device)
                    with torch.amp.autocast("cuda", enabled=True):
                        logits = model(volumes)
                    preds = torch.sigmoid(logits).cpu().numpy()
                    batch_end = batch_start + len(preds)
                    oof_preds[val_idx[batch_start:batch_end], model_idx] = preds
                    batch_start = batch_end
            del model
            torch.cuda.empty_cache()

        oof_mask[val_idx] = True
        fold_preds = oof_preds[val_idx].mean(axis=1)
        fold_ll = log_loss(val_labels_np, np.clip(fold_preds, 1e-7, 1 - 1e-7))
        fold_auc = roc_auc_score(val_labels_np, fold_preds)
        print(f"\n  Fold {fold} LogLoss: {fold_ll:.4f}, AUROC: {fold_auc:.4f}")

    # Overall
    overall_preds = oof_preds[oof_mask].mean(axis=1)
    overall_true = labels_arr[oof_mask]
    overall_ll = log_loss(overall_true, np.clip(overall_preds, 1e-7, 1 - 1e-7))
    overall_auc = roc_auc_score(overall_true, overall_preds)
    print(f"\n{'='*60} OVERALL {'='*60}")
    print(f"LogLoss: {overall_ll:.4f}, AUROC: {overall_auc:.4f}")

    np.save(os.path.join(OUTPUT_DIR, "oof_preds.npy"), oof_preds)
    np.save(os.path.join(OUTPUT_DIR, "oof_labels.npy"), labels_arr)
    np.save(os.path.join(OUTPUT_DIR, "oof_mask.npy"), oof_mask)
    with open(os.path.join(LOG_DIR, "oof_metrics.json"), "w") as f:
        json.dump({"overall_logloss": float(overall_ll), "overall_auc": float(overall_auc)}, f, indent=2)
    print("Done")


if __name__ == "__main__":
    main()
