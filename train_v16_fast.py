"""V16: Train 2.5D DenseNet on cached nifti slices + blend with tabular."""
import json, os, sys, time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

CACHE_DIR = "nifti_cache"
LABELS = "Dataset/train_labels.csv"
SITE = "Dataset/site_labels.csv"
SEEDS = [42, 777]
N_FOLDS = 5
CROP_SIZE = 96
N_SLICES = 5
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class CachedDaT(Dataset):
    def __init__(self, uids, labels, augment=False):
        self.uids = uids
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        label = self.labels[idx]
        cache_path = os.path.join(CACHE_DIR, uid + ".npy")
        if os.path.exists(cache_path):
            volume = np.load(cache_path)
        else:
            volume = np.zeros((CROP_SIZE, CROP_SIZE, N_SLICES), dtype=np.float32)

        if self.augment:
            if np.random.random() < 0.3:
                volume = volume * (0.9 + 0.2 * np.random.random())
            if np.random.random() < 0.2:
                volume = volume + np.random.normal(0, 0.05, volume.shape).astype(np.float32)
            if np.random.random() < 0.3:
                shift = np.random.randint(-2, 3)
                volume = np.roll(volume, shift, axis=2)

        volume = np.transpose(volume, (2, 0, 1))
        return torch.FloatTensor(volume), torch.tensor(label, dtype=torch.float32)


class DenseNet2D(nn.Module):
    def __init__(self, n_slices=5, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        orig_conv = self.densenet.features.conv0
        new_conv = nn.Conv2d(n_slices, orig_conv.out_channels,
                            kernel_size=orig_conv.kernel_size,
                            stride=orig_conv.stride,
                            padding=orig_conv.padding,
                            bias=orig_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_conv.weight
            for c in range(3, n_slices):
                new_conv.weight[:, c] = orig_conv.weight.mean(dim=1)
        self.densenet.features.conv0 = new_conv
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.densenet(x).squeeze(-1)


def optimize_temperature(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    def loss(log_t):
        p = expit(logit(oof) / np.exp(log_t))
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    r = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return float(np.exp(r.x[0]))


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            probs = torch.sigmoid(model(X)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    return np.array(all_probs), np.array(all_labels)


def train_oof(uids, y, groups, seeds):
    n_samples = len(y)
    oof = np.zeros(n_samples)
    counts = np.zeros(n_samples)

    for seed in seeds:
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(uids, y, groups)):
            print("  Seed %d, Fold %d: train=%d, val=%d" % (seed, fold, len(tr_idx), len(va_idx)))

            train_ds = CachedDaT(uids[tr_idx], y[tr_idx], augment=True)
            val_ds = CachedDaT(uids[va_idx], y[va_idx], augment=False)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model = DenseNet2D(n_slices=N_SLICES, pretrained=True).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            criterion = nn.BCEWithLogitsLoss()

            best_ll = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(EPOCHS):
                train_one_epoch(model, train_loader, optimizer, criterion)
                scheduler.step()

                if (epoch + 1) % 3 == 0 or epoch == EPOCHS - 1:
                    probs, labels = evaluate(model, val_loader)
                    ll = log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6))
                    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
                    print("    Epoch %d: LL=%.4f AUC=%.4f" % (epoch + 1, ll, auc))
                    if ll < best_ll:
                        best_ll = ll
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= 5:
                            print("    Early stop")
                            break

            if best_state:
                model.load_state_dict(best_state)
                model.to(DEVICE)
            probs, _ = evaluate(model, val_loader)
            oof[va_idx] += probs
            counts[va_idx] += 1
            torch.save(model.state_dict(), "submission_v16/weights/fold_s%d_f%d.pth" % (seed, fold))
            print("    Best LL=%.4f" % best_ll)

    return np.where(counts > 0, oof / counts, 0.5)


def main():
    labels = pd.read_csv(LABELS)
    site = pd.read_csv(SITE)
    df = labels.merge(site, on="uid")
    y = df["is_pathologic"].values
    groups = df["pseudo_site"].astype(str).values
    uids = df["uid"].values

    # Only use cached files
    cached_uids = np.array([uid for uid in uids if os.path.exists(os.path.join(CACHE_DIR, uid + ".npy"))])
    mask = np.isin(uids, cached_uids)
    print("Cached: %d/%d samples" % (mask.sum(), len(y)))

    y_cached = y[mask]
    groups_cached = groups[mask]
    uids_cached = uids[mask]

    print("Training 2.5D DenseNet121...")
    oof_image = train_oof(uids_cached, y_cached, groups_cached, SEEDS)

    ll_raw = log_loss(y_cached, np.clip(oof_image, 1e-6, 1 - 1e-6))
    auc_raw = roc_auc_score(y_cached, oof_image)
    print("Image OOF: AUC=%.4f, LL=%.4f" % (auc_raw, ll_raw))

    temp = optimize_temperature(oof_image, y_cached)
    p_temp = expit(logit(np.clip(oof_image, 1e-6, 1 - 1e-6)) / temp)
    ll_temp = log_loss(y_cached, np.clip(p_temp, 1e-6, 1 - 1e-6))
    auc_temp = roc_auc_score(y_cached, p_temp)
    print("Image+Temp (T=%.4f): AUC=%.4f, LL=%.4f" % (temp, auc_temp, ll_temp))

    results = {
        "n_samples": int(mask.sum()),
        "image_auroc": round(float(auc_raw), 4),
        "image_ll": round(float(ll_raw), 4),
        "image_temp_auroc": round(float(auc_temp), 4),
        "image_temp_ll": round(float(ll_temp), 4),
        "temperature": round(float(temp), 4),
    }
    with open("submission_v16/image_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save("submission_v16/oof_image.npy", oof_image)

    print("Training final models on all cached data...")
    for seed in SEEDS:
        ds = CachedDaT(uids_cached, y_cached, augment=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model = DenseNet2D(n_slices=N_SLICES, pretrained=True).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(5):
            for X, yb in loader:
                X, yb = X.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X), yb)
                loss.backward()
                optimizer.step()
            print("  Final seed %d, epoch %d done" % (seed, epoch + 1))
        torch.save(model.state_dict(), "submission_v16/weights/final_s%d.pth" % seed)

    print("Done!")


if __name__ == "__main__":
    main()
