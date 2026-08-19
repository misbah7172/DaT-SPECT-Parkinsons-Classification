"""V17: 3-projection DenseNet - axial/coronal/sagittal max projections as RGB channels."""
import json, os, sys, time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

NIFTI_DIR = r"E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip"
CACHE_DIR = "proj_cache"
LABELS = "Dataset/train_labels.csv"
SITE = "Dataset/site_labels.csv"
SEEDS = [42, 777]
N_FOLDS = 5
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 25
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_volume(data):
    data = np.nan_to_num(data).astype(np.float32)
    brain_mask = data > (data.max() * 0.01)
    if brain_mask.sum() < 100:
        brain_mask = data > 0
    brain_vals = data[brain_mask]
    if len(brain_vals) < 10:
        return data
    p1, p99 = np.percentile(brain_vals, [1, 99])
    data = np.clip(data, p1, p99)
    data = (data - brain_vals.mean()) / (brain_vals.std() + 1e-8)
    return data


def make_projections(data, size=128):
    """Create 3 orthogonal max-projections and resize to size x size."""
    # Axial: max along z (axis 2)
    axial = np.max(data, axis=2)
    # Coronal: max along y (axis 1)
    coronal = np.max(data, axis=1)
    # Sagittal: max along x (axis 0)
    sagittal = np.max(data, axis=0)

    def resize_2d(img, target_size):
        factors = (target_size / img.shape[0], target_size / img.shape[1])
        return zoom(img, factors, order=3).astype(np.float32)

    axial = resize_2d(axial, size)
    coronal = resize_2d(coronal, size)
    sagittal = resize_2d(sagittal, size)

    # Stack as 3 channels
    return np.stack([axial, coronal, sagittal], axis=0)  # (3, H, W)


def preprocess_nifti(uid):
    nifti_path = os.path.join(NIFTI_DIR, uid + ".nii.gz")
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)
        vs = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    except:
        return None

    data = normalize_volume(data)

    # Resample to 2mm isotropic
    target_shape = np.round(data.shape * vs / 2.0).astype(int)
    factors = target_shape / np.array(data.shape)
    data = zoom(data, factors, order=3)

    # Create projections
    return make_projections(data, IMG_SIZE).astype(np.float32)


def cache_all(uids):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = 0
    t0 = time.time()
    for i, uid in enumerate(uids):
        cache_path = os.path.join(CACHE_DIR, uid + ".npy")
        if os.path.exists(cache_path):
            continue
        vol = preprocess_nifti(uid)
        if vol is not None:
            np.save(cache_path, vol)
            cached += 1
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print("  %d/%d cached=%d time=%.0fs" % (i + 1, len(uids), cached, elapsed))
    print("  Done: cached=%d, time=%.0fs" % (cached, time.time() - t0))


class ProjDaT(Dataset):
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
            proj = np.load(cache_path)
        else:
            proj = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

        if self.augment:
            if np.random.random() < 0.3:
                proj = proj * (0.9 + 0.2 * np.random.random())
            if np.random.random() < 0.2:
                proj = proj + np.random.normal(0, 0.05, proj.shape).astype(np.float32)
            # Random horizontal flip (on axial view = flip axis 1 of channel 0)
            if np.random.random() < 0.3:
                proj = np.flip(proj, axis=2).copy()

        return torch.FloatTensor(proj), torch.tensor(label, dtype=torch.float32)


class ProjDenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        # 3 channels = RGB, no modification needed
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
            probs = torch.sigmoid(model(X.to(DEVICE))).cpu().numpy()
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

            train_ds = ProjDaT(uids[tr_idx], y[tr_idx], augment=True)
            val_ds = ProjDaT(uids[va_idx], y[va_idx], augment=False)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model = ProjDenseNet(pretrained=True).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            criterion = nn.BCEWithLogitsLoss()

            best_ll = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(EPOCHS):
                train_one_epoch(model, train_loader, optimizer, criterion)
                scheduler.step()

                if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
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
            torch.save(model.state_dict(), "submission_v17/weights/fold_s%d_f%d.pth" % (seed, fold))
            print("    Best LL=%.4f" % best_ll)

    return np.where(counts > 0, oof / counts, 0.5)


def main():
    labels = pd.read_csv(LABELS)
    site = pd.read_csv(SITE)
    df = labels.merge(site, on="uid")
    y = df["is_pathologic"].values
    groups = df["pseudo_site"].astype(str).values
    uids = df["uid"].values

    # Cache projections
    print("Step 1: Caching 3-projection images...")
    cache_all(uids)

    cached_uids = np.array([uid for uid in uids if os.path.exists(os.path.join(CACHE_DIR, uid + ".npy"))])
    mask = np.isin(uids, cached_uids)
    print("Cached: %d/%d samples" % (mask.sum(), len(y)))

    y_c = y[mask]
    groups_c = groups[mask]
    uids_c = uids[mask]

    print("\nStep 2: Training 3-projection DenseNet121 OOF...")
    oof_image = train_oof(uids_c, y_c, groups_c, SEEDS)

    ll_raw = log_loss(y_c, np.clip(oof_image, 1e-6, 1 - 1e-6))
    auc_raw = roc_auc_score(y_c, oof_image)
    print("\nImage OOF: AUC=%.4f, LL=%.4f" % (auc_raw, ll_raw))

    temp = optimize_temperature(oof_image, y_c)
    p_temp = expit(logit(np.clip(oof_image, 1e-6, 1 - 1e-6)) / temp)
    ll_temp = log_loss(y_c, np.clip(p_temp, 1e-6, 1 - 1e-6))
    auc_temp = roc_auc_score(y_c, p_temp)
    print("Image+Temp (T=%.4f): AUC=%.4f, LL=%.4f" % (temp, auc_temp, ll_temp))

    # Train final models
    print("\nStep 3: Training final models on all data...")
    for seed in SEEDS:
        ds = ProjDaT(uids_c, y_c, augment=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model = ProjDenseNet(pretrained=True).to(DEVICE)
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
            print("  Final seed %d, epoch %d" % (seed, epoch + 1))
        torch.save(model.state_dict(), "submission_v17/weights/final_s%d.pth" % seed)

    results = {
        "n_samples": int(mask.sum()),
        "image_auroc": round(float(auc_raw), 4),
        "image_ll": round(float(ll_raw), 4),
        "image_temp_auroc": round(float(auc_temp), 4),
        "image_temp_ll": round(float(ll_temp), 4),
        "temperature": round(float(temp), 4),
    }
    with open("submission_v17/results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save("submission_v17/oof_image.npy", oof_image)
    print("\nDone!")


if __name__ == "__main__":
    os.makedirs("submission_v17/weights", exist_ok=True)
    main()
