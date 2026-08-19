"""
V16: Pre-cache niftis + efficient 2.5D DenseNet training.
Step 1: Pre-cache all niftis as numpy arrays
Step 2: Train 2.5D model with OOF
Step 3: Blend with tabular model
"""
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

NIFTI_DIR = r"E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip"
CACHE_DIR = "nifti_cache"
LABELS = "Dataset/train_labels.csv"
SITE = "Dataset/site_labels.csv"

SEEDS = [42, 777]
N_FOLDS = 5
CROP_SIZE = 96
N_SLICES = 5
TARGET_SPACING = 2.0
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def preprocess_nifti(uid):
    nifti_path = os.path.join(NIFTI_DIR, f"{uid}.nii.gz")
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)
        vs = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    except:
        return None

    data = normalize_volume(data)

    target_shape = np.round(data.shape * vs / TARGET_SPACING).astype(int)
    factors = target_shape / np.array(data.shape)
    data = zoom(data, factors, order=3)

    half = CROP_SIZE // 2
    cx, cy, cz = [s // 2 for s in data.shape]
    
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(data.shape[0], x0 + CROP_SIZE)
    y1 = min(data.shape[1], y0 + CROP_SIZE)
    
    vol = data[x0:x1, y0:y1, :]
    
    if vol.shape[0] < CROP_SIZE or vol.shape[1] < CROP_SIZE:
        padded = np.zeros((CROP_SIZE, CROP_SIZE, vol.shape[2]), dtype=np.float32)
        padded[:vol.shape[0], :vol.shape[1], :] = vol
        vol = padded
    
    z_center = vol.shape[2] // 2
    half_slices = N_SLICES // 2
    z_start = max(0, z_center - half_slices)
    z_end = min(vol.shape[2], z_start + N_SLICES)
    if z_end - z_start < N_SLICES:
        z_start = max(0, z_end - N_SLICES)
    
    slices = vol[:, :, z_start:z_start + N_SLICES]
    if slices.shape[2] < N_SLICES:
        padded = np.zeros((CROP_SIZE, CROP_SIZE, N_SLICES), dtype=np.float32)
        padded[:, :, :slices.shape[2]] = slices
        slices = padded
    
    return slices.astype(np.float32)


def cache_all_niftis(uids):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = 0
    skipped = 0
    t0 = time.time()
    
    for i, uid in enumerate(uids):
        cache_path = os.path.join(CACHE_DIR, f"{uid}.npy")
        if os.path.exists(cache_path):
            skipped += 1
            continue
        
        vol = preprocess_nifti(uid)
        if vol is not None:
            np.save(cache_path, vol)
            cached += 1
        else:
            skipped += 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(uids) - i - 1) / rate
            print(f"  {i+1}/{len(uids)}: cached={cached}, skipped={skipped}, rate={rate:.1f}/s, ETA={eta:.0f}s")
    
    print(f"Done: cached={cached}, skipped={skipped}, time={time.time()-t0:.1f}s")


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
        
        cache_path = os.path.join(CACHE_DIR, f"{uid}.npy")
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
        
        volume = np.transpose(volume, (2, 0, 1))  # (N_SLICES, CROP_SIZE, CROP_SIZE)
        return torch.FloatTensor(volume), torch.tensor(label, dtype=torch.float32)


class DenseNet2D(nn.Module):
    def __init__(self, n_slices=5, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        
        # DenseNet expects 3 channels; we have 5 slices
        # Modify first conv to accept 5 channels
        orig_conv = self.densenet.features.conv0
        new_conv = nn.Conv2d(n_slices, orig_conv.out_channels, 
                            kernel_size=orig_conv.kernel_size,
                            stride=orig_conv.stride,
                            padding=orig_conv.padding,
                            bias=orig_conv.bias is not None)
        
        # Initialize: copy RGB weights for first 3 channels, mean for extra channels
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_conv.weight
            for c in range(3, n_slices):
                new_conv.weight[:, c] = orig_conv.weight.mean(dim=1)
        
        self.densenet.features.conv0 = new_conv
        
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )
    
    def forward(self, x):
        return self.densenet(x).squeeze(-1)


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
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    return np.array(all_probs), np.array(all_labels)


def optimize_temperature(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    def loss(log_t):
        p = expit(logit(oof) / np.exp(log_t))
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    from scipy.optimize import minimize
    r = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return float(np.exp(r.x[0]))


def train_oof(uids, y, groups, seeds):
    n_samples = len(y)
    oof = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    
    for seed in seeds:
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(uids, y, groups)):
            print(f"  Seed {seed}, Fold {fold}: train={len(tr_idx)}, val={len(va_idx)}")
            
            train_ds = CachedDaT(uids[tr_idx], y[tr_idx], augment=True)
            val_ds = CachedDaT(uids[va_idx], y[va_idx], augment=False)
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=0, pin_memory=True)
            
            model = DenseNet2D(n_slices=N_SLICES, pretrained=True).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            criterion = nn.BCEWithLogitsLoss()
            
            best_ll = float("inf")
            best_state = None
            patience = 7
            no_improve = 0
            
            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
                scheduler.step()
                
                if (epoch + 1) % 3 == 0 or epoch == EPOCHS - 1:
                    probs, labels = evaluate(model, val_loader)
                    ll = log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6))
                    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
                    print(f"    Epoch {epoch+1}: loss={train_loss:.4f} LL={ll:.4f} AUC={auc:.4f}")
                    
                    if ll < best_ll:
                        best_ll = ll
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            print(f"    Early stopping at epoch {epoch+1}")
                            break
            
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(DEVICE)
            
            probs, _ = evaluate(model, val_loader)
            oof[va_idx] += probs
            counts[va_idx] += 1
            
            # Save fold model
            torch.save(model.state_dict(), f"submission_v16/weights/fold_seed{seed}_fold{fold}.pth")
            print(f"    Val LL={best_ll:.4f}")
    
    oof = np.where(counts > 0, oof / counts, 0.5)
    return oof


def main():
    print("Loading labels...")
    labels = pd.read_csv(LABELS)
    site = pd.read_csv(SITE)
    df = labels.merge(site, on="uid")
    y = df["is_pathologic"].values
    groups = df["pseudo_site"].astype(str).values
    uids = df["uid"].values
    print(f"  n={len(y)}, pos={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
    print(f"  Device: {DEVICE}")
    
    print("\nStep 1: Pre-cache all niftis...")
    cache_all_niftis(uids)
    
    print("\nStep 2: Training 2.5D DenseNet121 OOF...")
    oof_image = train_oof(uids, y, groups, SEEDS)
    
    ll_raw = log_loss(y, np.clip(oof_image, 1e-6, 1-1e-6))
    auc_raw = roc_auc_score(y, oof_image)
    print(f"\nImage OOF: AUC={auc_raw:.4f}, LL={ll_raw:.4f}")
    
    temp = optimize_temperature(oof_image, y)
    p_temp = expit(logit(np.clip(oof_image, 1e-6, 1-1e-6)) / temp)
    ll_temp = log_loss(y, np.clip(p_temp, 1e-6, 1-1e-6))
    auc_temp = roc_auc_score(y, p_temp)
    print(f"Image + Temp (T={temp:.4f}): AUC={auc_temp:.4f}, LL={ll_temp:.4f}")
    
    results = {
        "image_auroc": round(float(auc_raw), 4),
        "image_ll": round(float(ll_raw), 4),
        "image_temp_auroc": round(float(auc_temp), 4),
        "image_temp_ll": round(float(ll_temp), 4),
        "temperature": round(float(temp), 4),
    }
    with open("submission_v16/image_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    np.save("submission_v16/oof_image.npy", oof_image)
    print(f"\nResults saved to submission_v16/")
    
    # Train final model on all data
    print("\nStep 3: Training final model on all data...")
    for seed in SEEDS:
        ds = CachedDaT(uids, y, augment=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        
        model = DenseNet2D(n_slices=N_SLICES, pretrained=True).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        model.train()
        for epoch in range(5):
            for X, y_batch in loader:
                X, y_batch = X.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            print(f"  Seed {seed}, Epoch {epoch+1} done")
        
        torch.save(model.state_dict(), f"submission_v16/weights/final_seed{seed}.pth")
        print(f"  Saved final seed {seed}")
    
    print("\nDone!")


if __name__ == "__main__":
    os.makedirs("submission_v16/weights", exist_ok=True)
    main()
