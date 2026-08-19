"""
V19: Fine-tuned ResNet18 + Tabular hybrid for DaT-SPECT.

Key differences from v18:
- CNN is TRAINED (not frozen) on the classification task
- Fine-tuned layer4 + custom classification head
- Site-aware OOF with StratifiedGroupKFold
- Tested on projections (3ch) and slices (5ch)
- Fuses fine-tuned CNN OOF probabilities with tabular ensemble
"""
import os, json, warnings, time, copy
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb, lightgbm as lgb, catboost as cb
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
PVE_REF = 15.625
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


def load_data():
    labels = pd.read_csv("Dataset/train_labels.csv")
    site = pd.read_csv("Dataset/site_labels.csv")
    geom = pd.read_csv("Dataset/voxel_geometry.csv")
    sbr = pd.read_csv("Dataset/sbr_features_train.csv")
    phys = pd.read_csv("Dataset/phys_features_train.csv")
    morph = pd.read_csv("Dataset/sbr_features_morph_train.csv")
    atlas = pd.read_csv("Dataset/atlas_features_train.csv")
    merged = labels.merge(site, on="uid").merge(geom, on="uid")
    raw1 = [c for c in sbr.columns if c not in ("uid", "label")]
    merged = merged.merge(sbr[["uid"] + raw1], on="uid")
    merged = merged.merge(phys[["uid"] + [c for c in phys.columns if c != "uid"]].rename(
        columns={c: f"ph_{c}" for c in phys.columns if c != "uid"}), on="uid")
    merged = merged.merge(morph[["uid"] + [c for c in morph.columns if c != "uid"]].rename(
        columns={c: f"mo_{c}" for c in morph.columns if c != "uid"}), on="uid")
    merged = merged.merge(atlas[["uid"] + [c for c in atlas.columns if c != "uid"]].rename(
        columns={c: f"at_{c}" for c in atlas.columns if c != "uid"}), on="uid")
    merged = merged.dropna()
    y = merged["is_pathologic"].values
    groups = merged["pseudo_site"].astype(str).values
    fac = (PVE_REF / merged["voxel_vol"].values) ** (1.0 / 3.0)
    for c in [c for c in raw1 if c.startswith("sbr_") and c in merged.columns]:
        merged[c] = merged[c] * fac
    return merged, y, groups


def add_bio(df):
    eps = 1e-6
    if "sbr_left_putamen" in df.columns and "sbr_right_putamen" in df.columns:
        L, R = df["sbr_left_putamen"].values, df["sbr_right_putamen"].values
        df["lr_put_asym"] = (L - R) / (L + R + eps)
        df["lr_put_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
        df["lr_put_sum"] = L + R
    if "sbr_left_caudate" in df.columns and "sbr_right_caudate" in df.columns:
        L, R = df["sbr_left_caudate"].values, df["sbr_right_caudate"].values
        df["lr_cau_asym"] = (L - R) / (L + R + eps)
        df["lr_cau_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
    if "sbr_left_total" in df.columns and "sbr_right_total" in df.columns:
        L, R = df["sbr_left_total"].values, df["sbr_right_total"].values
        df["total_str"] = L + R
        df["lr_total_asym"] = (L - R) / (L + R + eps)
    return df


def build_matrix(raw, raw_cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def build_tabular(merged):
    skip = ("uid", "is_pathologic", "pseudo_site", "site_id", "sx", "sy", "sz",
            "voxel_vol", "voxel_side", "label")
    raw1 = [c for c in merged.columns if not c.startswith("ph_") and not c.startswith("mo_")
            and not c.startswith("at_") and c not in skip]
    raw2 = [c for c in merged.columns if c.startswith("ph_")]
    raw3 = [c for c in merged.columns if c.startswith("mo_")]
    raw4 = [c for c in merged.columns if c.startswith("at_")]
    return (build_matrix(merged, raw1), build_matrix(merged, raw2),
            build_matrix(merged, raw3), build_matrix(merged, raw4))


# ============================================================
# CNN FINE-TUNING
# ============================================================
class ProjDataset(Dataset):
    def __init__(self, imgs, labels=None, augment=False):
        self.imgs = imgs
        self.labels = labels
        self.augment = augment
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]) if augment else None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = self.imgs[idx].copy()
        if self.transform:
            x = self.transform(torch.from_numpy(x))
        else:
            x = torch.from_numpy(x)
        if self.labels is not None:
            return x, torch.tensor(self.labels[idx], dtype=torch.float32)
        return x


class SliceDataset(Dataset):
    def __init__(self, imgs, labels=None, augment=False):
        self.imgs = imgs
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = self.imgs[idx].copy()
        if self.augment:
            if np.random.random() < 0.5:
                x = x[:, :, ::-1].copy()
            if np.random.random() < 0.2:
                x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        if self.labels is not None:
            return x, torch.tensor(self.labels[idx], dtype=torch.float32)
        return x


def build_finetuned_resnet18(channels=3, n_classes=1):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if channels != 3:
        old = m.conv1
        m.conv1 = nn.Conv2d(channels, old.out_channels, old.kernel_size,
                             old.stride, old.padding, bias=False)
        with torch.no_grad():
            m.conv1.weight = nn.Parameter(old.weight.mean(1, keepdim=True).repeat(1, channels, 1, 1))

    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer4.parameters():
        p.requires_grad = True

    feat_dim = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(feat_dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes)
    )
    return m


def finetune_cnn(model, train_loader, val_loader, epochs=15, lr=1e-4, wd=1e-4):
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_train, lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_val_loss = float("inf")
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batch = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                out = model(x).squeeze(-1)
                val_preds.append(torch.sigmoid(out).cpu().numpy())
                val_labels.append(y.numpy())
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_loss = criterion(torch.from_numpy(val_preds), torch.from_numpy(val_labels)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    return model


def extract_cnn_oof(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for x in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(DEVICE)
            out = model(x).squeeze(-1)
            preds.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(preds)


def extract_cnn_features(model, loader):
    model.eval()
    feats_list = []
    feat_model = nn.Sequential(*list(model.children())[:-1])
    feat_model.to(DEVICE)
    with torch.no_grad():
        for x in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(DEVICE)
            f = feat_model(x).squeeze(-1).squeeze(-1).cpu().numpy()
            feats_list.append(f)
    del feat_model
    return np.concatenate(feats_list)


# ============================================================
# LOADING CACHE
# ============================================================
def load_projection_cache(uids):
    imgs = []
    valid = []
    for uid in uids:
        p = f"proj_cache/{uid}.npy"
        if os.path.exists(p):
            imgs.append(np.load(p))
            valid.append(True)
        else:
            imgs.append(np.zeros((3, 128, 128), dtype=np.float32))
            valid.append(False)
    return np.stack(imgs), np.array(valid)


def load_slice_cache(uids):
    imgs = []
    valid = []
    for uid in uids:
        p = f"nifti_cache/{uid}.npy"
        if os.path.exists(p):
            imgs.append(np.load(p))
            valid.append(True)
        else:
            imgs.append(np.zeros((96, 96, 5), dtype=np.float32))
            valid.append(False)
    return np.stack(imgs), np.array(valid)


# ============================================================
# TABULAR OOF
# ============================================================
def oof_tabular(Xs, y, groups):
    n_ds = len(Xs)
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    n_m = len(mnames)
    n = len(y)
    oof = [np.zeros(n) for _ in range(n_ds * n_m)]
    cnt = [np.zeros(n) for _ in range(n_ds * n_m)]
    for seed in SEEDS:
        for xi, X in enumerate(Xs):
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                mi_top = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mi_top]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                b = xi * n_m
                oof[b][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt[b][va] += 1
                oof[b+1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[b+1][va] += 1
                oof[b+2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[b+2][va] += 1
                oof[b+3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[b+3][va] += 1
                oof[b+4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[b+4][va] += 1
                oof[b+5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt[b+5][va] += 1
        print(f"  seed {seed}")
    for i in range(n_ds * n_m):
        oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)
    names = [f"t_{ds}_{m}" for ds in ["sbr", "phys", "morph", "atlas"][:n_ds] for m in mnames]
    return oof, names


def oof_tabular_fast(Xs, y, groups):
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    n_m = len(mnames)
    n = len(y)
    oof = [np.zeros(n) for _ in range(n_m)]
    cnt = [np.zeros(n) for _ in range(n_m)]
    X_all = np.hstack(Xs)
    for seed in SEEDS:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_all, y, groups):
            sc = StandardScaler().fit(X_all[tr])
            Xtr, Xva = sc.transform(X_all[tr]), sc.transform(X_all[va])
            mi_top = min(44, Xtr.shape[1])
            sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mi_top]
            Xt, Xv = Xtr[:, sel], Xva[:, sel]
            oof[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt[0][va] += 1
            oof[1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[1][va] += 1
            oof[2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[2][va] += 1
            oof[3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[3][va] += 1
            oof[4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[4][va] += 1
            oof[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt[5][va] += 1
        print(f"  seed {seed}")
    for i in range(n_m):
        oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)
    return oof, [f"tab_{m}" for m in mnames]


# ============================================================
# BLENDING
# ============================================================
def optim_blend(M, y):
    def loss(w):
        return log_loss(y, np.clip(M @ w, 1e-6, 1 - 1e-6))
    r = minimize(loss, np.ones(M.shape[1]) / M.shape[1],
                 method="L-BFGS-B", bounds=[(0, 1)] * M.shape[1])
    return r.x / r.x.sum()


def optimize_platt(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    lg = logit(oof)
    def loss(p):
        return log_loss(y, np.clip(expit(p[0] * lg + p[1]), 1e-6, 1 - 1e-6))
    r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
    a, b = r.x
    return a, b, expit(a * lg + b)


def eval_oof(oof_list, y, names, tag=""):
    M = np.column_stack(oof_list)
    w = optim_blend(M, y)
    blend = M @ w
    a, b, p = optimize_platt(blend, y)
    auc = roc_auc_score(y, p)
    ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    print(f"  [{tag}] AUC={auc:.4f}  LL={ll:.4f}")
    return auc, ll


def eval_combined(tab_oof, cnn_oof, y, tag=""):
    M = np.column_stack(tab_oof + cnn_oof)
    w = optim_blend(M, y)
    blend = M @ w
    a, b, p = optimize_platt(blend, y)
    auc = roc_auc_score(y, p)
    ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    print(f"  [{tag}] AUC={auc:.4f}  LL={ll:.4f}")
    return auc, ll


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    Xs = build_tabular(merged)
    uids = merged["uid"].values

    # Load image caches
    print("\nLoading image caches...")
    proj_imgs, proj_valid = load_projection_cache(uids)
    slice_imgs, slice_valid = load_slice_cache(uids)
    print(f"  Projections: {proj_imgs.shape}, valid={proj_valid.sum()}")
    print(f"  Slices: {slice_imgs.shape}, valid={slice_valid.sum()}")

    # Tabular OOF
    print("\n=== TABULAR OOF ===")
    tab_oof, tab_names = oof_tabular(Xs, y, groups)
    tab_auc, tab_ll = eval_oof(tab_oof, y, tab_names, "TABULAR")

    results = {"tabular": {"auc": float(tab_auc), "ll": float(tab_ll)}}

    # ============================================================
    # CNN FINE-TUNING: Projections (3-channel, 128x128)
    # ============================================================
    print(f"\n{'='*60}")
    print("CNN FINE-TUNING: Projections (3ch, 128x128)")
    print(f"{'='*60}")

    SEEDS_CNN = [42, 777]
    n = len(y)

    # Store OOF predictions and features
    cnn_proj_oof = np.zeros(n)
    cnn_proj_feat = np.zeros((n, 512), dtype=np.float32)
    cnn_proj_counts = np.zeros(n)

    for seed in SEEDS_CNN:
        print(f"\n  Seed {seed}")
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(proj_imgs, y, groups)):
            model = build_finetuned_resnet18(channels=3)

            train_ds = ProjDataset(proj_imgs[tr], y[tr], augment=True)
            val_ds = ProjDataset(proj_imgs[va], y[va], augment=False)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

            model = finetune_cnn(model, train_loader, val_loader, epochs=15, lr=1e-4)

            preds = extract_cnn_oof(model, val_loader)
            cnn_proj_oof[va] += preds
            cnn_proj_counts[va] += 1

            # Extract features
            full_loader = DataLoader(ProjDataset(proj_imgs[va], augment=False), batch_size=64, shuffle=False, num_workers=0)
            feats = extract_cnn_features(model, full_loader)
            cnn_proj_feat[va] = feats

            del model, train_ds, val_ds, train_loader, val_loader
            torch.cuda.empty_cache()
            print(f"    fold {fold} done, val_size={len(va)}")

    cnn_proj_oof = np.where(cnn_proj_counts > 0, cnn_proj_oof / cnn_proj_counts, 0.5)

    try:
        cnn_proj_auc = roc_auc_score(y, cnn_proj_oof)
        cnn_proj_ll = log_loss(y, np.clip(cnn_proj_oof, 1e-6, 1-1e-6))
        print(f"\n  CNN Proj OOF: AUC={cnn_proj_auc:.4f}  LL={cnn_proj_ll:.4f}")
    except:
        print("  CNN Proj OOF failed")
        cnn_proj_auc, cnn_proj_ll = 0.5, 1.0

    results["cnn_proj"] = {"auc": float(cnn_proj_auc), "ll": float(cnn_proj_ll)}

    # ============================================================
    # CNN FINE-TUNING: Slices (5-channel, 96x96)
    # ============================================================
    print(f"\n{'='*60}")
    print("CNN FINE-TUNING: Slices (5ch, 96x96)")
    print(f"{'='*60}")

    cnn_sl_oof = np.zeros(n)
    cnn_sl_feat = np.zeros((n, 512), dtype=np.float32)
    cnn_sl_counts = np.zeros(n)

    for seed in SEEDS_CNN:
        print(f"\n  Seed {seed}")
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(slice_imgs, y, groups)):
            model = build_finetuned_resnet18(channels=5)

            train_ds = SliceDataset(slice_imgs[tr], y[tr], augment=True)
            val_ds = SliceDataset(slice_imgs[va], y[va], augment=False)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

            model = finetune_cnn(model, train_loader, val_loader, epochs=15, lr=1e-4)

            preds = extract_cnn_oof(model, val_loader)
            cnn_sl_oof[va] += preds
            cnn_sl_counts[va] += 1

            full_loader = DataLoader(SliceDataset(slice_imgs[va], augment=False), batch_size=64, shuffle=False, num_workers=0)
            feats = extract_cnn_features(model, full_loader)
            cnn_sl_feat[va] = feats

            del model, train_ds, val_ds, train_loader, val_loader
            torch.cuda.empty_cache()
            print(f"    fold {fold} done, val_size={len(va)}")

    cnn_sl_oof = np.where(cnn_sl_counts > 0, cnn_sl_oof / cnn_sl_counts, 0.5)

    try:
        cnn_sl_auc = roc_auc_score(y, cnn_sl_oof)
        cnn_sl_ll = log_loss(y, np.clip(cnn_sl_oof, 1e-6, 1-1e-6))
        print(f"\n  CNN Slice OOF: AUC={cnn_sl_auc:.4f}  LL={cnn_sl_ll:.4f}")
    except:
        print("  CNN Slice OOF failed")
        cnn_sl_auc, cnn_sl_ll = 0.5, 1.0

    results["cnn_slice"] = {"auc": float(cnn_sl_auc), "ll": float(cnn_sl_ll)}

    # ============================================================
    # FUSION: Tabular + CNN OOF
    # ============================================================
    print(f"\n{'='*60}")
    print("FUSION: Tabular + CNN OOF probabilities")
    print(f"{'='*60}")

    # Tabular + CNN proj OOF
    a, ll = eval_combined(tab_oof, [cnn_proj_oof], y, "TAB + CNN_Proj_OOF")
    results["tab_cnn_proj_oof"] = {"auc": float(a), "ll": float(ll)}

    # Tabular + CNN slice OOF
    a, ll = eval_combined(tab_oof, [cnn_sl_oof], y, "TAB + CNN_Slice_OOF")
    results["tab_cnn_slice_oof"] = {"auc": float(a), "ll": float(ll)}

    # Tabular + CNN proj + CNN slice OOF
    a, ll = eval_combined(tab_oof, [cnn_proj_oof, cnn_sl_oof], y, "TAB + CNN_Both_OOF")
    results["tab_cnn_both_oof"] = {"auc": float(a), "ll": float(ll)}

    # ============================================================
    # FUSION: Tabular + CNN features (from fine-tuned model)
    # ============================================================
    print(f"\n{'='*60}")
    print("FUSION: Tabular + CNN features (fine-tuned ResNet)")
    print(f"{'='*60}")

    X_tab = np.hstack(Xs)

    for n_pca in [None, 20, 50]:
        tag = f"TAB+Feat_Proj" + (f"_PCA{n_pca}" if n_pca else "")
        print(f"\n  {tag}")

        sc = StandardScaler()
        X_feat = sc.fit_transform(cnn_proj_feat)

        if n_pca and n_pca < X_feat.shape[1]:
            pca = PCA(n_components=n_pca)
            X_feat = pca.fit_transform(X_feat)

        X_both = np.hstack([X_tab, X_feat])
        mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
        n_m = len(mnames)
        oof_b = [np.zeros(n) for _ in range(n_m)]
        cnt_b = [np.zeros(n) for _ in range(n_m)]

        for seed in SEEDS:
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_both, y, groups):
                sc2 = StandardScaler().fit(X_both[tr])
                Xtr, Xva = sc2.transform(X_both[tr]), sc2.transform(X_both[va])
                mi_top = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mi_top]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]

                oof_b[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt_b[0][va] += 1
                oof_b[1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_b[1][va] += 1
                oof_b[2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_b[2][va] += 1
                oof_b[3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_b[3][va] += 1
                oof_b[4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_b[4][va] += 1
                oof_b[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt_b[5][va] += 1
            print(f"  seed {seed}")
        for i in range(n_m):
            oof_b[i] = np.where(cnt_b[i] > 0, oof_b[i] / cnt_b[i], 0.5)
        a, ll = eval_oof(oof_b, y, [f"f_{m}" for m in mnames], tag)
        results[f"tab_feat_proj" + (f"_pca{n_pca}" if n_pca else "")] = {"auc": float(a), "ll": float(ll)}

    # Same for slice features
    for n_pca in [None, 20]:
        tag = f"TAB+Feat_Slice" + (f"_PCA{n_pca}" if n_pca else "")
        print(f"\n  {tag}")

        sc = StandardScaler()
        X_feat = sc.fit_transform(cnn_sl_feat)
        if n_pca and n_pca < X_feat.shape[1]:
            pca = PCA(n_components=n_pca)
            X_feat = pca.fit_transform(X_feat)

        X_both = np.hstack([X_tab, X_feat])
        mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
        n_m = len(mnames)
        oof_b = [np.zeros(n) for _ in range(n_m)]
        cnt_b = [np.zeros(n) for _ in range(n_m)]

        for seed in SEEDS:
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_both, y, groups):
                sc2 = StandardScaler().fit(X_both[tr])
                Xtr, Xva = sc2.transform(X_both[tr]), sc2.transform(X_both[va])
                mi_top = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mi_top]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                oof_b[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt_b[0][va] += 1
                oof_b[1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_b[1][va] += 1
                oof_b[2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_b[2][va] += 1
                oof_b[3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_b[3][va] += 1
                oof_b[4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_b[4][va] += 1
                oof_b[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt_b[5][va] += 1
        for i in range(n_m):
            oof_b[i] = np.where(cnt_b[i] > 0, oof_b[i] / cnt_b[i], 0.5)
        a, ll = eval_oof(oof_b, y, [f"f_{m}" for m in mnames], tag)
        results[f"tab_feat_slice" + (f"_pca{n_pca}" if n_pca else "")] = {"auc": float(a), "ll": float(ll)}

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    base_ll = results["tabular"]["ll"]
    print(f"{'Experiment':40s} {'AUC':>8s} {'LogLoss':>8s} {'Delta LL':>10s}")
    print("-" * 70)
    for name in sorted(results.keys()):
        r = results[name]
        delta = base_ll - r["ll"]
        print(f"  {name:38s} {r['auc']:8.4f} {r['ll']:8.4f} {delta:+10.4f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    with open("v19_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved v19_results.json")


if __name__ == "__main__":
    main()
