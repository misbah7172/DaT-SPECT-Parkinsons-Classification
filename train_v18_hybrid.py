"""
V18: Hybrid 2D-to-3D + Tabular ensemble for DaT-SPECT.
Fast version: ResNet18 only, batched extraction.
"""
import os, json, warnings, time
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
import torch, torch.nn as nn
import torchvision.models as models

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
    if "sbr_left_putamen_post" in df.columns and "sbr_left_putamen_ant" in df.columns:
        df["lp_ratio"] = df["sbr_left_putamen_post"].values / (df["sbr_left_putamen_ant"].values + eps)
    if "sbr_right_putamen_post" in df.columns and "sbr_right_putamen_ant" in df.columns:
        df["rp_ratio"] = df["sbr_right_putamen_post"].values / (df["sbr_right_putamen_ant"].values + eps)
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
# BATCHED CNN FEATURE EXTRACTION
# ============================================================
def get_resnet18_ext(channels=3):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if channels != 3:
        old = m.conv1
        m.conv1 = nn.Conv2d(channels, old.out_channels, old.kernel_size,
                             old.stride, old.padding, bias=False)
        with torch.no_grad():
            m.conv1.weight = nn.Parameter(old.weight.mean(1, keepdim=True).repeat(1, channels, 1, 1))
    ext = nn.Sequential(*list(m.children())[:-1])
    ext.eval().to(DEVICE)
    return ext, 512


def extract_proj_feats(uids, batch_size=128):
    ext, dim = get_resnet18_ext(3)
    feats = np.zeros((len(uids), dim), dtype=np.float32)
    valid = np.zeros(len(uids), dtype=bool)

    batch_imgs = []
    batch_idx = []

    with torch.no_grad():
        for i, uid in enumerate(uids):
            p = f"proj_cache/{uid}.npy"
            if os.path.exists(p):
                batch_imgs.append(np.load(p))
                batch_idx.append(i)
                valid[i] = True
            if len(batch_imgs) == batch_size or (i == len(uids) - 1 and batch_imgs):
                x = torch.from_numpy(np.stack(batch_imgs)).to(DEVICE)
                f = ext(x).squeeze(-1).squeeze(-1).cpu().numpy()
                for j, idx in enumerate(batch_idx):
                    feats[idx] = f[j]
                batch_imgs, batch_idx = [], []
    del ext; torch.cuda.empty_cache()
    print(f"  Projections: {valid.sum()}/{len(uids)} valid, shape={feats.shape}")
    return feats, valid


def extract_slice_feats(uids, batch_size=16):
    ext, dim = get_resnet18_ext(5)
    n = len(uids)
    methods = ["mean", "max", "mean_std", "mean_max"]
    results = {}
    for m in methods:
        od = dim * 2 if m in ("mean_std", "mean_max") else dim
        results[m] = np.zeros((n, od), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    total_slices = 0
    t0 = time.time()

    with torch.no_grad():
        i = 0
        while i < n:
            batch_data = []
            batch_idx = []
            batch_nslices = []
            j = i
            while j < n and len(batch_data) < batch_size:
                uid = uids[j]
                p = f"nifti_cache/{uid}.npy"
                if os.path.exists(p):
                    sl = np.load(p)  # (96, 96, 5)
                    batch_data.append(sl)
                    batch_idx.append(j)
                    batch_nslices.append(sl.shape[2])
                    valid[j] = True
                j += 1

            if not batch_data:
                i = j
                continue

            all_slices = []
            slice_owner = []
            for k, sl in enumerate(batch_data):
                ns = sl.shape[2]
                for s in range(ns):
                    all_slices.append(sl[:, :, s])
                    slice_owner.append(k)

            x = torch.from_numpy(np.stack(all_slices)).unsqueeze(1).repeat(1, 5, 1, 1).float().to(DEVICE)
            f = ext(x).squeeze(-1).squeeze(-1).cpu().numpy()
            total_slices += len(all_slices)

            for k in range(len(batch_data)):
                owner_mask = np.array(slice_owner) == k
                sf = f[owner_mask]
                idx = batch_idx[k]
                results["mean"][idx] = sf.mean(0)
                results["max"][idx] = sf.max(0)
                results["mean_std"][idx] = np.concatenate([sf.mean(0), sf.std(0)])
                results["mean_max"][idx] = np.concatenate([sf.mean(0), sf.max(0)])

            i = j
            if (i // batch_size) % 3 == 0:
                print(f"    {i}/{n} subjects, {total_slices} slices ({time.time()-t0:.0f}s)")

    del ext; torch.cuda.empty_cache()
    print(f"  Slices: {valid.sum()}/{n} valid, {total_slices} total slices ({time.time()-t0:.0f}s)")
    return results, valid


# ============================================================
# BLENDING & CALIBRATION
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


def eval_blend(oof_list, y, names, tag=""):
    M = np.column_stack(oof_list)
    print(f"\n  [{tag}] Matrix: {M.shape}")
    best_per_stream = 0
    for i in range(M.shape[1]):
        try:
            auc = roc_auc_score(y, M[:, i])
            ll = log_loss(y, np.clip(M[:, i], 1e-6, 1 - 1e-6))
            print(f"    {names[i]:25s} AUC={auc:.4f}  LL={ll:.4f}")
            best_per_stream = max(best_per_stream, auc)
        except:
            pass
    w = optim_blend(M, y)
    blend = M @ w
    a, b, p_platt = optimize_platt(blend, y)
    ll_p = log_loss(y, np.clip(p_platt, 1e-6, 1 - 1e-6))
    auc_p = roc_auc_score(y, p_platt)
    print(f"  [{tag}] PLATT: AUC={auc_p:.4f}  LL={ll_p:.4f}")
    return auc_p, ll_p


# ============================================================
# OOF TRAINING FUNCTIONS
# ============================================================
def oof_tabular(Xs, y, groups):
    n_ds = len(Xs)
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    n_m = len(mnames)
    n = len(y)
    oof = [[np.zeros(n) for _ in range(n_ds * n_m)] for _ in range(2)]
    cnt = [[np.zeros(n) for _ in range(n_ds * n_m)] for _ in range(2)]

    for seed in SEEDS:
        for xi, X in enumerate(Xs):
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                mi_top = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mi_top]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                b = xi * n_m
                oof[0][b][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt[0][b][va] += 1
                oof[0][b+1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[0][b+1][va] += 1
                oof[0][b+2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[0][b+2][va] += 1
                oof[0][b+3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[0][b+3][va] += 1
                oof[0][b+4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[0][b+4][va] += 1
                oof[0][b+5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt[0][b+5][va] += 1
        print(f"  seed {seed} done")
    for i in range(n_ds * n_m):
        oof[0][i] = np.where(cnt[0][i] > 0, oof[0][i] / cnt[0][i], 0.5)
    names = [f"t_{ds}_{m}" for ds in ["sbr", "phys", "morph", "atlas"][:n_ds] for m in mnames]
    return oof[0], names


def oof_hybrid(X_tab, X_img, y, groups, tag="HYB"):
    n = len(y)
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    n_m = len(mnames)

    oof_tab = [np.zeros(n) for _ in range(n_m)]
    cnt_tab = [np.zeros(n) for _ in range(n_m)]
    oof_hyb = [np.zeros(n) for _ in range(n_m)]
    cnt_hyb = [np.zeros(n) for _ in range(n_m)]
    oof_img = [np.zeros(n) for _ in range(n_m)]
    cnt_img = [np.zeros(n) for _ in range(n_m)]

    X_both = np.hstack([X_tab, X_img])
    print(f"\n  [{tag}] Tab={X_tab.shape[1]}, Img={X_img.shape[1]}, Both={X_both.shape[1]}")

    for seed in SEEDS:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_tab, y, groups):
            sc_t = StandardScaler().fit(X_tab[tr])
            Xtr_t, Xva_t = sc_t.transform(X_tab[tr]), sc_t.transform(X_tab[va])
            sc_i = StandardScaler().fit(X_img[tr])
            Xtr_i, Xva_i = sc_i.transform(X_img[tr]), sc_i.transform(X_img[va])
            sc_b = StandardScaler().fit(X_both[tr])
            Xtr_b, Xva_b = sc_b.transform(X_both[tr]), sc_b.transform(X_both[va])

            mi_tab = min(44, Xtr_t.shape[1])
            sel_tab = np.argsort(mutual_info_classif(Xtr_t, y[tr], random_state=seed))[::-1][:mi_tab]
            mi_both = min(44, Xtr_b.shape[1])
            sel_both = np.argsort(mutual_info_classif(Xtr_b, y[tr], random_state=seed))[::-1][:mi_both]
            mi_img = min(44, Xtr_i.shape[1])
            sel_img = np.argsort(mutual_info_classif(Xtr_i, y[tr], random_state=seed))[::-1][:mi_img]

            Xtr_t2, Xva_t2 = Xtr_t[:, sel_tab], Xva_t[:, sel_tab]
            Xtr_b2, Xva_b2 = Xtr_b[:, sel_both], Xva_b[:, sel_both]
            Xtr_i2, Xva_i2 = Xtr_i[:, sel_img], Xva_i[:, sel_img]

            # Tabular only
            oof_tab[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr_t, y[tr]).predict_proba(Xva_t)[:, 1]; cnt_tab[0][va] += 1
            oof_tab[1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xtr_t2, y[tr]).predict_proba(Xva_t2)[:, 1]; cnt_tab[1][va] += 1
            oof_tab[2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xtr_t2, y[tr]).predict_proba(Xva_t2)[:, 1]; cnt_tab[2][va] += 1
            oof_tab[3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xtr_t2, y[tr]).predict_proba(Xva_t2)[:, 1]; cnt_tab[3][va] += 1
            oof_tab[4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xtr_t2, y[tr]).predict_proba(Xva_t2)[:, 1]; cnt_tab[4][va] += 1
            oof_tab[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr_t, y[tr]).decision_function(Xva_t)); cnt_tab[5][va] += 1

            # Hybrid
            oof_hyb[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr_b, y[tr]).predict_proba(Xva_b)[:, 1]; cnt_hyb[0][va] += 1
            oof_hyb[1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xtr_b2, y[tr]).predict_proba(Xva_b2)[:, 1]; cnt_hyb[1][va] += 1
            oof_hyb[2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xtr_b2, y[tr]).predict_proba(Xva_b2)[:, 1]; cnt_hyb[2][va] += 1
            oof_hyb[3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xtr_b2, y[tr]).predict_proba(Xva_b2)[:, 1]; cnt_hyb[3][va] += 1
            oof_hyb[4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xtr_b2, y[tr]).predict_proba(Xva_b2)[:, 1]; cnt_hyb[4][va] += 1
            oof_hyb[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr_b, y[tr]).decision_function(Xva_b)); cnt_hyb[5][va] += 1

            # Image only
            if Xtr_i2.shape[1] > 0:
                oof_img[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr_i, y[tr]).predict_proba(Xva_i)[:, 1]; cnt_img[0][va] += 1
                oof_img[1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xtr_i2, y[tr]).predict_proba(Xva_i2)[:, 1]; cnt_img[1][va] += 1
                oof_img[2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xtr_i2, y[tr]).predict_proba(Xva_i2)[:, 1]; cnt_img[2][va] += 1
                oof_img[3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xtr_i2, y[tr]).predict_proba(Xva_i2)[:, 1]; cnt_img[3][va] += 1
                oof_img[4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xtr_i2, y[tr]).predict_proba(Xva_i2)[:, 1]; cnt_img[4][va] += 1
                oof_img[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr_i, y[tr]).decision_function(Xva_i)); cnt_img[5][va] += 1
        print(f"  seed {seed} done")

    for i in range(n_m):
        oof_tab[i] = np.where(cnt_tab[i] > 0, oof_tab[i] / cnt_tab[i], 0.5)
        oof_hyb[i] = np.where(cnt_hyb[i] > 0, oof_hyb[i] / cnt_hyb[i], 0.5)
        oof_img[i] = np.where(cnt_img[i] > 0, oof_img[i] / cnt_img[i], 0.5)

    tab_names = [f"tab_{m}" for m in mnames]
    hyb_names = [f"hyb_{m}" for m in mnames]
    img_names = [f"img_{m}" for m in mnames]
    return (oof_tab, tab_names), (oof_img, img_names), (oof_hyb, hyb_names)


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    Xs = build_tabular(merged)
    X_tab = np.hstack(Xs)
    print(f"Tabular: {X_tab.shape}")
    uids = merged["uid"].values

    results = {}

    # === TABULAR ONLY BASELINE ===
    print("\n" + "=" * 60)
    print("A: TABULAR ONLY (baseline)")
    print("=" * 60)
    oof_t, n_t = oof_tabular(Xs, y, groups)
    auc_t, ll_t = eval_blend(oof_t, y, n_t, "TAB")
    results["A_tabular"] = {"tab_auc": auc_t, "tab_ll": ll_t}

    # === CNN FEATURES ===
    print("\n" + "=" * 60)
    print("Extracting ResNet18 projection features...")
    print("=" * 60)
    X_proj, proj_valid = extract_proj_feats(uids)

    print("\nExtracting ResNet18 slice features...")
    X_sl_dict, sl_valid = extract_slice_feats(uids)

    # === EXPERIMENT B: Projection features only (tabular+proj hybrid) ===
    print("\n" + "=" * 60)
    print("B: PROJECTION FEATURES")
    print("=" * 60)
    (oof_tab2, n_tab2), (oof_img2, n_img2), (oof_hyb2, n_hyb2) = oof_hybrid(X_tab, X_proj, y, groups, "PROJ")
    auc_tab2, ll_tab2 = eval_blend(oof_tab2, y, n_tab2, "TAB-REPRO")
    auc_img2, ll_img2 = eval_blend(oof_img2, y, n_img2, "PROJ-IMG")
    auc_hyb2, ll_hyb2 = eval_blend(oof_hyb2, y, n_hyb2, "PROJ-HYB")
    results["B_proj"] = {
        "tab_auc": auc_tab2, "tab_ll": ll_tab2,
        "img_auc": auc_img2, "img_ll": ll_img2,
        "hyb_auc": auc_hyb2, "hyb_ll": ll_hyb2
    }

    # === EXPERIMENT C: Projection + PCA ===
    for npc in [20, 50]:
        print(f"\n{'='*60}")
        print(f"C: PROJECTION + PCA {npc}")
        print(f"{'='*60}")
        pca = PCA(n_components=npc).fit(StandardScaler().fit_transform(X_proj))
        X_proj_pca = pca.fit_transform(StandardScaler().fit_transform(X_proj))
        (_, _), (_, _), (oof_h, n_h) = oof_hybrid(X_tab, X_proj_pca, y, groups, f"PROJ_PCA{npc}")
        a_h, l_h = eval_blend(oof_h, y, n_h, f"PROJ_PCA{npc}")
        results[f"C_proj_pca{npc}"] = {"hyb_auc": a_h, "hyb_ll": l_h}

    # === EXPERIMENT D: Slice features (various aggregations) ===
    for method in ["mean", "max", "mean_std", "mean_max"]:
        print(f"\n{'='*60}")
        print(f"D: SLICE FEATURES ({method})")
        print(f"{'='*60}")
        X_sl = X_sl_dict[method]
        X_sl = StandardScaler().fit_transform(X_sl)
        (_, _), (_, _), (oof_h, n_h) = oof_hybrid(X_tab, X_sl, y, groups, f"SLICE_{method}")
        a_h, l_h = eval_blend(oof_h, y, n_h, f"SLICE_{method}")
        results[f"D_slice_{method}"] = {"hyb_auc": a_h, "hyb_ll": l_h}

    # === EXPERIMENT E: Projection + Slice (both) ===
    valid_both = proj_valid & sl_valid
    if valid_both.sum() > 100:
        for method in ["mean", "max", "mean_std"]:
            print(f"\n{'='*60}")
            print(f"E: PROJ + SLICE ({method}), n={valid_both.sum()}")
            print(f"{'='*60}")
            X_both_img = np.hstack([X_proj, StandardScaler().fit_transform(X_sl_dict[method])])
            (_, _), (_, _), (oof_h, n_h) = oof_hybrid(X_tab, X_both_img, y, groups, f"BOTH_{method}")
            a_h, l_h = eval_blend(oof_h, y, n_h, f"BOTH_{method}")
            results[f"E_both_{method}"] = {"hyb_auc": a_h, "hyb_ll": l_h}

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':40s} {'AUC':>8s} {'LogLoss':>8s} {'vs baseline':>12s}")
    print("-" * 80)
    base = results["A_tabular"]["tab_ll"]
    for name, r in sorted(results.items()):
        a = r.get("hyb_auc", r.get("tab_auc", 0))
        l = r.get("hyb_ll", r.get("tab_ll", 0))
        delta = base - l
        sign = "+" if delta > 0 else ""
        print(f"{name:40s} {a:8.4f} {l:8.4f} {sign}{delta:.4f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    with open("v18_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved v18_results.json")


if __name__ == "__main__":
    main()
