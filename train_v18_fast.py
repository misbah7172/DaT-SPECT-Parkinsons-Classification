"""V18-fast: Quick test of projection features + tabular."""
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
import xgboost as xgb, lightgbm as lgb, catboost as cb
import torch, torch.nn as nn
import torchvision.models as models

SEEDS = [42, 777]
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


def extract_proj(uids, batch=256):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    ext = nn.Sequential(*list(m.children())[:-1]).eval().to(DEVICE)
    feats = np.zeros((len(uids), 512), dtype=np.float32)
    bi, bd = [], []
    with torch.no_grad():
        for i, uid in enumerate(uids):
            p = f"proj_cache/{uid}.npy"
            if os.path.exists(p):
                bd.append(np.load(p)); bi.append(i)
            if len(bd) == batch or (i == len(uids)-1 and bd):
                x = torch.from_numpy(np.stack(bd)).to(DEVICE)
                f = ext(x).squeeze(-1).squeeze(-1).cpu().numpy()
                for j, idx in enumerate(bi): feats[idx] = f[j]
                bd, bi = [], []
    del ext; torch.cuda.empty_cache()
    return feats


def extract_slices(uids, batch=8):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old = m.conv1
    m.conv1 = nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        m.conv1.weight = nn.Parameter(old.weight.mean(1, keepdim=True).repeat(1, 5, 1, 1))
    ext = nn.Sequential(*list(m.children())[:-1]).eval().to(DEVICE)
    n = len(uids)
    agg = {"mean": np.zeros((n, 512), dtype=np.float32),
           "max": np.zeros((n, 512), dtype=np.float32),
           "mean_std": np.zeros((n, 1024), dtype=np.float32),
           "mean_max": np.zeros((n, 1024), dtype=np.float32)}
    valid = np.zeros(n, dtype=bool)
    t0 = time.time()

    with torch.no_grad():
        i = 0
        while i < n:
            batch_d, batch_i = [], []
            j = i
            while j < n and len(batch_d) < batch:
                uid = uids[j]
                p = f"nifti_cache/{uid}.npy"
                if os.path.exists(p):
                    batch_d.append(np.load(p)); batch_i.append(j); valid[j] = True
                j += 1
            if not batch_d:
                i = j; continue
            all_sl, owner = [], []
            for k, sl in enumerate(batch_d):
                for s in range(sl.shape[2]):
                    all_sl.append(sl[:, :, s]); owner.append(k)
            x = torch.from_numpy(np.stack(all_sl)).unsqueeze(1).repeat(1, 5, 1, 1).float().to(DEVICE)
            f = ext(x).squeeze(-1).squeeze(-1).cpu().numpy()
            for k in range(len(batch_d)):
                mask = np.array(owner) == k
                sf = f[mask]; idx = batch_i[k]
                agg["mean"][idx] = sf.mean(0)
                agg["max"][idx] = sf.max(0)
                agg["mean_std"][idx] = np.concatenate([sf.mean(0), sf.std(0)])
                agg["mean_max"][idx] = np.concatenate([sf.mean(0), sf.max(0)])
            i = j
            if (i // batch) % 2 == 0:
                print(f"    {i}/{n} ({time.time()-t0:.0f}s)")
    del ext; torch.cuda.empty_cache()
    print(f"  Slices: {valid.sum()}/{n} valid ({time.time()-t0:.0f}s)")
    return agg, valid


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


def run_oof(X_tab, X_img, y, groups, tag=""):
    X_both = np.hstack([X_tab, X_img])
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    n_m = len(mnames)
    n = len(y)

    oof_t = [np.zeros(n) for _ in range(n_m)]; c_t = [np.zeros(n) for _ in range(n_m)]
    oof_i = [np.zeros(n) for _ in range(n_m)]; c_i = [np.zeros(n) for _ in range(n_m)]
    oof_b = [np.zeros(n) for _ in range(n_m)]; c_b = [np.zeros(n) for _ in range(n_m)]

    for seed in SEEDS:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_tab, y, groups):
            sc_t = StandardScaler().fit(X_tab[tr]); Xtr_t, Xva_t = sc_t.transform(X_tab[tr]), sc_t.transform(X_tab[va])
            sc_i = StandardScaler().fit(X_img[tr]); Xtr_i, Xva_i = sc_i.transform(X_img[tr]), sc_i.transform(X_img[va])
            sc_b = StandardScaler().fit(X_both[tr]); Xtr_b, Xva_b = sc_b.transform(X_both[tr]), sc_b.transform(X_both[va])

            sel_t = np.argsort(mutual_info_classif(Xtr_t, y[tr], random_state=seed))[::-1][:min(44, Xtr_t.shape[1])]
            sel_b = np.argsort(mutual_info_classif(Xtr_b, y[tr], random_state=seed))[::-1][:min(44, Xtr_b.shape[1])]
            sel_i = np.argsort(mutual_info_classif(Xtr_i, y[tr], random_state=seed))[::-1][:min(44, Xtr_i.shape[1])]
            Xtr_t2, Xva_t2 = Xtr_t[:, sel_t], Xva_t[:, sel_t]
            Xtr_b2, Xva_b2 = Xtr_b[:, sel_b], Xva_b[:, sel_b]
            Xtr_i2, Xva_i2 = Xtr_i[:, sel_i], Xva_i[:, sel_i]

            # Tabular
            oof_t[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr_t, y[tr]).predict_proba(Xva_t)[:, 1]; c_t[0][va] += 1
            oof_t[1][va] += xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xtr_t2, y[tr]).predict_proba(Xva_t2)[:, 1]; c_t[1][va] += 1
            oof_t[2][va] += lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xtr_t2, y[tr]).predict_proba(Xva_t2)[:, 1]; c_t[2][va] += 1
            oof_t[3][va] += cb.CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xtr_t2, y[tr]).predict_proba(Xva_t2)[:, 1]; c_t[3][va] += 1
            oof_t[4][va] += ExtraTreesClassifier(500, max_depth=9, n_jobs=-1, random_state=seed).fit(Xtr_t2, y[tr]).predict_proba(Xva_t2)[:, 1]; c_t[4][va] += 1
            oof_t[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr_t, y[tr]).decision_function(Xva_t)); c_t[5][va] += 1

            # Hybrid
            oof_b[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr_b, y[tr]).predict_proba(Xva_b)[:, 1]; c_b[0][va] += 1
            oof_b[1][va] += xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xtr_b2, y[tr]).predict_proba(Xva_b2)[:, 1]; c_b[1][va] += 1
            oof_b[2][va] += lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xtr_b2, y[tr]).predict_proba(Xva_b2)[:, 1]; c_b[2][va] += 1
            oof_b[3][va] += cb.CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xtr_b2, y[tr]).predict_proba(Xva_b2)[:, 1]; c_b[3][va] += 1
            oof_b[4][va] += ExtraTreesClassifier(500, max_depth=9, n_jobs=-1, random_state=seed).fit(Xtr_b2, y[tr]).predict_proba(Xva_b2)[:, 1]; c_b[4][va] += 1
            oof_b[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr_b, y[tr]).decision_function(Xva_b)); c_b[5][va] += 1

            # Image only
            oof_i[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr_i, y[tr]).predict_proba(Xva_i)[:, 1]; c_i[0][va] += 1
            oof_i[1][va] += xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xtr_i2, y[tr]).predict_proba(Xva_i2)[:, 1]; c_i[1][va] += 1
            oof_i[2][va] += lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xtr_i2, y[tr]).predict_proba(Xva_i2)[:, 1]; c_i[2][va] += 1
            oof_i[3][va] += cb.CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xtr_i2, y[tr]).predict_proba(Xva_i2)[:, 1]; c_i[3][va] += 1
            oof_i[4][va] += ExtraTreesClassifier(500, max_depth=9, n_jobs=-1, random_state=seed).fit(Xtr_i2, y[tr]).predict_proba(Xva_i2)[:, 1]; c_i[4][va] += 1
            oof_i[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr_i, y[tr]).decision_function(Xva_i)); c_i[5][va] += 1

        print(f"  seed {seed}")

    for i in range(n_m):
        oof_t[i] = np.where(c_t[i] > 0, oof_t[i] / c_t[i], 0.5)
        oof_b[i] = np.where(c_b[i] > 0, oof_b[i] / c_b[i], 0.5)
        oof_i[i] = np.where(c_i[i] > 0, oof_i[i] / c_i[i], 0.5)

    results = {}
    for label, oof_l, names in [("tab", oof_t, [f"tab_{m}" for m in mnames]),
                                  ("img", oof_i, [f"img_{m}" for m in mnames]),
                                  ("hyb", oof_b, [f"hyb_{m}" for m in mnames])]:
        M = np.column_stack(oof_l)
        w = optim_blend(M, y)
        blend = M @ w
        a, b_pl, p = optimize_platt(blend, y)
        auc = roc_auc_score(y, p)
        ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
        print(f"  {tag}/{label}: AUC={auc:.4f} LL={ll:.4f}")
        results[label] = {"auc": float(auc), "ll": float(ll)}
    return results


def main():
    t0 = time.time()
    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    Xs = build_tabular(merged)
    X_tab = np.hstack(Xs)
    print(f"Tabular: {X_tab.shape}")
    uids = merged["uid"].values

    all_r = {}

    # Extract features
    print("\nExtracting projection features...")
    t1 = time.time()
    X_proj = StandardScaler().fit_transform(extract_proj(uids))
    print(f"  Done: {X_proj.shape} ({time.time()-t1:.0f}s)")

    print("Extracting slice features...")
    t2 = time.time()
    sl_agg, sl_valid = extract_slices(uids)
    print(f"  Done ({time.time()-t2:.0f}s)")

    # A: Tabular only
    print("\n=== A: TABULAR ONLY ===")
    r = run_oof(X_tab, X_tab[:, :2], y, groups, "TAB_ONLY")
    all_r["A_tabular"] = r

    # B: Tabular + Projection
    print("\n=== B: TABULAR + PROJECTION ===")
    r = run_oof(X_tab, X_proj, y, groups, "TAB+PROJ")
    all_r["B_proj"] = r

    # C: Tabular + Slice (mean)
    print("\n=== C: TABULAR + SLICE (mean) ===")
    X_sl_m = StandardScaler().fit_transform(sl_agg["mean"])
    r = run_oof(X_tab, X_sl_m, y, groups, "TAB+SLICE_mean")
    all_r["C_slice_mean"] = r

    # D: Tabular + Slice (mean+std)
    print("\n=== D: TABULAR + SLICE (mean+std) ===")
    X_sl_ms = StandardScaler().fit_transform(sl_agg["mean_std"])
    r = run_oof(X_tab, X_sl_ms, y, groups, "TAB+SLICE_meanstd")
    all_r["D_slice_meanstd"] = r

    # E: Tabular + Proj + Slice
    print("\n=== E: TABULAR + PROJ + SLICE (mean) ===")
    X_both = np.hstack([X_proj, X_sl_m])
    r = run_oof(X_tab, X_both, y, groups, "TAB+PROJ+SLICE")
    all_r["E_proj+slice"] = r

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    base_ll = all_r["A_tabular"]["tab"]["ll"]
    print(f"{'Experiment':35s} {'Tab AUC':>8s} {'Tab LL':>8s} {'Hyb AUC':>8s} {'Hyb LL':>8s} {'Delta':>8s}")
    print("-" * 70)
    for name, r in sorted(all_r.items()):
        t = r.get("tab", {}); h = r.get("hyb", {})
        delta = base_ll - h.get("ll", 0)
        print(f"{name:35s} {t.get('auc',0):8.4f} {t.get('ll',0):8.4f} {h.get('auc',0):8.4f} {h.get('ll',0):8.4f} {delta:+8.4f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    with open("v18_results.json", "w") as f:
        json.dump(all_r, f, indent=2, default=str)
    print("Saved v18_results.json")


if __name__ == "__main__":
    main()
