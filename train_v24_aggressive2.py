"""
V24: Per-site models + augment + MLP for DaT-SPECT.
Truly aggressive: per-site training, SMOTE, MLP, advanced stacking.
"""
import os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb, catboost as cb

SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
PVE_REF = 15.625


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
        df["lr_put_diff"] = L - R
    if "sbr_left_caudate" in df.columns and "sbr_right_caudate" in df.columns:
        L, R = df["sbr_left_caudate"].values, df["sbr_right_caudate"].values
        df["lr_cau_asym"] = (L - R) / (L + R + eps)
        df["lr_cau_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
        df["lr_cau_sum"] = L + R
    if "sbr_left_putamen_post" in df.columns and "sbr_left_putamen_ant" in df.columns:
        df["lp_post_ant"] = df["sbr_left_putamen_post"].values / (df["sbr_left_putamen_ant"].values + eps)
    if "sbr_right_putamen_post" in df.columns and "sbr_right_putamen_ant" in df.columns:
        df["rp_post_ant"] = df["sbr_right_putamen_post"].values / (df["sbr_right_putamen_ant"].values + eps)
    if "sbr_left_putamen" in df.columns and "sbr_left_caudate" in df.columns:
        df["lp_cr"] = df["sbr_left_putamen"].values / (df["sbr_left_caudate"].values + eps)
    if "sbr_right_putamen" in df.columns and "sbr_right_caudate" in df.columns:
        df["rp_cr"] = df["sbr_right_putamen"].values / (df["sbr_right_caudate"].values + eps)
    if "sbr_left_total" in df.columns and "sbr_right_total" in df.columns:
        L, R = df["sbr_left_total"].values, df["sbr_right_total"].values
        df["total_str"] = L + R
        df["lr_total_asym"] = (L - R) / (L + R + eps)
    return df


def add_adv(df):
    eps = 1e-6
    region_cols = [c for c in ["sbr_left_putamen", "sbr_right_putamen",
                                "sbr_left_caudate", "sbr_right_caudate"] if c in df.columns]
    if len(region_cols) >= 2:
        vals = df[region_cols].values
        df["striatal_mean"] = vals.mean(axis=1)
        df["striatal_std"] = vals.std(axis=1)
        df["striatal_cv"] = df["striatal_std"] / (df["striatal_mean"] + eps)
        df["striatal_range"] = vals.max(axis=1) - vals.min(axis=1)
    if "sbr_left_putamen" in df.columns and "sbr_right_putamen" in df.columns:
        df["lp_x_rp"] = df["sbr_left_putamen"].values * df["sbr_right_putamen"].values
    if "sbr_left_caudate" in df.columns and "sbr_right_caudate" in df.columns:
        df["lc_x_rc"] = df["sbr_left_caudate"].values * df["sbr_right_caudate"].values
    return df


def build_matrix(raw, raw_cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def build_tabular(merged):
    skip = {"uid", "is_pathologic", "pseudo_site", "site_id", "sx", "sy", "sz",
            "voxel_vol", "voxel_side", "label"}
    raw1 = [c for c in merged.columns if not c.startswith("ph_") and not c.startswith("mo_")
            and not c.startswith("at_") and c not in skip]
    raw2 = [c for c in merged.columns if c.startswith("ph_")]
    raw3 = [c for c in merged.columns if c.startswith("mo_")]
    raw4 = [c for c in merged.columns if c.startswith("at_")]
    return (build_matrix(merged, raw1), build_matrix(merged, raw2),
            build_matrix(merged, raw3), build_matrix(merged, raw4))


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
    return r.x[0], r.x[1], expit(r.x[0] * lg + r.x[1])


def site_calibrate(blend, y, groups):
    result = blend.copy()
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            a, b, _ = optimize_platt(blend[mask], y[mask])
            result[mask] = expit(a * logit(np.clip(blend[mask], 1e-6, 1-1e-6)) + b)
    return result


def run_oof(Xs, y, groups, seeds=SEEDS, weights=None, tag=""):
    n_ds = len(Xs)
    n_m = 6
    n = len(y)
    oof = [np.zeros(n) for _ in range(n_ds * n_m)]
    cnt = [np.zeros(n) for _ in range(n_ds * n_m)]
    for seed in seeds:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(Xs[0], y, groups):
            w_tr = weights[tr] if weights is not None else None
            for xi, X in enumerate(Xs):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                mt = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mt]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                b = xi * n_m
                kw = {"sample_weight": w_tr} if w_tr is not None else {}
                oof[b][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr], **kw).predict_proba(Xva)[:, 1]; cnt[b][va] += 1
                oof[b+1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr], **kw).predict_proba(Xv)[:, 1]; cnt[b+1][va] += 1
                oof[b+2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr], **kw).predict_proba(Xv)[:, 1]; cnt[b+2][va] += 1
                oof[b+3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr], **kw).predict_proba(Xv)[:, 1]; cnt[b+3][va] += 1
                oof[b+4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr], **kw).predict_proba(Xv)[:, 1]; cnt[b+4][va] += 1
                oof[b+5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr], **kw).decision_function(Xva)); cnt[b+5][va] += 1
        print(f"  {tag} seed {seed}")
    for i in range(n_ds * n_m):
        oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)
    return oof


def run_per_site_oof(Xs, y, groups, seeds=SEEDS, tag=""):
    """Train separate models per site. OOF by holding out entire site."""
    n = len(y)
    sites = np.unique(groups)
    oof_lr = np.zeros(n)
    oof_xgb = np.zeros(n)
    oof_lgb = np.zeros(n)
    cnt = np.zeros(n)

    for site in sites:
        site_mask = groups == site
        if site_mask.sum() < 20 or len(np.unique(y[site_mask])) < 2:
            continue
        site_idx = np.where(site_mask)[0]
        site_y = y[site_idx]
        site_Xs = [X[site_idx] for X in Xs]
        site_X = np.hstack(site_Xs)
        n_site = len(site_idx)

        for seed in seeds[:3]:
            for tr, va in StratifiedKFold(min(3, n_site // 5 + 1), shuffle=True, random_state=seed).split(site_X, site_y):
                if len(va) < 3:
                    continue
                sc = StandardScaler().fit(site_X[tr])
                Xtr, Xva = sc.transform(site_X[tr]), sc.transform(site_X[va])
                mt = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, site_y[tr], random_state=seed))[::-1][:mt]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                global_idx = site_idx[va]

                oof_lr[global_idx] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, site_y[tr]).predict_proba(Xva)[:, 1]
                oof_xgb[global_idx] += xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, site_y[tr]).predict_proba(Xv)[:, 1]
                oof_lgb[global_idx] += lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=5, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, site_y[tr]).predict_proba(Xv)[:, 1]
                cnt[global_idx] += 1

    oof_lr = np.where(cnt > 0, oof_lr / cnt, 0.5)
    oof_xgb = np.where(cnt > 0, oof_xgb / cnt, 0.5)
    oof_lgb = np.where(cnt > 0, oof_lgb / cnt, 0.5)
    return [oof_lr, oof_xgb, oof_lgb]


def run_mlp_oof(X_tab, y, groups, seeds=SEEDS, tag=""):
    """Simple MLP OOF."""
    try:
        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print(f"  {tag}: torch not available, skipping")
        return None

    n = len(y)
    oof = np.zeros(n)
    cnt = np.zeros(n)

    class MLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.net(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_tab, y, groups):
            sc = StandardScaler().fit(X_tab[tr])
            Xtr = sc.transform(X_tab[tr]).astype(np.float32)
            Xva = sc.transform(X_tab[va]).astype(np.float32)
            ytr = y[tr].astype(np.float32)
            yva = y[va].astype(np.float32)

            train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

            model = MLP(Xtr.shape[1]).to(device)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            best_state = None
            best_vl = float("inf")
            patience = 10
            no_improve = 0

            for epoch in range(100):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = criterion(model(xb).squeeze(-1), yb)
                    loss.backward()
                    opt.step()

                model.eval()
                with torch.no_grad():
                    va_pred = torch.sigmoid(model(torch.from_numpy(Xva).to(device)).squeeze(-1)).cpu().numpy()
                vl = criterion(torch.from_numpy(va_pred), torch.from_numpy(yva)).item()
                if vl < best_vl:
                    best_vl = vl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(torch.from_numpy(Xva).to(device)).squeeze(-1)).cpu().numpy()
            oof[va] += pred
            cnt[va] += 1
            del model
            torch.cuda.empty_cache()
        print(f"  {tag} seed {seed}")

    oof = np.where(cnt > 0, oof / cnt, 0.5)
    return oof


def main():
    t0 = time.time()
    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    merged_adv = add_adv(merged.copy())
    Xs_adv = build_tabular(merged_adv)
    X_tab = np.hstack(Xs_adv)
    print(f"Tab: {X_tab.shape}")

    results = {}

    # A: Advanced features baseline (from v22)
    print(f"\n{'='*60}")
    print("A: ADVANCED BASELINE")
    print("=" * 60)
    oof_a = run_oof(Xs_adv, y, groups, tag="A")
    M_a = np.column_stack(oof_a)
    w_a = optim_blend(M_a, y)
    blend_a = M_a @ w_a
    a, b, p = optimize_platt(blend_a, y)
    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
    auc = roc_auc_score(y, p)
    p_sc = site_calibrate(blend_a, y, groups)
    ll_sc = log_loss(y, np.clip(p_sc, 1e-6, 1-1e-6))
    auc_sc = roc_auc_score(y, p_sc)
    print(f"  BASE: AUC={auc:.4f} LL={ll:.4f} | +SITE_CAL: AUC={auc_sc:.4f} LL={ll_sc:.4f}")
    results["A_base"] = {"auc": float(auc), "ll": float(ll), "auc_sc": float(auc_sc), "ll_sc": float(ll_sc)}

    # B: Per-site models
    print(f"\n{'='*60}")
    print("B: PER-SITE MODELS")
    print("=" * 60)
    oof_b = run_per_site_oof(Xs_adv, y, groups, tag="B")
    if oof_b:
        M_b = np.column_stack(oof_b)
        w_b = optim_blend(M_b, y)
        blend_b = M_b @ w_b
        a, b_p, p = optimize_platt(blend_b, y)
        ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
        auc = roc_auc_score(y, p)
        p_sc = site_calibrate(blend_b, y, groups)
        ll_sc = log_loss(y, np.clip(p_sc, 1e-6, 1-1e-6))
        auc_sc = roc_auc_score(y, p_sc)
        print(f"  PER-SITE: AUC={auc:.4f} LL={ll:.4f} | +SITE_CAL: AUC={auc_sc:.4f} LL={ll_sc:.4f}")
        results["B_per_site"] = {"auc": float(auc), "ll": float(ll), "auc_sc": float(auc_sc), "ll_sc": float(ll_sc)}

    # C: MLP
    print(f"\n{'='*60}")
    print("C: MLP")
    print("=" * 60)
    oof_c = run_mlp_oof(X_tab, y, groups, tag="C")
    if oof_c is not None:
        a, b_p, p = optimize_platt(oof_c, y)
        ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
        auc = roc_auc_score(y, p)
        p_sc = site_calibrate(oof_c, y, groups)
        ll_sc = log_loss(y, np.clip(p_sc, 1e-6, 1-1e-6))
        auc_sc = roc_auc_score(y, p_sc)
        print(f"  MLP: AUC={auc:.4f} LL={ll:.4f} | +SITE_CAL: AUC={auc_sc:.4f} LL={ll_sc:.4f}")
        results["C_mlp"] = {"auc": float(auc), "ll": float(ll), "auc_sc": float(auc_sc), "ll_sc": float(ll_sc)}

    # D: Ensemble of global + MLP + per-site
    print(f"\n{'='*60}")
    print("D: ENSEMBLE OF BEST")
    print("=" * 60)
    blends_to_ensemble = [blend_a]
    blend_names = ["global"]
    if oof_b:
        blends_to_ensemble.append(blend_b)
        blend_names.append("per_site")
    if oof_c is not None:
        blends_to_ensemble.append(oof_c)
        blend_names.append("mlp")

    M_d = np.column_stack(blends_to_ensemble)
    w_d = optim_blend(M_d, y)
    blend_d = M_d @ w_d
    a, b_p, p = optimize_platt(blend_d, y)
    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
    auc = roc_auc_score(y, p)
    p_sc = site_calibrate(blend_d, y, groups)
    ll_sc = log_loss(y, np.clip(p_sc, 1e-6, 1-1e-6))
    auc_sc = roc_auc_score(y, p_sc)
    print(f"  ENSEMBLE: AUC={auc:.4f} LL={ll:.4f} | +SITE_CAL: AUC={auc_sc:.4f} LL={ll_sc:.4f}")
    print(f"    weights: {dict(zip(blend_names, w_d.tolist()))}")
    results["D_ensemble"] = {"auc": float(auc), "ll": float(ll), "auc_sc": float(auc_sc), "ll_sc": float(ll_sc)}

    # E: Global ensemble + per-site OOF blended
    print(f"\n{'='*60}")
    print("E: GLOBAL 24-STREAM + 3 PER-SITE + MLP")
    print("=" * 60)
    all_streams = list(oof_a)
    if oof_b:
        all_streams.extend(oof_b)
    if oof_c is not None:
        all_streams.append(oof_c)
    M_e = np.column_stack(all_streams)
    w_e = optim_blend(M_e, y)
    blend_e = M_e @ w_e
    a, b_p, p = optimize_platt(blend_e, y)
    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
    auc = roc_auc_score(y, p)
    p_sc = site_calibrate(blend_e, y, groups)
    ll_sc = log_loss(y, np.clip(p_sc, 1e-6, 1-1e-6))
    auc_sc = roc_auc_score(y, p_sc)
    print(f"  MEGA_ENSEMBLE: AUC={auc:.4f} LL={ll:.4f} | +SITE_CAL: AUC={auc_sc:.4f} LL={ll_sc:.4f}")
    results["E_mega_ensemble"] = {"auc": float(auc), "ll": float(ll), "auc_sc": float(auc_sc), "ll_sc": float(ll_sc)}

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    base_ll = results["A_base"]["ll_sc"]
    print(f"{'Experiment':40s} {'AUC':>8s} {'LL':>8s} {'AUC+SC':>8s} {'LL+SC':>8s} {'Delta':>10s}")
    print("-" * 80)
    for name in sorted(results.keys()):
        r = results[name]
        delta = base_ll - r.get("ll_sc", r.get("ll", 0))
        marker = " **" if delta > 0.01 else (" *" if delta > 0.003 else "")
        print(f"  {name:38s} {r.get('auc',0):8.4f} {r.get('ll',0):8.4f} {r.get('auc_sc',0):8.4f} {r.get('ll_sc',0):8.4f} {delta:+10.4f}{marker}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    with open("v24_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved v24_results.json")


if __name__ == "__main__":
    main()
