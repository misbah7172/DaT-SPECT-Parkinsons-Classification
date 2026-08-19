"""
V22 Submission: Train with sbr + phys only (the two we can extract at inference).
Mega ensemble: global 12-stream + per-site + MLP.
"""
import os, json, warnings, time, shutil
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb, catboost as cb

SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
PVE_REF = 15.625
OUT = "submission_v22"


def load_data():
    labels = pd.read_csv("Dataset/train_labels.csv")
    site = pd.read_csv("Dataset/site_labels.csv")
    geom = pd.read_csv("Dataset/voxel_geometry.csv")
    sbr = pd.read_csv("Dataset/sbr_features_train.csv")
    phys = pd.read_csv("Dataset/phys_features_train.csv")
    merged = labels.merge(site, on="uid").merge(geom, on="uid")
    raw1 = [c for c in sbr.columns if c not in ("uid", "label")]
    merged = merged.merge(sbr[["uid"] + raw1], on="uid")
    merged = merged.merge(phys[["uid"] + [c for c in phys.columns if c != "uid"]].rename(
        columns={c: f"ph_{c}" for c in phys.columns if c != "uid"}), on="uid")
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
    params = {}
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            a, b, _ = optimize_platt(blend[mask], y[mask])
            params[site] = (float(a), float(b))
    return params


def train_mlp(X_tr, y_tr, X_va, y_va, input_dim, seed, arch="256_128"):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            layers = []
            if arch == "256_128":
                layers = [nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(128, 1)]
            elif arch == "128_64_32":
                layers = [nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
                          nn.Linear(32, 1)]
            else:
                layers = [nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                          nn.Linear(64, 1)]
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    sc = StandardScaler().fit(X_tr)
    Xtr_t = torch.FloatTensor(sc.transform(X_tr))
    Xva_t = torch.FloatTensor(sc.transform(X_va))
    ytr_t = torch.FloatTensor(y_tr).unsqueeze(1)
    yva_t = torch.FloatTensor(y_va).unsqueeze(1)

    pos_w = torch.FloatTensor([(len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-6)])
    model = MLP(input_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_ll, best_state = 1e9, None
    for ep in range(100):
        model.train()
        bs = 64
        perm = np.random.permutation(len(Xtr_t))
        for i in range(0, len(Xtr_t), bs):
            xb = Xtr_t[perm[i:i+bs]]
            yb = ytr_t[perm[i:i+bs]]
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(Xva_t)).numpy().ravel()
            ll = log_loss(y_va, np.clip(pred, 1e-6, 1-1e-6))
        if ll < best_ll:
            best_ll = ll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xva_t)).numpy().ravel(), sc, model


def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "weights"), exist_ok=True)

    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    merged_adv = add_adv(merged.copy())

    # Build 2 datasets: sbr (with bio/adv) + phys
    skip = {"uid", "is_pathologic", "pseudo_site", "site_id", "sx", "sy", "sz",
            "voxel_vol", "voxel_side", "label"}
    raw1 = [c for c in merged_adv.columns if not c.startswith("ph_") and c not in skip]
    raw2 = [c for c in merged_adv.columns if c.startswith("ph_")]
    X1 = build_matrix(merged_adv, raw1)
    X2 = build_matrix(merged_adv, raw2)
    Xs = [X1, X2]
    uids = merged["uid"].values
    print(f"Features: {[X.shape[1] for X in Xs]}")

    n_ds = 2
    n_m = 6
    n_streams = n_ds * n_m  # 12
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    ds_names = ["sbr", "phys"]
    stream_names = [f"{ds}_{m}" for ds in ds_names for m in mnames]

    # ============================================================
    # PART 1: Train global 12-stream models on full data
    # ============================================================
    print("\n=== Training global 12-stream models ===")
    for seed_idx, seed in enumerate(SEEDS):
        for xi, X in enumerate(Xs):
            sc = StandardScaler().fit(X)
            Xs_scaled = sc.transform(X)
            mt = min(44, Xs_scaled.shape[1])
            sel = np.argsort(mutual_info_classif(Xs_scaled, y, random_state=seed))[::-1][:mt]
            Xt = Xs_scaled[:, sel]
            b = xi * n_m
            prefix = os.path.join(OUT, "weights", f"s{seed_idx}")

            lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xs_scaled, y)
            joblib.dump(lr, f"{prefix}_ds{xi}_lr.pkl")
            joblib.dump(sc, f"{prefix}_ds{xi}_scaler.pkl")
            joblib.dump(sel, f"{prefix}_ds{xi}_mi.pkl")

            xgb_m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                use_label_encoder=False, eval_metric="logloss",
                random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y)
            joblib.dump(xgb_m, f"{prefix}_ds{xi}_xgb.pkl")

            lgb_m = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y)
            joblib.dump(lgb_m, f"{prefix}_ds{xi}_lgb.pkl")

            cb_m = cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y)
            joblib.dump(cb_m, f"{prefix}_ds{xi}_cb.pkl")

            et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y)
            joblib.dump(et, f"{prefix}_ds{xi}_et.pkl")

            ridge = RidgeClassifier(alpha=0.1, random_state=seed).fit(Xs_scaled, y)
            joblib.dump(ridge, f"{prefix}_ds{xi}_ridge.pkl")

        print(f"  Global seed {seed_idx} done")

    # ============================================================
    # PART 2: Train per-site models
    # ============================================================
    print("\n=== Training per-site models ===")
    X_tab = np.hstack(Xs)
    persite_models = {}
    for site in np.unique(groups):
        site_mask = groups == site
        if site_mask.sum() < 20 or len(np.unique(y[site_mask])) < 2:
            continue
        site_idx = np.where(site_mask)[0]
        site_y = y[site_idx]
        site_X = X_tab[site_idx]
        sc = StandardScaler().fit(site_X)
        Xs_s = sc.transform(site_X)
        mt = min(44, Xs_s.shape[1])
        sel = np.argsort(mutual_info_classif(Xs_s, site_y, random_state=42))[::-1][:mt]
        Xt_s = Xs_s[:, sel]

        lr = LogisticRegression(C=1.31, max_iter=3000, random_state=42).fit(Xs_s, site_y)
        xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0).fit(Xt_s, site_y)
        lgb_m = lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1).fit(Xt_s, site_y)
        site_safe = site.replace(".", "_")
        joblib.dump(lr, os.path.join(OUT, "weights", f"ps_{site_safe}_lr.pkl"))
        joblib.dump(xgb_m, os.path.join(OUT, "weights", f"ps_{site_safe}_xgb.pkl"))
        joblib.dump(lgb_m, os.path.join(OUT, "weights", f"ps_{site_safe}_lgb.pkl"))
        joblib.dump(sc, os.path.join(OUT, "weights", f"ps_{site_safe}_scaler.pkl"))
        joblib.dump(sel, os.path.join(OUT, "weights", f"ps_{site_safe}_mi.pkl"))
        persite_models[site] = True
        print(f"  Per-site: {site} (n={site_mask.sum()})")

    # ============================================================
    # PART 3: Train MLP ensemble
    # ============================================================
    print("\n=== Training MLP ensemble ===")
    mlp_preds = []
    for seed in SEEDS:
        for arch in ["256_128", "128_64_32"]:
            mlp_folds = np.zeros(len(y))
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_tab, y, groups):
                pred, sc, model = train_mlp(X_tab[tr], y[tr], X_tab[va], y[va], X_tab.shape[1], seed, arch)
                mlp_preds.append(pred)
                mlp_folds[va] = pred
            ll = log_loss(y, np.clip(mlp_folds, 1e-6, 1-1e-6))
            auc = roc_auc_score(y, mlp_folds)
            print(f"  MLP {arch} seed={seed}: AUC={auc:.4f} LL={ll:.4f}")

    # ============================================================
    # PART 4: Compute OOF for blend weights
    # ============================================================
    print("\n=== Computing OOF for blend weights ===")
    n = len(y)
    oof_global = [np.zeros(n) for _ in range(n_streams)]
    cnt_global = [np.zeros(n) for _ in range(n_streams)]
    oof_ps_lr = np.zeros(n)
    oof_ps_xgb = np.zeros(n)
    oof_ps_lgb = np.zeros(n)
    cnt_ps = np.zeros(n)
    oof_mlp = np.zeros(n)
    cnt_mlp = np.zeros(n)

    for seed in SEEDS:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_tab, y, groups):
            for xi, X in enumerate(Xs):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                mt = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mt]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                b = xi * n_m
                oof_global[b][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt_global[b][va] += 1
                oof_global[b+1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_global[b+1][va] += 1
                oof_global[b+2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_global[b+2][va] += 1
                oof_global[b+3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_global[b+3][va] += 1
                oof_global[b+4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt_global[b+4][va] += 1
                oof_global[b+5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt_global[b+5][va] += 1

            # Per-site OOF
            for site in np.unique(groups[tr]):
                s_mask = groups[tr] == site
                if s_mask.sum() < 15:
                    continue
                s_tr = tr[s_mask]
                s_va_idx = np.intersect1d(va, np.where(groups == site)[0])
                if len(s_va_idx) < 2:
                    continue
                site_X_tr = X_tab[s_tr]
                site_X_va = X_tab[s_va_idx]
                site_y_tr = y[s_tr]
                sc_ps = StandardScaler().fit(site_X_tr)
                Xtr_ps = sc_ps.transform(site_X_tr)
                Xva_ps = sc_ps.transform(site_X_va)
                mt_ps = min(44, Xtr_ps.shape[1])
                sel_ps = np.argsort(mutual_info_classif(Xtr_ps, site_y_tr, random_state=seed))[::-1][:mt_ps]

                oof_ps_lr[s_va_idx] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr_ps, site_y_tr).predict_proba(Xva_ps)[:, 1]
                oof_ps_xgb[s_va_idx] += xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xtr_ps[:, sel_ps], site_y_tr).predict_proba(Xva_ps[:, sel_ps])[:, 1]
                oof_ps_lgb[s_va_idx] += lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=5, random_state=seed, n_jobs=-1, verbose=-1).fit(Xtr_ps[:, sel_ps], site_y_tr).predict_proba(Xva_ps[:, sel_ps])[:, 1]
                cnt_ps[s_va_idx] += 1

            # MLP OOF
            pred_mlp, _, _ = train_mlp(X_tab[tr], y[tr], X_tab[va], y[va], X_tab.shape[1], seed, "128_64_32")
            oof_mlp[va] += pred_mlp
            cnt_mlp[va] += 1

        print(f"  OOF seed {seed}")

    for i in range(n_streams):
        oof_global[i] = np.where(cnt_global[i] > 0, oof_global[i] / cnt_global[i], 0.5)
    oof_ps_lr = np.where(cnt_ps > 0, oof_ps_lr / cnt_ps, 0.5)
    oof_ps_xgb = np.where(cnt_ps > 0, oof_ps_xgb / cnt_ps, 0.5)
    oof_ps_lgb = np.where(cnt_ps > 0, oof_ps_lgb / cnt_ps, 0.5)
    oof_mlp = np.where(cnt_mlp > 0, oof_mlp / cnt_mlp, 0.5)

    # ============================================================
    # PART 5: Optimize blend weights
    # ============================================================
    print("\n=== Optimizing blend weights ===")
    # Global blend
    M_g = np.column_stack(oof_global)
    w_g = optim_blend(M_g, y)
    blend_global = M_g @ w_g
    a_g, b_g, p_g = optimize_platt(blend_global, y)
    ll_g = log_loss(y, np.clip(p_g, 1e-6, 1-1e-6))
    auc_g = roc_auc_score(y, p_g)
    print(f"  Global: AUC={auc_g:.4f} LL={ll_g:.4f}")

    # Per-site blend
    M_ps = np.column_stack([oof_ps_lr, oof_ps_xgb, oof_ps_lgb])
    w_ps = optim_blend(M_ps, y)
    blend_ps = M_ps @ w_ps
    a_ps, b_ps, p_ps = optimize_platt(blend_ps, y)
    ll_ps = log_loss(y, np.clip(p_ps, 1e-6, 1-1e-6))
    auc_ps = roc_auc_score(y, p_ps)
    print(f"  Per-site: AUC={auc_ps:.4f} LL={ll_ps:.4f}")

    # MLP
    a_m, b_m, p_mlp = optimize_platt(oof_mlp, y)
    ll_mlp = log_loss(y, np.clip(p_mlp, 1e-6, 1-1e-6))
    auc_mlp = roc_auc_score(y, p_mlp)
    print(f"  MLP: AUC={auc_mlp:.4f} LL={ll_mlp:.4f}")

    # Mega blend: global + per-site + MLP
    M_mega = np.column_stack([p_g, p_ps, p_mlp])
    w_mega = optim_blend(M_mega, y)
    blend_mega = M_mega @ w_mega
    a_mega, b_mega, p_mega = optimize_platt(blend_mega, y)
    ll_mega = log_loss(y, np.clip(p_mega, 1e-6, 1-1e-6))
    auc_mega = roc_auc_score(y, p_mega)
    print(f"\n  MEGA (raw): AUC={auc_mega:.4f} LL={ll_mega:.4f}")

    # Site calibration
    sc_params = site_calibrate(blend_mega, y, groups)
    p_sc = blend_mega.copy()
    for site, (sa, sb) in sc_params.items():
        mask = groups == site
        p_sc[mask] = expit(sa * logit(np.clip(blend_mega[mask], 1e-6, 1-1e-6)) + sb)
    ll_sc = log_loss(y, np.clip(p_sc, 1e-6, 1-1e-6))
    auc_sc = roc_auc_score(y, p_sc)
    print(f"  MEGA + SITE_CAL: AUC={auc_sc:.4f} LL={ll_sc:.4f}")

    # ============================================================
    # PART 6: Train MLPs on full data and save
    # ============================================================
    print("\n=== Training MLPs on full data ===")
    for seed in SEEDS:
        for arch in ["256_128", "128_64_32"]:
            import torch
            import torch.nn as nn
            torch.manual_seed(seed)
            np.random.seed(seed)

            class MLP(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    layers = []
                    if arch == "256_128":
                        layers = [nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(128, 1)]
                    elif arch == "128_64_32":
                        layers = [nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(32, 1)]
                    else:
                        layers = [nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Linear(64, 1)]
                    self.net = nn.Sequential(*layers)
                def forward(self, x):
                    return self.net(x)

            sc = StandardScaler().fit(X_tab)
            Xtr_t = torch.FloatTensor(sc.transform(X_tab))
            ytr_t = torch.FloatTensor(y).unsqueeze(1)
            pos_w = torch.FloatTensor([(len(y) - y.sum()) / (y.sum() + 1e-6)])
            model = MLP(X_tab.shape[1])
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            best_ll, best_state = 1e9, None
            for ep in range(100):
                model.train()
                bs = 64
                perm = np.random.permutation(len(Xtr_t))
                for i in range(0, len(Xtr_t), bs):
                    xb = Xtr_t[perm[i:i+bs]]
                    yb = ytr_t[perm[i:i+bs]]
                    loss = crit(model(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    pred = torch.sigmoid(model(Xtr_t)).numpy().ravel()
                    ll = log_loss(y, np.clip(pred, 1e-6, 1-1e-6))
                if ll < best_ll:
                    best_ll = ll
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": X_tab.shape[1],
                "arch": arch,
                "scaler_mean": sc.mean_,
                "scaler_scale": sc.scale_,
            }, os.path.join(OUT, "weights", f"mlp_{arch}_s{seed}.pt"))
            print(f"  Saved MLP {arch} seed={seed} (train LL={best_ll:.4f})")

    # ============================================================
    # Save config
    # ============================================================
    # Get raw sbr columns for inference
    sbr_raw = pd.read_csv("Dataset/sbr_features_train.csv")
    raw1_cols = [c for c in sbr_raw.columns if c not in ("uid", "label")]

    cfg = {
        "n_seeds": len(SEEDS),
        "n_datasets": n_ds,
        "model_names": stream_names,
        "global_blend_weights": w_g.tolist(),
        "persite_blend_weights": w_ps.tolist(),
        "mega_blend_weights": w_mega.tolist(),
        "platt_a_global": float(a_g), "platt_b_global": float(b_g),
        "platt_a_persite": float(a_ps), "platt_b_persite": float(b_ps),
        "platt_a_mlp": float(a_m), "platt_b_mlp": float(b_m),
        "platt_a_mega": float(a_mega), "platt_b_mega": float(b_mega),
        "site_cal_params": sc_params,
        "clip_min": 0.005, "clip_max": 0.995,
        "mi_top": 44,
        "persite_models": list(persite_models.keys()),
        "sbr_raw_columns": raw1_cols,
    }
    with open(os.path.join(OUT, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    with open(os.path.join(OUT, "oof_metrics.json"), "w") as f:
        json.dump({
            "oof_auroc": round(float(auc_sc), 4),
            "oof_ll": round(float(ll_sc), 4),
            "oof_auroc_raw": round(float(auc_mega), 4),
            "oof_ll_raw": round(float(ll_mega), 4),
        }, f, indent=2)

    # Copy extractors
    shutil.copy("src/sbr_extractor.py", os.path.join(OUT, "sbr_extractor.py"))
    shutil.copy("submission_v7/sbr_extractor_phys.py", os.path.join(OUT, "sbr_extractor_phys.py"))

    print(f"\nSaved to {OUT} ({time.time()-t0:.0f}s)")
    print(f"  OOF AUC={auc_sc:.4f} LL={ll_sc:.4f}")


if __name__ == "__main__":
    main()
