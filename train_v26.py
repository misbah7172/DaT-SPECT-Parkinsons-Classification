"""V26: v25 mega ensemble + radiomics 5th stream. Saves all OOF streams for stacking."""
import os, json, warnings, time, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb, catboost as cb

SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
PVE_REF = 15.625
OUT = r'E:\DaT\v26_oof'


def load_data():
    labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
    site = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv')
    geom = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
    sbr = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv').drop(columns=['label'], errors='ignore')
    phys_raw = pd.read_csv(r'E:\DaT\Dataset\phys_features_train.csv')
    phys = phys_raw.rename(columns={c: f'ph_{c}' for c in phys_raw.columns if c != 'uid'})
    morph = pd.read_csv(r'E:\DaT\Dataset\sbr_features_morph_train.csv').drop(columns=['label'], errors='ignore')
    atlas = pd.read_csv(r'E:\DaT\Dataset\atlas_features_train.csv').drop(columns=['label'], errors='ignore')
    radiom = pd.read_csv(r'E:\DaT\radiomic_features.csv')
    merged = labels.merge(site, on='uid').merge(geom, on='uid').set_index('uid')
    raw1 = [c for c in sbr.columns if c != 'uid']
    fac = (PVE_REF / merged['voxel_vol'].values) ** (1.0 / 3.0)
    sbr.set_index('uid', inplace=True)
    sbr = sbr.loc[merged.index]
    for c in [c for c in raw1 if c.startswith('sbr_')]:
        sbr[c] = sbr[c] * fac
    parts = {
        'sbr': sbr,
        'phys': phys.set_index('uid').loc[merged.index],
        'morph': morph.set_index('uid').loc[merged.index],
        'atlas': atlas.set_index('uid').loc[merged.index],
        'radiom': radiom.set_index('uid').loc[merged.index],
    }
    y = merged['is_pathologic'].values
    groups = merged['pseudo_site'].astype(str).values
    return merged, parts, y, groups


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


def build_matrix(raw, raw_cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def build_streams(parts, merged):
    raw1 = [c for c in parts['sbr'].columns]
    sbr_m = add_bio(parts['sbr'].copy())
    streams = []
    for key in ['sbr', 'phys', 'morph', 'atlas', 'radiom']:
        d = sbr_m if key == 'sbr' else parts[key]
        streams.append(build_matrix(d, [c for c in d.columns]))
    return streams


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
    return r.x[0], r.x[1]


def site_calibrate(blend, y, groups):
    result = blend.copy()
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            a, b = optimize_platt(blend[mask], y[mask])
            result[mask] = expit(a * logit(np.clip(blend[mask], 1e-6, 1-1e-6)) + b)
    return result


def run_global_oof(Xs, y, groups, seeds=SEEDS, tag=""):
    n_ds = len(Xs)
    n_m = 6
    n = len(y)
    oof = [np.zeros(n) for _ in range(n_ds * n_m)]
    cnt = [np.zeros(n) for _ in range(n_ds * n_m)]
    for seed in seeds:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(Xs[0], y, groups):
            for xi, X in enumerate(Xs):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                mt = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mt]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                b = xi * n_m
                oof[b][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt[b][va] += 1
                oof[b+1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[b+1][va] += 1
                oof[b+2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[b+2][va] += 1
                oof[b+3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[b+3][va] += 1
                oof[b+4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[b+4][va] += 1
                oof[b+5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt[b+5][va] += 1
        print(f"  {tag} seed {seed}", flush=True)
    for i in range(n_ds * n_m):
        oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)
    return oof


def run_diverse_oof(X_tab, y, groups, seeds=SEEDS, tag=""):
    n = len(y)
    models_config = [
        ("xgb_deep", lambda s: xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, use_label_encoder=False, eval_metric="logloss", random_state=s, n_jobs=-1, verbosity=0)),
        ("xgb_shallow", lambda s: xgb.XGBClassifier(n_estimators=800, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, use_label_encoder=False, eval_metric="logloss", random_state=s, n_jobs=-1, verbosity=0)),
        ("lgb_deep", lambda s: lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, min_child_samples=5, random_state=s, n_jobs=-1, verbose=-1)),
        ("lgb_wide", lambda s: lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0, min_child_samples=15, random_state=s, n_jobs=-1, verbose=-1)),
        ("cb_deep", lambda s: cb.CatBoostClassifier(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=s, verbose=0)),
        ("cb_wide", lambda s: cb.CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05, l2_leaf_reg=2.0, random_seed=s, verbose=0)),
        ("et_1000", lambda s: ExtraTreesClassifier(1000, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=s)),
        ("et_500_shallow", lambda s: ExtraTreesClassifier(500, max_depth=6, min_samples_leaf=5, n_jobs=-1, random_state=s)),
        ("rf_500", lambda s: RandomForestClassifier(500, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=s)),
        ("knn_7", lambda s: KNeighborsClassifier(7)),
        ("knn_15", lambda s: KNeighborsClassifier(15)),
        ("lr_C01", lambda s: LogisticRegression(C=0.1, max_iter=3000, random_state=s)),
        ("lr_C5", lambda s: LogisticRegression(C=5.0, max_iter=3000, random_state=s)),
    ]
    n_models = len(models_config)
    oof = [np.zeros(n) for _ in range(n_models)]
    cnt = [np.zeros(n) for _ in range(n_models)]
    for seed in seeds:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_tab, y, groups):
            sc = StandardScaler().fit(X_tab[tr])
            Xtr, Xva = sc.transform(X_tab[tr]), sc.transform(X_tab[va])
            mt = min(44, Xtr.shape[1])
            sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mt]
            Xt, Xv = Xtr[:, sel], Xva[:, sel]
            for i, (name, builder) in enumerate(models_config):
                model = builder(seed)
                if name.startswith("knn"):
                    model.fit(Xtr, y[tr]); oof[i][va] += model.predict_proba(Xva)[:, 1]
                elif name.startswith("lr"):
                    model.fit(Xtr, y[tr]); oof[i][va] += model.predict_proba(Xva)[:, 1]
                else:
                    model.fit(Xt, y[tr]); oof[i][va] += model.predict_proba(Xv)[:, 1]
                cnt[i][va] += 1
        print(f"  {tag} seed {seed}", flush=True)
    for i in range(n_models):
        oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)
    return oof, [m[0] for m in models_config]


def run_per_site_oof(Xs, y, groups, seeds=SEEDS[:3], tag=""):
    n = len(y)
    sites = np.unique(groups)
    oof_lr = np.zeros(n); oof_xgb = np.zeros(n); oof_lgb = np.zeros(n)
    cnt = np.zeros(n)
    for site in sites:
        site_mask = groups == site
        if site_mask.sum() < 20 or len(np.unique(y[site_mask])) < 2:
            continue
        site_idx = np.where(site_mask)[0]
        site_y = y[site_idx]
        site_X = np.hstack([X[site_idx] for X in Xs])
        n_site = len(site_idx)
        n_splits = min(3, n_site // 10 + 1)
        for seed in seeds:
            for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(site_X, site_y):
                if len(va) < 2:
                    continue
                sc = StandardScaler().fit(site_X[tr])
                Xtr, Xva = sc.transform(site_X[tr]), sc.transform(site_X[va])
                mt = min(44, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, site_y[tr], random_state=seed))[::-1][:mt]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                gi = site_idx[va]
                oof_lr[gi] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, site_y[tr]).predict_proba(Xva)[:, 1]
                oof_xgb[gi] += xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, site_y[tr]).predict_proba(Xv)[:, 1]
                oof_lgb[gi] += lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=5, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, site_y[tr]).predict_proba(Xv)[:, 1]
                cnt[gi] += 1
    oof_lr = np.where(cnt > 0, oof_lr / cnt, 0.5)
    oof_xgb = np.where(cnt > 0, oof_xgb / cnt, 0.5)
    oof_lgb = np.where(cnt > 0, oof_lgb / cnt, 0.5)
    return [oof_lr, oof_xgb, oof_lgb]


def run_mlp_oof(X_tab, y, groups, seeds=SEEDS, tag=""):
    try:
        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print(f"  {tag}: torch not available")
        return None
    n = len(y)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        ("mlp_256_128", [256, 128], 0.3, 0.2),
        ("mlp_512_256", [512, 256], 0.4, 0.3),
        ("mlp_128_64_32", [128, 64, 32], 0.2, 0.1),
        ("mlp_512_256_128", [512, 256, 128], 0.4, 0.3),
    ]
    all_oof = {}
    for name, hidden, dp1, dp2 in configs:
        oof = np.zeros(n); cnt = np.zeros(n)
        for seed in seeds:
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_tab, y, groups):
                sc = StandardScaler().fit(X_tab[tr])
                Xtr = sc.transform(X_tab[tr]).astype(np.float32)
                Xva = sc.transform(X_tab[va]).astype(np.float32)
                class MLP(nn.Module):
                    def __init__(self):
                        super().__init__()
                        layers = []; in_d = Xtr.shape[1]
                        for h in hidden:
                            layers.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dp1)])
                            in_d = h
                        layers.extend([nn.Linear(in_d, 1)])
                        self.net = nn.Sequential(*layers)
                    def forward(self, x): return self.net(x)
                model = MLP().to(device)
                opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                crit = nn.BCEWithLogitsLoss()
                ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y[tr].astype(np.float32)))
                dl = DataLoader(ds, batch_size=64, shuffle=True)
                best_state = None; best_vl = float("inf"); patience = 10; ni = 0
                for ep in range(100):
                    model.train()
                    for xb, yb in dl:
                        xb, yb = xb.to(device), yb.to(device)
                        opt.zero_grad()
                        crit(model(xb).squeeze(-1), yb).backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        vp = torch.sigmoid(model(torch.from_numpy(Xva).to(device)).squeeze(-1)).cpu().numpy()
                    vl = log_loss(y[va], np.clip(vp, 1e-6, 1-1e-6))
                    if vl < best_vl:
                        best_vl = vl
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        ni = 0
                    else:
                        ni += 1
                        if ni >= patience: break
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    pred = torch.sigmoid(model(torch.from_numpy(Xva).to(device)).squeeze(-1)).cpu().numpy()
                oof[va] += pred; cnt[va] += 1
                del model; torch.cuda.empty_cache()
            print(f"  {tag}/{name} seed {seed}", flush=True)
        all_oof[name] = np.where(cnt > 0, oof / cnt, 0.5)
    return all_oof


def main():
    t0 = time.time()
    merged, parts, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum()}")
    Xs = build_streams(parts, merged)
    X_tab = np.hstack(Xs)
    print(f"Streams: {[X.shape for X in Xs]}, Tab {X_tab.shape}")

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, 'y.npy'), y)
    np.save(os.path.join(OUT, 'groups.npy'), groups)
    uid_list = merged.index.tolist()
    with open(os.path.join(OUT, 'uids.json'), 'w') as f:
        json.dump(uid_list, f)

    print("A: GLOBAL 30-STREAM")
    oof_a = run_global_oof(Xs, y, groups, tag="A")
    with open(os.path.join(OUT, 'oof_a.pkl'), 'wb') as f:
        pickle.dump({'oof': oof_a, 'names': [f'{d}_{m}' for d in ['sbr','phys','morph','atlas','radiom'] for m in ['lr','xgb','lgb','cb','et','ridge']]}, f)
    M_a = np.column_stack(oof_a)
    w_a = optim_blend(M_a, y)
    blend_a = M_a @ w_a
    a, b = optimize_platt(blend_a, y)
    p = expit(a * logit(np.clip(blend_a, 1e-6, 1-1e-6)) + b)
    p_sc = site_calibrate(blend_a, y, groups)
    print(f"  A: AUC={roc_auc_score(y,p):.4f} LL={log_loss(y,p):.4f} | SC AUC={roc_auc_score(y,p_sc):.4f} LL={log_loss(y,p_sc):.4f}")
    np.save(os.path.join(OUT, 'blend_a.npy'), blend_a)
    np.save(os.path.join(OUT, 'blend_a_sc.npy'), p_sc)
    results = {"A": {"auc": float(roc_auc_score(y, p)), "ll": float(log_loss(y, p)),
                     "auc_sc": float(roc_auc_score(y, p_sc)), "ll_sc": float(log_loss(y, p_sc))}}

    print("B: DIVERSE MODELS (13 variants on all features)")
    oof_b, names_b = run_diverse_oof(X_tab, y, groups, tag="B")
    with open(os.path.join(OUT, 'oof_b.pkl'), 'wb') as f:
        pickle.dump({'oof': oof_b, 'names': names_b}, f)
    M_b = np.column_stack(oof_b)
    w_b = optim_blend(M_b, y)
    blend_b = M_b @ w_b
    a, b = optimize_platt(blend_b, y)
    p = expit(a * logit(np.clip(blend_b, 1e-6, 1-1e-6)) + b)
    p_sc = site_calibrate(blend_b, y, groups)
    print(f"  B: AUC={roc_auc_score(y,p):.4f} LL={log_loss(y,p):.4f} | SC AUC={roc_auc_score(y,p_sc):.4f} LL={log_loss(y,p_sc):.4f}")
    results["B"] = {"auc": float(roc_auc_score(y, p)), "ll": float(log_loss(y, p)),
                    "auc_sc": float(roc_auc_score(y, p_sc)), "ll_sc": float(log_loss(y, p_sc))}

    print("C: PER-SITE MODELS")
    oof_c = run_per_site_oof(Xs, y, groups, tag="C")
    with open(os.path.join(OUT, 'oof_c.pkl'), 'wb') as f:
        pickle.dump({'oof': oof_c, 'names': ['lr','xgb','lgb']}, f)
    M_c = np.column_stack(oof_c)
    w_c = optim_blend(M_c, y)
    blend_c = M_c @ w_c
    a, b = optimize_platt(blend_c, y)
    p = expit(a * logit(np.clip(blend_c, 1e-6, 1-1e-6)) + b)
    p_sc = site_calibrate(blend_c, y, groups)
    print(f"  C: AUC={roc_auc_score(y,p):.4f} LL={log_loss(y,p):.4f} | SC AUC={roc_auc_score(y,p_sc):.4f} LL={log_loss(y,p_sc):.4f}")
    results["C"] = {"auc": float(roc_auc_score(y, p)), "ll": float(log_loss(y, p)),
                    "auc_sc": float(roc_auc_score(y, p_sc)), "ll_sc": float(log_loss(y, p_sc))}

    print("D: MLP VARIANTS")
    d_oof = run_mlp_oof(X_tab, y, groups, tag="D")
    mlp_blends = []
    if d_oof:
        for name, oof_arr in d_oof.items():
            a, b = optimize_platt(oof_arr, y)
            p = expit(a * logit(np.clip(oof_arr, 1e-6, 1-1e-6)) + b)
            print(f"  {name}: AUC={roc_auc_score(y,p):.4f} LL={log_loss(y,p):.4f}")
            mlp_blends.append(p)
        M_d = np.column_stack(mlp_blends)
        w_d = optim_blend(M_d, y)
        blend_d = M_d @ w_d
        a, b = optimize_platt(blend_d, y)
        p = expit(a * logit(np.clip(blend_d, 1e-6, 1-1e-6)) + b)
        print(f"  MLP_ENSEMBLE: AUC={roc_auc_score(y,p):.4f} LL={log_loss(y,p):.4f}")
        np.save(os.path.join(OUT, 'blend_d.npy'), p)
        results["D"] = {"auc": float(roc_auc_score(y, p)), "ll": float(log_loss(y, p))}

    print("E: MEGA ENSEMBLE")
    all_blends = [blend_a, blend_b, blend_c]
    blend_names = ["global", "diverse", "persite"]
    if d_oof:
        all_blends.append(blend_d); blend_names.append("mlp")
    M_e = np.column_stack(all_blends)
    w_e = optim_blend(M_e, y)
    blend_e = M_e @ w_e
    a, b = optimize_platt(blend_e, y)
    p = expit(a * logit(np.clip(blend_e, 1e-6, 1-1e-6)) + b)
    p_sc = site_calibrate(blend_e, y, groups)
    print(f"  MEGA: AUC={roc_auc_score(y,p):.4f} LL={log_loss(y,p):.4f} | SC AUC={roc_auc_score(y,p_sc):.4f} LL={log_loss(y,p_sc):.4f}")
    for n, wv in zip(blend_names, w_e):
        print(f"    {n}: {wv:.4f}")
    np.save(os.path.join(OUT, 'blend_e.npy'), blend_e)
    np.save(os.path.join(OUT, 'blend_e_sc.npy'), p_sc)
    results["E"] = {"auc": float(roc_auc_score(y, p)), "ll": float(log_loss(y, p)),
                    "auc_sc": float(roc_auc_score(y, p_sc)), "ll_sc": float(log_loss(y, p_sc))}

    print("F: MEGA + TOP DIVERSE")
    mega_streams = [blend_a, blend_c]
    mega_names = ["global", "persite"]
    if d_oof:
        mega_streams.append(blend_d); mega_names.append("mlp")
    top_diverse_idx = np.argsort(w_b)[::-1][:5]
    for idx in top_diverse_idx:
        mega_streams.append(oof_b[idx]); mega_names.append(names_b[idx])
    M_f = np.column_stack(mega_streams)
    w_f = optim_blend(M_f, y)
    blend_f = M_f @ w_f
    a, b = optimize_platt(blend_f, y)
    p = expit(a * logit(np.clip(blend_f, 1e-6, 1-1e-6)) + b)
    p_sc = site_calibrate(blend_f, y, groups)
    print(f"  MEGA+: AUC={roc_auc_score(y,p):.4f} LL={log_loss(y,p):.4f} | SC AUC={roc_auc_score(y,p_sc):.4f} LL={log_loss(y,p_sc):.4f}")
    np.save(os.path.join(OUT, 'blend_f_sc.npy'), p_sc)
    results["F"] = {"auc": float(roc_auc_score(y, p)), "ll": float(log_loss(y, p)),
                    "auc_sc": float(roc_auc_score(y, p_sc)), "ll_sc": float(log_loss(y, p_sc))}

    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nRESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"Total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()