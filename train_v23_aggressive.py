"""
V23: Aggressive tabular improvements - site-aware approaches.
Per-site normalization, site features, per-site models, harmonization, stacking.
"""
import os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata, zscore
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        df["lr_cau_diff"] = L - R
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


def harmonize_features(X, groups):
    """Simple ComBat-like: z-score normalize within each site, then standardize."""
    X_h = X.copy()
    unique_sites = np.unique(groups)
    for site in unique_sites:
        mask = groups == site
        if mask.sum() < 3:
            continue
        mean = X_h[mask].mean(axis=0)
        std = X_h[mask].std(axis=0) + 1e-8
        X_h[mask] = (X_h[mask] - mean) / std
    return X_h


def per_site_zscore(X, groups):
    """Z-score within each site."""
    X_z = X.copy()
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() < 3:
            continue
        mean = X_z[mask].mean(axis=0)
        std = X_z[mask].std(axis=0) + 1e-8
        X_z[mask] = (X_z[mask] - mean) / std
    return X_z


def add_site_features(X, groups, n_sites=None):
    """Add one-hot site features."""
    unique_sites = sorted(np.unique(groups))
    if n_sites is None:
        n_sites = len(unique_sites)
    site_map = {s: i for i, s in enumerate(unique_sites)}
    site_onehot = np.zeros((len(groups), min(n_sites, len(unique_sites))))
    for i, g in enumerate(groups):
        idx = site_map[g]
        if idx < site_onehot.shape[1]:
            site_onehot[i, idx] = 1.0
    return np.hstack([X, site_onehot])


def target_encode_site(X, y, groups, n_splits=5, seed=42):
    """Target encoding for site: per-fold to avoid leakage."""
    X_te = np.zeros((len(y), X.shape[1] + len(np.unique(groups))))
    X_te[:, :X.shape[1]] = X
    unique_sites = sorted(np.unique(groups))
    site_map = {s: i for i, s in enumerate(unique_sites)}
    global_mean = y.mean()

    for tr, va in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        for site in unique_sites:
            s_mask = groups[tr] == site
            if s_mask.sum() > 0:
                mean_pos = y[tr][s_mask].mean()
                count = s_mask.sum()
                smooth = mean_pos * count / (count + 10) + global_mean * 10 / (count + 10)
            else:
                smooth = global_mean
            va_mask = groups[va] == site
            X_te[va, X.shape[1] + site_map[site]] = smooth

    full_mask = np.ones(len(y), dtype=bool)
    for site in unique_sites:
        s_mask = groups == site
        if s_mask.sum() > 0:
            mean_pos = y[s_mask].mean()
            count = s_mask.sum()
            smooth = mean_pos * count / (count + 10) + global_mean * 10 / (count + 10)
        else:
            smooth = global_mean
        not_assigned = s_mask & (X_te[:, X.shape[1]:].sum(axis=1) == 0)
        X_te[not_assigned, X.shape[1] + site_map[site]] = smooth

    return X_te


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
                fit_kw = {"sample_weight": w_tr} if w_tr is not None else {}

                oof[b][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr], **fit_kw).predict_proba(Xva)[:, 1]; cnt[b][va] += 1
                oof[b+1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr], **fit_kw).predict_proba(Xv)[:, 1]; cnt[b+1][va] += 1
                oof[b+2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr], **fit_kw).predict_proba(Xv)[:, 1]; cnt[b+2][va] += 1
                oof[b+3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr], **fit_kw).predict_proba(Xv)[:, 1]; cnt[b+3][va] += 1
                oof[b+4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr], **fit_kw).predict_proba(Xv)[:, 1]; cnt[b+4][va] += 1
                oof[b+5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr], **fit_kw).decision_function(Xva)); cnt[b+5][va] += 1
        print(f"  {tag} seed {seed}")
    for i in range(n_ds * n_m):
        oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)
    return oof


def evaluate(oof_list, y, groups, tag="", do_site_cal=True):
    M = np.column_stack(oof_list)
    w = optim_blend(M, y)
    blend = M @ w
    a, b_pl, p = optimize_platt(blend, y)
    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
    auc = roc_auc_score(y, p)
    print(f"  {tag}: AUC={auc:.4f} LL={ll:.4f}")
    results = {"auc": float(auc), "ll": float(ll)}
    if do_site_cal:
        p_sc = site_calibrate(blend, y, groups)
        ll_sc = log_loss(y, np.clip(p_sc, 1e-6, 1-1e-6))
        auc_sc = roc_auc_score(y, p_sc)
        print(f"  {tag}+SITE_CAL: AUC={auc_sc:.4f} LL={ll_sc:.4f}")
        results["auc_sc"] = float(auc_sc)
        results["ll_sc"] = float(ll_sc)
    return results


def main():
    t0 = time.time()
    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    merged_adv = add_adv(merged.copy())
    Xs_base = build_tabular(merged)
    Xs_adv = build_tabular(merged_adv)
    print(f"Base: {[X.shape[1] for X in Xs_base]}, Adv: {[X.shape[1] for X in Xs_adv]}")

    X_tab = np.hstack(Xs_adv)
    results = {}

    # A: Baseline with advanced features
    print(f"\n{'='*60}")
    print("A: ADVANCED FEATURES BASELINE")
    print("=" * 60)
    oof_a = run_oof(Xs_adv, y, groups, tag="A")
    results["A_adv_baseline"] = evaluate(oof_a, y, groups, "A")

    # B: Harmonized features (ComBat-like)
    print(f"\n{'='*60}")
    print("B: HARMONIZED FEATURES")
    print("=" * 60)
    Xs_harm = []
    for X in Xs_adv:
        Xs_harm.append(harmonize_features(X, groups))
    oof_b = run_oof(Xs_harm, y, groups, tag="B")
    results["B_harmonized"] = evaluate(oof_b, y, groups, "B")

    # C: Per-site z-score normalized
    print(f"\n{'='*60}")
    print("C: PER-SITE Z-SCORE")
    print("=" * 60)
    Xs_zscore = []
    for X in Xs_adv:
        Xs_zscore.append(per_site_zscore(X, groups))
    oof_c = run_oof(Xs_zscore, y, groups, tag="C")
    results["C_per_site_zscore"] = evaluate(oof_c, y, groups, "C")

    # D: Site one-hot features (added to each stream)
    print(f"\n{'='*60}")
    print("D: SITE ONE-HOT FEATURES")
    print("=" * 60)
    Xs_siteoh = []
    for X in Xs_adv:
        Xs_siteoh.append(add_site_features(X, groups))
    oof_d = run_oof(Xs_siteoh, y, groups, tag="D")
    results["D_site_onehot"] = evaluate(oof_d, y, groups, "D")

    # E: Harmonized + site one-hot
    print(f"\n{'='*60}")
    print("E: HARMONIZED + SITE ONE-HOT")
    print("=" * 60)
    Xs_e = []
    for X in Xs_adv:
        Xh = harmonize_features(X, groups)
        Xs_e.append(add_site_features(Xh, groups))
    oof_e = run_oof(Xs_e, y, groups, tag="E")
    results["E_harm_siteoh"] = evaluate(oof_e, y, groups, "E")

    # F: Single matrix with ALL features + site one-hot
    print(f"\n{'='*60}")
    print("F: SINGLE MATRIX ALL FEATURES + SITE ONE-HOT")
    print("=" * 60)
    X_f = add_site_features(X_tab, groups)
    Xs_f = [X_f]
    oof_f = run_oof(Xs_f, y, groups, tag="F")
    results["F_single_matrix"] = evaluate(oof_f, y, groups, "F")

    # G: Single matrix + harmonized + site one-hot
    print(f"\n{'='*60}")
    print("G: SINGLE MATRIX + HARMONIZED + SITE ONE-HOT")
    print("=" * 60)
    X_g = harmonize_features(X_tab, groups)
    X_g = add_site_features(X_g, groups)
    Xs_g = [X_g]
    oof_g = run_oof(Xs_g, y, groups, tag="G")
    results["G_single_harm_site"] = evaluate(oof_g, y, groups, "G")

    # H: Best combo = harmonized per-stream + advanced + site cal
    print(f"\n{'='*60}")
    print("H: ENSEMBLE OF BEST (A+B+D+E) + SITE CAL")
    print("=" * 60)
    best_blends = []
    for oof_list, nm in [(oof_a, "adv"), (oof_b, "harm"), (oof_d, "siteoh"), (oof_e, "harm_siteoh")]:
        M = np.column_stack(oof_list)
        w = optim_blend(M, y)
        blend = M @ w
        a, b, p = optimize_platt(blend, y)
        best_blends.append(p)
        print(f"  {nm}: AUC={roc_auc_score(y,p):.4f} LL={log_loss(y,np.clip(p,1e-6,1-1e-6)):.4f}")
    M_h = np.column_stack(best_blends)
    w_h = optim_blend(M_h, y)
    blend_h = M_h @ w_h
    a, b, p_h = optimize_platt(blend_h, y)
    ll_h = log_loss(y, np.clip(p_h, 1e-6, 1-1e-6))
    auc_h = roc_auc_score(y, p_h)
    print(f"  ENSEMBLE: AUC={auc_h:.4f} LL={ll_h:.4f}")
    results["H_ensemble"] = {"auc": float(auc_h), "ll": float(ll_h)}
    p_h_sc = site_calibrate(blend_h, y, groups)
    ll_h_sc = log_loss(y, np.clip(p_h_sc, 1e-6, 1-1e-6))
    auc_h_sc = roc_auc_score(y, p_h_sc)
    print(f"  ENSEMBLE+SITE_CAL: AUC={auc_h_sc:.4f} LL={ll_h_sc:.4f}")
    results["H_ensemble_sc"] = {"auc": float(auc_h_sc), "ll": float(ll_h_sc)}

    # I: Per-stream harmonized + site one-hot + advanced + site cal
    print(f"\n{'='*60}")
    print("I: ALL IMPROVEMENTS COMBINED")
    print("=" * 60)
    oof_i = run_oof(Xs_e, y, groups, weights=np.ones(len(y)), tag="I")
    M_i = np.column_stack(oof_i)
    w_i = optim_blend(M_i, y)
    blend_i = M_i @ w_i
    a, b, p_i = optimize_platt(blend_i, y)
    ll_i = log_loss(y, np.clip(p_i, 1e-6, 1-1e-6))
    auc_i = roc_auc_score(y, p_i)
    print(f"  ALL_COMBINED: AUC={auc_i:.4f} LL={ll_i:.4f}")
    results["I_all_combined"] = {"auc": float(auc_i), "ll": float(ll_i)}
    p_i_sc = site_calibrate(blend_i, y, groups)
    ll_i_sc = log_loss(y, np.clip(p_i_sc, 1e-6, 1-1e-6))
    auc_i_sc = roc_auc_score(y, p_i_sc)
    print(f"  ALL_COMBINED+SITE_CAL: AUC={auc_i_sc:.4f} LL={ll_i_sc:.4f}")
    results["I_all_combined_sc"] = {"auc": float(auc_i_sc), "ll": float(ll_i_sc)}

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    base_ll = results["A_adv_baseline"]["ll"]
    print(f"{'Experiment':40s} {'AUC':>8s} {'LogLoss':>8s} {'Delta':>10s}")
    print("-" * 70)
    for name in sorted(results.keys()):
        r = results[name]
        ll = r.get("ll_sc", r.get("ll", 0))
        auc = r.get("auc_sc", r.get("auc", 0))
        delta = base_ll - ll
        marker = " **" if delta > 0.01 else (" *" if delta > 0.003 else "")
        print(f"  {name:38s} {auc:8.4f} {ll:8.4f} {delta:+10.4f}{marker}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    with open("v23_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved v23_results.json")


if __name__ == "__main__":
    main()
