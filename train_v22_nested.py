"""
V22: Nested site calibration + per-site models for DaT-SPECT.
Best improvement from v21 was site calibration (+0.013 LL).
This version implements it properly with nested OOF to avoid leakage.
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
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
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
        df["lr_cau_diff"] = L - R
    if "sbr_left_putamen_post" in df.columns and "sbr_left_putamen_ant" in df.columns:
        df["lp_post_ant"] = df["sbr_left_putamen_post"].values / (df["sbr_left_putamen_ant"].values + eps)
    if "sbr_right_putamen_post" in df.columns and "sbr_right_putamen_ant" in df.columns:
        df["rp_post_ant"] = df["sbr_right_putamen_post"].values / (df["sbr_right_putamen_ant"].values + eps)
    if "sbr_left_caudate_post" in df.columns and "sbr_left_caudate_ant" in df.columns:
        df["lc_post_ant"] = df["sbr_left_caudate_post"].values / (df["sbr_left_caudate_ant"].values + eps)
    if "sbr_right_caudate_post" in df.columns and "sbr_right_caudate_ant" in df.columns:
        df["rc_post_ant"] = df["sbr_right_caudate_post"].values / (df["sbr_right_caudate_ant"].values + eps)
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
    for c in region_cols:
        df[f"{c}_rank"] = rankdata(df[c].values) / len(df)
    if "sbr_left_putamen" in df.columns and "sbr_right_putamen" in df.columns:
        df["lp_x_rp"] = df["sbr_left_putamen"].values * df["sbr_right_putamen"].values
    if "sbr_left_caudate" in df.columns and "sbr_right_caudate" in df.columns:
        df["lc_x_rc"] = df["sbr_left_caudate"].values * df["sbr_right_caudate"].values
    if "sbr_left_putamen" in df.columns and "sbr_right_caudate" in df.columns:
        df["lp_x_rc"] = df["sbr_left_putamen"].values * df["sbr_right_caudate"].values
    if "sbr_right_putamen" in df.columns and "sbr_left_caudate" in df.columns:
        df["rp_x_lc"] = df["sbr_right_putamen"].values * df["sbr_left_caudate"].values
    return df


from scipy.stats import rankdata


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


def full_oof(Xs, y, groups, seeds=SEEDS, weights=None, tag=""):
    n_ds = len(Xs)
    n_m = 6
    n_streams = n_ds * n_m
    n = len(y)
    oof = [np.zeros(n) for _ in range(n_streams)]
    cnt = [np.zeros(n) for _ in range(n_streams)]

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

                lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed)
                lr.fit(Xtr, y[tr], sample_weight=w_tr) if w_tr is not None else lr.fit(Xtr, y[tr])
                oof[b][va] += lr.predict_proba(Xva)[:, 1]; cnt[b][va] += 1

                xgb_m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0)
                xgb_m.fit(Xt, y[tr], sample_weight=w_tr) if w_tr is not None else xgb_m.fit(Xt, y[tr])
                oof[b+1][va] += xgb_m.predict_proba(Xv)[:, 1]; cnt[b+1][va] += 1

                lgb_m = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1)
                lgb_m.fit(Xt, y[tr], sample_weight=w_tr) if w_tr is not None else lgb_m.fit(Xt, y[tr])
                oof[b+2][va] += lgb_m.predict_proba(Xv)[:, 1]; cnt[b+2][va] += 1

                cb_m = cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0)
                cb_m.fit(Xt, y[tr], sample_weight=w_tr) if w_tr is not None else cb_m.fit(Xt, y[tr])
                oof[b+3][va] += cb_m.predict_proba(Xv)[:, 1]; cnt[b+3][va] += 1

                et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed)
                et.fit(Xt, y[tr], sample_weight=w_tr) if w_tr is not None else et.fit(Xt, y[tr])
                oof[b+4][va] += et.predict_proba(Xv)[:, 1]; cnt[b+4][va] += 1

                ridge = RidgeClassifier(alpha=0.1, random_state=seed)
                ridge.fit(Xtr, y[tr], sample_weight=w_tr) if w_tr is not None else ridge.fit(Xtr, y[tr])
                oof[b+5][va] += expit(ridge.decision_function(Xva)); cnt[b+5][va] += 1
        print(f"  {tag} seed {seed}")
    for i in range(n_streams):
        oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)
    return oof


def site_calibrate(blend, y, groups):
    result = blend.copy()
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            a, b, _ = optimize_platt(blend[mask], y[mask])
            result[mask] = expit(a * logit(np.clip(blend[mask], 1e-6, 1-1e-6)) + b)
    return result


def nested_site_calibrate(oof, y, groups, n_inner_folds=3):
    """Properly nested: for each site, fit site calib via inner StratifiedKFold."""
    from sklearn.model_selection import StratifiedKFold
    result = np.zeros_like(oof)
    sites = np.unique(groups)
    for site in sites:
        site_mask = groups == site
        site_idx = np.where(site_mask)[0]
        if len(site_idx) < 15 or len(np.unique(y[site_idx])) < 2:
            result[site_idx] = oof[site_idx]
            continue
        site_y = y[site_idx]
        site_oof = oof[site_idx]
        site_preds = np.zeros(len(site_idx))
        site_cnt = np.zeros(len(site_idx))
        for seed in range(n_inner_folds):
            for tr, va in StratifiedKFold(n_inner_folds, shuffle=True, random_state=seed*10+42).split(site_oof, site_y):
                a, b, _ = optimize_platt(site_oof[tr], site_y[tr])
                site_preds[va] += expit(a * logit(np.clip(site_oof[va], 1e-6, 1-1e-6)) + b)
                site_cnt[va] += 1
        site_preds = np.where(site_cnt > 0, site_preds / site_cnt, oof[site_idx])
        result[site_idx] = site_preds
    return result


def main():
    t0 = time.time()
    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    merged_adv = add_adv(merged.copy())
    Xs_adv = build_tabular(merged_adv)
    Xs_base = build_tabular(merged)
    print(f"Base: {[X.shape[1] for X in Xs_base]}, Adv: {[X.shape[1] for X in Xs_adv]}")

    n_ds = len(Xs_adv)
    n_m = 6
    n_streams = n_ds * n_m
    ds_names = ["sbr", "phys", "morph", "atlas"][:n_ds]
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    stream_names = [f"{ds}_{m}" for ds in ds_names for m in mnames]

    results = {}

    # A: BASELINE
    print(f"\n{'='*60}")
    print("A: BASELINE")
    print("=" * 60)
    oof_a = full_oof(Xs_base, y, groups, tag="A")
    M_a = np.column_stack(oof_a)
    w_a = optim_blend(M_a, y)
    blend_a = M_a @ w_a
    a, b, p = optimize_platt(blend_a, y)
    ll_a = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
    auc_a = roc_auc_score(y, p)
    print(f"  BASELINE: AUC={auc_a:.4f} LL={ll_a:.4f}")
    results["A_baseline"] = {"auc": float(auc_a), "ll": float(ll_a)}

    # B: ADVANCED FEATURES
    print(f"\n{'='*60}")
    print("B: ADVANCED FEATURES")
    print("=" * 60)
    oof_b = full_oof(Xs_adv, y, groups, tag="B")
    M_b = np.column_stack(oof_b)
    w_b = optim_blend(M_b, y)
    blend_b = M_b @ w_b
    a, b, p = optimize_platt(blend_b, y)
    ll_b = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
    auc_b = roc_auc_score(y, p)
    print(f"  ADVANCED: AUC={auc_b:.4f} LL={ll_b:.4f}")
    results["B_adv"] = {"auc": float(auc_b), "ll": float(ll_b)}

    # C: ADVANCED + SITE CAL (naive)
    print(f"\n{'='*60}")
    print("C: ADVANCED + NAIVE SITE CAL")
    print("=" * 60)
    p_site = site_calibrate(blend_b, y, groups)
    ll_c = log_loss(y, np.clip(p_site, 1e-6, 1-1e-6))
    auc_c = roc_auc_score(y, p_site)
    print(f"  NAIVE SITE CAL: AUC={auc_c:.4f} LL={ll_c:.4f}")
    results["C_naive_site_cal"] = {"auc": float(auc_c), "ll": float(ll_c)}

    # D: ADVANCED + NESTED SITE CAL (proper, no leakage)
    print(f"\n{'='*60}")
    print("D: ADVANCED + NESTED SITE CAL (no leakage)")
    print("=" * 60)
    p_nested = nested_site_calibrate(blend_b, y, groups)
    ll_d = log_loss(y, np.clip(p_nested, 1e-6, 1-1e-6))
    auc_d = roc_auc_score(y, p_nested)
    print(f"  NESTED SITE CAL: AUC={auc_d:.4f} LL={ll_d:.4f}")
    results["D_nested_site_cal"] = {"auc": float(auc_d), "ll": float(ll_d)}

    # E: BASELINE + NESTED SITE CAL
    print(f"\n{'='*60}")
    print("E: BASELINE + NESTED SITE CAL")
    print("=" * 60)
    p_base_nested = nested_site_calibrate(blend_a, y, groups)
    ll_e = log_loss(y, np.clip(p_base_nested, 1e-6, 1-1e-6))
    auc_e = roc_auc_score(y, p_base_nested)
    print(f"  BASE+NESTED: AUC={auc_e:.4f} LL={ll_e:.4f}")
    results["E_base_nested"] = {"auc": float(auc_e), "ll": float(ll_e)}

    # F: ENSEMBLE of best approaches + nested site cal
    print(f"\n{'='*60}")
    print("F: ENSEMBLE + NESTED SITE CAL")
    print("=" * 60)
    ensemble_blend = np.column_stack([blend_a, blend_b])
    w_ens = optim_blend(ensemble_blend, y)
    blend_ens = ensemble_blend @ w_ens
    p_ens = nested_site_calibrate(blend_ens, y, groups)
    ll_f = log_loss(y, np.clip(p_ens, 1e-6, 1-1e-6))
    auc_f = roc_auc_score(y, p_ens)
    print(f"  ENS+NESTED: AUC={auc_f:.4f} LL={ll_f:.4f}")
    results["F_ensemble_nested"] = {"auc": float(auc_f), "ll": float(ll_f)}

    # G: Greedy elimination on ADV streams
    print(f"\n{'='*60}")
    print("G: GREEDY ELIMINATION on ADV")
    print("=" * 60)
    best_ll_g = ll_b
    best_mask_g = np.ones(n_streams, dtype=bool)
    for _ in range(n_streams):
        improved = False
        for j in range(n_streams):
            if not best_mask_g[j]:
                continue
            mask = best_mask_g.copy()
            mask[j] = False
            Mc = M_b[:, mask]
            if Mc.shape[1] == 0:
                continue
            wc = optim_blend(Mc, y)
            bc = Mc @ wc
            a, b_p, pc = optimize_platt(bc, y)
            ll_cand = log_loss(y, np.clip(pc, 1e-6, 1-1e-6))
            if ll_cand < best_ll_g:
                best_ll_g = ll_cand
                best_auc_g = roc_auc_score(y, pc)
                best_mask_g = mask.copy()
                improved = True
                print(f"  Drop {stream_names[j]} -> LL={best_ll_g:.4f} AUC={best_auc_g:.4f}")
        if not improved:
            break

    M_elim = M_b[:, best_mask_g]
    w_elim = optim_blend(M_elim, y)
    blend_elim = M_elim @ w_elim
    p_elim_nest = nested_site_calibrate(blend_elim, y, groups)
    ll_g = log_loss(y, np.clip(p_elim_nest, 1e-6, 1-1e-6))
    auc_g = roc_auc_score(y, p_elim_nest)
    print(f"  ELIM+NESTED: AUC={auc_g:.4f} LL={ll_g:.4f} (n_streams={best_mask_g.sum()})")
    active = [stream_names[i] for i in range(n_streams) if best_mask_g[i]]
    for n in active:
        print(f"    {n}")
    results["G_elim_nested"] = {"auc": float(auc_g), "ll": float(ll_g)}

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    base_ll = results["A_baseline"]["ll"]
    print(f"{'Experiment':40s} {'AUC':>8s} {'LogLoss':>8s} {'Delta':>10s}")
    print("-" * 70)
    for name in sorted(results.keys()):
        r = results[name]
        delta = base_ll - r["ll"]
        marker = " **" if delta > 0.01 else (" *" if delta > 0.003 else "")
        print(f"  {name:38s} {r['auc']:8.4f} {r['ll']:8.4f} {delta:+10.4f}{marker}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    with open("v22_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved v22_results.json")


if __name__ == "__main__":
    main()
