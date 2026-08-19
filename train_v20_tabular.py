"""
V20: Tabular-only improvements for DaT-SPECT.
Focus: calibration, site harmonization, feature engineering, 2-stage boosting.
"""
import os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    if "sbr_left_putamen" in df.columns and "sbr_left_putamen_post" in df.columns:
        df["lp_x_post"] = df["sbr_left_putamen"].values * df["sbr_left_putamen_post"].values
    if "sbr_right_putamen" in df.columns and "sbr_right_putamen_post" in df.columns:
        df["rp_x_post"] = df["sbr_right_putamen"].values * df["sbr_right_putamen_post"].values
    return df


def add_advanced_features(df):
    eps = 1e-6
    # Cross-region interactions
    for c1 in ["sbr_left_putamen", "sbr_right_putamen", "sbr_left_caudate", "sbr_right_caudate"]:
        if c1 in df.columns:
            for c2 in ["sbr_left_putamen", "sbr_right_putamen", "sbr_left_caudate", "sbr_right_caudate"]:
                if c2 in df.columns and c1 != c2:
                    name = f"cross_{c1.replace('sbr_','')[:3]}_{c2.replace('sbr_','')[:3]}"
                    df[name] = df[c1].values * df[c2].values

    # Quartile / rank features
    for c in ["sbr_left_putamen", "sbr_right_putamen", "sbr_left_caudate", "sbr_right_caudate"]:
        if c in df.columns:
            df[f"{c}_rank"] = rankdata(df[c].values) / len(df)

    # Variance across regions
    region_cols = [c for c in ["sbr_left_putamen", "sbr_right_putamen",
                                "sbr_left_caudate", "sbr_right_caudate"] if c in df.columns]
    if len(region_cols) >= 2:
        region_vals = df[region_cols].values
        df["striatal_mean"] = region_vals.mean(axis=1)
        df["striatal_std"] = region_vals.std(axis=1)
        df["striatal_cv"] = df["striatal_std"] / (df["striatal_mean"] + eps)
        df["striatal_range"] = region_vals.max(axis=1) - region_vals.min(axis=1)

    return df


def build_matrix(raw, raw_cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def build_tabular(merged, skip_extra=frozenset()):
    skip = {"uid", "is_pathologic", "pseudo_site", "site_id", "sx", "sy", "sz",
            "voxel_vol", "voxel_side", "label"} | skip_extra
    raw1 = [c for c in merged.columns if not c.startswith("ph_") and not c.startswith("mo_")
            and not c.startswith("at_") and c not in skip]
    raw2 = [c for c in merged.columns if c.startswith("ph_")]
    raw3 = [c for c in merged.columns if c.startswith("mo_")]
    raw4 = [c for c in merged.columns if c.startswith("at_")]
    return (build_matrix(merged, raw1), build_matrix(merged, raw2),
            build_matrix(merged, raw3), build_matrix(merged, raw4))


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


def optimize_temp(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    lg = logit(oof)
    def loss(log_t):
        return log_loss(y, np.clip(expit(lg / np.exp(log_t[0])), 1e-6, 1 - 1e-6))
    r = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return float(np.exp(r.x[0]))


def site_calibrate(oof, y, groups):
    """Calibrate separately per site."""
    result = oof.copy()
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            a, b, _ = optimize_platt(oof[mask], y[mask])
            result[mask] = expit(a * logit(np.clip(oof[mask], 1e-6, 1-1e-6)) + b)
    return result


def nested_oof_calibrate(oof, y, groups, n_seeds_cal=3):
    """Use nested OOF to calibrate without leakage."""
    result = np.zeros_like(oof)
    for seed in range(n_seeds_cal):
        for tr, va in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(oof, y, groups):
            a, b, _ = optimize_platt(oof[tr], y[tr])
            result[va] += expit(a * logit(np.clip(oof[va], 1e-6, 1-1e-6)) + b)
    result = result / n_seeds_cal
    mask = result == 0
    result[mask] = oof[mask]
    return result


# ============================================================
# OOF TRAINING
# ============================================================
def run_full_oof(Xs, y, groups, seeds=SEEDS, mi_top=44, extra_tag=""):
    n_ds = len(Xs)
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    n_m = len(mnames)
    n = len(y)
    oof = [np.zeros(n) for _ in range(n_ds * n_m)]
    cnt = [np.zeros(n) for _ in range(n_ds * n_m)]

    for seed in seeds:
        for xi, X in enumerate(Xs):
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                mt = min(mi_top, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mt]
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


def run_two_stage_oof(Xs, y, groups, seeds=SEEDS, mi_top=44, conf_thresh=0.85, boost_weight=2.0):
    """2-stage: train on full, identify confident, retrain with boosted weights."""
    n_ds = len(Xs)
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    n_m = len(mnames)
    n = len(y)

    # Stage 1: normal OOF
    oof1 = [np.zeros(n) for _ in range(n_ds * n_m)]
    cnt1 = [np.zeros(n) for _ in range(n_ds * n_m)]
    for seed in seeds:
        for xi, X in enumerate(Xs):
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                mt = min(mi_top, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mt]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                b = xi * n_m
                oof1[b][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt1[b][va] += 1
                oof1[b+1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt1[b+1][va] += 1
                oof1[b+2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt1[b+2][va] += 1
                oof1[b+3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt1[b+3][va] += 1
                oof1[b+4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt1[b+4][va] += 1
                oof1[b+5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt1[b+5][va] += 1
        print(f"  stage1 seed {seed}")
    for i in range(n_ds * n_m):
        oof1[i] = np.where(cnt1[i] > 0, oof1[i] / cnt1[i], 0.5)

    # Get stage1 blend
    M1 = np.column_stack(oof1)
    w1 = optim_blend(M1, y)
    blend1 = M1 @ w1

    # Identify confident predictions
    p1 = np.clip(blend1, 1e-6, 1 - 1e-6)
    confident = (p1 > conf_thresh) | (p1 < (1 - conf_thresh))
    weights = np.ones(n)
    weights[confident] = boost_weight
    print(f"  Stage1: {confident.sum()}/{n} confident ({confident.mean()*100:.1f}%)")

    # Stage 2: retrain with boosted weights
    oof2 = [np.zeros(n) for _ in range(n_ds * n_m)]
    cnt2 = [np.zeros(n) for _ in range(n_ds * n_m)]
    for seed in seeds:
        for xi, X in enumerate(Xs):
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                mt = min(mi_top, Xtr.shape[1])
                sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mt]
                Xt, Xv = Xtr[:, sel], Xva[:, sel]
                b = xi * n_m
                w_tr = weights[tr]
                oof2[b][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr], sample_weight=w_tr).predict_proba(Xva)[:, 1]; cnt2[b][va] += 1
                oof2[b+1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr], sample_weight=w_tr).predict_proba(Xv)[:, 1]; cnt2[b+1][va] += 1
                oof2[b+2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr], sample_weight=w_tr).predict_proba(Xv)[:, 1]; cnt2[b+2][va] += 1
                oof2[b+3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr], sample_weight=w_tr).predict_proba(Xv)[:, 1]; cnt2[b+3][va] += 1
                oof2[b+4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr], sample_weight=w_tr).predict_proba(Xv)[:, 1]; cnt2[b+4][va] += 1
                oof2[b+5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr], sample_weight=w_tr).decision_function(Xva)); cnt2[b+5][va] += 1
        print(f"  stage2 seed {seed}")
    for i in range(n_ds * n_m):
        oof2[i] = np.where(cnt2[i] > 0, oof2[i] / cnt2[i], 0.5)

    names = [f"t_{ds}_{m}" for ds in ["sbr", "phys", "morph", "atlas"][:n_ds] for m in mnames]
    return oof2, names


# ============================================================
# SITE ANALYSIS
# ============================================================
def analyze_sites(blend, y, groups, tag=""):
    print(f"\n--- Per-site analysis [{tag}] ---")
    sites = np.unique(groups)
    worst_ll = 10
    for site in sorted(sites):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            try:
                ll = log_loss(y[mask], np.clip(blend[mask], 1e-6, 1-1e-6))
                auc = roc_auc_score(y[mask], blend[mask])
                n = mask.sum()
                marker = " ***" if ll > 0.5 else ""
                print(f"  {site:20s} n={n:5d} AUC={auc:.3f} LL={ll:.3f}{marker}")
            except:
                pass


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    Xs_base = build_tabular(merged)
    print(f"Base features: {[X.shape[1] for X in Xs_base]}")

    results = {}

    # ============================================================
    # EXPERIMENT A: Baseline (same as v15)
    # ============================================================
    print(f"\n{'='*60}")
    print("A: BASELINE (4-dataset, 6 models)")
    print("=" * 60)
    oof_a, names_a = run_full_oof(Xs_base, y, groups)
    M_a = np.column_stack(oof_a)
    w_a = optim_blend(M_a, y)
    blend_a = M_a @ w_a
    a, b_pl, p_a = optimize_platt(blend_a, y)
    ll_a = log_loss(y, np.clip(p_a, 1e-6, 1-1e-6))
    auc_a = roc_auc_score(y, p_a)
    print(f"  BASELINE: AUC={auc_a:.4f} LL={ll_a:.4f}")
    results["A_baseline"] = {"auc": float(auc_a), "ll": float(ll_a)}
    analyze_sites(blend_a, y, groups, "baseline")

    # ============================================================
    # EXPERIMENT B: 2-stage boosting
    # ============================================================
    print(f"\n{'='*60}")
    print("B: TWO-STAGE BOOSTING (conf_thresh=0.85, weight=1.5)")
    print("=" * 60)
    oof_b, names_b = run_two_stage_oof(Xs_base, y, groups, conf_thresh=0.85, boost_weight=1.5)
    M_b = np.column_stack(oof_b)
    w_b = optim_blend(M_b, y)
    blend_b = M_b @ w_b
    a, b_pl, p_b = optimize_platt(blend_b, y)
    ll_b = log_loss(y, np.clip(p_b, 1e-6, 1-1e-6))
    auc_b = roc_auc_score(y, p_b)
    print(f"  2-STAGE: AUC={auc_b:.4f} LL={ll_b:.4f}")
    results["B_two_stage"] = {"auc": float(auc_b), "ll": float(ll_b)}

    # ============================================================
    # EXPERIMENT C: Advanced features + 2-stage
    # ============================================================
    print(f"\n{'='*60}")
    print("C: ADVANCED FEATURES + 2-STAGE")
    print("=" * 60)
    merged_adv = add_advanced_features(merged.copy())
    Xs_adv = build_tabular(merged_adv)
    print(f"Advanced features: {[X.shape[1] for X in Xs_adv]}")
    oof_c, names_c = run_two_stage_oof(Xs_adv, y, groups, conf_thresh=0.85, boost_weight=1.5)
    M_c = np.column_stack(oof_c)
    w_c = optim_blend(M_c, y)
    blend_c = M_c @ w_c
    a, b_pl, p_c = optimize_platt(blend_c, y)
    ll_c = log_loss(y, np.clip(p_c, 1e-6, 1-1e-6))
    auc_c = roc_auc_score(y, p_c)
    print(f"  ADV+2STAGE: AUC={auc_c:.4f} LL={ll_c:.4f}")
    results["C_adv_2stage"] = {"auc": float(auc_c), "ll": float(ll_c)}

    # ============================================================
    # EXPERIMENT D: Advanced features only (baseline comparison)
    # ============================================================
    print(f"\n{'='*60}")
    print("D: ADVANCED FEATURES ONLY")
    print("=" * 60)
    oof_d, names_d = run_full_oof(Xs_adv, y, groups)
    M_d = np.column_stack(oof_d)
    w_d = optim_blend(M_d, y)
    blend_d = M_d @ w_d
    a, b_pl, p_d = optimize_platt(blend_d, y)
    ll_d = log_loss(y, np.clip(p_d, 1e-6, 1-1e-6))
    auc_d = roc_auc_score(y, p_d)
    print(f"  ADV_BASELINE: AUC={auc_d:.4f} LL={ll_d:.4f}")
    results["D_adv_baseline"] = {"auc": float(auc_d), "ll": float(ll_d)}

    # ============================================================
    # EXPERIMENT E: Site calibration on best blend
    # ============================================================
    print(f"\n{'='*60}")
    print("E: SITE-SPECIFIC CALIBRATION")
    print("=" * 60)
    for tag_e, blend_e in [("baseline", blend_a), ("2stage", blend_b), ("adv_2stage", blend_c)]:
        p_site = site_calibrate(blend_e, y, groups)
        ll_e = log_loss(y, np.clip(p_site, 1e-6, 1-1e-6))
        auc_e = roc_auc_score(y, p_site)
        print(f"  site_cal [{tag_e}]: AUC={auc_e:.4f} LL={ll_e:.4f}")
        results[f"E_site_cal_{tag_e}"] = {"auc": float(auc_e), "ll": float(ll_e)}

    # ============================================================
    # EXPERIMENT F: Nested OOF calibration
    # ============================================================
    print(f"\n{'='*60}")
    print("F: NESTED OOF CALIBRATION")
    print("=" * 60)
    for tag_f, blend_f in [("baseline", blend_a), ("2stage", blend_b)]:
        p_nested = nested_oof_calibrate(blend_f, y, groups)
        ll_f = log_loss(y, np.clip(p_nested, 1e-6, 1-1e-6))
        auc_f = roc_auc_score(y, p_nested)
        print(f"  nested_cal [{tag_f}]: AUC={auc_f:.4f} LL={ll_f:.4f}")
        results[f"F_nested_cal_{tag_f}"] = {"auc": float(auc_f), "ll": float(ll_f)}

    # ============================================================
    # EXPERIMENT G: Single threshold sweep (best from literature)
    # ============================================================
    print(f"\n{'='*60}")
    print("G: 2-STAGE with different thresholds")
    print("=" * 60)
    for thresh, weight in [(0.80, 1.5), (0.90, 1.5), (0.90, 2.0)]:
        oof_g, _ = run_two_stage_oof(Xs_base, y, groups, conf_thresh=thresh, boost_weight=weight)
        M_g = np.column_stack(oof_g)
        w_g = optim_blend(M_g, y)
        blend_g = M_g @ w_g
        a, b_pl, p_g = optimize_platt(blend_g, y)
        ll_g = log_loss(y, np.clip(p_g, 1e-6, 1-1e-6))
        auc_g = roc_auc_score(y, p_g)
        print(f"  thresh={thresh:.2f} weight={weight:.1f}: AUC={auc_g:.4f} LL={ll_g:.4f}")
        results[f"G_t{thresh}_w{weight}"] = {"auc": float(auc_g), "ll": float(ll_g)}

    # ============================================================
    # EXPERIMENT H: Greedy stream elimination on best
    # ============================================================
    print(f"\n{'='*60}")
    print("H: GREEDY ELIMINATION on ADV+2STAGE")
    print("=" * 60)
    best_ll_h = ll_c
    best_mask_h = np.ones(M_c.shape[1], dtype=bool)
    for _ in range(M_c.shape[1]):
        improved = False
        for j in range(M_c.shape[1]):
            if not best_mask_h[j]:
                continue
            mask = best_mask_h.copy()
            mask[j] = False
            Mc = M_c[:, mask]
            if Mc.shape[1] == 0:
                continue
            wc = optim_blend(Mc, y)
            bc = Mc @ wc
            a, b_pl, pc = optimize_platt(bc, y)
            ll_cand = log_loss(y, np.clip(pc, 1e-6, 1-1e-6))
            if ll_cand < best_ll_h:
                best_ll_h = ll_cand
                best_mask_h = mask.copy()
                best_auc_h = roc_auc_score(y, pc)
                improved = True
                print(f"  Dropped stream {j} -> LL={best_ll_h:.4f} AUC={best_auc_h:.4f}")
        if not improved:
            break
    a, b_pl, p_h = optimize_platt(M_c[:, best_mask_h] @ optim_blend(M_c[:, best_mask_h], y), y)
    ll_h = log_loss(y, np.clip(p_h, 1e-6, 1-1e-6))
    auc_h = roc_auc_score(y, p_h)
    print(f"  ELIMINATED: AUC={auc_h:.4f} LL={ll_h:.4f} (streams={best_mask_h.sum()})")
    results["H_elim"] = {"auc": float(auc_h), "ll": float(ll_h)}

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
        marker = " **" if delta > 0.005 else (" *" if delta > 0.001 else "")
        print(f"  {name:38s} {r['auc']:8.4f} {r['ll']:8.4f} {delta:+10.4f}{marker}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    with open("v20_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved v20_results.json")


if __name__ == "__main__":
    main()
