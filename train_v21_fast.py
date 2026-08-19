"""
V21: Focused tabular improvements for DaT-SPECT.
Only the most promising experiments, fast execution.
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
from sklearn.model_selection import StratifiedGroupKFold
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


def optimize_temp(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    lg = logit(oof)
    def loss(lt):
        return log_loss(y, np.clip(expit(lg / np.exp(lt[0])), 1e-6, 1 - 1e-6))
    r = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    T = float(np.exp(r.x[0]))
    return T, expit(lg / T)


def site_calibrate(blend, y, groups):
    result = blend.copy()
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            a, b, _ = optimize_platt(blend[mask], y[mask])
            result[mask] = expit(a * logit(np.clip(blend[mask], 1e-6, 1-1e-6)) + b)
    return result


def train_one_fold(Xs, y_tr, tr, va, seed, weights=None):
    """Train 6 models on one fold. Returns OOF predictions for val."""
    preds = []
    n_m = 6
    for xi, X in enumerate(Xs):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        mt = min(44, Xtr.shape[1])
        sel = np.argsort(mutual_info_classif(Xtr, y_tr, random_state=seed))[::-1][:mt]
        Xt, Xv = Xtr[:, sel], Xva[:, sel]
        w_tr = weights[tr] if weights is not None else None

        lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed)
        if w_tr is not None: lr.fit(Xtr, y_tr, sample_weight=w_tr)
        else: lr.fit(Xtr, y_tr)
        preds.append(lr.predict_proba(Xva)[:, 1])

        xgb_m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0)
        if w_tr is not None: xgb_m.fit(Xt, y_tr, sample_weight=w_tr)
        else: xgb_m.fit(Xt, y_tr)
        preds.append(xgb_m.predict_proba(Xv)[:, 1])

        lgb_m = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1)
        if w_tr is not None: lgb_m.fit(Xt, y_tr, sample_weight=w_tr)
        else: lgb_m.fit(Xt, y_tr)
        preds.append(lgb_m.predict_proba(Xv)[:, 1])

        cb_m = cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0)
        if w_tr is not None: cb_m.fit(Xt, y_tr, sample_weight=w_tr)
        else: cb_m.fit(Xt, y_tr)
        preds.append(cb_m.predict_proba(Xv)[:, 1])

        et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed)
        if w_tr is not None: et.fit(Xt, y_tr, sample_weight=w_tr)
        else: et.fit(Xt, y_tr)
        preds.append(et.predict_proba(Xv)[:, 1])

        ridge = RidgeClassifier(alpha=0.1, random_state=seed)
        if w_tr is not None: ridge.fit(Xtr, y_tr, sample_weight=w_tr)
        else: ridge.fit(Xtr, y_tr)
        preds.append(expit(ridge.decision_function(Xva)))

    return preds  # list of 24 arrays (4 datasets x 6 models)


def full_oof(Xs, y, groups, seeds=SEEDS, weights=None, tag=""):
    n_ds = len(Xs)
    n_m = 6
    n_streams = n_ds * n_m
    n = len(y)
    oof = [np.zeros(n) for _ in range(n_streams)]
    cnt = [np.zeros(n) for _ in range(n_streams)]

    for seed in seeds:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(Xs[0], y, groups):
            preds = train_one_fold(Xs, y[tr], tr, va, seed, weights)
            for i, p in enumerate(preds):
                oof[i][va] += p
                cnt[i][va] += 1
        print(f"  {tag} seed {seed}")
    for i in range(n_streams):
        oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)
    return oof


def evaluate(oof_list, y, tag=""):
    M = np.column_stack(oof_list)
    w = optim_blend(M, y)
    blend = M @ w
    auc_raw = roc_auc_score(y, blend)
    ll_raw = log_loss(y, np.clip(blend, 1e-6, 1-1e-6))

    a, b, p_platt = optimize_platt(blend, y)
    ll_platt = log_loss(y, np.clip(p_platt, 1e-6, 1-1e-6))
    auc_platt = roc_auc_score(y, p_platt)

    T, p_temp = optimize_temp(blend, y)
    ll_temp = log_loss(y, np.clip(p_temp, 1e-6, 1-1e-6))
    auc_temp = roc_auc_score(y, p_temp)

    best_ll = min(ll_platt, ll_temp)
    best_auc = auc_platt if ll_platt <= ll_temp else auc_temp
    best_method = "platt" if ll_platt <= ll_temp else "temp"
    print(f"  {tag}: raw(AUC={auc_raw:.4f} LL={ll_raw:.4f}) platt(LL={ll_platt:.4f}) temp(T={T:.3f} LL={ll_temp:.4f}) -> BEST={best_method} AUC={best_auc:.4f} LL={best_ll:.4f}")
    return best_auc, best_ll, blend


def greedy_eliminate(M, y, names, init_auc, init_ll):
    best_ll = init_ll
    best_auc = init_auc
    best_mask = np.ones(M.shape[1], dtype=bool)
    for _ in range(M.shape[1]):
        improved = False
        for j in range(M.shape[1]):
            if not best_mask[j]:
                continue
            mask = best_mask.copy()
            mask[j] = False
            Mc = M[:, mask]
            if Mc.shape[1] == 0:
                continue
            wc = optim_blend(Mc, y)
            bc = Mc @ wc
            a, b, pc = optimize_platt(bc, y)
            ll_c = log_loss(y, np.clip(pc, 1e-6, 1-1e-6))
            if ll_c < best_ll:
                best_ll = ll_c
                best_auc = roc_auc_score(y, pc)
                best_mask = mask.copy()
                improved = True
        if not improved:
            break
    active = [names[i] for i in range(len(names)) if best_mask[i]]
    print(f"  Eliminated to {best_mask.sum()} streams: LL={best_ll:.4f} AUC={best_auc:.4f}")
    return best_auc, best_ll, best_mask


def main():
    t0 = time.time()
    merged, y, groups = load_data()
    print(f"n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")
    merged = add_bio(merged)
    Xs_base = build_tabular(merged)
    n_ds = len(Xs_base)
    n_m = 6
    print(f"Base: {[X.shape[1] for X in Xs_base]}")

    ds_names = ["sbr", "phys", "morph", "atlas"][:n_ds]
    mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    stream_names = [f"{ds}_{m}" for ds in ds_names for m in mnames]

    results = {}

    # ============================================================
    # A: BASELINE
    # ============================================================
    print(f"\n{'='*60}")
    print("A: BASELINE")
    print("=" * 60)
    oof_a = full_oof(Xs_base, y, groups, tag="A")
    M_a = np.column_stack(oof_a)
    auc_a, ll_a, blend_a = evaluate(oof_a, y, "baseline")
    results["A_baseline"] = {"auc": float(auc_a), "ll": float(ll_a)}

    # Greedy elimination
    auc_elim, ll_elim, mask_elim = greedy_eliminate(M_a, y, stream_names, auc_a, ll_a)
    results["A_baseline_elim"] = {"auc": float(auc_elim), "ll": float(ll_elim)}

    # Per-site analysis
    print("\n  --- Per-site analysis ---")
    for site in sorted(np.unique(groups)):
        s_mask = groups == site
        if s_mask.sum() > 10 and len(np.unique(y[s_mask])) > 1:
            ll = log_loss(y[s_mask], np.clip(blend_a[s_mask], 1e-6, 1-1e-6))
            auc = roc_auc_score(y[s_mask], blend_a[s_mask])
            print(f"  {site:20s} n={s_mask.sum():5d} AUC={auc:.3f} LL={ll:.3f}")

    # ============================================================
    # B: TWO-STAGE (conf=0.85, w=1.5)
    # ============================================================
    print(f"\n{'='*60}")
    print("B: TWO-STAGE (conf=0.85, w=1.5)")
    print("=" * 60)
    # Stage 1 already done, blend_a is stage1 blend
    p1 = np.clip(blend_a, 1e-6, 1 - 1e-6)
    confident = (p1 > 0.85) | (p1 < 0.15)
    weights_b = np.ones(len(y))
    weights_b[confident] = 1.5
    print(f"  Confident: {confident.sum()}/{len(y)} ({confident.mean()*100:.1f}%)")
    oof_b = full_oof(Xs_base, y, groups, weights=weights_b, tag="B")
    auc_b, ll_b, blend_b = evaluate(oof_b, y, "2stage")
    results["B_two_stage"] = {"auc": float(auc_b), "ll": float(ll_b)}

    # ============================================================
    # C: ADVANCED FEATURES
    # ============================================================
    print(f"\n{'='*60}")
    print("C: ADVANCED FEATURES")
    print("=" * 60)
    merged_adv = add_adv(merged.copy())
    Xs_adv = build_tabular(merged_adv)
    print(f"  Adv: {[X.shape[1] for X in Xs_adv]}")
    oof_c = full_oof(Xs_adv, y, groups, tag="C")
    auc_c, ll_c, blend_c = evaluate(oof_c, y, "adv")
    results["C_adv"] = {"auc": float(auc_c), "ll": float(ll_c)}

    # ============================================================
    # D: ADVANCED + TWO-STAGE
    # ============================================================
    print(f"\n{'='*60}")
    print("D: ADVANCED + TWO-STAGE")
    print("=" * 60)
    oof_d = full_oof(Xs_adv, y, groups, weights=weights_b, tag="D")
    auc_d, ll_d, blend_d = evaluate(oof_d, y, "adv_2stage")
    results["D_adv_2stage"] = {"auc": float(auc_d), "ll": float(ll_d)}
    M_d = np.column_stack(oof_d)
    auc_elim_d, ll_elim_d, _ = greedy_eliminate(M_d, y, stream_names, auc_d, ll_d)
    results["D_adv_2stage_elim"] = {"auc": float(auc_elim_d), "ll": float(ll_elim_d)}

    # ============================================================
    # E: SITE CALIBRATION on best
    # ============================================================
    print(f"\n{'='*60}")
    print("E: SITE CALIBRATION")
    print("=" * 60)
    for tag_e, blend_e in [("baseline", blend_a), ("2stage", blend_b), ("adv_2stage", blend_d)]:
        p_site = site_calibrate(blend_e, y, groups)
        ll_e = log_loss(y, np.clip(p_site, 1e-6, 1-1e-6))
        auc_e = roc_auc_score(y, p_site)
        print(f"  site_cal [{tag_e}]: AUC={auc_e:.4f} LL={ll_e:.4f}")
        results[f"E_site_cal_{tag_e}"] = {"auc": float(auc_e), "ll": float(ll_e)}

    # ============================================================
    # F: STACKING (use OOF from different seeds as meta-features)
    # ============================================================
    print(f"\n{'='*60}")
    print("F: ENSEMBLE OF 2-STAGE RESULTS")
    print("=" * 60)
    # Blend the best results
    best_blend = np.column_stack([blend_a, blend_b, blend_c, blend_d])
    best_names = ["baseline", "2stage", "adv", "adv_2stage"]
    w_best = optim_blend(best_blend, y)
    final_blend = best_blend @ w_best
    a, b_pl, p_final = optimize_platt(final_blend, y)
    ll_final = log_loss(y, np.clip(p_final, 1e-6, 1-1e-6))
    auc_final = roc_auc_score(y, p_final)
    print(f"  ENSEMBLE: AUC={auc_final:.4f} LL={ll_final:.4f}")
    for n, wv in zip(best_names, w_best):
        print(f"    {n}: {wv:.4f}")
    results["F_ensemble"] = {"auc": float(auc_final), "ll": float(ll_final)}

    # Site cal on ensemble
    p_site_final = site_calibrate(final_blend, y, groups)
    ll_site_final = log_loss(y, np.clip(p_site_final, 1e-6, 1-1e-6))
    auc_site_final = roc_auc_score(y, p_site_final)
    print(f"  ENSEMBLE+SITE_CAL: AUC={auc_site_final:.4f} LL={ll_site_final:.4f}")
    results["F_ensemble_site_cal"] = {"auc": float(auc_site_final), "ll": float(ll_site_final)}

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
    with open("v21_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved v21_results.json")


if __name__ == "__main__":
    main()
