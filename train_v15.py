"""
V15: Aggressive approach - nested calibration + site-aware blending + feature interactions.
Paper insight: "A model can have AUROC > 0.93 but poor log loss if probabilities are overconfident."
Key: Need to fundamentally improve the model, not just calibrate.
"""
import json
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

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

    phys_keep = [c for c in phys.columns if c != "uid"]
    merged = merged.merge(phys[["uid"] + phys_keep].rename(columns={c: f"ph_{c}" for c in phys_keep}), on="uid")

    morph_keep = [c for c in morph.columns if c != "uid"]
    merged = merged.merge(morph[["uid"] + morph_keep].rename(columns={c: f"mo_{c}" for c in morph_keep}), on="uid")

    atlas_keep = [c for c in atlas.columns if c != "uid"]
    merged = merged.merge(atlas[["uid"] + atlas_keep].rename(columns={c: f"at_{c}" for c in atlas_keep}), on="uid")

    merged = merged.dropna()
    y = merged["is_pathologic"].values
    groups = merged["pseudo_site"].astype(str).values

    fac = (PVE_REF / merged["voxel_vol"].values) ** (1.0 / 3.0)
    for c in [c for c in raw1 if c.startswith("sbr_") and c in merged.columns]:
        merged[c] = merged[c] * fac

    return merged, y, groups


def add_biological_features(df):
    eps = 1e-6
    
    # Asymmetry features
    if "sbr_left_putamen" in df.columns and "sbr_right_putamen" in df.columns:
        L = df["sbr_left_putamen"].values
        R = df["sbr_right_putamen"].values
        df["lr_putamen_asym"] = (L - R) / (L + R + eps)
        df["lr_putamen_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
        df["lr_putamen_sum"] = L + R
        df["lr_putamen_diff"] = L - R
        
    if "sbr_left_caudate" in df.columns and "sbr_right_caudate" in df.columns:
        L = df["sbr_left_caudate"].values
        R = df["sbr_right_caudate"].values
        df["lr_caudate_asym"] = (L - R) / (L + R + eps)
        df["lr_caudate_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
        df["lr_caudate_sum"] = L + R
        df["lr_caudate_diff"] = L - R
    
    # Posterior-anterior ratios
    if "sbr_left_putamen_post" in df.columns and "sbr_left_putamen_ant" in df.columns:
        df["left_post_ant_ratio"] = df["sbr_left_putamen_post"].values / (df["sbr_left_putamen_ant"].values + eps)
    if "sbr_right_putamen_post" in df.columns and "sbr_right_putamen_ant" in df.columns:
        df["right_post_ant_ratio"] = df["sbr_right_putamen_post"].values / (df["sbr_right_putamen_ant"].values + eps)
    
    # Putamen-to-caudate ratios
    if "sbr_left_putamen" in df.columns and "sbr_left_caudate" in df.columns:
        df["left_putamen_caudate_ratio"] = df["sbr_left_putamen"].values / (df["sbr_left_caudate"].values + eps)
    if "sbr_right_putamen" in df.columns and "sbr_right_caudate" in df.columns:
        df["right_putamen_caudate_ratio"] = df["sbr_right_putamen"].values / (df["sbr_right_caudate"].values + eps)
    
    # Total striatal features
    if "sbr_left_total" in df.columns and "sbr_right_total" in df.columns:
        L = df["sbr_left_total"].values
        R = df["sbr_right_total"].values
        df["total_striatal"] = L + R
        df["lr_total_asym"] = (L - R) / (L + R + eps)
    
    # Feature interactions (key for nonlinear patterns)
    if "sbr_left_putamen" in df.columns and "sbr_left_putamen_post" in df.columns:
        df["lp_x_post"] = df["sbr_left_putamen"].values * df["sbr_left_putamen_post"].values
    if "sbr_right_putamen" in df.columns and "sbr_right_putamen_post" in df.columns:
        df["rp_x_post"] = df["sbr_right_putamen"].values * df["sbr_right_putamen_post"].values
    
    return df


def build_matrix(raw, raw_cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(
        np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def build_all_matrices(merged):
    raw1 = [c for c in merged.columns if not c.startswith("ph_") and not c.startswith("mo_") and not c.startswith("at_")
            and c not in ("uid", "is_pathologic", "pseudo_site", "site_id", "sx", "sy", "sz",
                          "voxel_vol", "voxel_side", "label")]
    raw2 = [c for c in merged.columns if c.startswith("ph_")]
    raw3 = [c for c in merged.columns if c.startswith("mo_")]
    raw4 = [c for c in merged.columns if c.startswith("at_")]
    return (
        build_matrix(merged, raw1),
        build_matrix(merged, raw2),
        build_matrix(merged, raw3),
        build_matrix(merged, raw4),
    ), (raw1, raw2, raw3, raw4)


def optim_blend(M, y):
    def loss(w):
        p = M @ w
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    r = minimize(loss, np.ones(M.shape[1]) / M.shape[1],
                 method="L-BFGS-B", bounds=[(0, 1)] * M.shape[1])
    return r.x / r.x.sum()


def optimize_temperature(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    def loss(log_t):
        p = expit(logit(oof) / np.exp(log_t[0]))
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    r = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return float(np.exp(r.x[0]))


def optimize_platt_scalar(oof, y):
    from scipy.optimize import minimize
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    logits = logit(oof)
    def loss(params):
        a, b = params
        p = expit(a * logits + b)
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
    a, b = r.x
    calibrated = expit(a * logits + b)
    return a, b, calibrated


def train_and_evaluate(Xs, y, groups, seeds):
    n_datasets = len(Xs)
    model_names = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
    n_models = len(model_names)
    n_streams = n_datasets * n_models
    n_samples = len(y)

    oof = {i: np.zeros(n_samples) for i in range(n_streams)}
    counts = {i: np.zeros(n_samples) for i in range(n_streams)}

    for seed in seeds:
        for xi, X in enumerate(Xs):
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])

                n_feat = Xtr.shape[1]
                mi_top = min(44, n_feat)
                mi = mutual_info_classif(Xtr, y[tr], random_state=seed)
                sel = np.argsort(mi)[::-1][:mi_top]
                Xtr_t, Xva_t = Xtr[:, sel], Xva[:, sel]

                base = xi * n_models

                lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr])
                oof[base][va] += lr.predict_proba(Xva)[:, 1]
                counts[base][va] += 1

                xgb_model = xgb.XGBClassifier(
                    n_estimators=500, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=seed, n_jobs=-1, verbosity=0
                ).fit(Xtr_t, y[tr])
                oof[base + 1][va] += xgb_model.predict_proba(Xva_t)[:, 1]
                counts[base + 1][va] += 1

                lgb_model = lgb.LGBMClassifier(
                    n_estimators=500, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                    min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1
                ).fit(Xtr_t, y[tr])
                oof[base + 2][va] += lgb_model.predict_proba(Xva_t)[:, 1]
                counts[base + 2][va] += 1

                cb_model = cb.CatBoostClassifier(
                    iterations=500, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_seed=seed, verbose=0
                ).fit(Xtr_t, y[tr])
                oof[base + 3][va] += cb_model.predict_proba(Xva_t)[:, 1]
                counts[base + 3][va] += 1

                et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xtr_t, y[tr])
                oof[base + 4][va] += et.predict_proba(Xva_t)[:, 1]
                counts[base + 4][va] += 1

                ridge = RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr])
                oof[base + 5][va] += expit(ridge.decision_function(Xva))
                counts[base + 5][va] += 1

        print(f"  seed {seed} done")

    for i in range(n_streams):
        oof[i] = np.where(counts[i] > 0, oof[i] / counts[i], 0.5)

    ds_names = ["sbr", "phys", "morph", "atlas"]
    names = [f"{ds}_{m}" for ds in ds_names[:n_datasets] for m in model_names]

    return oof, names


def analyze_site_performance(M, y, groups, names):
    """Analyze per-site performance to identify weak sites."""
    print("\n--- Site Performance Analysis ---")
    sites = np.unique(groups)
    site_aucs = {}
    for site in sites:
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            blend = M[mask].mean(axis=1)
            try:
                auc = roc_auc_score(y[mask], blend)
                site_aucs[site] = auc
            except:
                pass
    
    if site_aucs:
        sorted_sites = sorted(site_aucs.items(), key=lambda x: x[1])
        print(f"  Worst sites: {[(s, f'{a:.3f}') for s, a in sorted_sites[:5]]}")
        print(f"  Best sites: {[(s, f'{a:.3f}') for s, a in sorted_sites[-5:]]}")
    
    return site_aucs


def main():
    print("Loading ALL 4 feature sets...")
    merged, y, groups = load_data()
    print(f"  n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")

    merged = add_biological_features(merged)
    print(f"  Total columns: {len(merged.columns)}")

    Xs, raw_cols = build_all_matrices(merged)
    for i, (name, X) in enumerate(zip(["sbr", "phys", "morph", "atlas"], Xs)):
        print(f"  {name}: {X.shape}")

    print("\nTraining OOF...")
    oof, names = train_and_evaluate(Xs, y, groups, SEEDS)

    M = np.column_stack([oof[i] for i in range(len(oof))])
    print(f"\nOOF matrix: {M.shape}")

    # Analyze per-site performance
    site_aucs = analyze_site_performance(M, y, groups, names)

    print("\nPer-stream metrics:")
    for i, name in enumerate(names):
        try:
            auc = roc_auc_score(y, M[:, i])
            ll = log_loss(y, np.clip(M[:, i], 1e-6, 1 - 1e-6))
            print(f"  {name:20s} AUC={auc:.4f}  LL={ll:.4f}")
        except Exception as e:
            print(f"  {name:20s} error: {e}")

    # Optimize blend
    w = optim_blend(M, y)
    blend = M @ w
    print(f"\nBLEND AUC={roc_auc_score(y, blend):.4f}  LL={log_loss(y, np.clip(blend, 1e-6, 1-1e-6)):.4f}")

    # Try Platt scaling
    a_platt, b_platt, p_platt = optimize_platt_scalar(blend, y)
    ll_platt = log_loss(y, np.clip(p_platt, 1e-6, 1-1e-6))
    print(f"PLATT (a={a_platt:.4f}, b={b_platt:.4f}): AUC={roc_auc_score(y, p_platt):.4f}  LL={ll_platt:.4f}")

    # Try temperature scaling
    temp = optimize_temperature(blend, y)
    p_temp = expit(logit(np.clip(blend, 1e-6, 1 - 1e-6)) / temp)
    ll_temp = log_loss(y, np.clip(p_temp, 1e-6, 1-1e-6))
    print(f"TEMP (T={temp:.4f}): AUC={roc_auc_score(y, p_temp):.4f}  LL={ll_temp:.4f}")

    # Select best
    best_p = p_platt if ll_platt < ll_temp else p_temp
    best_method = "platt" if ll_platt < ll_temp else "temp"
    best_ll = min(ll_platt, ll_temp)
    best_auc = roc_auc_score(y, best_p)
    print(f"\nBEST: {best_method} AUC={best_auc:.4f}  LL={best_ll:.4f}")

    # Greedy elimination
    print("\n--- Greedy stream elimination ---")
    best_ll_elim = best_ll
    best_auc_elim = best_auc
    best_w_elim = w.copy()
    best_mask = np.ones(M.shape[1], dtype=bool)
    best_method_elim = best_method
    best_calib_params = {}

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
            
            # Try Platt
            a_c, b_c, p_c_platt = optimize_platt_scalar(bc, y)
            ll_c_platt = log_loss(y, np.clip(p_c_platt, 1e-6, 1-1e-6))
            
            # Try temperature
            t_c = optimize_temperature(bc, y)
            p_c_temp = expit(logit(np.clip(bc, 1e-6, 1 - 1e-6)) / t_c)
            ll_c_temp = log_loss(y, np.clip(p_c_temp, 1e-6, 1-1e-6))
            
            for mname, p, ll_c in [("platt", p_c_platt, ll_c_platt), ("temp", p_c_temp, ll_c_temp)]:
                if ll_c < best_ll_elim:
                    best_mask = mask.copy()
                    best_w_full = np.zeros(M.shape[1])
                    best_w_full[mask] = wc
                    best_w_elim = best_w_full
                    best_ll_elim = ll_c
                    best_auc_elim = roc_auc_score(y, p)
                    best_method_elim = mname
                    best_calib_params = {"a": a_c, "b": b_c, "temp": t_c} if mname == "platt" else {"temp": t_c}
                    improved = True
                    print(f"  Dropped {names[j]} ({mname}) -> AUC={best_auc_elim:.4f} LL={best_ll_elim:.4f}")
        if not improved:
            break

    print(f"\nFINAL: AUC={best_auc_elim:.4f}  LL={best_ll_elim:.4f}")
    active = [(names[i], float(best_w_elim[i])) for i in range(len(names)) if best_mask[i]]
    for n, wt in sorted(active, key=lambda x: -x[1]):
        print(f"  {n:20s} w={wt:.4f}")

    return merged, y, groups, Xs, best_w_elim, best_mask, names, best_auc_elim, best_ll_elim, best_method_elim, best_calib_params


if __name__ == "__main__":
    merged, y, groups, Xs, best_w, best_mask, names, final_auc, final_ll, best_method, calib_params = main()

    seeds = SEEDS
    out_dir = "submission_v15"
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)

    print("\nTraining final models on full data...")
    model_names_base = ["lr", "xgb", "lgb", "cb", "et", "ridge"]

    for si, seed in enumerate(seeds):
        for xi, X in enumerate(Xs):
            sc = StandardScaler().fit(X)
            Xs_scaled = sc.transform(X)
            mi = mutual_info_classif(Xs_scaled, y, random_state=seed)
            mi_top = min(44, Xs_scaled.shape[1])
            sel = np.argsort(mi)[::-1][:mi_top]
            Xt = Xs_scaled[:, sel]

            base = xi * len(model_names_base)

            lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xs_scaled, y)
            joblib.dump(lr, os.path.join(out_dir, f"weights/s{si}_{names[base]}.pkl"))

            xgb_model = xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                use_label_encoder=False, eval_metric="logloss",
                random_state=seed, n_jobs=-1, verbosity=0
            ).fit(Xt, y)
            joblib.dump(xgb_model, os.path.join(out_dir, f"weights/s{si}_{names[base+1]}.pkl"))

            lgb_model = lgb.LGBMClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1
            ).fit(Xt, y)
            joblib.dump(lgb_model, os.path.join(out_dir, f"weights/s{si}_{names[base+2]}.pkl"))

            cb_model = cb.CatBoostClassifier(
                iterations=500, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_seed=seed, verbose=0
            ).fit(Xt, y)
            joblib.dump(cb_model, os.path.join(out_dir, f"weights/s{si}_{names[base+3]}.pkl"))

            et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y)
            joblib.dump(et, os.path.join(out_dir, f"weights/s{si}_{names[base+4]}.pkl"))

            ridge = RidgeClassifier(alpha=0.1, random_state=seed).fit(Xs_scaled, y)
            joblib.dump(ridge, os.path.join(out_dir, f"weights/s{si}_{names[base+5]}.pkl"))

            joblib.dump(sc, os.path.join(out_dir, f"weights/s{si}_ds{xi}_scaler.pkl"))
            joblib.dump(sel, os.path.join(out_dir, f"weights/s{si}_ds{xi}_mi.pkl"))

        print(f"  seed {seed} done")

    active_weights = [float(best_w[i]) for i in range(len(best_w)) if best_mask[i]]
    active_indices = [int(i) for i in range(len(best_w)) if best_mask[i]]

    cfg = {
        "n_seeds": len(seeds),
        "model_names": names,
        "blend_weights": active_weights,
        "blend_indices": active_indices,
        "calibration_method": best_method,
        "calibration_params": calib_params,
        "clip_min": 0.005,
        "clip_max": 0.995,
        "mi_top": 44,
        "n_datasets": 4,
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    with open(os.path.join(out_dir, "oof_metrics.json"), "w") as f:
        json.dump({
            "oof_auroc": round(float(final_auc), 4),
            "oof_ll": round(float(final_ll), 4),
            "active_models": [names[i] for i in range(len(names)) if best_mask[i]],
        }, f, indent=2)

    shutil.copy("src/sbr_extractor.py", os.path.join(out_dir, "sbr_extractor.py"))
    shutil.copy("submission_v7/sbr_extractor_phys.py", os.path.join(out_dir, "sbr_extractor_phys.py"))
    print(f"\nSaved to {out_dir}")
