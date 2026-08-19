"""
Comprehensive OOF training using ALL 4 feature sets:
  1. sbr_features (93 cols) - original SBR extractor features
  2. phys_features (109 cols) - physical features with volumes/shape
  3. morph_features (125 cols) - shape morphology at different thresholds
  4. atlas_features (34 cols) - atlas-based ROI features

Key fix: morph + atlas features were NEVER loaded in v6 pipeline.
"""
import json
import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
PVE_REF = 15.625

RAW1_COLS = None
RAW2_COLS = None
RAW3_COLS = None
RAW4_COLS = None


def load_data():
    global RAW1_COLS, RAW2_COLS, RAW3_COLS, RAW4_COLS

    labels = pd.read_csv("Dataset/train_labels.csv")
    site = pd.read_csv("Dataset/site_labels.csv")
    geom = pd.read_csv("Dataset/voxel_geometry.csv")
    sbr = pd.read_csv("Dataset/sbr_features_train.csv")
    phys = pd.read_csv("Dataset/phys_features_train.csv")
    morph = pd.read_csv("Dataset/sbr_features_morph_train.csv")
    atlas = pd.read_csv("Dataset/atlas_features_train.csv")

    merged = labels.merge(site, on="uid").merge(geom, on="uid")

    RAW1_COLS = [c for c in sbr.columns if c not in ("uid", "label")]
    merged = merged.merge(sbr[["uid"] + RAW1_COLS], on="uid")

    phys_keep = [c for c in phys.columns if c != "uid"]
    RAW2_COLS = [f"ph_{c}" for c in phys_keep]
    merged = merged.merge(phys[["uid"] + phys_keep].rename(columns={c: f"ph_{c}" for c in phys_keep}), on="uid")

    morph_keep = [c for c in morph.columns if c != "uid"]
    RAW3_COLS = [f"mo_{c}" for c in morph_keep]
    merged = merged.merge(morph[["uid"] + morph_keep].rename(columns={c: f"mo_{c}" for c in morph_keep}), on="uid")

    # For atlas: rename to at_ prefix to avoid column name collision with sbr features
    atlas_keep = [c for c in atlas.columns if c != "uid"]
    RAW4_COLS = [f"at_{c}" for c in atlas_keep]
    merged = merged.merge(atlas[["uid"] + atlas_keep].rename(columns={c: f"at_{c}" for c in atlas_keep}), on="uid")

    merged = merged.dropna()
    y = merged["is_pathologic"].values
    groups = merged["pseudo_site"].astype(str).values

    fac = (PVE_REF / merged["voxel_vol"].values) ** (1.0 / 3.0)
    for c in [c for c in RAW1_COLS if c.startswith("sbr_") and c in merged.columns]:
        merged[c] = merged[c] * fac

    return merged, y, groups


def build_matrix(raw, raw_cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(
        np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def build_combined_matrix(merged):
    X1 = build_matrix(merged, RAW1_COLS)
    X2 = build_matrix(merged, RAW2_COLS)
    X3 = build_matrix(merged, RAW3_COLS)
    X4 = build_matrix(merged, RAW4_COLS)
    return X1, X2, X3, X4


def optim_blend(M, y):
    def loss(w):
        return log_loss(y, np.clip(M @ w, 1e-6, 1 - 1e-6))
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


def train_and_evaluate(Xs, y, groups, seeds, mi_top=44, lr_c=1.31, ridge_alpha=0.1):
    n_streams = len(Xs) * 5
    n_samples = len(y)
    oof = {i: np.zeros(n_samples) for i in range(n_streams)}
    counts = {i: np.zeros(n_samples) for i in range(n_streams)}

    for seed in seeds:
        for xi, X in enumerate(Xs):
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])

                mi = mutual_info_classif(Xtr, y[tr], random_state=seed)
                sel = np.argsort(mi)[::-1][:mi_top]
                Xtr_t, Xva_t = Xtr[:, sel], Xva[:, sel]

                stream_idx = xi * 5

                lr = LogisticRegression(C=lr_c, max_iter=3000, random_state=seed).fit(Xtr, y[tr])
                oof[stream_idx][va] += lr.predict_proba(Xva)[:, 1]
                counts[stream_idx][va] += 1

                et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xtr_t, y[tr])
                oof[stream_idx + 1][va] += et.predict_proba(Xva_t)[:, 1]
                counts[stream_idx + 1][va] += 1

                hgb = HistGradientBoostingClassifier(
                    max_iter=500, max_depth=4, learning_rate=0.0489,
                    l2_regularization=4.6, min_samples_leaf=10, random_state=seed
                ).fit(Xtr_t, y[tr])
                oof[stream_idx + 2][va] += hgb.predict_proba(Xva_t)[:, 1]
                counts[stream_idx + 2][va] += 1

                rf = RandomForestClassifier(300, max_depth=14, n_jobs=-1, random_state=seed).fit(Xtr_t, y[tr])
                oof[stream_idx + 3][va] += rf.predict_proba(Xva_t)[:, 1]
                counts[stream_idx + 3][va] += 1

                ridge = RidgeClassifier(alpha=ridge_alpha, random_state=seed).fit(Xtr, y[tr])
                oof[stream_idx + 4][va] += expit(ridge.decision_function(Xva))
                counts[stream_idx + 4][va] += 1

    for i in range(n_streams):
        oof[i] = np.where(counts[i] > 0, oof[i] / counts[i], 0.5)

    model_names = ["lr", "et", "hgb", "rf", "ridge"]
    ds_names = ["sbr", "phys", "morph", "atlas"]
    names = [f"{ds}_{m}" for ds in ds_names for m in model_names]

    return oof, names


def main():
    print("Loading ALL 4 feature sets...")
    merged, y, groups = load_data()
    print(f"  n={len(y)}, positive={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")

    X1, X2, X3, X4 = build_combined_matrix(merged)
    print(f"  X1(sbr)={X1.shape}, X2(phys)={X2.shape}, X3(morph)={X3.shape}, X4(atlas)={X4.shape}")

    seeds = SEEDS
    mi_top = 44
    lr_c = 1.31
    ridge_alpha = 0.1

    Xs = [X1, X2, X3, X4]
    oof, names = train_and_evaluate(Xs, y, groups, seeds, mi_top, lr_c, ridge_alpha)

    M = np.column_stack([oof[i] for i in range(len(oof))])
    print(f"  OOF matrix: {M.shape}")

    # Per-stream metrics
    print("\nPer-stream AUC:")
    for i, name in enumerate(names):
        try:
            auc = roc_auc_score(y, M[:, i])
            ll = log_loss(y, np.clip(M[:, i], 1e-6, 1 - 1e-6))
            print(f"  {name:20s} AUC={auc:.4f}  LL={ll:.4f}")
        except:
            print(f"  {name:20s} (degenerate)")

    w = optim_blend(M, y)
    blend = M @ w
    temp = optimize_temperature(blend, y)
    p_temp = expit(logit(np.clip(blend, 1e-6, 1 - 1e-6)) / temp)
    final = np.clip(p_temp, 0.005, 0.995)

    print(f"\nBLEND AUC={roc_auc_score(y, blend):.4f}  LL={log_loss(y, np.clip(blend, 1e-6, 1-1e-6)):.4f}")
    print(f"TEMP={temp:.4f}")
    print(f"FINAL AUC={roc_auc_score(y, final):.4f}  LL={log_loss(y, final):.4f}")

    # Try dropping weak streams
    print("\n--- Greedy stream elimination ---")
    best_auc = roc_auc_score(y, final)
    best_ll = log_loss(y, final)
    best_w, best_temp = w.copy(), temp
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
            tc = optimize_temperature(bc, y)
            pc = expit(logit(np.clip(bc, 1e-6, 1 - 1e-6)) / tc)
            pc = np.clip(pc, 0.005, 0.995)
            auc_c = roc_auc_score(y, pc)
            ll_c = log_loss(y, pc)
            if ll_c < best_ll:
                best_mask = mask
                best_w_full = np.zeros(M.shape[1])
                best_w_full[mask] = wc
                best_w = best_w_full
                best_temp = tc
                best_auc = auc_c
                best_ll = ll_c
                improved = True
                print(f"  Dropped {names[j]} -> AUC={auc_c:.4f} LL={ll_c:.4f}")
        if not improved:
            break

    print(f"\nAfter elimination: AUC={best_auc:.4f}  LL={best_ll:.4f}")
    print(f"  Active streams: {[names[i] for i in range(len(names)) if best_mask[i]]}")
    print(f"  Weights: {np.round(best_w[best_mask], 3)}")

    return merged, y, groups, Xs, best_w, best_temp, best_mask, names, best_auc, best_ll


if __name__ == "__main__":
    merged, y, groups, Xs, best_w, best_temp, best_mask, names, final_auc, final_ll = main()

    seeds = SEEDS
    out_dir = "submission_v8"
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)

    print("\nTraining final models on full data...")
    for si, seed in enumerate(seeds):
        for xi, X in enumerate(Xs):
            sc = StandardScaler().fit(X)
            Xs_all = sc.transform(X)
            mi = mutual_info_classif(X, y, random_state=seed)
            sel = np.argsort(mi)[::-1][:44]
            Xt = Xs_all[:, sel]

            lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xs_all, y)
            et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y)
            hgb = HistGradientBoostingClassifier(
                max_iter=500, max_depth=4, learning_rate=0.0489,
                l2_regularization=4.6, min_samples_leaf=10, random_state=seed
            ).fit(Xt, y)
            rf = RandomForestClassifier(300, max_depth=14, n_jobs=-1, random_state=seed).fit(Xt, y)
            ridge = RidgeClassifier(alpha=0.1, random_state=seed).fit(Xs_all, y)

            joblib.dump(sc, os.path.join(out_dir, f"weights/s{si}_x{xi}_scaler.pkl"))
            joblib.dump(sel, os.path.join(out_dir, f"weights/s{si}_x{xi}_mi.pkl"))
            joblib.dump(lr, os.path.join(out_dir, f"weights/s{si}_x{xi}_lr.pkl"))
            joblib.dump(et, os.path.join(out_dir, f"weights/s{si}_x{xi}_et.pkl"))
            joblib.dump(hgb, os.path.join(out_dir, f"weights/s{si}_x{xi}_hgb.pkl"))
            joblib.dump(rf, os.path.join(out_dir, f"weights/s{si}_x{xi}_rf.pkl"))
            joblib.dump(ridge, os.path.join(out_dir, f"weights/s{si}_x{xi}_ridge.pkl"))
        print(f"  seed {seed} done")

    active_weights = []
    active_indices = []
    for i in range(len(best_w)):
        if best_mask[i]:
            active_weights.append(float(best_w[i]))
            active_indices.append(int(i))

    ds_map = {}
    for xi, ds_name in enumerate(["sbr", "phys", "morph", "atlas"]):
        for mi_idx, mname in enumerate(["lr", "et", "hgb", "rf", "ridge"]):
            stream_idx = xi * 5 + mi_idx
            ds_map[str(stream_idx)] = {"dataset": ds_name, "model": mname}

    cfg = {
        "n_seeds": len(seeds),
        "n_datasets": len(Xs),
        "dataset_names": ["sbr", "phys", "morph", "atlas"],
        "model_names": ["lr", "et", "hgb", "rf", "ridge"],
        "blend_weights": active_weights,
        "blend_indices": active_indices,
        "stream_map": ds_map,
        "temperature": float(best_temp),
        "clip_min": 0.005,
        "clip_max": 0.995,
        "mi_top": 44,
        "lr_c": 1.31,
        "ridge_alpha": 0.1,
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    with open(os.path.join(out_dir, "oof_metrics.json"), "w") as f:
        json.dump({
            "oof_auroc": round(float(final_auc), 4),
            "oof_ll": round(float(final_ll), 4),
            "active_streams": [names[i] for i in range(len(names)) if best_mask[i]],
        }, f, indent=2)

    shutil.copy("src/sbr_extractor.py", os.path.join(out_dir, "sbr_extractor.py"))
    shutil.copy("src/sbr_extractor_phys.py", os.path.join(out_dir, "sbr_extractor_phys.py"))
    print(f"\nSaved to {out_dir}")
