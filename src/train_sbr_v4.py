import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from sbr_extractor import FEATURE_COLUMNS
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS

MODEL_NAMES = ("lr", "et", "hgb")
RAW1 = list(FEATURE_COLUMNS)
RAW2 = list(PHYS_FEATURE_COLUMNS)
SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
PVE_REF = 15.625

MAIN_TEMPLATE = '''import json
import os

import joblib
import numpy as np
import pandas as pd

from sbr_extractor import FEATURE_COLUMNS, extract_features
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS, extract_physical_features

RAW1 = list(FEATURE_COLUMNS)
RAW2 = list(PHYS_FEATURE_COLUMNS)
PVE_REF = 15.625


def _voxel_vol(path):
    import nibabel as nib
    z = nib.load(str(path)).header.get_zooms()[:3]
    return float(z[0] * z[1] * z[2])


def _pve_factor(voxel_vol):
    return (PVE_REF / voxel_vol) ** (1.0 / 3.0)


def build_matrix(raw, raw_cols):
    a = np.array([raw.get(c, np.nan) for c in raw_cols], dtype=np.float64).reshape(1, -1)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def main():
    data_dir = os.path.join(os.getcwd(), "data")
    nifti_dir = os.path.join(data_dir, "niftis")
    fmt = pd.read_csv(os.path.join(data_dir, "submission_format.csv"))
    cfg = json.load(open("model_config.json"))
    weights_dir = "weights"

    order = fmt["uid"].tolist()
    n_total = len(order)
    probas = []
    for idx, uid in enumerate(order):
        path = os.path.join(nifti_dir, f"{uid}.nii.gz")
        fac = _pve_factor(_voxel_vol(path))
        f1 = extract_features(path)
        f2 = extract_physical_features(path)
        for k in list(f2.keys()):
            if k.startswith("sbr_"):
                f2[k] = f2[k] * fac
        f2["voxel_vol"] = float(PVE_REF / fac ** 3)
        f2["pve_factor"] = fac
        X1 = build_matrix(f1, RAW1)
        X2 = build_matrix(f2, RAW2)
        matrices = (X1, X2)
        p_all = []
        for seed in range(cfg["n_seeds"]):
            fold_scores = []
            for xi in range(2):
                X = matrices[xi]
                for mk in ("lr", "et", "hgb"):
                    model = joblib.load(os.path.join(weights_dir, f"s{seed}_x{xi}_{mk}.pkl"))
                    scaler = joblib.load(os.path.join(weights_dir, f"s{seed}_x{xi}_scaler.pkl"))
                    if mk == "lr":
                        inp = scaler.transform(X)
                    else:
                        mi_cols = joblib.load(os.path.join(weights_dir, f"s{seed}_x{xi}_mi.pkl"))
                        inp = scaler.transform(X)[:, mi_cols]
                    fold_scores.append(model.predict_proba(inp)[:, 1])
            s = np.zeros_like(fold_scores[0])
            for w, sc_ in zip(cfg["blend_weights"], fold_scores):
                s = s + w * sc_
            p_all.append(s)
        p = float(np.mean(np.concatenate(p_all), axis=0))
        p = min(max(expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / cfg["temperature"]), cfg["clip_min"]), cfg["clip_max"])
        probas.append(p)
        if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
            print(f"[progress] {idx + 1}/{n_total}")

    pd.DataFrame({"uid": order, "is_pathologic": probas}).to_csv("submission.csv", index=False)
    print("submission written")


def logit(x):
    return np.log(x / (1.0 - x))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    main()
'''


def build_matrix(raw: pd.DataFrame, raw_cols: list) -> np.ndarray:
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1), nan=0.0)


def optim_blend(matrices, y):
    M = np.column_stack(matrices) if not isinstance(matrices, np.ndarray) else matrices

    def loss(w):
        return log_loss(y, np.clip(M @ w, 1e-6, 1 - 1e-6))

    r = minimize(loss, np.ones(M.shape[1]) / M.shape[1], method="L-BFGS-B", bounds=[(0, 1)] * M.shape[1])
    return r.x / r.x.sum()


def optimize_temperature(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)

    def loss(log_t):
        p = expit(logit(oof) / np.exp(log_t[0]))
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

    r = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return float(np.exp(r.x[0]))


def main():
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SEEDS
    labels = pd.read_csv("Dataset/train_labels.csv")
    site = pd.read_csv("Dataset/site_labels.csv")
    geom = pd.read_csv("Dataset/voxel_geometry.csv")
    sbr1 = pd.read_csv("Dataset/sbr_features_train.csv")
    sbr2 = pd.read_csv("Dataset/phys_features_train.csv")
    merged = labels.merge(site, on="uid").merge(geom, on="uid").merge(sbr1, on="uid")
    sbr2_r = sbr2.rename(columns={c: f"p_{c}" for c in sbr2.columns if c != "uid"})
    merged = merged.merge(sbr2_r, on="uid").dropna()
    y = merged["is_pathologic"].values
    groups = merged["pseudo_site"].astype(str).values

    # PVE correction on phys SBR features
    fac = merged["pve_factor"].values if "pve_factor" in merged.columns else (PVE_REF / merged["voxel_vol"].values) ** (1.0 / 3.0)
    if "--inv" in sys.argv:
        fac = 1.0 / fac
    if "--raw" in sys.argv:
        fac = np.ones_like(fac)
    add_geom = "--no-geom" not in sys.argv
    p_cols = [f"p_{c}" for c in RAW2 if f"p_{c}" in merged.columns]
    for c in p_cols:
        if c[2:].startswith("sbr_"):
            merged[c] = merged[c] * fac
    merged["p_voxel_vol"] = merged["voxel_vol"]
    merged["p_pve_factor"] = fac
    X1 = build_matrix(merged, RAW1)
    geom_cols = ["p_voxel_vol", "p_pve_factor"] if add_geom else []
    X2 = build_matrix(merged[p_cols + geom_cols].rename(
        columns={c: c[2:] for c in p_cols + geom_cols}), RAW2 + (["voxel_vol", "pve_factor"] if add_geom else []))
    Xs = (X1, X2)
    print("X1", X1.shape, "X2", X2.shape, "n", len(y), "seeds", seeds)

    out_dir = "submission_v4"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)

    oof_seed = {}
    for si, seed in enumerate(seeds):
        for xi in range(2):
            X = Xs[xi]
            for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                Xtr_s, Xva_s = sc.transform(X[tr]), sc.transform(X[va])
                lr = LogisticRegression(C=0.3, max_iter=3000, random_state=seed).fit(Xtr_s, y[tr])
                oof_seed.setdefault((si, xi, "lr"), np.zeros(len(y)))[va] = lr.predict_proba(Xva_s)[:, 1]
                mi = mutual_info_classif(X[tr], y[tr], random_state=seed)
                sel = np.argsort(mi)[::-1][:80]
                et = ExtraTreesClassifier(500, max_depth=12, n_jobs=-1, random_state=seed).fit(Xtr_s[:, sel], y[tr])
                oof_seed.setdefault((si, xi, "et"), np.zeros(len(y)))[va] = et.predict_proba(Xva_s[:, sel])[:, 1]
                hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.02,
                                                     l2_regularization=1.0, min_samples_leaf=10,
                                                     random_state=seed).fit(Xtr_s[:, sel], y[tr])
                oof_seed.setdefault((si, xi, "hgb"), np.zeros(len(y)))[va] = hgb.predict_proba(Xva_s[:, sel])[:, 1]
        print(f"seed {seed} OOF done")

    names = [f"{'v' if xi==0 else 'p'}_{m}" for xi in (0, 1) for m in MODEL_NAMES]
    streams = {}
    for xi in (0, 1):
        for m in MODEL_NAMES:
            streams[f"{'v' if xi==0 else 'p'}_{m}"] = np.mean(
                [oof_seed[(si, xi, m)] for si in range(len(seeds))], axis=0)

    M = np.column_stack([streams[n] for n in names])
    w = optim_blend(M, y)
    blend = M @ w
    temp = optimize_temperature(blend, y)
    final = np.clip(expit(logit(np.clip(blend, 1e-6, 1 - 1e-6)) / temp), 0.005, 0.995)

    print("per-model (seed-avg):", dict(zip(names, [round(roc_auc_score(y, streams[n]), 4) for n in names])))
    print("blend w", np.round(w, 3), "AUC", round(roc_auc_score(y, blend), 4),
          "LL", round(log_loss(y, np.clip(blend, 0.005, 0.995)), 4))
    print("FINAL AUC", round(roc_auc_score(y, final), 4), "LL", round(log_loss(y, final), 4))

    for si, seed in enumerate(seeds):
        for xi in range(2):
            X = Xs[xi]
            sc = StandardScaler().fit(X)
            Xs_all = sc.transform(X)
            mi = mutual_info_classif(X, y, random_state=seed)
            sel = np.argsort(mi)[::-1][:80]
            lr = LogisticRegression(C=0.3, max_iter=3000, random_state=seed).fit(Xs_all, y)
            et = ExtraTreesClassifier(500, max_depth=12, n_jobs=-1, random_state=seed).fit(Xs_all[:, sel], y)
            hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.02,
                                                 l2_regularization=1.0, min_samples_leaf=10,
                                                 random_state=seed).fit(Xs_all[:, sel], y)
            joblib.dump(sc, os.path.join(out_dir, f"weights/s{si}_x{xi}_scaler.pkl"))
            joblib.dump(sel, os.path.join(out_dir, f"weights/s{si}_x{xi}_mi.pkl"))
            joblib.dump(lr, os.path.join(out_dir, f"weights/s{si}_x{xi}_lr.pkl"))
            joblib.dump(et, os.path.join(out_dir, f"weights/s{si}_x{xi}_et.pkl"))
            joblib.dump(hgb, os.path.join(out_dir, f"weights/s{si}_x{xi}_hgb.pkl"))
        print(f"seed {seed} final models done")

    cfg = {
        "n_seeds": len(seeds),
        "model_names": list(MODEL_NAMES),
        "blend_weights": [float(x) for x in w],
        "temperature": temp,
        "clip_min": 0.005,
        "clip_max": 0.995,
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(MAIN_TEMPLATE)
    shutil.copy("src/sbr_extractor.py", os.path.join(out_dir, "sbr_extractor.py"))
    shutil.copy("src/sbr_extractor_phys.py", os.path.join(out_dir, "sbr_extractor_phys.py"))
    with open(os.path.join(out_dir, "oof_metrics.json"), "w") as f:
        json.dump({"oof_auroc": round(float(roc_auc_score(y, final)), 4),
                   "oof_ll": round(float(log_loss(y, final)), 4)}, f, indent=2)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
