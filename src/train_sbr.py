import argparse
import json
import math
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbr_extractor import FEATURE_COLUMNS, extract_features

RAW_COLS = list(FEATURE_COLUMNS)
NO_LOG = {"brain_skew", "brain_kurt"}

MAIN_PY_TEMPLATE = '''import json
import os

import joblib
import numpy as np
import pandas as pd

from sbr_extractor import FEATURE_COLUMNS, extract_features

RAW_COLS = list(FEATURE_COLUMNS)
NO_LOG = {"brain_skew", "brain_kurt"}


def build_features(raw):
    log_idx = [i for i, c in enumerate(RAW_COLS) if c not in NO_LOG]
    return np.column_stack([raw] + [np.log1p(raw[:, log_idx])])


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
        feat = extract_features(path)
        raw = np.array([feat[c] for c in RAW_COLS], dtype=np.float64).reshape(1, -1)
        X = build_features(raw)
        p_all = []
        for i in range(cfg["n_folds"]):
            names = ["lr", "et", "hgb"]
            scalers = joblib.load(os.path.join(weights_dir, f"scaler_fold{i}.pkl"))
            Xs = scalers.transform(X)
            sel = joblib.load(os.path.join(weights_dir, f"mi_cols_fold{i}.pkl"))
            Xt = Xs[:, sel]
            scores = []
            for name in names:
                model = joblib.load(os.path.join(weights_dir, f"{name}_fold{i}.pkl"))
                if name == "lr":
                    scores.append(model.predict_proba(Xs)[:, 1])
                else:
                    scores.append(model.predict_proba(Xt)[:, 1])
            blend = np.zeros_like(scores[0])
            for w, s in zip(cfg["blend_weights"], scores):
                blend = blend + w * s
            p_all.append(blend[0])
        p = float(np.mean(p_all))
        logitp = logit(np.clip(p, 1e-6, 1 - 1e-6)) / cfg["temperature"]
        p = float(expit(logitp))
        p = min(max(p, cfg["clip_min"]), cfg["clip_max"])
        probas.append(p)
        if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
            print(f"[progress] {idx + 1}/{n_total}")

    sub = pd.DataFrame({"uid": order, "is_pathologic": probas})
    sub.to_csv("submission.csv", index=False)


def logit(x):
    return np.log(x / (1.0 - x))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    main()
'''


def build_features(raw_df):
    raw = raw_df[RAW_COLS].values
    log_idx = [i for i, c in enumerate(RAW_COLS) if c not in NO_LOG]
    X = np.column_stack([raw, np.log1p(raw[:, log_idx])])
    cols = RAW_COLS + [f"log1p_{RAW_COLS[i]}" for i in log_idx]
    return X, cols


def optimize_blend(oof_lr, oof_et, oof_hgb, y):
    matrix = np.column_stack([oof_lr, oof_et, oof_hgb])

    def loss(w):
        p = np.clip(matrix @ w, 1e-6, 1 - 1e-6)
        return log_loss(y, p)

    best = None
    for init in ([1 / 3, 1 / 3, 1 / 3], [0.6, 0.3, 0.1], [0.8, 0.2, 0.0]):
        res = minimize(loss, np.array(init), method="L-BFGS-B",
                       bounds=[(0, 1)] * 3, options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    w = best.x
    return w / w.sum(), best.fun


def optimize_temperature(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)

    def loss(log_t):
        p = expit(logit(oof) / math.exp(log_t[0]))
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

    res = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return math.exp(res.x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--use-cached-features", action="store_true")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--mi-selection", choices=["fold", "global"], default="fold")
    args = ap.parse_args()

    data_dir = args.data_dir
    labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    site = pd.read_csv(os.path.join(data_dir, "site_labels.csv"))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "weights"), exist_ok=True)

    cache_path = os.path.join(data_dir, "sbr_features_train.csv")
    if args.use_cached_features and os.path.exists(cache_path):
        feats = pd.read_csv(cache_path)
    else:
        nifti_candidates = [
            os.path.join(data_dir, "niftis"),
            os.path.join(data_dir, "DaT_Parkinsons_Challenge_-_niftis.zip"),
        ]
        nifti_dir = next((p for p in nifti_candidates if os.path.isdir(p)), None)
        if nifti_dir is None:
            raise SystemExit("no nifti directory found")
        rows = []
        for uid in labels["uid"]:
            path = os.path.join(nifti_dir, f"{uid}.nii.gz")
            row = extract_features(path)
            row["uid"] = uid
            rows.append(row)
        feats = pd.DataFrame(rows)
    feats = feats.rename(columns={"label": "label"})

    df = labels.merge(feats, on="uid", how="left").merge(site, on="uid", how="left")
    df = df.dropna(subset=["is_pathologic"])
    X_all, all_cols = build_features(df)
    y = df["is_pathologic"].values
    groups = df["pseudo_site"].astype(str).values

    oof = {name: np.zeros(len(df)) for name in ("lr", "et", "hgb")}
    saved = {"fold_models": [], "ocs": []}

    gkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    splits = list(gkf.split(X_all, y, groups))

    global_mi = None
    if getattr(args, "mi_selection", "fold") == "global":
        mi = mutual_info_classif(X_all, y, random_state=42)
        global_mi = np.argsort(mi)[::-1][:50]
        print("global MI selection:", global_mi[:8])

    for fold, (tr, va) in enumerate(splits):
        Xtr, Xva = X_all[tr], X_all[va]
        ytr = y[tr]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        lr = LogisticRegression(C=0.30, solver="lbfgs", max_iter=2000, random_state=42)
        lr.fit(Xtr_s, ytr)
        oof["lr"][va] = lr.predict_proba(Xva_s)[:, 1]

        mi = mutual_info_classif(Xtr, ytr, random_state=42)
        order_sel = np.argsort(mi)[::-1][:50]
        if global_mi is not None:
            order_sel = global_mi
        Xtr_t = Xtr_s[:, order_sel]
        Xva_t = Xva_s[:, order_sel]

        et = ExtraTreesClassifier(n_estimators=500, max_depth=10,
                                  max_features="sqrt", n_jobs=-1,
                                  random_state=42 + fold)
        et.fit(Xtr_t, ytr)
        oof["et"][va] = et.predict_proba(Xva_t)[:, 1]

        hgb = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.02,
            l2_regularization=1.0, min_samples_leaf=10, random_state=42 + fold)
        hgb.fit(Xtr_t, ytr)
        oof["hgb"][va] = hgb.predict_proba(Xva_t)[:, 1]

        joblib.dump(scaler, os.path.join(args.out_dir, f"weights/scaler_fold{fold}.pkl"))
        joblib.dump(order_sel, os.path.join(args.out_dir, f"weights/mi_cols_fold{fold}.pkl"))
        joblib.dump(lr, os.path.join(args.out_dir, f"weights/lr_fold{fold}.pkl"))
        joblib.dump(et, os.path.join(args.out_dir, f"weights/et_fold{fold}.pkl"))
        joblib.dump(hgb, os.path.join(args.out_dir, f"weights/hgb_fold{fold}.pkl"))

    blend_w, blend_loss = optimize_blend(oof["lr"], oof["et"], oof["hgb"], y)
    oof_blend = blend_w[0] * oof["lr"] + blend_w[1] * oof["et"] + blend_w[2] * oof["hgb"]
    temp = optimize_temperature(oof_blend, y)
    oof_cal = expit(logit(np.clip(oof_blend, 1e-6, 1 - 1e-6)) / temp)
    clip_min, clip_max = 0.005, 0.995
    oof_final = np.clip(oof_cal, clip_min, clip_max)

    for name in ("lr", "et", "hgb"):
        print(f"{name}: OOF AUROC={roc_auc_score(y, oof[name]):.4f}  "
              f"LL={log_loss(y, np.clip(oof[name], clip_min, clip_max)):.4f}")
    print(f"blend weights: {np.round(blend_w, 3)}  OOF AUROC="
          f"{roc_auc_score(y, oof_blend):.4f}")
    print(f"temperature: {temp:.4f}")
    print(f"FINAL OOF AUROC={roc_auc_score(y, oof_final):.4f}  "
          f"LL={log_loss(y, oof_final):.4f}")

    cfg = {
        "n_folds": args.n_folds,
        "raw_columns": RAW_COLS,
        "all_columns": all_cols,
        "n_features": len(all_cols),
        "blend_weights": [float(w) for w in blend_w],
        "temperature": float(temp),
        "clip_min": clip_min,
        "clip_max": clip_max,
    }
    with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    shutil.copy(os.path.join(os.path.dirname(__file__), "sbr_extractor.py"),
                os.path.join(args.out_dir, "sbr_extractor.py"))
    with open(os.path.join(args.out_dir, "main.py"), "w") as f:
        f.write(MAIN_PY_TEMPLATE)

    with open(os.path.join(args.out_dir, "oof_metrics.json"), "w") as f:
        json.dump({
            "oof_auroc_lr": roc_auc_score(y, oof["lr"]),
            "oof_auroc_et": roc_auc_score(y, oof["et"]),
            "oof_auroc_hgb": roc_auc_score(y, oof["hgb"]),
            "oof_auroc_blend": roc_auc_score(y, oof_blend),
            "oof_auroc_final": roc_auc_score(y, oof_final),
            "oof_ll_final": log_loss(y, oof_final),
            "blend_weights": [float(w) for w in blend_w],
            "temperature": float(temp),
        }, f, indent=2)


if __name__ == "__main__":
    main()