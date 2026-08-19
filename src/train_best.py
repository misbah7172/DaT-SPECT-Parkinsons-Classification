import argparse
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from sbr_extractor import FEATURE_COLUMNS as SBR_COLUMNS, extract_features
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS as PHYS_COLUMNS, extract_physical_features
from sbr_extractor_atlas import ATLAS_FEATURE_COLUMNS as ATLAS_COLUMNS, extract_atlas_features


VIEW_COLUMNS = {
    "sbr": list(SBR_COLUMNS),
    "phys": list(PHYS_COLUMNS),
    "atlas": list(ATLAS_COLUMNS),
}
MODEL_NAMES = ("lr", "et", "hgb", "rf")
DEFAULT_SEEDS = [42, 777, 2024, 12345, 999]
CLIP_MIN = 0.005
CLIP_MAX = 0.995


MAIN_TEMPLATE = r'''import json
import os
import zipfile

import joblib
import numpy as np
import pandas as pd

from sbr_extractor import FEATURE_COLUMNS as SBR_COLUMNS, extract_features
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS as PHYS_COLUMNS, extract_physical_features
from sbr_extractor_atlas import ATLAS_FEATURE_COLUMNS as ATLAS_COLUMNS, extract_atlas_features

VIEW_COLUMNS = {
    "sbr": list(SBR_COLUMNS),
    "phys": list(PHYS_COLUMNS),
    "atlas": list(ATLAS_COLUMNS),
}


def augment(x):
    return np.concatenate([x, np.log1p(np.clip(x, 0, None))], axis=1)


def feature_matrix(feat_dict, columns):
    values = np.array([feat_dict.get(col, np.nan) for col in columns], dtype=np.float64).reshape(1, -1)
    return np.nan_to_num(augment(values), nan=0.0, posinf=0.0, neginf=0.0)


def load_atlas_assets(data_dir):
    rois = np.load(os.path.join(data_dir, "atlas_rois.npy"), allow_pickle=True)
    template = np.load(os.path.join(data_dir, "atlas_template.npy"), allow_pickle=True)
    if isinstance(rois, np.ndarray) and rois.ndim == 4 and rois.shape[0] == 4:
        rois = {"lp": rois[0], "rp": rois[1], "lc": rois[2], "rc": rois[3]}
    elif isinstance(rois, np.ndarray) and rois.shape == ():
        rois = rois.item()
    return rois, template


def resolve_nifti_path(data_dir, uid):
    direct = os.path.join(data_dir, "niftis", f"{uid}.nii.gz")
    if os.path.exists(direct):
        return direct

    for root, _, files in os.walk(data_dir):
        candidate = os.path.join(root, f"{uid}.nii.gz")
        if os.path.exists(candidate):
            return candidate

    zip_path = os.path.join(data_dir, "DaT_Parkinsons_Challenge_-_niftis.zip")
    if os.path.exists(zip_path):
        cache_dir = os.path.join(data_dir, "_nifti_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached = os.path.join(cache_dir, f"{uid}.nii.gz")
        if os.path.exists(cached):
            return cached
        with zipfile.ZipFile(zip_path) as zf:
            member = next((name for name in zf.namelist() if name.endswith(f"{uid}.nii.gz")), None)
            if member is None:
                raise FileNotFoundError(uid)
            with zf.open(member) as src, open(cached, "wb") as dst:
                dst.write(src.read())
        return cached

    raise FileNotFoundError(uid)


def extract_view(data_dir, uid, view_name, atlas_assets):
    path = resolve_nifti_path(data_dir, uid)
    if view_name == "sbr":
        return extract_features(path)
    if view_name == "phys":
        return extract_physical_features(path)
    rois, template = atlas_assets
    return extract_atlas_features(path, rois, template)


def logit_clip(p):
    return np.log(p / (1.0 - p))


def expit_fn(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    data_dir = os.path.join(os.getcwd(), "data")
    fmt = pd.read_csv(os.path.join(data_dir, "submission_format.csv"))
    cfg = json.load(open("model_config.json", "r", encoding="utf-8"))
    atlas_assets = load_atlas_assets(data_dir)
    weights_dir = "weights"

    order = fmt["uid"].tolist()
    probas = []
    for idx, uid in enumerate(order, 1):
        stream_values = []
        for view_name in cfg["views"]:
            feat = extract_view(data_dir, uid, view_name, atlas_assets)
            x = feature_matrix(feat, VIEW_COLUMNS[view_name])
            for model_name in cfg["model_names"]:
                per_seed_preds = []
                for seed_idx, _seed in enumerate(cfg["seeds"]):
                    scaler = joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_scaler.pkl"))
                    sel = joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_sel.pkl"))
                    xs = scaler.transform(x)
                    xt = xs[:, sel]
                    if model_name == "lr":
                        per_seed_preds.append(joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_lr.pkl")).predict_proba(xs)[:, 1][0])
                    elif model_name == "et":
                        per_seed_preds.append(joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_et.pkl")).predict_proba(xt)[:, 1][0])
                    elif model_name == "hgb":
                        per_seed_preds.append(joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_hgb.pkl")).predict_proba(xt)[:, 1][0])
                    else:
                        per_seed_preds.append(joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_rf.pkl")).predict_proba(xt)[:, 1][0])
                stream_values.append(float(np.mean(per_seed_preds)))

        blend = float(np.dot(cfg["blend_weights"], np.array(stream_values, dtype=np.float64)))
        p = expit_fn(logit_clip(np.clip(blend, 1e-6, 1 - 1e-6)) / cfg["temperature"])
        p = float(np.clip(p, cfg["clip_min"], cfg["clip_max"]))
        probas.append(p)
        if idx % 10 == 0 or idx == len(order):
            print(f"[progress] {idx}/{len(order)}")

    pd.DataFrame({"uid": order, "is_pathologic": probas}).to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
'''


def augment(x):
    return np.concatenate([x, np.log1p(np.clip(x, 0, None))], axis=1)


def build_matrix(df, columns):
    arr = df[columns].to_numpy(dtype=np.float64)
    return np.nan_to_num(augment(arr), nan=0.0, posinf=0.0, neginf=0.0)


def log_loss_clipped(y_true, y_pred):
    return log_loss(y_true, np.clip(y_pred, CLIP_MIN, CLIP_MAX))


def optimize_blend(matrix, y):
    def objective(weights):
        p = np.clip(matrix @ weights, 1e-6, 1 - 1e-6)
        return log_loss(y, p)

    starts = [np.ones(matrix.shape[1]) / matrix.shape[1]]
    if matrix.shape[1] >= 3:
        starts.append(
            np.array([0.6] + [0.4 / (matrix.shape[1] - 1)] * (matrix.shape[1] - 1)))
        starts.append(
            np.array([0.8] + [0.2 / (matrix.shape[1] - 1)] * (matrix.shape[1] - 1)))

    best = None
    for start in starts:
        res = minimize(objective, start, method="L-BFGS-B",
                       bounds=[(0, 1)] * matrix.shape[1], options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    weights = best.x
    return weights / weights.sum()


def optimize_temperature(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)

    def objective(log_t):
        temperature = np.exp(log_t[0])
        p = expit(logit(oof) / temperature)
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

    res = minimize(objective, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return float(np.exp(res.x[0]))


def fit_fold_models(X, y, train_idx, valid_idx, seed, fold, top_k):
    Xtr, Xva = X[train_idx], X[valid_idx]
    ytr = y[train_idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)

    lr = LogisticRegression(C=0.3, solver="lbfgs",
                            max_iter=3000, random_state=seed)
    lr.fit(Xtr_s, ytr)

    mi = mutual_info_classif(Xtr, ytr, random_state=seed)
    sel = np.argsort(mi)[::-1][: min(top_k, Xtr.shape[1])]
    Xtr_t = Xtr_s[:, sel]
    Xva_t = Xva_s[:, sel]

    et = ExtraTreesClassifier(n_estimators=600, max_depth=9,
                              max_features="sqrt", n_jobs=-1, random_state=seed + fold)
    et.fit(Xtr_t, ytr)

    hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=4, learning_rate=0.03,
                                         l2_regularization=1.0, min_samples_leaf=10, random_state=seed + fold)
    hgb.fit(Xtr_t, ytr)

    rf = RandomForestClassifier(n_estimators=400, max_depth=12,
                                min_samples_leaf=2, n_jobs=-1, random_state=seed + fold)
    rf.fit(Xtr_t, ytr)

    return {
        "lr_pred": lr.predict_proba(Xva_s)[:, 1],
        "et_pred": et.predict_proba(Xva_t)[:, 1],
        "hgb_pred": hgb.predict_proba(Xva_t)[:, 1],
        "rf_pred": rf.predict_proba(Xva_t)[:, 1],
    }


def fit_full_models(X, y, seed, fold, top_k):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    lr = LogisticRegression(C=0.3, solver="lbfgs",
                            max_iter=3000, random_state=seed)
    lr.fit(Xs, y)

    mi = mutual_info_classif(X, y, random_state=seed)
    sel = np.argsort(mi)[::-1][: min(top_k, X.shape[1])]
    Xt = Xs[:, sel]

    et = ExtraTreesClassifier(n_estimators=600, max_depth=9,
                              max_features="sqrt", n_jobs=-1, random_state=seed + fold)
    et.fit(Xt, y)

    hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=4, learning_rate=0.03,
                                         l2_regularization=1.0, min_samples_leaf=10, random_state=seed + fold)
    hgb.fit(Xt, y)

    rf = RandomForestClassifier(n_estimators=400, max_depth=12,
                                min_samples_leaf=2, n_jobs=-1, random_state=seed + fold)
    rf.fit(Xt, y)

    return scaler, sel, lr, et, hgb, rf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Dataset")
    parser.add_argument("--out-dir", default="artifacts_best")
    parser.add_argument("--views", default="sbr,phys,atlas")
    parser.add_argument("--seeds", default=",".join(str(s)
                        for s in DEFAULT_SEEDS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--tree-top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional row limit for smoke testing")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = Path(args.out_dir)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    selected_views = [v.strip() for v in args.views.split(",") if v.strip()]
    invalid = [v for v in selected_views if v not in VIEW_COLUMNS]
    if invalid:
        raise ValueError(f"Unknown views: {invalid}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    site = pd.read_csv(os.path.join(data_dir, "site_labels.csv"))
    base = labels.merge(site, on="uid", how="inner")
    if args.limit and args.limit > 0:
        base = base.iloc[: args.limit].copy()
    base = base.dropna(
        subset=["is_pathologic", "pseudo_site"]).reset_index(drop=True)

    feature_frames = {
        "sbr": pd.read_csv(os.path.join(data_dir, "sbr_features_train.csv")),
        "phys": pd.read_csv(os.path.join(data_dir, "phys_features_train.csv")),
        "atlas": pd.read_csv(os.path.join(data_dir, "atlas_features_train.csv")),
    }

    X_views = {}
    for view_name in selected_views:
        view_df = base[["uid"]].merge(
            feature_frames[view_name], on="uid", how="left")
        X_views[view_name] = build_matrix(view_df, VIEW_COLUMNS[view_name])

    y = base["is_pathologic"].astype(int).to_numpy()
    groups = base["pseudo_site"].astype(str).to_numpy()

    print("rows=", len(base), "seeds=", seeds, "views=", selected_views)
    for view_name, X in X_views.items():
        print(view_name, X.shape)

    splits = list(StratifiedGroupKFold(n_splits=args.n_folds,
                  shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups))

    oof_streams = {}
    for view_name in selected_views:
        for model_name in MODEL_NAMES:
            for seed_idx in range(len(seeds)):
                oof_streams[f"{view_name}_{model_name}_s{seed_idx}"] = np.zeros(
                    len(y), dtype=np.float64)

    for seed_idx, seed in enumerate(seeds):
        print("[seed]", seed)
        for fold, (tr, va) in enumerate(splits):
            for view_name, X in X_views.items():
                out = fit_fold_models(X, y, tr, va, seed,
                                      fold, args.tree_top_k)
                oof_streams[f"{view_name}_lr_s{seed_idx}"][va] = out["lr_pred"]
                oof_streams[f"{view_name}_et_s{seed_idx}"][va] = out["et_pred"]
                oof_streams[f"{view_name}_hgb_s{seed_idx}"][va] = out["hgb_pred"]
                oof_streams[f"{view_name}_rf_s{seed_idx}"][va] = out["rf_pred"]
        print("[seed]", seed, "OOF complete")

    stream_names = []
    stream_values = []
    stream_scores = {}
    for view_name in selected_views:
        for model_name in MODEL_NAMES:
            per_seed = [oof_streams[f"{view_name}_{model_name}_s{seed_idx}"]
                        for seed_idx in range(len(seeds))]
            stream = np.mean(np.column_stack(per_seed), axis=1)
            stream_names.append(f"{view_name}_{model_name}")
            stream_values.append(stream)
            stream_scores[f"{view_name}_{model_name}"] = {
                "auroc": float(roc_auc_score(y, stream)),
                "log_loss": float(log_loss_clipped(y, stream)),
            }

    M = np.column_stack(stream_values)
    blend_weights = optimize_blend(M, y)
    blend = M @ blend_weights
    temperature = optimize_temperature(blend, y)
    final = expit(logit(np.clip(blend, 1e-6, 1 - 1e-6)) / temperature)
    final = np.clip(final, CLIP_MIN, CLIP_MAX)

    print("blend_weights=", np.round(blend_weights, 4).tolist())
    print("temperature=", round(temperature, 4))
    print("FINAL AUROC=", round(float(roc_auc_score(y, final)), 5))
    print("FINAL LL=", round(float(log_loss_clipped(y, final)), 5))

    for seed_idx, seed in enumerate(seeds):
        for view_name, X in X_views.items():
            scaler, sel, lr, et, hgb, rf = fit_full_models(
                X, y, seed, 0, args.tree_top_k)
            prefix = f"s{seed_idx}_{view_name}"
            joblib.dump(scaler, weights_dir / f"{prefix}_scaler.pkl")
            joblib.dump(sel, weights_dir / f"{prefix}_sel.pkl")
            joblib.dump(lr, weights_dir / f"{prefix}_lr.pkl")
            joblib.dump(et, weights_dir / f"{prefix}_et.pkl")
            joblib.dump(hgb, weights_dir / f"{prefix}_hgb.pkl")
            joblib.dump(rf, weights_dir / f"{prefix}_rf.pkl")

    cfg = {
        "views": selected_views,
        "seeds": seeds,
        "model_names": list(MODEL_NAMES),
        "blend_weights": [float(w) for w in blend_weights],
        "temperature": float(temperature),
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "tree_top_k": int(args.tree_top_k),
    }
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    with open(out_dir / "oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "oof_auroc_final": float(roc_auc_score(y, final)),
                "oof_log_loss_final": float(log_loss_clipped(y, final)),
                "blend_weights": [float(w) for w in blend_weights],
                "temperature": float(temperature),
                "streams": stream_scores,
            },
            f,
            indent=2,
        )

    shutil.copy(os.path.join(os.path.dirname(__file__),
                "sbr_extractor.py"), out_dir / "sbr_extractor.py")
    shutil.copy(os.path.join(os.path.dirname(__file__),
                "sbr_extractor_phys.py"), out_dir / "sbr_extractor_phys.py")
    shutil.copy(os.path.join(os.path.dirname(__file__),
                "sbr_extractor_atlas.py"), out_dir / "sbr_extractor_atlas.py")
    with open(out_dir / "main.py", "w", encoding="utf-8") as f:
        f.write(MAIN_TEMPLATE)

    print("saved", out_dir)


if __name__ == "__main__":
    main()
