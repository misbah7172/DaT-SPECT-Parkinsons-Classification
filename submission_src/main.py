import json
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
