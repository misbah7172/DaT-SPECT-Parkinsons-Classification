import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbr_extractor import FEATURE_COLUMNS
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS

RAW1 = list(FEATURE_COLUMNS)
RAW2 = list(PHYS_FEATURE_COLUMNS)
NO_LOG = {"brain_skew", "brain_kurt"}

MODEL_NAMES = ("lr", "et", "hgb")


DERIVED_COLS = [
    "caud_preserve", "post_asym", "ant_asym", "ap_grad",
    "pc_p95", "pc_p97", "pc_p99",
    "focus_lp", "focus_rp", "focus_lc", "focus_rc",
    "vol_asym_95", "vol_asym_97", "vol_asym_99",
    "vol_pc_95", "vol_pc_97", "vol_pc_99", "put_focus",
]


def derived_frame(df: pd.DataFrame) -> pd.DataFrame:
    g = df
    one = lambda c: g[c] if c in g.columns else np.nan
    d = pd.DataFrame(index=g.index)
    d["caud_preserve"] = (one("sbr_left_caudate") + one("sbr_right_caudate")) / (
        one("sbr_left_putamen") + one("sbr_right_putamen") + 1e-6
    )
    d["post_asym"] = ((one("sbr_left_putamen_post") - one("sbr_right_putamen_post")).abs()) / (
        one("sbr_left_putamen_post") + one("sbr_right_putamen_post") + 1e-6
    )
    d["ant_asym"] = ((one("sbr_left_putamen_ant") - one("sbr_right_putamen_ant")).abs()) / (
        one("sbr_left_putamen_ant") + one("sbr_right_putamen_ant") + 1e-6
    )
    d["ap_grad"] = one("left_ap_ratio") + one("right_ap_ratio")
    for q in (95, 97, 99):
        d[f"pc_p{q}"] = (one(f"sbr_lp_p{q}") + one(f"sbr_rp_p{q}")) / (
            one(f"sbr_lc_p{q}") + one(f"sbr_rc_p{q}") + 1e-6
        )
    for side in ("lp", "rp", "lc", "rc"):
        d[f"focus_{side}"] = one(f"sbr_{side}_p99") / (one(f"sbr_{side}_p95") + 1e-6)
    for q in (95, 97, 99):
        d[f"vol_asym_{q}"] = ((one(f"vol_lp_p{q}") - one(f"vol_rp_p{q}")).abs()) / (
            one(f"vol_lp_p{q}") + one(f"vol_rp_p{q}") + 1e-6
        )
        d[f"vol_pc_{q}"] = (one(f"vol_lc_p{q}") + one(f"vol_rc_p{q}")) / (
            one(f"vol_lp_p{q}") + one(f"vol_rp_p{q}") + 1e-6
        )
    d["put_focus"] = one("min_putamen_sbr") / (one("sbr_left_putamen") + one("sbr_right_putamen") + 1e-6)
    return d[DERIVED_COLS]


def build_matrix(raw: pd.DataFrame, raw_cols: list) -> np.ndarray:
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    d = derived_frame(raw).values.astype(np.float64)
    return np.nan_to_num(
        np.concatenate([a, np.log1p(np.clip(a, 0, None)), d], axis=1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


MAIN_TEMPLATE = '''import json
import os

import joblib
import numpy as np
import pandas as pd

from sbr_extractor import FEATURE_COLUMNS, extract_features
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS, extract_physical_features

RAW1 = list(FEATURE_COLUMNS)
RAW2 = list(PHYS_FEATURE_COLUMNS)
NO_LOG = {"brain_skew", "brain_kurt"}

DERIVED_COLS = [
    "caud_preserve", "post_asym", "ant_asym", "ap_grad",
    "pc_p95", "pc_p97", "pc_p99",
    "focus_lp", "focus_rp", "focus_lc", "focus_rc",
    "vol_asym_95", "vol_asym_97", "vol_asym_99",
    "vol_pc_95", "vol_pc_97", "vol_pc_99", "put_focus",
]


def derived_row(d):
    one = lambda c: d.get(c, np.nan)
    out = {}
    out["caud_preserve"] = (one("sbr_left_caudate") + one("sbr_right_caudate")) / (
        one("sbr_left_putamen") + one("sbr_right_putamen") + 1e-6
    )
    out["post_asym"] = abs(one("sbr_left_putamen_post") - one("sbr_right_putamen_post")) / (
        one("sbr_left_putamen_post") + one("sbr_right_putamen_post") + 1e-6
    )
    out["ant_asym"] = abs(one("sbr_left_putamen_ant") - one("sbr_right_putamen_ant")) / (
        one("sbr_left_putamen_ant") + one("sbr_right_putamen_ant") + 1e-6
    )
    out["ap_grad"] = one("left_ap_ratio") + one("right_ap_ratio")
    for q in (95, 97, 99):
        out[f"pc_p{q}"] = (one(f"sbr_lp_p{q}") + one(f"sbr_rp_p{q}")) / (
            one(f"sbr_lc_p{q}") + one(f"sbr_rc_p{q}") + 1e-6
        )
    for side in ("lp", "rp", "lc", "rc"):
        out[f"focus_{side}"] = one(f"sbr_{side}_p99") / (one(f"sbr_{side}_p95") + 1e-6)
    for q in (95, 97, 99):
        out[f"vol_asym_{q}"] = abs(one(f"vol_lp_p{q}") - one(f"vol_rp_p{q}")) / (
            one(f"vol_lp_p{q}") + one(f"vol_rp_p{q}") + 1e-6
        )
        out[f"vol_pc_{q}"] = (one(f"vol_lc_p{q}") + one(f"vol_rc_p{q}")) / (
            one(f"vol_lp_p{q}") + one(f"vol_rp_p{q}") + 1e-6
        )
    out["put_focus"] = one("min_putamen_sbr") / (one("sbr_left_putamen") + one("sbr_right_putamen") + 1e-6)
    return np.array([out[c] for c in DERIVED_COLS], dtype=np.float64).reshape(1, -1)


def build_matrix(raw, raw_cols):
    a = np.array([raw.get(c, np.nan) for c in raw_cols], dtype=np.float64).reshape(1, -1)
    d = derived_row(raw)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None)), d], axis=1),
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
        f1 = extract_features(path)
        f2 = extract_physical_features(path)
        X1 = build_matrix(f1, RAW1)
        X2 = build_matrix(f2, RAW2)
        p_all = []
        for i in range(cfg["n_folds"]):
            fold_scores = []
            for si, Xs_name in enumerate(("X1", "X2")):
                X = X1 if Xs_name == "X1" else X2
                scaler = joblib.load(os.path.join(weights_dir, f"scaler_{Xs_name}_fold{i}.pkl"))
                mi_cols = joblib.load(os.path.join(weights_dir, f"mi_{Xs_name}_fold{i}.pkl"))
                Xst = scaler.transform(X)
                Xt = Xst[:, mi_cols]
                for name in cfg["model_names"]:
                    model = joblib.load(os.path.join(weights_dir, f"{Xs_name}_{name}_fold{i}.pkl"))
                    inp = Xst if name == "lr" else Xt
                    fold_scores.append(model.predict_proba(inp)[:, 1])
            blend = sum(w * s for w, s in zip(cfg["blend_weights"], fold_scores))
            p_all.append(blend[0])
        p = float(np.mean(p_all))
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


def optimize_blend(matrices, y):
    M = np.column_stack(matrices)

    def loss(w):
        return log_loss(y, np.clip(M @ w, 1e-6, 1 - 1e-6))

    best = None
    for init in (np.ones(M.shape[1]) / M.shape[1],):
        r = minimize(loss, init, method="L-BFGS-B", bounds=[(0, 1)] * M.shape[1])
        if best is None or r.fun < best.fun:
            best = r
    w = best.x / best.x.sum()
    return w, M @ w


def optimize_temperature(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)

    def loss(log_t):
        p = expit(logit(oof) / np.exp(log_t[0]))
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

    r = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return float(np.exp(r.x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    data_dir = args.data_dir
    labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    site = pd.read_csv(os.path.join(data_dir, "site_labels.csv"))
    sbr1 = pd.read_csv(os.path.join(data_dir, "sbr_features_train.csv"))
    sbr2 = pd.read_csv(os.path.join(data_dir, "phys_features_train.csv"))

    merged = labels.merge(site, on="uid").merge(sbr1, on="uid")
    sbr2_r = sbr2.rename(columns={c: f"p_{c}" for c in sbr2.columns if c != "uid"})
    merged = merged.merge(sbr2_r, on="uid").dropna()
    print("merged rows:", len(merged))

    X1 = build_matrix(merged, RAW1)
    p_cols = [f"p_{c}" for c in RAW2 if f"p_{c}" in merged.columns]
    p_df = merged[p_cols].rename(columns={c: c[2:] for c in p_cols})
    X2 = build_matrix(p_df, RAW2)

    y = merged["is_pathologic"].values
    groups = merged["pseudo_site"].astype(str).values
    print("X1", X1.shape, "X2", X2.shape, "pos rate", round(y.mean(), 3))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "weights"), exist_ok=True)

    oof = {}
    for Xname, X in (("X1", X1), ("X2", X2)):
        for m in MODEL_NAMES:
            oof[f"{Xname}_{m}"] = np.zeros(len(merged))
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(args.n_folds, shuffle=True, random_state=42).split(X, y, groups)):
            Xtr, Xva, ytr = X[tr], X[va], y[tr]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
            lr = LogisticRegression(C=0.3, max_iter=3000, random_state=42).fit(Xtr_s, ytr)
            oof[f"{Xname}_lr"][va] = lr.predict_proba(Xva_s)[:, 1]
            mi = mutual_info_classif(Xtr, ytr, random_state=42)
            sel = np.argsort(mi)[::-1][:80]
            Xtr_t, Xva_t = Xtr_s[:, sel], Xva_s[:, sel]
            et = ExtraTreesClassifier(500, max_depth=12, n_jobs=-1, random_state=42 + fold).fit(Xtr_t, ytr)
            oof[f"{Xname}_et"][va] = et.predict_proba(Xva_t)[:, 1]
            hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.02,
                                                 l2_regularization=1.0, min_samples_leaf=10,
                                                 random_state=42 + fold).fit(Xtr_t, ytr)
            oof[f"{Xname}_hgb"][va] = hgb.predict_proba(Xva_t)[:, 1]
            joblib.dump(scaler, os.path.join(args.out_dir, f"weights/scaler_{Xname}_fold{fold}.pkl"))
            joblib.dump(sel, os.path.join(args.out_dir, f"weights/mi_{Xname}_fold{fold}.pkl"))
            joblib.dump(lr, os.path.join(args.out_dir, f"weights/{Xname}_lr_fold{fold}.pkl"))
            joblib.dump(et, os.path.join(args.out_dir, f"weights/{Xname}_et_fold{fold}.pkl"))
            joblib.dump(hgb, os.path.join(args.out_dir, f"weights/{Xname}_hgb_fold{fold}.pkl"))
        print(f"{Xname} done")

    names = list(oof.keys())
    print("\nper-model:", {m: round(roc_auc_score(y, oof[m]), 4) for m in names})

    w, blend = optimize_blend([oof[m] for m in names], y)
    temp = optimize_temperature(blend, y)
    final = np.clip(expit(logit(np.clip(blend, 1e-6, 1 - 1e-6)) / temp), 0.005, 0.995)

    print("\nblend weights", dict(zip(names, np.round(w, 3))))
    print("AUC", round(roc_auc_score(y, blend), 4), "LL", round(log_loss(y, np.clip(blend, 0.005, 0.995)), 4))
    print("temperature", round(temp, 4))
    print("FINAL AUC", round(roc_auc_score(y, final), 4), "LL", round(log_loss(y, final), 4))

    keep = names
    cfg = {
        "n_folds": args.n_folds,
        "model_names": MODEL_NAMES,
        "blend_weights": [float(wi) for wi in w],
        "blend_order": keep,
        "temperature": temp,
        "clip_min": 0.005,
        "clip_max": 0.995,
    }
    with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out_dir, "main.py"), "w") as f:
        f.write(MAIN_TEMPLATE)
    shutil.copy(os.path.join(os.path.dirname(__file__), "sbr_extractor.py"), os.path.join(args.out_dir, "sbr_extractor.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "sbr_extractor_phys.py"), os.path.join(args.out_dir, "sbr_extractor_phys.py"))

    with open(os.path.join(args.out_dir, "oof_metrics.json"), "w") as f:
        json.dump({"oof_auroc": round(float(roc_auc_score(y, final)), 4),
                   "oof_ll": round(float(log_loss(y, final)), 4),
                   "blend_weights": [float(x) for x in w],
                   "temperature": temp}, f, indent=2)
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()