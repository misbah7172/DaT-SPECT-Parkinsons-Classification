import warnings

warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from sbr_extractor import FEATURE_COLUMNS
from sbr_extractor_atlas import ATLAS_FEATURE_COLUMNS
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS

MODEL_NAMES = ("lr", "et", "hgb")

RAW1 = list(FEATURE_COLUMNS)
RAW2 = list(PHYS_FEATURE_COLUMNS)


def build_plain(df, cols):
    a = df[[c for c in cols if c in df.columns]].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1), nan=0.0)


def build_derived(df, cols):
    from train_sbr_v2 import derived_frame
    a = df[[c for c in cols if c in df.columns]].values.astype(np.float64)
    d = derived_frame(df)[["caud_preserve", "post_asym", "ant_asym", "ap_grad", "pc_p95", "pc_p97", "pc_p99",
                           "put_focus"]].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None)), d], axis=1), nan=0.0)


def oof_run(X, y, groups):
    oof = {m: np.zeros(len(y)) for m in MODEL_NAMES}
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr_s, Xva_s = sc.transform(X[tr]), sc.transform(X[va])
        lr = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr_s, y[tr])
        oof["lr"][va] = lr.predict_proba(Xva_s)[:, 1]
        mi = mutual_info_classif(X[tr], y[tr], random_state=42)
        sel = np.argsort(mi)[::-1][:80]
        Xtr_t, Xva_t = Xtr_s[:, sel], Xva_s[:, sel]
        et = ExtraTreesClassifier(500, max_depth=12, n_jobs=-1, random_state=42).fit(Xtr_t, y[tr])
        oof["et"][va] = et.predict_proba(Xva_t)[:, 1]
        hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.02).fit(Xtr_t, y[tr])
        oof["hgb"][va] = hgb.predict_proba(Xva_t)[:, 1]
    return oof


def blend(M, y):
    loss = lambda w: log_loss(y, np.clip(M @ w, 1e-6, 1 - 1e-6))
    r = minimize(loss, np.ones(M.shape[1]) / M.shape[1], method="L-BFGS-B", bounds=[(0, 1)] * M.shape[1])
    w = r.x / r.x.sum()
    return w, M @ w


def main():
    lab = pd.read_csv("Dataset/train_labels.csv").set_index("uid")
    site = pd.read_csv("Dataset/site_labels.csv").set_index("uid")
    v1 = pd.read_csv("Dataset/sbr_features_train.csv").set_index("uid")
    ph = pd.read_csv("Dataset/phys_features_train.csv").set_index("uid")
    y = lab.loc[v1.index, "is_pathologic"].values
    groups = site.loc[v1.index, "pseudo_site"].astype(str).values

    X1p = build_plain(v1, RAW1)
    X1d = build_derived(v1, RAW1)
    X2p = build_plain(ph, RAW2)

    o1p = oof_run(X1p, y, groups)
    o1d = oof_run(X1d, y, groups)
    o2p = oof_run(X2p, y, groups)

    for name, o in (("v1 plain", o1p), ("v1 derived", o1d), ("phys plain", o2p)):
        b = blend(np.column_stack([o[m] for m in MODEL_NAMES]), y)[1]
        print(f"{name:12s} AUC {roc_auc_score(y, b):.4f} LL {log_loss(y, np.clip(b, 0.005, 0.995)):.4f}")

    # combos at model level
    def comb(sets, label):
        M = np.column_stack([o[m] for sname, o in sets for m in MODEL_NAMES])
        w, b = blend(M, y)
        print(f"{label:28s} AUC {roc_auc_score(y, b):.4f} LL {log_loss(y, np.clip(b, 0.005, 0.995)):.4f} w {np.round(w,2)}")

    comb([("v", o1p), ("p", o2p)], "v1plain+phys")
    comb([("v", o1d), ("p", o2p)], "v1derived+phys")


if __name__ == "__main__":
    main()