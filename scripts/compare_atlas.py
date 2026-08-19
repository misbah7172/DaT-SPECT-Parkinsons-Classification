import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("lr", "et", "hgb")
NO_LOG = {"brain_skew", "brain_kurt"}


def build_matrix(raw: pd.DataFrame, raw_cols: list) -> np.ndarray:
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(
        np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def oof_run(X, y, groups, seed=42):
    oof = {m: np.zeros(len(y)) for m in MODEL_NAMES}
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr_s, Xva_s = sc.transform(X[tr]), sc.transform(X[va])
        lr = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr_s, y[tr])
        oof["lr"][va] = lr.predict_proba(Xva_s)[:, 1]
        mi = mutual_info_classif(X[tr], y[tr], random_state=seed)
        sel = np.argsort(mi)[::-1][:80]
        Xtr_t, Xva_t = Xtr_s[:, sel], Xva_s[:, sel]
        et = ExtraTreesClassifier(500, max_depth=12, n_jobs=-1, random_state=seed).fit(Xtr_t, y[tr])
        oof["et"][va] = et.predict_proba(Xva_t)[:, 1]
        hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.02).fit(Xtr_t, y[tr])
        oof["hgb"][va] = hgb.predict_proba(Xva_t)[:, 1]
    return oof


def blend(M, y):
    loss = lambda w: log_loss(y, np.clip(M @ w, 1e-6, 1 - 1e-6))
    r = minimize(loss, np.ones(M.shape[1]) / M.shape[1], method="L-BFGS-B", bounds=[(0, 1)] * M.shape[1])
    return r.x / r.x.sum()


def main():
    lab = pd.read_csv("Dataset/train_labels.csv").set_index("uid")
    site = pd.read_csv("Dataset/site_labels.csv").set_index("uid")
    v1 = pd.read_csv("Dataset/sbr_features_train.csv").set_index("uid")
    ph = pd.read_csv("Dataset/phys_features_train.csv").set_index("uid")
    at = pd.read_csv("Dataset/atlas_features_train.csv").set_index("uid")
    y = lab.loc[v1.index, "is_pathologic"].values
    groups = site.loc[v1.index, "pseudo_site"].astype(str).values
    base1 = [c for c in v1.columns if c not in ("uid", "label")]
    base2 = [c for c in ph.columns if c != "uid"]
    base3 = [c for c in at.columns if c != "uid"]

    X1 = build_matrix(v1, base1)
    X2 = build_matrix(ph, base2)
    X3 = build_matrix(at, base3)

    o1 = oof_run(X1, y, groups)
    o2 = oof_run(X2, y, groups)
    o3 = oof_run(X3, y, groups)

    print("per-model (lr/et/hgb):")
    print("  v1  :", [round(roc_auc_score(y, o1[m]), 4) for m in MODEL_NAMES])
    print("  ph  :", [round(roc_auc_score(y, o2[m]), 4) for m in MODEL_NAMES])
    print("  at  :", [round(roc_auc_score(y, o3[m]), 4) for m in MODEL_NAMES])

    # single-family blends
    def fam_blend(o):
        M = np.column_stack([o[m] for m in MODEL_NAMES])
        w = blend(M, y)
        return M @ w

    b1, b2, b3 = fam_blend(o1), fam_blend(o2), fam_blend(o3)
    print("\nfam blends: v1", round(roc_auc_score(y, b1), 4), "ph", round(roc_auc_score(y, b2), 4),
          "at", round(roc_auc_score(y, b3), 4))

    # all models across all families
    all_names = [f"{f}_{m}" for f in ("v", "p", "a") for m in MODEL_NAMES]
    M9 = np.column_stack([o1[m] for m in MODEL_NAMES] + [o2[m] for m in MODEL_NAMES] + [o3[m] for m in MODEL_NAMES])
    w9 = blend(M9, y)
    b9 = M9 @ w9
    print("\nall-9 blend AUC", round(roc_auc_score(y, b9), 4), "LL", round(log_loss(y, np.clip(b9, 0.005, 0.995)), 4))
    print("weights:", dict(zip(all_names, np.round(w9, 3))))

    # stack as columns (feature-level) for at+v1
    a1 = np.concatenate([o1["lr"], o2["lr"], o3["lr"]])  # placeholder not used


if __name__ == "__main__":
    main()