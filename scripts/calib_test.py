import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("lr", "et", "hgb")
NO_LOG = {"brain_skew", "brain_kurt"}


def build_matrix(raw, raw_cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    log_idx = [i for i, c in enumerate(cols) if c not in NO_LOG]
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a[:, log_idx], 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def main():
    lab = pd.read_csv("Dataset/train_labels.csv").set_index("uid")
    site = pd.read_csv("Dataset/site_labels.csv").set_index("uid")
    v1 = pd.read_csv("Dataset/sbr_features_train.csv").set_index("uid")
    ph = pd.read_csv("Dataset/phys_features_train.csv").set_index("uid")
    y = lab.loc[v1.index, "is_pathologic"].values
    groups = site.loc[v1.index, "pseudo_site"].astype(str).values
    base1 = [c for c in v1.columns if c not in ("uid", "label")]
    base2 = [c for c in ph.columns if c != "uid"]
    X1 = build_matrix(v1, base1)
    X2 = build_matrix(ph, base2)

    oof = {}
    for Xname, X in (("X1", X1), ("X2", X2)):
        for m in MODEL_NAMES:
            oof[f"{Xname}_{m}"] = np.zeros(len(y))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            Xtr_s, Xva_s = sc.transform(X[tr]), sc.transform(X[va])
            lr = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr_s, y[tr])
            oof[f"{Xname}_lr"][va] = lr.predict_proba(Xva_s)[:, 1]
            mi = mutual_info_classif(X[tr], y[tr], random_state=42)
            sel = np.argsort(mi)[::-1][:80]
            Xtr_t, Xva_t = Xtr_s[:, sel], Xva_s[:, sel]
            et = ExtraTreesClassifier(500, max_depth=12, n_jobs=-1, random_state=42).fit(Xtr_t, y[tr])
            oof[f"{Xname}_et"][va] = et.predict_proba(Xva_t)[:, 1]
            hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.02).fit(Xtr_t, y[tr])
            oof[f"{Xname}_hgb"][va] = hgb.predict_proba(Xva_t)[:, 1]

    names = list(oof)
    M = np.column_stack([oof[n] for n in names])
    loss = lambda w: log_loss(y, np.clip(M @ w, 1e-6, 1 - 1e-6))
    r = minimize(loss, np.ones(len(names)) / len(names), method="L-BFGS-B", bounds=[(0, 1)] * len(names))
    w = r.x / r.x.sum()
    blend = M @ w
    print("blend AUC", round(roc_auc_score(y, blend), 4), "raw LL", round(log_loss(y, np.clip(blend, 0.005, 0.995)), 4))

    # temperature
    def temp_loss(log_t):
        p = expit(logit(np.clip(blend, 1e-6, 1 - 1e-6)) / np.exp(log_t[0]))
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

    tr = minimize(temp_loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    temp = float(np.exp(tr.x[0]))
    p_temp = expit(logit(np.clip(blend, 1e-6, 1 - 1e-6)) / temp)
    print("temperature", round(temp, 4), "LL", round(log_loss(y, np.clip(p_temp, 0.005, 0.995)), 4))

    # isotonic nested within OOF folds
    iso_pred = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr_i, va_i in skf.split(blend, y):
        iso = IsotonicRegression(out_of_bounds="clip").fit(blend[tr_i], y[tr_i])
        iso_pred[va_i] = iso.predict(blend[va_i])
    print("isotonic LL", round(log_loss(y, np.clip(iso_pred, 0.005, 0.995)), 4),
          "AUC", round(roc_auc_score(y, iso_pred), 4))

    # theoretical LL floor: entropy bound via binning (perfect calibration of blend)
    bins = np.percentile(blend, np.linspace(0, 100, 21))[1:-1]
    cuts = np.digitize(blend, bins)
    ll_floor = 0.0
    for c in np.unique(cuts):
        m = cuts == c
        p = np.clip(y[m].mean(), 1e-6, 1 - 1e-6)
        ll_floor += np.sum(-(y[m] * np.log(p) + (1 - y[m]) * np.log(1 - p)))
    print("binned-oracle LL floor", round(ll_floor / len(y), 4))


if __name__ == "__main__":
    main()