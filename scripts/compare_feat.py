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


def oof_run(X, Y, G, name, seed=42):
    oof = {m: np.zeros(len(Y)) for m in ("lr", "et", "hgb")}
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, Y, groups=G):
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform
        lr = LogisticRegression(C=0.3, max_iter=3000).fit(Xs(X[tr]), Y[tr])
        oof["lr"][va] = lr.predict_proba(Xs(X[va]))[:, 1]
        mi = mutual_info_classif(X[tr], Y[tr], random_state=seed)
        sel = np.argsort(mi)[::-1][:80]
        et = ExtraTreesClassifier(500, max_depth=12, n_jobs=-1, random_state=seed).fit(X[tr][:, sel], Y[tr])
        oof["et"][va] = et.predict_proba(X[va][:, sel])[:, 1]
        hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.02).fit(X[tr][:, sel], Y[tr])
        oof["hgb"][va] = hgb.predict_proba(X[va][:, sel])[:, 1]
    return oof


def blend(M, Y):
    loss = lambda w: log_loss(Y, np.clip(M @ w, 1e-6, 1 - 1e-6))
    r = minimize(loss, np.ones(M.shape[1]) / M.shape[1], method="L-BFGS-B", bounds=[(0, 1)] * M.shape[1])
    w = r.x / r.x.sum()
    return w, M @ w


def main():
    lab = pd.read_csv("Dataset/train_labels.csv").set_index("uid")
    site = pd.read_csv("Dataset/site_labels.csv").set_index("uid")
    v1 = pd.read_csv("Dataset/sbr_features_train.csv").set_index("uid")
    ph = pd.read_csv("Dataset/phys_features_train.csv").set_index("uid")
    Y = lab.loc[v1.index, "is_pathologic"].values
    G = site.loc[v1.index, "pseudo_site"].astype(str).values
    base1 = [c for c in v1.columns if c not in ("uid", "label")]
    base2 = [c for c in ph.columns if c != "uid"]
    X1 = np.nan_to_num(np.concatenate([v1[base1].values, np.log1p(v1[base1].clip(lower=0).values)], axis=1))
    Xp = np.nan_to_num(np.concatenate([ph[base2].values, np.log1p(ph[base2].clip(lower=0).values)], axis=1))

    o1 = oof_run(X1, Y, G, "v1")
    op = oof_run(Xp, Y, G, "phys")

    names = [m for m in o1]
    print("model  v1  phys")
    for m in names:
        print(f"{m:4s} {roc_auc_score(Y, o1[m]):.4f} {roc_auc_score(Y, op[m]):.4f}")

    # model-level blend across all 6 OOFs (v1 lr/et/hgb + phys lr/et/hgb)
    M6 = np.column_stack([o1[m] for m in names] + [op[m] for m in names])
    w6, b6 = blend(M6, Y)
    print("\nall-6 blend: AUC", round(roc_auc_score(Y, b6), 4), "LL", round(log_loss(Y, np.clip(b6, 0.005, 0.995)), 4))
    print("weights:", dict(zip(["v1_lr", "v1_et", "v1_hgb", "p_lr", "p_et", "p_hgb"], np.round(w6, 3))))

    # two-space blend: v1-blend + phys-blend (each pre-blended per-space)
    w1, b1 = blend(np.column_stack([o1[m] for m in names]), Y)
    wp, bp = blend(np.column_stack([op[m] for m in names]), Y)
    M2 = np.column_stack([b1, bp])
    w2, b2 = blend(M2, Y)
    print("space-blend: AUC", round(roc_auc_score(Y, b2), 4), "LL", round(log_loss(Y, np.clip(b2, 0.005, 0.995)), 4))
    print("space w:", np.round(w2, 3))

    # phys-only blend for reference
    print("phys-internal blend AUC", round(roc_auc_score(Y, bp), 4), "LL", round(log_loss(Y, np.clip(bp, 0.005, 0.995)), 4))


if __name__ == "__main__":
    main()