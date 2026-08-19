import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.special import expit, logit
import math

pd.set_option("display.width", 200)

lab = pd.read_csv("Dataset/train_labels.csv")
site = pd.read_csv("Dataset/site_labels.csv")
sbr = pd.read_csv("Dataset/sbr_features_train.csv")
morph = pd.read_csv("Dataset/sbr_features_morph_train.csv")

labels = lab.set_index("uid")["is_pathologic"]
sites = site.set_index("uid")["pseudo_site"]

base_feats = [c for c in sbr.columns if c not in ("uid", "label")]
morph_feats = [c for c in morph.columns if c not in ("uid", "label")]

def featurize(use_morph=False):
    X = sbr[["uid"] + base_feats].copy()
    Y = labels.loc[X.uid].values
    G = sites.loc[X.uid].values.astype(str)
    raw = X[base_feats].values
    log_idx = [i for i, c in enumerate(base_feats) if c not in ("brain_skew", "brain_kurt")]
    cols = base_feats + [f"log1p_{c}" for c in base_feats]
    F = np.concatenate([raw, np.log1p(raw[:, log_idx])], axis=1)
    if use_morph:
        mm = morph.set_index("uid").loc[X.uid].values  # already has some overlap; keep as extra
        # drop columns that duplicate sbr base feats to avoid double counting
        extra = [c for c in morph_feats if c not in base_feats]
        mm = morph.set_index("uid")[extra].values
        F = np.concatenate([F, mm], axis=1)
        cols = cols + extra
    return F, Y, G, cols

def evaluate_cv(splitter, use_morph=False, n_models="all"):
    F, Y, G, cols = featurize(use_morph)
    n = len(Y)
    oof = {m: np.zeros(n) for m in ("lr", "et", "hgb", "rf")}
    folds = list(splitter.split(F, Y, groups=G) if isinstance(splitter, StratifiedGroupKFold) else splitter.split(F, Y))
    for fold, (tr, va) in enumerate(folds):
        Xtr, Xva, ytr = F[tr], F[va], Y[tr]
        sc = StandardScaler().fit(Xtr)
        Xs_tr, Xs_va = sc.transform(Xtr), sc.transform(Xva)
        lr = LogisticRegression(C=0.30, solver="lbfgs", max_iter=3000).fit(Xs_tr, ytr)
        oof["lr"][va] = lr.predict_proba(Xs_va)[:, 1]
        mi = mutual_info_classif(Xtr, ytr, random_state=42)
        sel = np.argsort(mi)[::-1][:60]
        Xt_tr, Xt_va = Xs_tr[:, sel], Xs_va[:, sel]
        et = ExtraTreesClassifier(500, max_depth=12, max_features="sqrt", n_jobs=-1, random_state=42 + fold).fit(Xt_tr, ytr)
        oof["et"][va] = et.predict_proba(Xt_va)[:, 1]
        hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.02, l2_regularization=1.0, min_samples_leaf=10, random_state=42 + fold).fit(Xt_tr, ytr)
        oof["hgb"][va] = hgb.predict_proba(Xt_va)[:, 1]
        if n_models == "all":
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(500, max_depth=12, n_jobs=-1, random_state=42 + fold).fit(Xt_tr, ytr)
            oof["rf"][va] = rf.predict_proba(Xt_va)[:, 1]
    rows = {m: (roc_auc_score(Y, oof[m]), np.round(log_loss(Y, np.clip(oof[m], 0.001, 0.999)), 4)) for m in oof}
    M = np.column_stack(list(oof.values()) if n_models == "all" else [oof[m] for m in ("lr", "et", "hgb")])
    names = list(oof.keys()) if n_models == "all" else ["lr", "et", "hgb"]
    def loss(w):
        return log_loss(Y, np.clip(M @ w, 1e-6, 1 - 1e-6))
    best = None
    for init0 in (np.ones(len(names)) / len(names),):
        r = minimize(loss, init0, method="L-BFGS-B", bounds=[(0, 1)] * len(names))
        if best is None or r.fun < best.fun:
            best = r
    w = best.x / best.x.sum()
    blend = M @ w
    return rows, blend, w, Y

split_site = StratifiedGroupKFold(5, shuffle=True, random_state=42)
split_naive = StratifiedKFold(5, shuffle=True, random_state=42)

for name, sp, morph_ in [("site aware sbr", split_site, False), ("site aware sbr+morph", split_site, True), ("naive sbr", split_naive, False)]:
    rows, blend, w, Y = evaluate_cv(sp, use_morph=morph_, n_models="all")
    print("=" * 60)
    print(name)
    for m, (a, l) in rows.items():
        print(f"   {m}: AUC={a:.4f} LL={l}")
    print("   blend", np.round(w, 3), "AUC", round(roc_auc_score(Y, blend), 4), "LL", round(log_loss(Y, np.clip(blend, 0.005, 0.995)), 4))