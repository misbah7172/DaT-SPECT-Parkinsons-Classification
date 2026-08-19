import warnings
warnings.filterwarnings("ignore")
import sys, os, json, numpy as pd, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, "E:/DaT/src")
from sbr_extractor import FEATURE_COLUMNS

PVE_REF = 15.625
RAW1 = list(FEATURE_COLUMNS)

labels = pd.read_csv("E:/DaT/Dataset/train_labels.csv")
site = pd.read_csv("E:/DaT/Dataset/site_labels.csv")
geom = pd.read_csv("E:/DaT/Dataset/voxel_geometry.csv")
sbr1 = pd.read_csv("E:/DaT/Dataset/sbr_features_train.csv")
merged = pd.read_csv("E:/DaT/Dataset/voxel_geometry.csv")
# Actually load properly
labels = pd.read_csv("E:/DaT/Dataset/train_labels.csv")
site = pd.read_csv("E:/DaT/Dataset/site_labels.csv")
geom = pd.read_csv("E:/DaT/Dataset/voxel_geometry.csv")
sbr1 = pd.read_csv("E:/DaT/Dataset/sbr_features_train.csv")
merged = labels.merge(site, on="uid").merge(geom, on="uid").merge(sbr1, on="uid").dropna()
y = merged["is_pathologic"].values
groups = merged["pseudo_site"].astype(str).values
fac = (15.625 / merged["voxel_vol"].values) ** (1.0 / 3.0)
for c in [c for c in RAW1 if c.startswith("sbr_")]:
    merged[c] = merged[c] * fac

def build_matrix(raw, cols):
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1), nan=0.0)

X1 = build_matrix(merged, RAW1)

# Test n_sel values
seed = 42
print("=== n_sel SWEEP ===")
for n in [44, 58, 80]:
    mi = mutual_info_classif(X1, y, random_state=seed)
    sel = np.argsort(mi)[::-1][:n]
    X1_s = X1[:, sel]
    
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    
    lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(X1_s, y)
    ridge = RidgeClassifier(alpha=1.0, random_state=seed).fit(X1_s, y)
    et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(X1_s, y)
    hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.0489,
                                         l2_regularization=4.6, min_samples_leaf=10,
                                         random_state=seed).fit(X1_s, y)
    rf = RandomForestClassifier(300, max_depth=14, n_jobs=-1, random_state=seed).fit(X1_s, y)
    
    # Simple blend: weights from v6 concept
    # lr 0.081, et 0.209, hgb 0.216, rf 0.0, ridge 0.052
    preds = np.zeros(len(y))
    preds += lr.predict_proba(X1_s)[:, 1] * 0.081
    preds += et.predict_proba(X1_s)[:, 1] * 0.209
    preds += hgb.predict_proba(X1_s)[:, 1] * 0.216
    preds += rf.predict_proba(X1_s)[:, 1] * 0.0
    preds += ridge.predict_proba(X1_s)[:, 1] * 0.052
    
    preds = np.clip(preds, 0.005, 0.995)
    
    auc = roc_auc_score(y, preds)
    ll = log_loss(y, preds)
    print(f"n_sel={n}: AUC={round(auc,4)}, LL={round(ll,4)}")

print()
print("=== Ridge alpha SWEEP (with n_sel=58) ===")
for alpha in [0.1, 1.0, 5.0, 10.0]:
    ridge = RidgeClassifier(alpha=alpha, random_state=seed).fit(X1, y)
    preds_r = ridge.predict_proba(X1)[:, 1]
    auc = roc_auc_score(y, preds_r)
    ll = log_loss(y, np.clip(preds_r, 0.005, 0.995))
    print(f"ridge alpha={alpha}: AUC={round(auc,4)}, LL={round(ll,4)}")

print()
print("=== LR C SWEEP (with n_sel=58, ridge alpha=1.0) ===")
for c in [0.1, 0.3, 1.0, 1.31, 2.0]:
    lr = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(X1, y)
    preds_lr = lr.predict_proba(X1)[:, 1]
    auc = roc_auc_score(y, preds_lr)
    ll = log_loss(y, np.clip(preds_lr, 0.005, 0.995))
    print(f"LR C={c}: AUC={round(auc,4)}, LL={round(ll,4)}")