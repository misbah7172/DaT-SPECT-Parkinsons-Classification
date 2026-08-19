import warnings
warnings.filterwarnings("ignore")
import sys, os, json, numpy as pd, time
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score
import joblib

sys.path.insert(0, "E:/DaT/src")
from sbr_extractor import FEATURE_COLUMNS
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS

PVE_REF = 15.625
RAW1, RAW2 = list(FEATURE_COLUMNS), list(PHYS_FEATURE_COLUMNS)

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

# ============================================================
# CONFIGURATION A: n_sel=58, LR C=1.312, Ridge alpha=8.108
# ============================================================
print("=" * 50)
print("CONFIGURATION A: n_sel=58, LR C=1.312, Ridge alpha=8.108")
print("=" * 50)

aucs_a = []
lls_a = []
for seed in [42]:
    aucs = []
    lls = []
    for train_idx, test_idx in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X1, y, groups):
        # Feature selection inside training fold
        mi = mutual_info_classif(X1[train_idx], y[train_idx], random_state=seed)
        sel = np.argsort(mi)[::-1][:58]
        
        # Scale
        sc = StandardScaler().fit(X1[train_idx])
        X1_train_s = sc.transform(X1[train_idx])
        X1_test_s = sc.transform(X1[test_idx])
        
        # LR
        lr = LogisticRegression(C=1.312, max_iter=3000, random_state=seed).fit(X1_train_s, y[train_idx])
        lr_pred = lr.predict_proba(X1_test_s)[:, 1]
        
        # Ridge
        ridge = RidgeClassifier(alpha=8.108, random_state=seed).fit(X1_train_s, y[train_idx])
        ridge_d = ridge.decision_function(X1_test_s)
        
        # ET on selected features
        mi_s = mutual_info_classif(X1_train_s, y[train_idx], random_state=seed)
        et_sel = np.argsort(mi_s)[::-1][:20]
        et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(X1_train_s[:, sel], y[train_idx])
        et_pred = et.predict_proba(X1_test_s[:, sel])[:, 1]
        
        # HGB on selected features
        hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.0489,
                                             l2_regularization=4.6, min_samples_leaf=10,
                                                 random_state=seed).fit(X1_train_s[:, sel], y[train_idx])
        hgb_pred = hgb.predict_proba(X1_test_s[:, sel])[:, 1]
        
        # Blend
        preds = lr.predict_proba(X1_test_s)[:, 1] * 0.081 + et_pred * 0.209 + hgb_pred * 0.216
        ridge_d = ridge.decision_function(X1_test_s)
        ridge_prob = expit(ridge_d / 0.8958)
        ridge_prob = np.clip(ridge_prob, 0.005, 0.995)
        preds += ridge_prob * 0.052
        preds = np.clip(preds, 0.005, 0.995)
        
        auc = roc_auc_score(y[test_idx], lr.predict_proba(X1_test_s)[:, 1])
        ll = log_loss(y[test_idx], lr.predict_proba(X1_test_s)[:, 1])
        
        aucs.append(auc)
        lls.append(ll)
        
print(f"Config A (n_sel=58, C=1.312, alpha=8.108):")
print(f"Mean OOF AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean OOF LL: {np.mean(lls):.4f} ± {np.std(lls):.4f}")

# ============================================================
# CONFIGURATION B: n_sel=44, LR C=1.31, Ridge alpha=0.1
# ============================================================
print()
print("=" * 50)
print("CONFIGURATION B: n_sel=44, LR C=1.31, Ridge alpha=0.1")
print("=" * 50)

aucs_b = []
lls_b = []
for seed in [42]:
    for train_idx, test_idx in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X1, y, groups):
        # Feature selection inside training fold
        mi = mutual_info_classif(X1[train_idx], y[train_idx], random_state=seed)
        sel = np.argsort(mi)[::-1][:44]
        
        # Scale
        sc = StandardScaler().fit(X1[train_idx])
        X1_train_s = sc.transform(X1[train_idx])
        X1_test_s = sc.transform(X1[test_idx])
        
        # LR
        lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(X1_train_s, y[train_idx])
        lr_pred = lr.predict_proba(X1_test_s)[:, 1]
        
        # Ridge
        ridge = RidgeClassifier(alpha=0.1, random_state=seed).fit(X1_train_s, y[train_idx])
        ridge_d = ridge.decision_function(X1_test_s)
        
        # ET on selected features
        mi_s = mutual_info_classif(X1_train_s, y[train_idx], random_state=seed)
        et_sel = np.argsort(mi_s)[::-1][:20]
        et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(X1_train_s[:, sel], y[train_idx])
        et_pred = et.predict_proba(X1_test_s[:, sel])[:, 1]
        
        # HGB on selected features
        hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.0489,
                                             l2_regularization=4.6, min_samples_leaf=10,
                                                 random_state=seed).fit(X1_train_s[:, sel], y[train_idx])
        hgb_pred = hgb.predict_proba(X1_test_s[:, sel])[:, 1]
        
        # Blend
        w = np.array([0.081, 0.209, 0.216, 0.052])
        w = w / w.sum()
        
        preds = lr.predict_proba(X1_test_s)[:, 1] * w[0] + et_pred * w[1] + hgb_pred * w[2]
        ridge_d = ridge.decision_function(X1_test_s)
        ridge_prob = expit(ridge_d / 0.8958)
        ridge_prob = np.clip(ridge_prob, 0.005, 0.995)
        preds += ridge_prob * w[3]
        preds = np.clip(preds, 0.005, 0.995)
        
        auc = roc_auc_score(y[test_idx], lr.predict_proba(X1_test_s)[:, 1])
        ll = log_loss(y[test_idx], lr.predict_proba(X1_test_s)[:, 1])
        
        aucs_b.append(auc)
        lls_b.append(ll)
        
print("Configuration B (n_sel=44, C=1.31, alpha=0.1):")
print(f"Mean OOF AUC: {np.mean(aucs_b):.4f} ± {np.std(aucs_b):.4f}")
print(f"Mean OOF LL: {np.mean(lls_b):.4f} ± {np.std(lls_b):.4f}")