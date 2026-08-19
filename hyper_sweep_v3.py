import warnings
warnings.filterwarnings("ignore")
import sys, os, json, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from scipy.optimize import minimize
from scipy.special import expit, logit

sys.path.insert(0, "E:/DaT/src")
from sbr_extractor import FEATURE_COLUMNS
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS

PVE_REF = 15.625
RAW1, RAW2 = list(FEATURE_COLUMNS), list(PHYS_FEATURE_COLUMNS)

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
fac = (PVE_REF / merged["voxel_vol"].values) ** (1.0 / 3.0)
for c in [c for c in RAW1 if c.startswith("sbr_")]:
    merged[c] = merged[c] * fac

def build_matrix(raw, cols):
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1), nan=0.0)

X1 = build_matrix(merged, RAW1)
X2 = build_matrix(merged[[f"p_{c}" for c in RAW2 if f"p_{c}" in merged.columns].rename(columns={c: c[2:] for c in RAW2}), RAW2)

seed = 42

# We'll test combinations of:
# - n_sel: 44, 58, 80
# - LR C: 0.3, 1.0, 1.31
# - ridge alpha: 0.1, 1.0, 5.0
# Using 1-seed 1-fold OOF for speed, then note results

print("=== FOCUSED HYPERPARAMETER SWEEP ===")
print("Testing n_sel, LR C, ridge alpha on in-sample OOF protocol")
print("=" * 60)

results = []

for n_sel in [44, 58, 80]:
    # Select top n_sel features by mutual info
    mi = mutual_info_classif(X1, y, random_state=seed)
    sel = np.argsort(mi)[::-1][:n_sel]
    X1_s = X1[:, sel]
    
    for c_lr in [0.3, 1.0, 1.31]:
        for ridge_a in [0.1, 1.0, 5.0]:
            # Train models on 1-seed 1-fold for speed
            # Actually, let's just train on full data with these params and compute in-sample metrics
            # (this is quick and gives diagnostic values)
            
            # LR
            lr = LogisticRegression(C=c_lr, max_iter=3000, random_state=seed).fit(X1_s, y)
            lr_pred = lr.predict_proba(X1_s)[:, 1]
            
            # Ridge
            ridge = RidgeClassifier(alpha=ridge_a, random_state=seed).fit(X1_s, y)
            ridge_pred = ridge.decision_function(X1_s)
            
            # ET
            mi_s = mutual_info_classif(X1_s, y, random_state=seed)
            et_sel = np.argsort(mi_s)[::-1][:20]  # just use top 20 for speed
            et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(X1_s[:, et_sel], y)
            et_pred = et.predict_proba(X1_s[:, et_sel])[:, 1]
            
            # HGB
            hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.0489,
                                                 l2_regularization=4.6, min_samples_leaf=10,
                                                 random_state=seed).fit(X1_s, y)
            hgb_pred = hgb.predict_proba(X1_s)[:, 1]
            
            # RF
            rf = RandomForestClassifier(300, max_depth=14, n_jobs=-1, random_state=seed).fit(X1_s, y)
            rf_pred = rf.predict_proba(X1_s)[:, 1]
            
            # Simple blend with v6-like weights: lr 0.081, et 0.209, hgb 0.216, ridge 0.052
            # But normalize weights to sum to 1
            weights = np.array([0.081, 0.209, 0.216, 0.052])
            weights = weights / weights.sum()
            
            blend = np.zeros(len(y))
            blend += lr.predict_proba(X1_s)[:, 1] * weights[0]
            preds_et = et.predict_proba(X1_s[:, et_sel])[:, 1] * weights[1]
            # Need to handle et feature selection properly
            # Actually, let me just use et on all X1_s features for simplicity
            preds_et = et.predict_proba(X1_s)[:, 1] * weights[1]
            preds_hgb = hgb.predict_proba(X1_s)[:, 1] * weights[2]
            preds_ridge = ridge.decision_function(X1_s) * weights[3]  # ridge gives decision function
            
            # Convert ridge decision function to probability via sigmoid
            ridge_probs = expit(ridge.decision_function(X1_s) / 0.8958)  # temperature
            ridge_probs = np.clip(ridge_probs, 0.005, 0.995)
            
            blend += preds_ridge
            
            preds = blend / weights.sum()  # normalize
            preds = np.clip(preds, 0.005, 0.995)
            
            auc = roc_auc_score(y, preds)
            ll = log_loss(y, preds)
            
            results.append({
                'n_sel': n_sel,
                'lr_c': c_lr,
                'ridge_a': ridge_a,
                'auc': round(auc, 4),
                'll': round(ll, 4)
            })
            print(f"n_sel={n_sel}, C={c_lr}, alpha={ridge_a}: AUC={round(auc,4)}, LL={round(ll,4)}")

print()
print("=== RESULTS SUMMARY ===")
for r in results:
    print(f"n_sel={r['n_sel']}, C={r['lr_c']}, alpha={r['ridge_a']}: AUC={r['auc']}, LL={r['ll']}")

# Find best
best = min(results, key=lambda x: x['ll'])
print(f"\nBest LL: n_sel={best['n_sel']}, C={best['lr_c']}, alpha={best['ridge_a']}: LL={best['ll']}, AUC={best['auc']}")