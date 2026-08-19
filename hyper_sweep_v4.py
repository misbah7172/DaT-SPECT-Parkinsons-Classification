import warnings
warnings.filterwarnings("ignore")
import sys, os, json, numpy as np, pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import mutual_info_classif
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

seed = 42
print("=== FOCUSED SWEEP: n_sel, LR C, ridge alpha ===")
print("=" * 50)

results = []

for n_sel in [44, 58, 80]:
    mi = mutual_info_classif(X1, y, random_state=seed)
    sel = np.argsort(mi)[::-1][:n_sel]
    X1_s = X1[:, sel]
    
    for c_lr in [0.3, 1.0, 1.31]:
        for ridge_a in [0.1, 1.0, 5.0]:
            # Train models on X1_s (selected features)
            lr = LogisticRegression(C=c_lr, max_iter=3000, random_state=seed).fit(X1_s, y)
            ridge = RidgeClassifier(alpha=ridge_a, random_state=seed).fit(X1_s, y)
            et = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(X1_s, y)
            hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.0489,
                                                 l2_regularization=4.6, min_samples_leaf=10,
                                                 random_state=seed).fit(X1_s, y)
            rf = RandomForestClassifier(300, max_depth=14, n_jobs=-1, random_state=seed).fit(X1_s, y)
            
            # Blend: lr 0.081, et 0.209, hgb 0.216, ridge 0.052 (v6 concept)
            # But normalize so weights sum to 1
            w = np.array([0.081, 0.209, 0.216, 0.052])
            w = w / w.sum()
            
            # Evaluate on X1_s for all models
            preds = np.zeros(len(y))
            preds += lr.predict_proba(X1_s)[:, 1] * w[0]
            preds += et.predict_proba(X1_s)[:, 1] * w[1]
            preds += hgb.predict_proba(X1_s)[:, 1] * w[2]
            # ridge decision function converted to prob
            ridge_d = ridge.decision_function(X1_s)
            ridge_prob = expit(ridge_d / 0.8958)  # temperature
            ridge_prob = np.clip(ridge_prob, 0.005, 0.995)
            preds += ridge_prob * w[3]
            
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
print("=== BEST CONFIGURATION ===")
best = min(results, key=lambda x: x['ll'])
print(f"Best LL: n_sel={best['n_sel']}, C={best['lr_c']}, alpha={best['ridge_a']}")
print(f"  AUC={best['auc']}, LL={best['ll']}")
print()
print("=== ALL RESULTS ===")
for r in results:
    print(f"n_sel={r['n_sel']}, C={r['lr_c']}, alpha={r['ridge_a']}: AUC={r['auc']}, LL={r['ll']}")