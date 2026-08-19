import warnings
warnings.filterwarnings("ignore")
import sys, os, json, numpy as np, pandas as pd, time
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
import joblib

sys.path.insert(0, "E:/DaT/src")
from sbr_extractor import FEATURE_COLUMNS
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS

# CONFIGURATION
PVE_REF = 15.625
RAW1, RAW2 = list(FEATURE_COLUMNS), list(PHYS_FEATURE_COLUMNS)
SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5

# Load data
labels = pd.read_csv("E:/DaT/Dataset/train_labels.csv")
site = pd.read_csv("E:/DaT/Dataset/site_labels.csv")
geom = pd.read_csv("E:/DaT/Dataset/voxel_geometry.csv")
sbr1 = pd.read_csv("E:/DaT/Dataset/sbr_features_train.csv")
sbr2 = pd.read_csv("E:/DaT/Dataset/phys_features_train.csv")
merged = pd.read_csv("E:/DaT/Dataset/voxel_geometry.csv")
# Actually load properly
labels = pd.read_csv("E:/DaT/Dataset/train_labels.csv")
site = pd.read_csv("E:/DaT/Dataset/site_labels.csv")
geom = pd.read_csv("E:/DaT/Dataset/voxel_geometry.csv")
sbr1 = pd.read_csv("E:/DaT/Dataset/sbr_features_train.csv")
sbr2 = pd.read_csv("E:/DaT/Dataset/phys_features_train.csv")
merged = labels.merge(site, on="uid").merge(geom, on="uid").merge(sbr1, on="uid").dropna()
y = merged["is_pathologic"].values
groups = merged["pseudo_site"].astype(str).values
fac = (PVE_REF / merged["voxel_vol"].values) ** (1.0 / 3.0)
for c in [c for c in RAW1 if c.startswith("sbr_")]:
    merged[c] = merged[c] * fac

def build_matrix(raw, raw_cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1), nan=0.0)

X1 = build_matrix(merged, RAW1)
p_cols = [f"p_{c}" for c in RAW2 if f"p_{c}" in merged.columns]
X2 = build_matrix(merged[p_cols].rename(columns={c: c[2:] for c in p_cols}), RAW2)

# ============================================================
# CONFIGURATION A: n_sel=58, LR C=1.312, Ridge alpha=8.108
# ============================================================
print("=" * 60)
print("CONFIGURATION A: n_sel=58, LR C=1.312, Ridge alpha=8.108")
print("=" * 60)

results_a = []
for si, seed in enumerate(SEEDS):
    fold_results = []
    for nfolds in [5]:
        fold_aucs = []
        fold_lls = []
        for train_idx, test_idx in StratifiedGroupKFold(nfolds, shuffle=True, random_state=seed).split(X1, y, groups):
            # Feature selection inside training fold only
            mi = mutual_info_classif(X1[train_idx], y[train_idx], random_state=seed)
            sel = np.argsort(mi)[::-1][:58]
            
            # Scale
            sc = StandardScaler().fit(X1[train_idx])
            X1_train_s = sc.transform(X1[train_idx])
            X1_test_s = sc.transform(X1[test_idx])
            
            # LR on scaled features
            lr = LogisticRegression(C=1.312, max_iter=3000, random_state=seed).fit(X1_train_s, y[train_idx])
            lr_pred = lr.predict_proba(X1_test_s)[:, 1]
            
            # Ridge on scaled features
            ridge = RidgeClassifier(alpha=8.108, random_state=seed).fit(X1_train_s, y[train_idx])
            ridge_pred = ridge.decision_function(X1_test_s)
            
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
            
            # Simple blend: lr 0.081, et 0.209, hgb 0.216, ridge 0.052
            # But normalize so weights sum to 1
            w = np.array([0.081, 0.209, 0.216, 0.052])
            w = w / w.sum()
            
            preds = np.zeros(len(y[test_idx]))
            preds += lr.predict_proba(X1_test_s)[:, 1] * w[0]
            preds += et_pred * w[1]
            preds += hgb_pred * w[2]
            # Ridge: use decision function converted to prob
            ridge_d = ridge.decision_function(X1_test_s)
            ridge_prob = expit(ridge_d / 0.8958)  # temperature
            ridge_prob = np.clip(ridge_prob, 0.005, 0.995)
            preds += ridge_prob * w[3]
            
            preds = np.clip(preds, 0.005, 0.995)
            
            auc = roc_auc_score(y[test_idx], lr.predict_proba(X1_test_s)[:, 1])
            ll = log_loss(y[test_idx], lr.predict_proba(X1_test_s)[:, 1])
            
            fold_aucs.append(auc)
            fold_lls.append(ll)
            
        fold_results.append({
            'fold': nfolds,
            'mean_auc': np.mean(fold_aucs),
            'mean_ll': np.mean(fold_lls),
            'auc_std': np.std(fold_aucs),
            'll_std': np.std(fold_lls)
        })
    
    results_a.append({
        'seed': si,
        'fold_results': fold_results
    })

print("Configuration A (n_sel=58, C=1.312, alpha=8.108) completed")
mean_auc_a = np.mean([r['mean_auc'] for r in results_a])
mean_ll_a = np.mean([r['mean_ll'] for r in results_a])
auc_std_a = np.mean([r['auc_std'] for r in results_a])
ll_std_a = np.mean([r['ll_std'] for r in results_a])
print(f"Mean OOF AUC: {mean_auc_a:.4f} ± {auc_std_a:.4f}")
print(f"Mean OOF LL: {mean_ll_a:.4f} ± {ll_std_a:.4f}")

# ============================================================
# CONFIGURATION B: n_sel=44, LR C=1.31, Ridge alpha=0.1
# ============================================================
print()
print("=" * 60)
print("CONFIGURATION B: n_sel=44, LR C=1.31, Ridge alpha=0.1")
print("=" * 60)

results_b = []
for si, seed in enumerate(SEEDS):
    fold_results = []
    for nfolds in [5]:
        fold_aucs = []
        fold_lls = []
        for train_idx, test_idx in StratifiedGroupKFold(nfolds, shuffle=True, random_state=seed).split(X1, y, groups):
            # Feature selection inside training fold only
            mi = mutual_info_classif(X1[train_idx], y[train_idx], random_state=seed)
            sel = np.argsort(mi)[::-1][:44]
            
            # Scale
            sc = StandardScaler().fit(X1[train_idx])
            X1_train_s = sc.transform(X1[train_idx])
            X1_test_s = sc.transform(X1[test_idx])
            
            # LR on scaled features
            lr = LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(X1_train_s, y[train_idx])
            lr_pred = lr.predict_proba(X1_test_s)[:, 1]
            
            # Ridge on scaled features
            ridge = RidgeClassifier(alpha=0.1, random_state=seed).fit(X1_train_s, y[train_idx])
            ridge_pred = ridge.decision_function(X1_test_s)
            
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
            
            # Simple blend
            w = np.array([0.081, 0.209, 0.216, 0.052])
            w = w / w.sum()
            
            preds = np.zeros(len(y[test_idx]))
            preds += lr.predict_proba(X1_test_s)[:, 1] * w[0]
            preds += et_pred * w[1]
            preds += hgb_pred * w[2]
            ridge_d = ridge.decision_function(X1_test_s)
            ridge_prob = expit(ridge_d / 0.8958)
            ridge_prob = np.clip(ridge_prob, 0.005, 0.995)
            preds += ridge_prob * w[3]
            
            preds = np.clip(preds, 0.005, 0.995)
            
            auc = roc_auc_score(y[test_idx], lr.predict_proba(X1_test_s)[:, 1])
            ll = log_loss(y[test_idx], lr.predict_proba(X1_test_s)[:, 1])
            
            fold_aucs.append(auc)
            fold_lls.append(ll)
            
        fold_results.append({
            'fold': nfolds,
            'mean_auc': np.mean(fold_aucs),
            'mean_ll': np.mean(fold_lls),
            'auc_std': np.std(fold_aucs),
            'll_std': np.std(fold_lls)
        })
    
    results_b.append({
        'seed': si,
        'fold_results': fold_results
    })

print("Configuration B (n_sel=44, C=1.31, alpha=0.1) completed")
mean_auc_b = np.mean([r['mean_auc'] for r in results_b])
mean_ll_b = np.mean([r['mean_ll'] for r in results_b])
auc_std_b = np.mean([r['auc_std'] for r in results_b])
ll_std_b = np.mean([r['ll_std'] for r in results_b])
print(f"Mean OOF AUC: {mean_auc_b:.4f} ± {auc_std_b:.4f}")
print(f"Mean OOF LL: {mean_ll_b:.4f} ± {ll_std_b:.4f}")

print()
print("=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"Config A (n_sel=58, C=1.312, alpha=8.108):")
print(f"  Mean OOF AUC: {np.mean([r['mean_auc'] for r in results_a]):.4f}")
print(f"  Mean OOF LL: {np.mean([r['mean_ll'] for r in results_a]):.4f}")
print()
print(f"Config B (n_sel=44, C=1.31, alpha=0.1):")
print(f"  Mean OOF AUC: {mean_auc_b:.4f}")
print(f"  Mean OOF LL: {mean_ll_b:.4f}")
print()
if mean_ll_b < np.mean([r['mean_ll'] for r in results_a]):
    print("Config B wins on LL")
else:
    print("Config A wins on LL")
if mean_auc_b > np.mean([r['mean_auc'] for r in results_a]):
    print("Config B wins on AUC")
else:
    print("Config A wins on AUC")