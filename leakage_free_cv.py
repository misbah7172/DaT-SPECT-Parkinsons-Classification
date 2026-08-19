import warnings
warnings.filterwarnings("ignore")
import sys, os, json, numpy as np, pandas as pd, time
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, brier_loss, calibration_curve
import joblib

sys.path.insert(0, "E:/DaT/src")
from sbr_extractor import FEATURE_COLUMNS
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS

PVE_REF = 15.625
RAW1 = list(FEATURE_COLUMNS)
RAW2 = list(PHYS_FEATURE_COLUMNS)

# Load data
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

def build_matrix(raw, cols):
    cols = [c for c in raw_cols if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1), nan=0.0)

X1 = build_matrix(merged, RAW1)
p_cols = [f"p_{c}" for c in RAW2 if f"p_{c}" in merged.columns]
X2 = build_matrix(merged[p_cols].rename(columns={c: c[2:] for c in p_cols}), RAW2)
Xs = (X1, X2)

# ============================================================
# CONFIGURATION A: n_sel=58, LR C=1.312, Ridge alpha=8.108
# ============================================================
print("=" * 60)
print("CONFIGURATION A: n_sel=58, LR C=1.312, Ridge alpha=8.108")
print("=" * 60)

results_a = []
for seed in [42]:
    aucs = []
    lls = []
    briers = []
    eces = []
    
    # Outer 5-fold CV
    for train_idx, test_idx in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X1, y, groups):
        # INNERS: 5-fold internal CV for hyperparameter tuning
        # But for this test, we'll use fixed hyperparameters
        # Feature selection INSIDE training fold
        mi = mutual_info_classif(X1[train_idx], y[train_idx], random_state=seed)
        sel = np.argsort(mi)[::-1][:58]
        
        # Scale - ONLY on training fold
        sc = StandardScaler().fit(X1[train_idx])
        X1_train_s = sc.transform(X1[train_idx])
        X1_test_s = sc.transform(X1[test_idx])
        
        # Ridge on selected features
        ridge = RidgeClassifier(alpha=8.108, random_state=seed).fit(X1_train_s[:, sel], y[train_idx])
        ridge_d = ridge.decision_function(X1_test_s[:, sel])
        
        # LR on selected features
        lr = LogisticRegression(C=1.312, max_iter=3000, random_state=seed).fit(X1_train_s[:, sel], y[train_idx])
        lr_pred = lr.predict_proba(X1_test_s[:, sel])[:, 1]
        
        # ET on selected features
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
        
        preds = lr.predict_proba(X1_test_s[:, sel])[:, 1] * w[0] + et_pred * w[1] + hgb_pred * w[2]
        ridge_d = ridge.decision_function(X1_test_s[:, sel])
        ridge_prob = expit(ridge_d / 0.8958)
        ridge_prob = np.clip(ridge_prob, 0.005, 0.995)
        preds += ridge_prob * w[3]
        preds = np.clip(preds, 0.005, 0.995)
        
        auc = roc_auc_score(y[test_idx], lr.predict_proba(X1_test_s[:, sel])[:, 1])
        ll = log_loss(y[test_idx], lr.predict_proba(X1_test_s[:, sel])[:, 1])
        
        aucs.append(auc)
        lls.append(ll)
        
    # Calculate Brier and ECE
    # Calibration curve
    prob_true, prob_pred = calibration_curve(lr.predict_proba(X1_test_s[:, sel])[:, 1], y[test_idx], n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred) * (1.0 / 10))
    
    results_a.append({
        'auc': np.mean(aucs),
        'll': np.mean(lls),
        'auc_std': np.std(aucs),
        'll_std': np.std(lls),
        'ece': np.mean(eces) if 'eces' in dir() else np.mean([np.abs(np.mean(np.random.rand(10)) - np.mean(np.random.rand(10)))])  # placeholder
    })

print(f"Config A (n_sel=58, C=1.312, alpha=8.108):")
print(f"  Mean OOF AUROC: {np.mean([r['auc'] for r in results_a]):.4f} ± {np.mean([r['auc_std'] for r in results_a]):.4f}")
print(f"  Mean OOF LogLoss: {np.mean([r['ll'] for r in results_a]):.4f} ± {np.std([r['ll'] for r in results_a]):.4f}")
print(f"  Mean Brier: N/A (need to compute)")
print(f"  Mean ECE: N/A (need to compute)")

# ============================================================
# CONFIGURATION B: n_sel=44, LR C=1.31, Ridge alpha=0.1
# ============================================================
print()
print("=" * 60)
print("CONFIGURATION B: n_sel=44, LR C=1.31, Ridge alpha=0.1")
print("=" * 60)

results_b = []
for seed in [42]:
    aucs_b = []
    lls_b = []
    for train_idx, test_idx in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X1, y, groups):
        # Feature selection INSIDE training fold
        mi = mutual_info_classif(X1[train_idx], y[train_idx], random_state=seed)
        sel = np.argsort(mi)[::-1][:44]
        
        # Scale - ONLY on training fold
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
print(f"  Mean OOF AUROC: {np.mean(aucs_b):.4f} ± {np.std(aucs_b):.4f}")
print(f"  Mean OOF LogLoss: {np.mean(lls_b):.4f} ± {np.std(lls_b):.4f}")
print(f"  Mean Brier: N/A (compute separately)")
print(f"  Mean ECE: N/A (need to compute)")

print()
print("=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"Config A (n_sel=58, C=1.312, alpha=8.108):")
print(f"  Mean OOF AUROC: {np.mean([r['auc'] for r in results_a]):.4f} ± {np.std([r['auc'] for r in results_a]):.4f}")
print(f"  Mean OOF LogLoss: {np.mean([r['ll'] for r in results_a]):.4f} ± {np.std([r['ll'] for r in results_a]):.4f}")
print()
print(f"Config B (n_sel=44, C=1.31, alpha=0.1):")
print(f"  Mean OOF AUROC: {np.mean(aucs_b):.4f} ± {np.std(aucs_b):.4f}")
print(f"  Mean OOF LogLoss: {np.mean(lls_b):.4f} ± {np.std(lls_b):.4f}")
print()
if np.mean(aucs_b) > np.mean([r['auc'] for r in results_a]):
    print("Config B wins on AUROC")
else:
    print("Config A wins on AUROC")
if np.mean(lls_b) < np.mean([r['ll'] for r in results_a]):
    print("Config B wins on LogLoss")
else:
    print("Config A wins on LogLoss")