"""Phase 2b: Quick asymmetry feature analysis with simple classifiers."""
import numpy as np
import json
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Load data
X = np.load(r'E:\DaT\cnn3d\X_reg.npy')
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()

# Load existing deep OOF
deep_raw = np.load(r'E:\DaT\submission_v26\weights\fold_oof.npy')

# Load existing SBR OOF
import pickle
with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names, oofs = A['names'], np.column_stack(A['oof'])
idx = [i for i, n in enumerate(names) if n in ('sbr_lr', 'sbr_xgb', 'sbr_lgb', 'sbr_et', 'sbr_ridge')]
sbr_mean = oofs[:, idx].mean(axis=1)

print("=" * 70)
print("PHASE 2b: ASYMMETRY FEATURE ANALYSIS")
print("=" * 70)

# ROI positions in the cropped volume (34x40x42)
# Left putamen: x=[18:24], y=[10:19], z=[12:32]
LP = X[:, 18:24, 10:19, 12:32]
# Right putamen: x=[11:17], y=[11:19], z=[14:22]
RP = X[:, 11:17, 11:19, 14:22]
# Left caudate: x=[18:24], y=[20:30], z=[10:22]
LC = X[:, 18:24, 20:30, 10:22]
# Right caudate: x=[10:17], y=[20:30], z=[12:22]
RC = X[:, 10:17, 20:30, 12:22]

print(f"\nROI Shapes:")
print(f"  LP: {LP.shape}, RP: {RP.shape}, LC: {LC.shape}, RC: {RC.shape}")

# Compute mean intensities per ROI
eps = 1e-8
def roi_mean(vol):
    mask = vol > 0
    return np.array([vol[i][mask[i]].mean() if mask[i].any() else 0 for i in range(len(vol))])

lp_mean = roi_mean(LP)
rp_mean = roi_mean(RP)
lc_mean = roi_mean(LC)
rc_mean = roi_mean(RC)

print(f"\nROI Mean Intensities:")
print(f"  LP: {lp_mean.mean():.4f} +/- {lp_mean.std():.4f}")
print(f"  RP: {rp_mean.mean():.4f} +/- {rp_mean.std():.4f}")
print(f"  LC: {lc_mean.mean():.4f} +/- {lc_mean.std():.4f}")
print(f"  RC: {rc_mean.mean():.4f} +/- {rc_mean.std():.4f}")

# Compute asymmetry features
put_asym = (lp_mean - rp_mean) / (lp_mean + rp_mean + eps)
cau_asym = (lc_mean - rc_mean) / (lc_mean + rc_mean + eps)
left_total = lp_mean + lc_mean
right_total = rp_mean + rc_mean
total_sbr = left_total + right_total
left_ratio = left_total / (total_sbr + eps)
right_ratio = right_total / (total_sbr + eps)
pc_ratio = (lp_mean + rp_mean) / (lc_mean + rc_mean + eps)
abs_put_asym = np.abs(put_asym)
abs_cau_asym = np.abs(cau_asym)

# Background region
bg = X[:, 0:5, 0:5, 0:5]
bg_mean = np.array([bg[i].mean() for i in range(len(X))])
sb_ratio = total_sbr / (bg_mean + eps)

# Stack all features
features = np.column_stack([
    lp_mean, rp_mean, lc_mean, rc_mean,
    put_asym, cau_asym,
    left_total, right_total, total_sbr,
    left_ratio, right_ratio,
    pc_ratio,
    sb_ratio,
    abs_put_asym, abs_cau_asym,
])
feature_names = [
    'lp_mean', 'rp_mean', 'lc_mean', 'rc_mean',
    'put_asym', 'cau_asym',
    'left_total', 'right_total', 'total_sbr',
    'left_ratio', 'right_ratio',
    'pc_ratio', 'sb_ratio',
    'abs_put_asym', 'abs_cau_asym',
]

print(f"\nAsymmetry features shape: {features.shape}")

# Feature importance analysis
print(f"\nFeature Analysis (Normal vs Abnormal):")
for i, name in enumerate(feature_names):
    feat = features[:, i]
    pos_mean = feat[y == 1].mean()
    neg_mean = feat[y == 0].mean()
    diff = pos_mean - neg_mean
    print(f"  {name:20s}: Normal={neg_mean:.4f}, Abnormal={pos_mean:.4f}, Diff={diff:.4f}")

# Train simple classifiers on asymmetry features only
print(f"\n" + "=" * 70)
print(f"CLASSIFIER COMPARISON (Asymmetry Features Only)")
print(f"=" * 70)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_classifier(clf_name, clf_factory, X, y, groups):
    oof = np.zeros(len(y))
    for trn_idx, val_idx in sgkf.split(X, y, groups):
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn = y[trn_idx]
        
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X_trn)
        X_val_s = scaler.transform(X_val)
        
        clf = clf_factory()
        clf.fit(X_trn_s, y_trn)
        
        if hasattr(clf, 'predict_proba'):
            oof[val_idx] = clf.predict_proba(X_val_s)[:, 1]
        else:
            oof[val_idx] = clf.decision_function(X_val_s)
    
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
    brier = brier_score_loss(y, oof)
    return auc, ll, brier, oof

# Test classifiers
classifiers = [
    ('LogReg', lambda: LogisticRegression(C=1.0, max_iter=1000)),
    ('XGBoost', lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0)),
    ('LightGBM', lambda: lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbose=-1)),
    ('ExtraTrees', lambda: ExtraTreesClassifier(n_estimators=100, max_depth=5)),
]

results = []
for name, factory in classifiers:
    auc, ll, brier, oof = evaluate_classifier(name, factory, features, y, groups)
    results.append({'name': name, 'auc': auc, 'll': ll, 'brier': brier, 'oof': oof})
    print(f"  {name:15s}: AUC={auc:.4f}, LL={ll:.4f}, Brier={brier:.4f}")

# Find best asymmetry classifier
best = min(results, key=lambda x: x['ll'])
print(f"\n  Best asymmetry classifier: {best['name']} (LL={best['ll']:.4f})")

# Now test: can asymmetry features IMPROVE the deep+SBR blend?
print(f"\n" + "=" * 70)
print(f"BLEND WITH ASYMMETRY FEATURES")
print(f"=" * 70)

# Create meta-features: deep + sbr + asymmetry
meta_features = np.column_stack([
    deep_raw,
    sbr_mean,
    best['oof'],  # best asymmetry classifier output
    deep_raw * sbr_mean,  # interaction
    np.abs(deep_raw - sbr_mean),  # disagreement
])

# Train meta-learner
oof_meta = np.zeros(len(y))
for trn_idx, val_idx in sgkf.split(meta_features, y, groups):
    X_trn, X_val = meta_features[trn_idx], meta_features[val_idx]
    y_trn = y[trn_idx]
    
    scaler = StandardScaler()
    X_trn_s = scaler.fit_transform(X_trn)
    X_val_s = scaler.transform(X_val)
    
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X_trn_s, y_trn)
    oof_meta[val_idx] = lr.predict_proba(X_val_s)[:, 1]

meta_auc = roc_auc_score(y, oof_meta)
meta_ll = log_loss(y, np.clip(oof_meta, 1e-7, 1-1e-7))
print(f"  Deep + SBR + Asymmetry: AUC={meta_auc:.4f}, LL={meta_ll:.4f}")

# Current best
with open(r'E:\DaT\submission_v26\weights\ship_final.json', 'r') as f:
    ship = json.load(f)
print(f"  Current best blend:     AUC={ship['final_auc']:.4f}, LL={ship['final_ll']:.4f}")

# Temperature scaling on the best combination
def temp_loss(T, y, p):
    p_cal = np.clip(expit(logit(np.clip(p, 1e-7, 1-1e-7)) / T), 1e-7, 1-1e-7)
    return log_loss(y, p_cal)

best_T = 1.0
best_ll = temp_loss(1.0, y, oof_meta)
for T_init in np.arange(0.5, 1.5, 0.05):
    r = minimize(lambda t: temp_loss(t[0], y, oof_meta), [T_init], method='Nelder-Mead')
    if r.fun < best_ll:
        best_ll = r.fun
        best_T = r.x[0]

oof_meta_cal = np.clip(expit(logit(np.clip(oof_meta, 1e-7, 1e-7)) / best_T), 1e-7, 1-1e-7)
meta_cal_auc = roc_auc_score(y, oof_meta_cal)
meta_cal_ll = log_loss(y, oof_meta_cal)
print(f"  + Temperature scaling:  AUC={meta_cal_auc:.4f}, LL={meta_cal_ll:.4f} (T={best_T:.4f})")

# Summary
print(f"\n" + "=" * 70)
print(f"SUMMARY")
print(f"=" * 70)
print(f"""
Current best:    AUC={ship['final_auc']:.4f}, LL={ship['final_ll']:.4f}
Asymmetry only:  AUC={best['auc']:.4f}, LL={best['ll']:.4f}
Meta-learner:    AUC={meta_auc:.4f}, LL={meta_ll:.4f}
Meta + temp:     AUC={meta_cal_auc:.4f}, LL={meta_cal_ll:.4f}

Conclusion: {'Asymmetry features help!' if meta_ll < ship['final_ll'] else 'Current blend is better.'}
""")
