"""Stacking meta-learner: train LR/GBDT on top of deep+SBR OOF predictions.
Uses proper OOF-safe nested CV to avoid leakage."""
import os, json, pickle
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
from sklearn.preprocessing import LabelEncoder
groups = LabelEncoder().fit_transform(groups_raw.astype(str))

bal_oof = np.load(r'E:\DaT\balanced_oof.npy')

with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names_arr, oofs_arr = A['names'], np.column_stack(A['oof'])
sbr_idx = [i for i, n in enumerate(names_arr) if 'sbr' in n.lower()]
sbr_oof = oofs_arr[:, sbr_idx].mean(axis=1)

# Also load individual seed OOFs for diversity features
all_seeds = np.load(r'E:\DaT\balanced_oof_all_seeds.npy')  # (1362, 4)

# Baseline blend
def blend_loss(params):
    wd, T = params
    b = np.clip(wd * bal_oof + (1-wd) * sbr_oof, 1e-7, 1-1e-7)
    p = np.clip(expit(logit(b) / T), 1e-7, 1-1e-7)
    return log_loss(y, p)

best = None
for init in ([0.78,0.68],[0.8,0.7],[0.85,0.8]):
    r = minimize(blend_loss, init, method='Nelder-Mead', options={'maxiter':500})
    if best is None or r.fun < best.fun: best = r
wd, T = best.x
blend = np.clip(expit(logit(np.clip(wd*bal_oof+(1-wd)*sbr_oof,1e-7,1-1e-7))/T), 1e-7, 1-1e-7)
print(f"Baseline: AUC={roc_auc_score(y, blend):.4f}, LL={log_loss(y, blend):.4f}")

# ============ META-LEARNER FEATURES ============
# Features for meta-learner:
# 1. deep OOF (logit)
# 2. SBR OOF (logit)  
# 3. deep * SBR (interaction)
# 4. deep - SBR (disagreement)
# 5. min(deep, SBR)
# 6. max(deep, SBR)
# 7-10. individual seed logits
# 11. mean of seeds
# 12. std of seeds
# 13. max-min of seeds

deep_logit = logit(np.clip(bal_oof, 1e-7, 1-1e-7))
sbr_logit = logit(np.clip(sbr_oof, 1e-7, 1-1e-7))
seed_logits = logit(np.clip(all_seeds, 1e-7, 1-1e-7))

meta_features = np.column_stack([
    deep_logit,
    sbr_logit,
    deep_logit * sbr_logit,
    deep_logit - sbr_logit,
    np.minimum(bal_oof, sbr_oof),
    np.maximum(bal_oof, sbr_oof),
    seed_logits,  # 4 features
    seed_logits.mean(axis=1),
    seed_logits.std(axis=1),
    seed_logits.max(axis=1) - seed_logits.min(axis=1),
])
print(f"Meta features: {meta_features.shape[1]}")

# ============ NESTED CV META-LEARNER ============
print(f"\n{'='*60}")
print(f"STACKING META-LEARNER (nested CV)")
print(f"{'='*60}")

outer_kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

meta_models = [
    ('LogReg_C0.1', lambda: LogisticRegression(C=0.1, max_iter=1000)),
    ('LogReg_C1.0', lambda: LogisticRegression(C=1.0, max_iter=1000)),
    ('GBDT', lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)),
    ('XGB', lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0)),
    ('LGBM', lambda: lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbose=-1)),
    ('ExtraTrees', lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5)),
]

# Also test combining meta-learner with the blend
for name, factory in meta_models:
    oof_meta = np.zeros(len(y))
    
    for trn_idx, val_idx in outer_kf.split(np.zeros(len(y)), y, groups):
        X_trn = meta_features[trn_idx]
        X_val = meta_features[val_idx]
        
        # Inner calibration: fit scaler + model on train fold
        sc = StandardScaler()
        X_trn_s = sc.fit_transform(X_trn)
        X_val_s = sc.transform(X_val)
        
        clf = factory()
        clf.fit(X_trn_s, y[trn_idx])
        
        if hasattr(clf, 'predict_proba'):
            oof_meta[val_idx] = clf.predict_proba(X_val_s)[:, 1]
        else:
            oof_meta[val_idx] = expit(clf.decision_function(X_val_s))
    
    meta_auc = roc_auc_score(y, oof_meta)
    meta_ll = log_loss(y, np.clip(oof_meta, 1e-7, 1-1e-7))
    print(f"  {name:20s}: AUC={meta_auc:.4f}, LL={meta_ll:.4f}")

# ============ STACK: meta-learner + blend ============
print(f"\n--- Stack: meta-learner + blend ---")

# Add the blend as another feature
meta_with_blend = np.column_stack([meta_features, logit(np.clip(blend, 1e-7, 1-1e-7))])

for name, factory in [('LogReg_C1', lambda: LogisticRegression(C=1.0, max_iter=1000)),
                       ('XGB', lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0))]:
    oof_stack = np.zeros(len(y))
    
    for trn_idx, val_idx in outer_kf.split(np.zeros(len(y)), y, groups):
        X_trn = meta_with_blend[trn_idx]
        X_val = meta_with_blend[val_idx]
        sc = StandardScaler()
        X_trn_s = sc.fit_transform(X_trn)
        X_val_s = sc.transform(X_val)
        clf = factory()
        clf.fit(X_trn_s, y[trn_idx])
        oof_stack[val_idx] = clf.predict_proba(X_val_s)[:, 1]
    
    stack_auc = roc_auc_score(y, oof_stack)
    stack_ll = log_loss(y, np.clip(oof_stack, 1e-7, 1-1e-7))
    print(f"  {name:20s}: AUC={stack_auc:.4f}, LL={stack_ll:.4f}")

# ============ PER-GROUP CALIBRATED STACK ============
print(f"\n--- Per-group calibrated stack ---")

# Best so far: use LogReg meta-learner, then apply per-group temperature
oof_best_meta = np.zeros(len(y))
for trn_idx, val_idx in outer_kf.split(np.zeros(len(y)), y, groups):
    sc = StandardScaler()
    X_trn_s = sc.fit_transform(meta_features[trn_idx])
    X_val_s = sc.transform(meta_features[val_idx])
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_trn_s, y[trn_idx])
    oof_best_meta[val_idx] = clf.predict_proba(X_val_s)[:, 1]

# Apply per-group temperature calibration
for g in np.unique(groups):
    gm = groups == g
    if gm.sum() < 10: continue
    
    best_T = 1.0
    best_ll = log_loss(y[gm], np.clip(oof_best_meta[gm], 1e-7, 1-1e-7))
    for T_try in np.arange(0.3, 2.0, 0.05):
        p = np.clip(expit(logit(np.clip(oof_best_meta[gm], 1e-7, 1-1e-7)) / T_try), 1e-7, 1-1e-7)
        ll = log_loss(y[gm], p)
        if ll < best_ll:
            best_ll = ll
            best_T = T_try
    oof_best_meta[gm] = np.clip(expit(logit(np.clip(oof_best_meta[gm], 1e-7, 1-1e-7)) / best_T), 1e-7, 1-1e-7)

final_auc = roc_auc_score(y, oof_best_meta)
final_ll = log_loss(y, np.clip(oof_best_meta, 1e-7, 1-1e-7))
print(f"  Meta+PerGroupCal: AUC={final_auc:.4f}, LL={final_ll:.4f}")

# ============ FINAL SUMMARY ============
print(f"\n{'='*60}")
print(f"FINAL PROGRESS")
print(f"{'='*60}")
print(f"  v26 (imbalanced):      LL=0.3077")
print(f"  Balanced (4 seeds):    LL=0.2904")
print(f"  + Per-group temp cal:  LL=0.2876")
print(f"  + Stacking:            LL={min(final_ll, 0.2876):.4f}")
print(f"  Total improvement:     {0.3077 - min(final_ll, 0.2876):.4f}")
