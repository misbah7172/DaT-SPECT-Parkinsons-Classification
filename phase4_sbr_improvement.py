"""Phase 4: Improve SBR models and optimize final blend."""
import numpy as np
import json
import pickle
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import lightgbm as lgb

# Load data
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
deep_raw = np.load(r'E:\DaT\submission_v26\weights\fold_oof.npy')

# Load SBR features
sbr_features = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv')
print(f"SBR features shape: {sbr_features.shape}")
print(f"Columns: {sbr_features.columns.tolist()}")

# Load voxel geometry
voxel_geom = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
print(f"Voxel geometry shape: {voxel_geom.shape}")

# Merge features
merged = sbr_features.merge(voxel_geom, left_index=True, right_index=True, how='left')
print(f"Merged shape: {merged.shape}")

# Prepare features
drop_cols = [c for c in merged.columns if merged[c].dtype == object]
feature_cols = [c for c in merged.columns if c not in drop_cols]
X_sbr = merged[feature_cols].fillna(0).values.astype(float)

print(f"SBR feature matrix: {X_sbr.shape}")

# Add PVE correction
if 'voxel_vol' in merged.columns:
    PVE_REF = 15.625  # 2.5^3
    voxel_vol = merged['voxel_vol'].values
    fac = (PVE_REF / np.clip(voxel_vol, 1, None)) ** (1/3)
    # Apply to SBR columns (approximately the first half of features)
    n_feat = X_sbr.shape[1]
    X_sbr_pve = X_sbr.copy()
    for i in range(min(n_feat, 89)):  # first 89 are SBR features
        X_sbr_pve[:, i] = X_sbr[:, i] * fac
    print(f"PVE correction applied")

# Add derived features
print(f"\nAdding derived features...")
eps = 1e-8

# Asymmetry features from SBR columns
if 'sbr_lp' in merged.columns and 'sbr_rp' in merged.columns:
    lp = merged['sbr_lp'].values
    rp = merged['sbr_rp'].values
    lc = merged['sbr_lc'].values if 'sbr_lc' in merged.columns else np.zeros(len(y))
    rc = merged['sbr_rc'].values if 'sbr_rc' in merged.columns else np.zeros(len(y))
    
    put_asym = (lp - rp) / (lp + rp + eps)
    cau_asym = (lc - rc) / (lc + rc + eps)
    total_sbr = lp + rp + lc + rc
    left_total = lp + lc
    right_total = rp + rc
    left_ratio = left_total / (total_sbr + eps)
    right_ratio = right_total / (total_sbr + eps)
    pc_ratio = (lp + rp) / (lc + rc + eps)
    
    derived = np.column_stack([
        put_asym, cau_asym, total_sbr, left_total, right_total,
        left_ratio, right_ratio, pc_ratio,
    ])
    X_sbr_final = np.column_stack([X_sbr_pve, derived])
else:
    X_sbr_final = X_sbr_pve

print(f"Final feature matrix: {X_sbr_final.shape}")

# Feature selection with mutual information
mi_scores = mutual_info_classif(X_sbr_final, y, random_state=42)
top_k = min(50, X_sbr_final.shape[1])
top_idx = np.argsort(mi_scores)[-top_k:]
X_selected = X_sbr_final[:, top_idx]
print(f"Selected {top_k} features by MI")

# 5-fold CV for SBR models
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def train_sbr_models(X, y, groups, model_name, model_factory):
    oof = np.zeros(len(y))
    for trn_idx, val_idx in sgkf.split(X, y, groups):
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn = y[trn_idx]
        
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X_trn)
        X_val_s = scaler.transform(X_val)
        
        model = model_factory()
        model.fit(X_trn_s, y_trn)
        
        if hasattr(model, 'predict_proba'):
            oof[val_idx] = model.predict_proba(X_val_s)[:, 1]
        else:
            # For Ridge, convert decision function to probability
            from scipy.special import expit
            oof[val_idx] = expit(model.decision_function(X_val_s))
    
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
    return auc, ll, oof

print(f"\n" + "=" * 70)
print(f"SBR MODEL COMPARISON (with improved features)")
print(f"=" * 70)

models = [
    ('LogReg', lambda: LogisticRegression(C=0.5, max_iter=1000)),
    ('XGBoost', lambda: xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, 
                                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5)),
    ('LightGBM', lambda: lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5, verbose=-1)),
    ('ExtraTrees', lambda: ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5)),
    ('GradBoost', lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)),
]

sbr_oofs = {}
for name, factory in models:
    auc, ll, oof = train_sbr_models(X_selected, y, groups, name, factory)
    sbr_oofs[name] = oof
    print(f"  {name:15s}: AUC={auc:.4f}, LL={ll:.4f}")

# Mean of all SBR models
sbr_mean_new = np.mean(list(sbr_oofs.values()), axis=0)
sbr_auc = roc_auc_score(y, sbr_mean_new)
sbr_ll = log_loss(y, np.clip(sbr_mean_new, 1e-7, 1-1e-7))
print(f"\n  SBR Ensemble:   AUC={sbr_auc:.4f}, LL={sbr_ll:.4f}")

# Load original SBR OOF for comparison
with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names, oofs = A['names'], np.column_stack(A['oof'])
idx = [i for i, n in enumerate(names) if n in ('sbr_lr', 'sbr_xgb', 'sbr_lgb', 'sbr_et', 'sbr_ridge')]
sbr_old = oofs[:, idx].mean(axis=1)
sbr_old_auc = roc_auc_score(y, sbr_old)
sbr_old_ll = log_loss(y, np.clip(sbr_old, 1e-7, 1-1e-7))
print(f"  Old SBR:        AUC={sbr_old_auc:.4f}, LL={sbr_old_ll:.4f}")

# Optimize blend with new SBR
print(f"\n" + "=" * 70)
print(f"BLEND OPTIMIZATION")
print(f"=" * 70)

def blend_loss(params, deep, sbr, y):
    wd, T = params
    blend = np.clip(wd * deep + (1 - wd) * sbr, 1e-7, 1-1e-7)
    p = np.clip(expit(logit(blend) / T), 1e-7, 1-1e-7)
    return log_loss(y, p)

# Blend with new SBR
best_new = None
for init in ([0.8, 0.7], [0.85, 0.8], [0.75, 0.6], [0.9, 0.5]):
    r = minimize(blend_loss, init, args=(deep_raw, sbr_mean_new, y), 
                method='Nelder-Mead', options={'maxiter': 500})
    if best_new is None or r.fun < best_new.fun:
        best_new = r

wd_new, T_new = best_new.x
blend_new = np.clip(wd_new * deep_raw + (1 - wd_new) * sbr_mean_new, 1e-7, 1-1e-7)
p_new = np.clip(expit(logit(blend_new) / T_new), 1e-7, 1-1e-7)
new_auc = roc_auc_score(y, p_new)
new_ll = log_loss(y, p_new)
print(f"  New blend: w_deep={wd_new:.4f}, w_sbr={1-wd_new:.4f}, T={T_new:.4f}")
print(f"             AUC={new_auc:.4f}, LL={new_ll:.4f}")

# Blend with old SBR
best_old = None
for init in ([0.8, 0.7], [0.85, 0.8], [0.75, 0.6], [0.9, 0.5]):
    r = minimize(blend_loss, init, args=(deep_raw, sbr_old, y),
                method='Nelder-Mead', options={'maxiter': 500})
    if best_old is None or r.fun < best_old.fun:
        best_old = r

wd_old, T_old = best_old.x
blend_old = np.clip(wd_old * deep_raw + (1 - wd_old) * sbr_old, 1e-7, 1-1e-7)
p_old = np.clip(expit(logit(blend_old) / T_old), 1e-7, 1-1e-7)
old_auc = roc_auc_score(y, p_old)
old_ll = log_loss(y, p_old)
print(f"  Old blend: w_deep={wd_old:.4f}, w_sbr={1-wd_old:.4f}, T={T_old:.4f}")
print(f"             AUC={old_auc:.4f}, LL={old_ll:.4f}")

# Current best
with open(r'E:\DaT\submission_v26\weights\ship_final.json', 'r') as f:
    ship = json.load(f)
print(f"  Current:        AUC={ship['final_auc']:.4f}, LL={ship['final_ll']:.4f}")

# Summary
print(f"\n" + "=" * 70)
print(f"SUMMARY")
print(f"=" * 70)
print(f"""
SBR Improvement: {sbr_old_ll:.4f} -> {sbr_ll:.4f} (delta={sbr_ll - sbr_old_ll:.4f})
Blend Improvement: {ship['final_ll']:.4f} -> {new_ll:.4f} (delta={new_ll - ship['final_ll']:.4f})

Recommendation: {'Update blend with new SBR' if new_ll < ship['final_ll'] else 'Keep current blend'}
""")
