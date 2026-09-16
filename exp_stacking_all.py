"""Stacking ensemble from ALL available CNN OOF predictions.
These come from different architectures, training regimes, and seeds."""
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import json

y = np.load('v26_oof/y.npy')
groups = np.load('v26_oof/groups.npy', allow_pickle=True)
print(f"y: {len(y)}, prevalence: {y.mean():.3f}")

# Load ALL available CNN OOFs
oof_dict = {}

# v26 individual fold models (different architectures)
for name in ['fold_oof_r_42', 'fold_oof_r_777', 'fold_oof_r_2024', 'fold_oof_r_100',
             'fold_oof_big_1984', 'fold_oof_big_2025',
             'fold_oof_big_11', 'fold_oof_big_22', 'fold_oof_big_33',
             'fold_oof_big_44', 'fold_oof_big_55', 'fold_oof_big_66']:
    path = f'submission_v26/weights/{name}.npy'
    if os.path.exists(path):
        oof_dict[f'v26_{name}'] = np.load(path)

# v27 balanced 4-seed models
for i in range(4):
    arr = np.load('balanced_oof_all_seeds.npy')
    oof_dict[f'v27_bal_seed{i}'] = arr[:, i]

# v30 diverse CNN architectures
for name in ['cnn3d_oof', 'cnn3d_oof_sc', 'cnn3dr_oof', 'cnn3dr_oof_sc',
             'cnn25v2_oof', 'cnn25v2_oof_sc', 'cnn25s_oof', 'cnn25s_oof_sc',
             'cnn25r2_oof', 'cnn25r2_oof_sc', 'cnn25b_oof', 'cnn25b_oof_sc']:
    path = f'v30_oof/{name}.npy'
    if os.path.exists(path):
        oof_dict[f'v30_{name}'] = np.load(path)

# v27 CNN OOF (very conservative)
if os.path.exists('v27_cnn_oof/oof.npy'):
    oof_dict['v27_cnn'] = np.load('v27_cnn_oof/oof.npy')

# Root-level experiment OOFs
for name in ['v29_honest_oof', 'v31_honest_oof', 'v34_honest_oof',
             'v40_honest_oof', 'v41_honest_oof', 'v42_temp_honest_oof',
             'exp_oof_60ep_bce', 'exp_oof_60ep_focal']:
    path = f'{name}.npy'
    if os.path.exists(path):
        oof_dict[name] = np.load(path)

print(f"\nLoaded {len(oof_dict)} OOF predictions")
for k, v in sorted(oof_dict.items()):
    auc = roc_auc_score(y, v)
    ll = log_loss(y, np.clip(v, 1e-7, 1-1e-7))
    print(f"  {k}: AUC={auc:.4f}, LL={ll:.4f}, std={v.std():.4f}")

# Build feature matrix
oof_names = sorted(oof_dict.keys())
X = np.column_stack([oof_dict[n] for n in oof_names])
print(f"\nFeature matrix: {X.shape}")

# Also include interactions
X_log = np.log(np.clip(X, 1e-7, 1-1e-7) / (1 - np.clip(X, 1e-7, 1-1e-7)))  # logit
X_both = np.hstack([X, X_log])
print(f"Extended matrix (raw + logit): {X_both.shape}")

# Evaluate with balanced SGKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def eval_oof(oof, name=""):
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
    return auc, ll

# Baseline: simple average of all models
X_avg = X.mean(axis=1)
print(f"\n{'='*60}")
print(f"Simple average of {len(oof_names)} models: AUC={eval_oof(X_avg)[0]:.4f}, LL={eval_oof(X_avg)[1]:.4f}")

# Fold-level stacking
print(f"\n{'='*60}")
print("Fold-level stacking:")

meta_oof = np.zeros(len(y))

for fold_idx, (trn_idx, val_idx) in enumerate(sgkf.split(X_both, y, groups)):
    X_trn, X_val = X_both[trn_idx], X_both[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]
    
    # Try several meta-learners
    for meta_name, meta_model in [
        ('LR', LogisticRegression(C=1.0, max_iter=1000)),
        ('LR_C10', LogisticRegression(C=10.0, max_iter=1000)),
        ('LR_C01', LogisticRegression(C=0.1, max_iter=1000)),
        ('Ridge', Ridge(alpha=1.0)),
        ('GB', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)),
        ('HGB', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1)),
    ]:
        if meta_name == 'Ridge':
            # Ridge for regression
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
            m.fit(X_trn, y_trn)
            pred = np.clip(m.predict(X_val), 0, 1)
        elif meta_name in ('GB', 'HGB'):
            m = meta_model
            m.fit(X_trn, y_trn)
            pred = m.predict_proba(X_val)[:, 1]
        else:
            m = meta_model
            m.fit(X_trn, y_trn)
            pred = m.predict_proba(X_val)[:, 1]
        
        meta_oof[val_idx] = pred
    
    # Average across all meta-learners
    break  # Just check first fold

# Actually, let me do it properly with a single best meta-learner
print(f"\n--- Testing meta-learners with 5-fold stacking ---")

results = {}
for meta_name, meta_factory in [
    ('LR', lambda: LogisticRegression(C=1.0, max_iter=1000)),
    ('LR_C10', lambda: LogisticRegression(C=10.0, max_iter=1000)),
    ('LR_C01', lambda: LogisticRegression(C=0.1, max_iter=1000)),
    ('Ridge', lambda: Ridge(alpha=1.0)),
    ('Ridge_01', lambda: Ridge(alpha=0.1)),
    ('GB_100', lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)),
    ('GB_200', lambda: GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05)),
    ('HGB_100', lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1)),
    ('HGB_200', lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05)),
]:
    meta_oof = np.zeros(len(y))
    
    for trn_idx, val_idx in sgkf.split(X_both, y, groups):
        X_trn, X_val = X_both[trn_idx], X_both[val_idx]
        y_trn = y[trn_idx]
        
        m = meta_factory()
        if isinstance(m, Ridge):
            pred = np.clip(m.fit(X_trn, y_trn).predict(X_val), 0, 1)
        else:
            m.fit(X_trn, y_trn)
            pred = m.predict_proba(X_val)[:, 1]
        meta_oof[val_idx] = pred
    
    auc, ll = eval_oof(meta_oof)
    results[meta_name] = (auc, ll, meta_oof.copy())
    print(f"  {meta_name:12s}: AUC={auc:.4f}, LL={ll:.4f}")

# Find best
best_name = min(results, key=lambda k: results[k][1])
best_auc, best_ll, best_oof = results[best_name]
print(f"\n{'='*60}")
print(f"BEST: {best_name} AUC={best_auc:.4f}, LL={best_ll:.4f}")

# Also try feature selection: top-K most predictive models
print(f"\n--- Feature importance (individual AUC) ---")
importances = []
for i, name in enumerate(oof_names):
    auc = roc_auc_score(y, X[:, i])
    importances.append((auc, name, i))
importances.sort(reverse=True)
for auc, name, i in importances[:10]:
    print(f"  {name}: AUC={auc:.4f}")

# Try top-K selection
print(f"\n--- Top-K model selection ---")
for K in [5, 10, 15, 20, 30]:
    top_idx = [imp[2] for imp in importances[:K]]
    X_top = X_both[:, np.array(top_idx + [i + len(oof_names) for i in top_idx])]
    
    meta_oof = np.zeros(len(y))
    for trn_idx, val_idx in sgkf.split(X_top, y, groups):
        X_trn, X_val = X_top[trn_idx], X_top[val_idx]
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(X_trn, y[trn_idx])
        meta_oof[val_idx] = m.predict_proba(X_val)[:, 1]
    
    auc, ll = eval_oof(meta_oof)
    print(f"  Top-{K:2d}: AUC={auc:.4f}, LL={ll:.4f}")

# Save best OOF
np.save('stacking_best_oof.npy', best_oof)
print(f"\nSaved best OOF to stacking_best_oof.npy")

# Also try: only raw features (no logit)
print(f"\n--- Raw features only (no logit) ---")
for meta_name, meta_factory in [
    ('LR', lambda: LogisticRegression(C=1.0, max_iter=1000)),
    ('LR_C10', lambda: LogisticRegression(C=10.0, max_iter=1000)),
    ('Ridge', lambda: Ridge(alpha=1.0)),
]:
    meta_oof = np.zeros(len(y))
    for trn_idx, val_idx in sgkf.split(X, y, groups):
        m = meta_factory()
        if isinstance(m, Ridge):
            pred = np.clip(m.fit(X[trn_idx], y[trn_idx]).predict(X[val_idx]), 0, 1)
        else:
            m.fit(X[trn_idx], y[trn_idx])
            pred = m.predict_proba(X[val_idx])[:, 1]
        meta_oof[val_idx] = pred
    auc, ll = eval_oof(meta_oof)
    print(f"  {meta_name:12s}: AUC={auc:.4f}, LL={ll:.4f}")
