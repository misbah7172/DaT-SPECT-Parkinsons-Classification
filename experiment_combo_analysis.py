"""Experiment 2b: Analyze individual combo OOFs and try optimal weighting."""
import json, pickle, numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

W = r'E:\DaT\submission_v26\weights'
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
eps = 1e-6

with open(W + r'\ship_final.json', 'r') as f:
    ship = json.load(f)

import os, glob

combo_files = sorted(glob.glob(W + r'\fold_oof_*.npy'))
print(f"Found {len(combo_files)} combo OOF files\n")

combo_oofs = {}
for cf in combo_files:
    name = os.path.basename(cf).replace('fold_oof_', '').replace('.npy', '')
    oof = np.load(cf)
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, eps, 1-eps))
    combo_oofs[name] = oof
    print(f"  {name:20s} AUC={auc:.4f}  LL={ll:.4f}")

# Ensemble mean
ensemble = np.mean(list(combo_oofs.values()), axis=0)
print(f"\n  {'ENSEMBLE (equal)':20s} AUC={roc_auc_score(y, ensemble):.4f}  LL={log_loss(y, ensemble):.4f}")
print(f"  {'Current blend':20s} AUC={ship['final_auc']:.4f}  LL={ship['final_ll']:.4f}")

# Pairwise correlation matrix
names = list(combo_oofs.keys())
n = len(names)
print(f"\nPairwise Correlation Matrix:")
header = "         " + "  ".join([f"{n[:6]:>6s}" for n in names])
print(header)
for i in range(n):
    row = f"{names[i][:8]:8s}"
    for j in range(n):
        corr = np.corrcoef(combo_oofs[names[i]], combo_oofs[names[j]])[0, 1]
        row += f"  {corr:.4f}"
    print(row)

# Optimal weighting
combo_stack = np.column_stack(list(combo_oofs.values()))
n_combos = len(combo_oofs)

def combo_loss(params):
    w = np.array(params)
    w = w / w.sum()
    blend = np.clip(combo_stack @ w, eps, 1-eps)
    return log_loss(y, blend)

init_w = [1.0/n_combos] * n_combos
result = minimize(combo_loss, init_w, method='Nelder-Mead',
                 options={'maxiter': 10000, 'xatol': 1e-7})
opt_w = np.array(result.x)
opt_w = opt_w / opt_w.sum()
opt_blend = np.clip(combo_stack @ opt_w, eps, 1-eps)

print(f"\nOptimal Combo Weights:")
for i, name in enumerate(names):
    print(f"  {name:20s} {opt_w[i]:.4f}")
print(f"\n  Optimal blend: AUC={roc_auc_score(y, opt_blend):.4f}  LL={log_loss(y, opt_blend):.4f}")

# Try temperature-scaled optimal blend
def temp_blend_loss(params):
    T = params[0]
    p = np.clip(expit(logit(np.clip(opt_blend, eps, 1-eps)) / T), eps, 1-eps)
    return log_loss(y, p)

best_T = 1.0
best_ll = temp_loss([1.0], y, opt_blend) if False else 999
for T_init in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]:
    r = minimize(temp_blend_loss, [T_init], method='Nelder-Mead', options={'maxiter': 200})
    if r.fun < best_ll:
        best_ll = r.fun
        best_T = r.x[0]

opt_cal = np.clip(expit(logit(np.clip(opt_blend, eps, 1-eps)) / best_T), eps, 1-eps)
print(f"  Optimal + temp(T={best_T:.4f}): AUC={roc_auc_score(y, opt_cal):.4f}  LL={log_loss(y, opt_cal):.4f}")

# Group-based analysis: are certain combos better for certain sites?
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\nPer-Fold Analysis:")
fold_results = []
for fold, (trn_idx, val_idx) in enumerate(sgkf.split(ensemble, y, groups)):
    y_val = y[val_idx]
    for name, oof in combo_oofs.items():
        pass  # skip per-fold for speed
    
    # Ensemble fold performance
    e_val = ensemble[val_idx]
    opt_val = opt_blend[val_idx]
    print(f"  Fold {fold}: n={len(val_idx)}, pos={y_val.sum()}, "
          f"ensemble AUC={roc_auc_score(y_val, e_val):.4f}, "
          f"optimal AUC={roc_auc_score(y_val, opt_val):.4f}")

# Analysis: which combos are most/least correlated with errors
print(f"\nError Correlation Analysis:")
ensemble_pred = np.clip(ensemble, eps, 1-eps)
ensemble_errors = np.abs(y - ensemble_pred)
for name, oof in combo_oofs.items():
    error_corr = np.corrcoef(np.abs(y - np.clip(oof, eps, 1-eps)), ensemble_errors)[0, 1]
    pred_corr = np.corrcoef(oof, ensemble)[0, 1]
    print(f"  {name:20s} pred_corr={pred_corr:.4f}  error_corr={error_corr:.4f}")
