"""Optimize blend weights for top CNN models + SBR."""
import numpy as np
import os
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from scipy.optimize import minimize

y = np.load('v26_oof/y.npy')
groups = np.load('v26_oof/groups.npy', allow_pickle=True)

# Load key OOFs
v27_bal = np.load('balanced_oof_all_seeds.npy')  # (1362, 4) - seeds 42, 777, 2024, 100
v27_cnn = np.load('v27_cnn_oof/oof.npy')
v26_fold_r42 = np.load('submission_v26/weights/fold_oof_r_42.npy')
v26_fold_r777 = np.load('submission_v26/weights/fold_oof_r_777.npy')
v26_fold_r2024 = np.load('submission_v26/weights/fold_oof_r_2024.npy')
v26_fold_r100 = np.load('submission_v26/weights/fold_oof_r_100.npy')
v26_fold_big1984 = np.load('submission_v26/weights/fold_oof_big_1984.npy')
v26_fold_big2025 = np.load('submission_v26/weights/fold_oof_big_2025.npy')
v40 = np.load('v40_honest_oof.npy')
v42 = np.load('v42_temp_honest_oof.npy')
exp_bce = np.load('exp_oof_60ep_bce.npy')

def ll(y, p):
    return log_loss(y, np.clip(p, 1e-7, 1-1e-7))

def auc(y, p):
    return roc_auc_score(y, p)

# Individual metrics
models = {
    'v27_bal_42': v27_bal[:, 0],
    'v27_bal_777': v27_bal[:, 1],
    'v27_bal_2024': v27_bal[:, 2],
    'v27_bal_100': v27_bal[:, 3],
    'v26_r42': v26_fold_r42,
    'v26_r777': v26_fold_r777,
    'v26_r2024': v26_fold_r2024,
    'v26_r100': v26_fold_r100,
    'v26_big1984': v26_fold_big1984,
    'v26_big2025': v26_fold_big2025,
    'v40': v40,
    'v42': v42,
    'exp_bce': exp_bce,
}

print("Individual model metrics:")
for name, oof in sorted(models.items(), key=lambda x: -auc(y, x[1])):
    print(f"  {name:16s}: AUC={auc(y,oof):.4f}, LL={ll(y,oof):.4f}")

# Optimize weights for top models
# Use top-7 most diverse strong models
top_models = ['v27_bal_42', 'v27_bal_777', 'v27_bal_2024', 'v27_bal_100',
              'v26_r42', 'v26_r777', 'v26_r2024', 'v26_r100',
              'v26_big1984', 'v26_big2025', 'v40', 'v42', 'exp_bce']

X = np.column_stack([models[n] for n in top_models])
print(f"\nStacking {len(top_models)} models")

# Grid search: weights that sum to 1
from itertools import product

print("\n--- Brute force top combinations ---")
best_ll = 999
best_w = None
best_auc = 0

# Use scipy optimize
def neg_ll(w):
    w = np.abs(w)
    w = w / w.sum()
    blend = X @ w
    return ll(y, blend)

# Start with equal weights
w0 = np.ones(len(top_models)) / len(top_models)
result = minimize(neg_ll, w0, method='Nelder-Mead', options={'maxiter': 10000})
w_opt = np.abs(result.x)
w_opt = w_opt / w_opt.sum()
blend_opt = X @ w_opt
print(f"Optimized weights:")
for name, w in sorted(zip(top_models, w_opt), key=lambda x: -x[1]):
    if w > 0.001:
        print(f"  {name:16s}: {w:.4f}")
print(f"  Optimized: AUC={auc(y,blend_opt):.4f}, LL={ll(y,blend_opt):.4f}")

# Also try temperature-scaled version
print("\n--- Temperature scaling on optimized blend ---")
for T in np.arange(0.5, 1.5, 0.05):
    logits = np.log(blend_opt / (1 - blend_opt))
    scaled = 1 / (1 + np.exp(-logits / T))
    l = ll(y, scaled)
    a = auc(y, scaled)
    if l < best_ll:
        best_ll = l
        best_auc = a
        best_T = T
        best_blend = scaled.copy()

print(f"  Best T={best_T:.2f}: AUC={best_auc:.4f}, LL={best_ll:.4f}")

# Try 2-model combinations (strong diverse pairs)
print("\n--- Best 2-model blends ---")
for i, n1 in enumerate(top_models):
    for j, n2 in enumerate(top_models):
        if j <= i:
            continue
        X2 = np.column_stack([models[n1], models[n2]])
        w0 = np.array([0.5, 0.5])
        def neg2(w):
            w = np.abs(w); w = w/w.sum()
            return ll(y, X2 @ w)
        r = minimize(neg2, w0, method='Nelder-Mead')
        w2 = np.abs(r.x); w2 = w2/w2.sum()
        blend2 = X2 @ w2
        a2 = auc(y, blend2)
        l2 = ll(y, blend2)
        if l2 < 0.30:
            print(f"  {n1}+{n2}: w=[{w2[0]:.3f},{w2[1]:.3f}] AUC={a2:.4f}, LL={l2:.4f}")

# Try combinations including SBR (v23-style deep + SBR)
print("\n--- Deep + SBR combo ---")
v26_oof = np.load('submission_v26/weights/fold_oof.npy')  # v26 overall OOF
print(f"v26 overall OOF: AUC={auc(y,v26_oof):.4f}, LL={ll(y,v26_oof):.4f}")

# Try all 3-model combos of top-3 + v26 overall
X3 = np.column_stack([v27_bal.mean(axis=1), v26_r2024, v40])
for T in [0.5, 0.6, 0.7, 0.77, 0.8, 0.9, 1.0]:
    logits = np.log(np.clip(X3.mean(axis=1), 1e-7, 1-1e-7) / 
                     (1 - np.clip(X3.mean(axis=1), 1e-7, 1-1e-7)))
    scaled = 1 / (1 + np.exp(-logits / T))
    l = ll(y, scaled)
    a = auc(y, scaled)
    print(f"  v27bal+v26r2024+v40 avg, T={T:.2f}: AUC={a:.4f}, LL={l:.4f}")
