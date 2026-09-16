"""Experiment 1: Analyze deep OOF calibration and try temperature scaling."""
import json, pickle, sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

W = r'E:\DaT\submission_v26\weights'
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)

# Load deep OOF
deep_raw = np.load(W + r'\fold_oof.npy')

# Load SBR OOF
with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names, oofs = A['names'], np.column_stack(A['oof'])
idx = [i for i, n in enumerate(names) if n in ('sbr_lr', 'sbr_xgb', 'sbr_lgb', 'sbr_et', 'sbr_ridge')]
sbr_raw = oofs[:, idx].mean(axis=1)

eps = 1e-6

print("=" * 60)
print("EXPERIMENT 1: Deep OOF Calibration Analysis")
print("=" * 60)

# 1. Raw deep performance
deep_clipped = np.clip(deep_raw, eps, 1 - eps)
print(f"\n1. Raw Deep OOF:")
print(f"   AUC: {roc_auc_score(y, deep_clipped):.4f}")
print(f"   LL:  {log_loss(y, deep_clipped):.4f}")
print(f"   Brier: {brier_score_loss(y, deep_clipped):.4f}")
print(f"   Mean pred: {deep_clipped.mean():.4f}")
print(f"   Std pred:  {deep_clipped.std():.4f}")
print(f"   Min pred:  {deep_clipped.min():.4f}")
print(f"   Max pred:  {deep_clipped.max():.4f}")

# 2. Temperature scaling on deep OOF
def temp_loss(params, y, p):
    T = params[0]
    p_cal = np.clip(expit(logit(np.clip(p, eps, 1-eps)) / T), eps, 1-eps)
    return log_loss(y, p_cal)

best_T = 1.0
best_ll = temp_loss([1.0], y, deep_raw)
for T_init in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
    r = minimize(temp_loss, [T_init], args=(y, deep_raw), method='Nelder-Mead',
                 options={'maxiter': 200, 'xatol': 1e-5})
    if r.fun < best_ll:
        best_ll = r.fun
        best_T = r.x[0]

deep_cal = np.clip(expit(logit(np.clip(deep_raw, eps, 1-eps)) / best_T), eps, 1-eps)
print(f"\n2. Temperature-Scaled Deep OOF (T={best_T:.4f}):")
print(f"   AUC: {roc_auc_score(y, deep_cal):.4f}")
print(f"   LL:  {log_loss(y, deep_cal):.4f}")
print(f"   Brier: {brier_score_loss(y, deep_cal):.4f}")
print(f"   Mean pred: {deep_cal.mean():.4f}")

# 3. SBR raw
sbr_clipped = np.clip(sbr_raw, eps, 1 - eps)
print(f"\n3. Raw SBR OOF:")
print(f"   AUC: {roc_auc_score(y, sbr_clipped):.4f}")
print(f"   LL:  {log_loss(y, sbr_clipped):.4f}")
print(f"   Brier: {brier_score_loss(y, sbr_clipped):.4f}")

# 4. Current blend (from ship_final.json)
with open(W + r'\ship_final.json', 'r') as f:
    ship = json.load(f)
print(f"\n4. Current Blend (from ship_final.json):")
print(f"   w_deep: {ship['w_deep']}")
print(f"   w_sbr:  {ship['w_sbr']}")
print(f"   T:      {ship['T']}")
print(f"   AUC:    {ship['final_auc']:.4f}")
print(f"   LL:     {ship['final_ll']:.4f}")

# 5. Try re-optimizing blend with calibrated deep
def blend_loss(params):
    wd, T = params
    blend = np.clip(wd * deep_cal + (1 - wd) * sbr_raw, eps, 1 - eps)
    p = np.clip(expit(logit(blend) / T), eps, 1 - eps)
    return log_loss(y, p)

best = None
for init in ([0.9, 0.7], [0.85, 0.8], [0.95, 0.6], [0.88, 1.0], [0.8, 0.5]):
    r = minimize(blend_loss, init, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6})
    if best is None or r.fun < best.fun:
        best = r

wd, T = best.x
blend = np.clip(wd * deep_cal + (1 - wd) * sbr_raw, eps, 1 - eps)
p = np.clip(expit(logit(blend) / T), eps, 1 - eps)
print(f"\n5. Re-optimized Blend (calibrated deep + SBR):")
print(f"   w_deep: {wd:.4f}")
print(f"   w_sbr:  {1-wd:.4f}")
print(f"   T:      {T:.4f}")
print(f"   AUC:    {roc_auc_score(y, p):.4f}")
print(f"   LL:     {log_loss(y, p):.4f}")
print(f"   Brier:  {brier_score_loss(y, p):.4f}")

# 6. Try blend with ONLY calibrated deep (no SBR)
def deep_only_loss(params):
    T = params[0]
    p = np.clip(expit(logit(np.clip(deep_cal, eps, 1-eps)) / T), eps, 1-eps)
    return log_loss(y, p)

r = minimize(deep_only_loss, [0.7], method='Nelder-Mead', options={'maxiter': 200})
T_deep = r.x[0]
p_deep_only = np.clip(expit(logit(np.clip(deep_cal, eps, 1-eps)) / T_deep), eps, 1-eps)
print(f"\n6. Calibrated Deep Only (T={T_deep:.4f}):")
print(f"   AUC: {roc_auc_score(y, p_deep_only):.4f}")
print(f"   LL:  {log_loss(y, p_deep_only):.4f}")

# 7. Prediction correlation
corr = np.corrcoef(deep_raw, sbr_raw)[0, 1]
print(f"\n7. Deep-SBR Correlation: {corr:.4f}")

# 8. Per-class analysis
pos_mask = y == 1
neg_mask = y == 0
print(f"\n8. Per-Class Analysis:")
print(f"   Positive samples: {pos_mask.sum()}")
print(f"   Negative samples: {neg_mask.sum()}")
print(f"   Deep mean (pos): {deep_raw[pos_mask].mean():.4f}")
print(f"   Deep mean (neg): {deep_raw[neg_mask].mean():.4f}")
print(f"   SBR mean (pos):  {sbr_raw[pos_mask].mean():.4f}")
print(f"   SBR mean (neg):  {sbr_raw[neg_mask].mean():.4f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
results = {
    "deep_raw_auc": float(roc_auc_score(y, deep_clipped)),
    "deep_raw_ll": float(log_loss(y, deep_clipped)),
    "deep_cal_T": float(best_T),
    "deep_cal_auc": float(roc_auc_score(y, deep_cal)),
    "deep_cal_ll": float(log_loss(y, deep_cal)),
    "sbr_raw_auc": float(roc_auc_score(y, sbr_clipped)),
    "sbr_raw_ll": float(log_loss(y, sbr_clipped)),
    "current_blend_auc": ship['final_auc'],
    "current_blend_ll": ship['final_ll'],
    "new_blend_wd": float(wd),
    "new_blend_T": float(T),
    "new_blend_auc": float(roc_auc_score(y, p)),
    "new_blend_ll": float(log_loss(y, p)),
    "deep_sbr_corr": float(corr),
}
print(json.dumps(results, indent=2))

# Save results
with open(r'E:\DaT\experiment1_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to E:\\DaT\\experiment1_results.json")
