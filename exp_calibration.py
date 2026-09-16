"""Quick experiments to improve LL beyond balanced retrain.
Tests: calibration, more epochs, mixup, focal loss on existing OOF."""
import os, json, pickle
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
from sklearn.preprocessing import LabelEncoder
groups = LabelEncoder().fit_transform(groups_raw.astype(str))

# Load balanced OOF (4 seeds)
bal_oof = np.load(r'E:\DaT\balanced_oof.npy')

# Load SBR OOF
with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names_arr, oofs_arr = A['names'], np.column_stack(A['oof'])
sbr_idx = [i for i, n in enumerate(names_arr) if 'sbr' in n.lower()]
sbr_oof = oofs_arr[:, sbr_idx].mean(axis=1)

# Optimal blend
def blend_loss(params):
    wd, T = params
    b = np.clip(wd * bal_oof + (1-wd) * sbr_oof, 1e-7, 1-1e-7)
    p = np.clip(expit(logit(b) / T), 1e-7, 1-1e-7)
    return log_loss(y, p)

best = None
for init in ([0.78,0.68],[0.8,0.7],[0.85,0.8],[0.75,0.6]):
    r = minimize(blend_loss, init, method='Nelder-Mead', options={'maxiter':500})
    if best is None or r.fun < best.fun: best = r
wd, T = best.x
blend = np.clip(expit(logit(np.clip(wd*bal_oof+(1-wd)*sbr_oof,1e-7,1-1e-7))/T), 1e-7, 1-1e-7)
print(f"Baseline blend: AUC={roc_auc_score(y, blend):.4f}, LL={log_loss(y, blend):.4f}")
print(f"  w_deep={wd:.4f}, w_sbr={1-wd:.4f}, T={T:.4f}")

# ============ EXPERIMENT 1: Per-group temperature calibration ============
print(f"\n{'='*60}")
print(f"EXP 1: Per-group temperature calibration")
print(f"{'='*60}")

# For each group, learn a group-specific temperature
g_temp = {}
for g in np.unique(groups):
    gm = groups == g
    if gm.sum() < 10: continue
    
    def group_temp_loss(T_val):
        p = np.clip(expit(logit(np.clip(blend[gm], 1e-7, 1-1e-7)) / T_val), 1e-7, 1-1e-7)
        return log_loss(y[gm], p)
    
    best_T = 1.0
    best_ll = group_temp_loss(1.0)
    for T_try in np.arange(0.3, 2.0, 0.05):
        ll = group_temp_loss(T_try)
        if ll < best_ll:
            best_ll = ll
            best_T = T_try
    g_temp[g] = best_T
    print(f"  Group {g}: T={best_T:.2f}, n={gm.sum()}, LL_before={group_temp_loss(1.0):.4f}, LL_after={best_ll:.4f}")

# Apply per-group calibration
calibrated = blend.copy()
for g, T_g in g_temp.items():
    gm = groups == g
    calibrated[gm] = np.clip(expit(logit(np.clip(blend[gm], 1e-7, 1-1e-7)) / T_g), 1e-7, 1-1e-7)

cal_auc = roc_auc_score(y, calibrated)
cal_ll = log_loss(y, calibrated)
print(f"\n  Before: LL={log_loss(y, blend):.4f}")
print(f"  After:  LL={cal_ll:.4f}, AUC={cal_auc:.4f}")
print(f"  Improvement: {log_loss(y, blend) - cal_ll:.4f}")

# ============ EXPERIMENT 2: Isotonic regression calibration ============
print(f"\n{'='*60}")
print(f"EXP 2: Isotonic regression (OOF-safe)")
print(f"{'='*60}")

# Use group-aware CV for isotonic fitting
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
iso_cal = np.zeros(len(y))

for trn_idx, val_idx in sgkf.split(np.zeros(len(y)), y, groups):
    # Fit isotonic on train fold
    ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    ir.fit(blend[trn_idx], y[trn_idx])
    iso_cal[val_idx] = ir.predict(blend[val_idx])

iso_auc = roc_auc_score(y, iso_cal)
iso_ll = log_loss(y, np.clip(iso_cal, 1e-7, 1-1e-7))
print(f"  Isotonic: LL={iso_ll:.4f}, AUC={iso_auc:.4f}")

# ============ EXPERIMENT 3: Platt scaling ============
print(f"\n{'='*60}")
print(f"EXP 3: Platt scaling (OOF-safe)")
print(f"{'='*60}")

platt_cal = np.zeros(len(y))
for trn_idx, val_idx in sgkf.split(np.zeros(len(y)), y, groups):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0)
    lr.fit(logit(blend[trn_idx]).reshape(-1, 1), y[trn_idx])
    platt_cal[val_idx] = lr.predict_proba(logit(blend[val_idx]).reshape(-1, 1))[:, 1]

platt_auc = roc_auc_score(y, platt_cal)
platt_ll = log_loss(y, np.clip(platt_cal, 1e-7, 1-1e-7))
print(f"  Platt: LL={platt_ll:.4f}, AUC={platt_auc:.4f}")

# ============ EXPERIMENT 4: Per-group isotonic ============
print(f"\n{'='*60}")
print(f"EXP 4: Per-group isotonic")
print(f"{'='*60}")

pg_iso = np.zeros(len(y))
for g in np.unique(groups):
    gm = groups == g
    if gm.sum() < 20: 
        pg_iso[gm] = blend[gm]
        continue
    
    g_idx = np.where(gm)[0]
    # Split group into train/test
    n = len(g_idx)
    n_trn = int(n * 0.8)
    rng = np.random.RandomState(42)
    perm = rng.permutation(g_idx)
    trn_g = perm[:n_trn]
    val_g = perm[n_trn:]
    
    ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    ir.fit(blend[trn_g], y[trn_g])
    pg_iso[val_g] = ir.predict(blend[val_g])
    # For train portion, use LOO-like approach
    pg_iso[trn_g] = blend[trn_g]  # Keep original for train

pg_iso_auc = roc_auc_score(y, np.clip(pg_iso, 1e-7, 1-1e-7))
pg_iso_ll = log_loss(y, np.clip(pg_iso, 1e-7, 1-1e-7))
print(f"  Per-group isotonic: LL={pg_iso_ll:.4f}, AUC={pg_iso_auc:.4f}")

# ============ EXPERIMENT 5: Optimal threshold analysis ============
print(f"\n{'='*60}")
print(f"EXP 5: Prediction statistics")
print(f"{'='*60}")
print(f"  Mean prediction: {blend.mean():.4f}")
print(f"  Class 0 mean: {blend[y==0].mean():.4f}")
print(f"  Class 1 mean: {blend[y==1].mean():.4f}")
print(f"  Fraction > 0.5: {(blend > 0.5).mean():.4f}")

# ============ SUMMARY ============
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  v26 baseline:      LL=0.3077")
print(f"  Balanced blend:    LL={log_loss(y, blend):.4f}")
print(f"  Per-group temp:    LL={cal_ll:.4f}")
print(f"  Isotonic:          LL={iso_ll:.4f}")
print(f"  Platt:             LL={platt_ll:.4f}")
print(f"  Best:              LL={min(cal_ll, iso_ll, platt_ll):.4f}")
