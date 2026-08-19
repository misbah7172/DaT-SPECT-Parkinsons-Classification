"""V29: HONEST nested evaluation of v26 mega blend (fit weights on outer-train only)."""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

OUT = r'E:\DaT\v26_oof'
y = np.load(os.path.join(OUT, 'y.npy'))
groups = np.load(os.path.join(OUT, 'groups.npy'), allow_pickle=True)

with open(os.path.join(OUT, 'oof_a.pkl'), 'rb') as f:
    A = pickle.load(f)
with open(os.path.join(OUT, 'oof_b.pkl'), 'rb') as f:
    B = pickle.load(f)
with open(os.path.join(OUT, 'oof_c.pkl'), 'rb') as f:
    C = pickle.load(f)
oof_a = np.column_stack(A['oof'])
oof_b = np.column_stack(B['oof'])
oof_c = np.column_stack(C['oof'])
blend_d = np.load(os.path.join(OUT, 'blend_d.npy'))

with open(r'E:\DaT\v30_oof\oof_roi.pkl', 'rb') as f:
    ROI = pickle.load(f)
oof_roi = np.column_stack(ROI['oof'])

eps = 1e-6
def logits(M):
    return np.log(np.clip(M, eps, 1-eps) / np.clip(1-M, eps, 1-eps))

def optim_blend(M, y):
    def loss(w):
        return log_loss(y, np.clip(M @ w, eps, 1-eps))
    r = minimize(loss, np.ones(M.shape[1]) / M.shape[1], method="L-BFGS-B",
                 bounds=[(0, 1)] * M.shape[1])
    return r.x / r.x.sum()

def platt_fit(p, y):
    lg = logit(np.clip(p, eps, 1-eps))
    def loss(q):
        return log_loss(y, np.clip(expit(q[0]*lg + q[1]), eps, 1-eps))
    r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
    return r.x[0], r.x[1]

def site_cal(blend, y, groups):
    result = blend.copy()
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            a, b = platt_fit(blend[mask], y[mask])
            result[mask] = expit(a * logit(np.clip(blend[mask], eps, 1-eps)) + b)
    return result

# streams for mega
M_all = np.hstack([oof_a, oof_b, oof_c, logits(blend_d.reshape(-1,1)), oof_roi])
streams = [oof_a, oof_b, oof_c, blend_d.reshape(-1,1), oof_roi]

# Honest nested: outer 5-fold group CV; fit blend weights + platt + site cal on tr only
oof = np.zeros(len(y))
for seed in [42, 777, 2024]:
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(M_all, y, groups):
        Mt = np.column_stack([s[tr] for s in streams])
        w = optim_blend(Mt, y[tr])
        pt = Mt @ w
        a, b = platt_fit(pt, y[tr])
        Mv = np.column_stack([s[va] for s in streams])
        pv = expit(a * logit(np.clip(Mv @ w, eps, 1-eps)) + b)
        # site cal on outer train predictions
        gtr = groups[tr]
        # fit per-site platt on pt (training)
        pv_sc = pv.copy()
        for site in np.unique(groups):
            mtr = gtr == site
            if mtr.sum() > 10 and len(np.unique(y[tr][mtr])) > 1:
                a2, b2 = platt_fit(pt[mtr], y[tr][mtr])
                pv_sc[groups[va] == site] = expit(a2 * logit(np.clip(pv[groups[va] == site], eps, 1-eps)) + b2)
        oof[va] += pv_sc
    print('seed', seed)
oof /= 3
print(f'\nHONEST nested mega OOF: AUC={roc_auc_score(y, oof):.4f} LL={log_loss(y, np.clip(oof, eps, 1-eps)):.4f}')

# equal-weight baseline (no fitting) for comparison
eq = np.column_stack([oof_a, oof_b, oof_c, blend_d.reshape(-1,1), oof_roi]).mean(axis=1)
print(f'EQUAL-WEIGHT all-stream mean: AUC={roc_auc_score(y, eq):.4f} LL={log_loss(y, np.clip(eq, eps, 1-eps)):.4f}')

# per-site breakdown
for s in np.unique(groups):
    m = groups == s
    if m.sum() < 20 or len(np.unique(y[m])) < 2: continue
    print(f'  {s}: n={m.sum()} AUC={roc_auc_score(y[m], oof[m]):.4f} LL={log_loss(y[m], np.clip(oof[m], eps, 1-eps)):.4f}')

np.save(r'E:\DaT\v29_honest_oof.npy', oof)