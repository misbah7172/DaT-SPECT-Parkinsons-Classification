"""Fit v26 blend honestly on OOF: deep (v52 strong fold-avg) + sbr (oof_a mean of 5)."""
import json, pickle, sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score

W = r'E:\DaT\submission_v26\weights'
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)

deep = np.load(W + r'\fold_oof.npy')
with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names, oofs = A['names'], np.column_stack(A['oof'])
idx = [i for i, n in enumerate(names) if n in ('sbr_lr', 'sbr_xgb', 'sbr_lgb', 'sbr_et', 'sbr_ridge')]
sbr = oofs[:, idx].mean(axis=1)

eps = 1e-6
def blend_loss(params):
    wd, T = params
    blend = np.clip(wd * deep + (1 - wd) * sbr, eps, 1 - eps)
    p = np.clip(expit(logit(blend) / T), eps, 1 - eps)
    return log_loss(y, p)

best = None
for init in ([0.9, 0.7], [0.85, 0.8], [0.95, 0.6], [0.88, 1.0]):
    r = minimize(blend_loss, init, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6})
    if best is None or r.fun < best.fun:
        best = r

wd, T = best.x
wde = np.clip(wd, 1e-4, 1 - 1e-4)
blend = np.clip(wde * deep + (1 - wde) * sbr, eps, 1 - eps)
p = np.clip(expit(logit(blend) / T), eps, 1 - eps)
res = {'w_deep': round(float(wde), 4), 'w_sbr': round(float(1 - wde), 4), 'T': round(float(T), 4),
       'deep_auc': float(roc_auc_score(y, deep)), 'deep_ll': float(log_loss(y, np.clip(deep, eps, 1 - eps))),
       'sbr_auc': float(roc_auc_score(y, sbr)), 'sbr_ll': float(log_loss(y, np.clip(sbr, eps, 1 - eps))),
       'final_auc': float(roc_auc_score(y, p)), 'final_ll': float(log_loss(y, p))}
with open(W + r'\ship_final.json', 'w') as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))