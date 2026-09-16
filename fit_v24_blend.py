"""Fit v24 blend constants honestly on OOF only (no test/target leakage).
Deep stream = fold-averaged CNN OOF (submission_v24/weights/fold_oof.npy).
SBR stream = mean of the 5 sbr classifier OOFs from v26_oof/oof_a.pkl
(lr, xgb, lgb, et, ridge) — the tabular ensemble used at submission time.
Optimizes [w_deep, w_sbr=1-w_deep, T] to minimize log loss on the OOF blend,
then writes submission_v24/weights/ship_final.json.
"""
import json, pickle
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score

W = r'E:\DaT\submission_v24\weights'
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
    w = wd
    blend = w * deep + (1 - w) * sbr
    blend = np.clip(blend, eps, 1 - eps)
    p = expit(logit(blend) / T)
    p = np.clip(p, eps, 1 - eps)
    return log_loss(y, p)


best = None
for init in ([0.89, 0.75], [0.85, 0.9], [0.93, 0.6]):
    r = minimize(blend_loss, init, method='Nelder-Mead',
                 options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6})
    if best is None or r.fun < best.fun:
        best = r

wd, T = best.x
wde = np.clip(wd, 1e-4, 1 - 1e-4)
blend = wde * deep + (1 - wde) * sbr
blend = np.clip(blend, eps, 1 - eps)
p = expit(logit(blend) / T)
p = np.clip(p, eps, 1 - eps)
res = {'w_deep6': round(float(wde), 4), 'w_sbr': round(float(1 - wde), 4),
       'T': round(float(T), 4),
       'deep_auc': float(roc_auc_score(y, deep)), 'deep_ll': float(log_loss(y, np.clip(deep, eps, 1 - eps))),
       'sbr_auc': float(roc_auc_score(y, sbr)), 'sbr_ll': float(log_loss(y, np.clip(sbr, eps, 1 - eps))),
       'blend_auc': float(roc_auc_score(y, blend)), 'blend_ll': float(log_loss(y, np.clip(blend, eps, 1 - eps))),
       'final_auc': float(roc_auc_score(y, p)), 'final_ll': float(log_loss(y, p))}
with open(W + r'\ship_final.json', 'w') as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))