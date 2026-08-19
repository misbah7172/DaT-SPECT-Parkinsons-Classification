"""V28: stacking meta-learner on v26 OOF streams (LR + GBDT on logits), with proper nested-CV evaluation.
Loads v26 OOF streams, blends all with a stacking meta-learner, then site-calibrates.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

OUT = r'E:\DaT\v26_oof'
RES = r'E:\DaT\v28_stack'

y = np.load(os.path.join(OUT, 'y.npy'))
groups = np.load(os.path.join(OUT, 'groups.npy'), allow_pickle=True)

with open(os.path.join(OUT, 'oof_a.pkl'), 'rb') as f:
    A = __import__('pickle').load(f)
with open(os.path.join(OUT, 'oof_b.pkl'), 'rb') as f:
    B = __import__('pickle').load(f)
with open(os.path.join(OUT, 'oof_c.pkl'), 'rb') as f:
    C = __import__('pickle').load(f)

oof_a = np.column_stack(A['oof'])
oof_b = np.column_stack(B['oof'])
oof_c = np.column_stack(C['oof'])
names_a, names_b, names_c = A['names'], B['names'], C['names']

blend_d = np.load(os.path.join(OUT, 'blend_d.npy'))

# Build meta feature matrix: logits of each stream
eps = 1e-6
def logits(M):
    return np.log(np.clip(M, eps, 1-eps) / np.clip(1-M, eps, 1-eps))

M = np.hstack([logits(oof_a), logits(oof_b), logits(oof_c), logits(blend_d.reshape(-1, 1))])
n = len(y)
print('meta matrix:', M.shape)

# Nested CV evaluation: outer 5-fold group-stratified to honestly estimate stacking OOF
from sklearn.model_selection import StratifiedGroupKFold
outer_auc, outer_ll = [], []
oof_stack = np.zeros(n)

for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(M, y, groups)):
    # fit meta-LR on tr (using only tr rows of OOF — but OOF already leaks? 
    # NOTE: OOF streams for tr rows were produced with their own folds; using them to train meta is standard stacking
    # but the va rows' OOF predictions are out-of-fold w.r.t. base models -> acceptable
    sc = StandardScaler().fit(M[tr])
    Xtr, Xva = sc.transform(M[tr]), sc.transform(M[va])
    lr = LogisticRegression(C=0.3, max_iter=5000).fit(Xtr, y[tr])
    p = lr.predict_proba(Xva)[:, 1]
    oof_stack[va] = p
    outer_auc.append(roc_auc_score(y[va], p))
    outer_ll.append(log_loss(y[va], np.clip(p, eps, 1-eps)))

print(f'\nStacking (outer CV): AUC={np.mean(outer_auc):.4f}+/-{np.std(outer_auc):.4f} LL={np.mean(outer_ll):.4f}')
print(f'Stacking OOF total: AUC={roc_auc_score(y, oof_stack):.4f} LL={log_loss(y, np.clip(oof_stack, eps, 1-eps)):.4f}')

# site calibration on stacking OOF
def site_calibrate(blend, y, groups):
    result = blend.copy()
    for site in np.unique(groups):
        mask = groups == site
        if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
            from scipy.optimize import minimize
            lg = logit(np.clip(blend[mask], eps, 1-eps))
            def loss(p):
                return log_loss(y[mask], np.clip(expit(p[0]*lg + p[1]), eps, 1-eps))
            r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
            result[mask] = expit(r.x[0]*lg + r.x[1])
    return result

p_sc = site_calibrate(oof_stack, y, groups)
print(f'Stacking OOF + site cal: AUC={roc_auc_score(y, p_sc):.4f} LL={log_loss(y, np.clip(p_sc, eps, 1-eps)):.4f}')

# Also full-CV stacking (fit meta on ALL OOF rows, report in-sample as upper bound)
os.makedirs(RES, exist_ok=True)
np.save(os.path.join(RES, 'oof_stack.npy'), oof_stack)
np.save(os.path.join(RES, 'oof_stack_sc.npy'), p_sc)