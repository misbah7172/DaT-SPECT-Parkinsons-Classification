"""V31: HONEST nested evaluation with greedy forward stream selection within outer folds.
Fitting all ~53 weights at once is noisy; selecting a small subset per fold should be
more robust and better-calibrated. Also test subset-size sensitivity.
"""
import os, pickle
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

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
cnn3d = np.load(r'E:\DaT\v30_oof\cnn3d_oof.npy')
cnn25b = np.load(r'E:\DaT\v30_oof\cnn25b_oof.npy')

eps = 1e-6
def lg(M): return np.log(np.clip(M, eps, 1-eps) / np.clip(1-M, eps, 1-eps))

S = np.hstack([oof_a, oof_b, oof_c, blend_d.reshape(-1,1), oof_roi, cnn3d.reshape(-1,1), cnn25b.reshape(-1,1)])
n, p = S.shape
print('streams:', p, 'n:', n)

def platt_fit(p, y):
    l = logit(np.clip(p, eps, 1-eps))
    def loss(q):
        return log_loss(y, np.clip(expit(q[0]*l + q[1]), eps, 1-eps))
    r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
    return r.x[0], r.x[1]

def opt_w(Mt, yt):
    def loss(w):
        return log_loss(yt, np.clip(Mt @ w, eps, 1-eps))
    r = minimize(loss, np.ones(Mt.shape[1]) / Mt.shape[1], method="L-BFGS-B",
                 bounds=[(0, 1)] * Mt.shape[1])
    return r.x / r.x.sum()

def site_cal(p, y, groups):
    res = p.copy()
    for site in np.unique(groups):
        m = groups == site
        if m.sum() > 10 and len(np.unique(y[m])) > 1:
            a, b = platt_fit(p[m], y[m])
            res[m] = expit(a * logit(np.clip(p[m], eps, 1-eps)) + b)
    return res

def greedy_select(St, yt, k):
    """Forward select k streams minimizing log_loss of weighted blend."""
    sel = []
    available = list(range(St.shape[1]))
    best_sub = None
    for _ in range(k):
        best_score = 1e9; best_j = None; best_w = None
        for j in available:
            cand = sel + [j]
            Mt = St[:, cand]
            w = opt_w(Mt, yt)
            score = log_loss(yt, np.clip(Mt @ w, eps, 1-eps))
            if score < best_score:
                best_score = score; best_j = j; best_w = w
        if best_j is None: break
        sel.append(best_j)
        available.remove(best_j)
        best_sub = (sel.copy(), best_w.copy())
    return best_sub

for K in [16, 20]:
    oof = np.zeros(n)
    oof_sc = np.zeros(n)
    for seed in [42, 777, 2024]:
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(S, y, groups):
            sel, w = greedy_select(S[tr], y[tr], K if K else p)
            Mt = S[tr][:, sel]
            Mv = S[va][:, sel]
            pv = Mv @ w
            oof[va] += pv
            pv_sc = site_cal(pv, y[va], groups[va])
            oof_sc[va] += np.clip(pv_sc, eps, 1-eps)
    oof /= 3
    oof_sc /= 3
    tag = f'K={K}' if K else 'K=all'
    print(f'{tag:8s}: AUC={roc_auc_score(y, oof):.4f} LL={log_loss(y, np.clip(oof, eps, 1-eps)):.4f} | SC AUC={roc_auc_score(y, oof_sc):.4f} LL={log_loss(y, oof_sc):.4f}', flush=True)

np.save(r'E:\DaT\v31_honest_oof.npy', oof_sc)



